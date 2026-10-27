"""检索：向量 + BM25 混合召回 → 页级去重 → LLM 重排。

在标准混合检索流程上做了两处适配多文档的改动：
- 召回后按 (文档, 页码) 聚合，同一页只保留得分最高的块；
- 上下文用「整页」而非「块」拼装（父页面检索），并在每页前标注文档名与页码。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import INDEX_DIR, PARSED_DIR, SETTINGS
from src.ingestion import load_index, tokenize_zh
from src.llm import LLMClient

RERANK_SYSTEM = """你是一个 RAG 检索重排专家。
你将收到一个查询和若干候选文本块（编号 C1、C2……），请分别判断每个文本块与查询的相关性。

评分说明：
1. reasoning：分析该文本块与查询的关系，简要说明理由（30 字以内）。
2. relevance_score：0-1 的分数，步长 0.1。
   0 = 完全无关；0.3 = 略有相关；0.5 = 一般相关；0.7 = 相关；0.9 = 高度相关；1 = 完全匹配。
3. 只基于内容客观评价，不做假设；不要遗漏任何一个编号。

严格输出 JSON：
{"block_rankings": [{"id": "C1", "reasoning": "...", "relevance_score": 0.8}, ...]}
"""


def _minmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.ones_like(scores)
    return (scores - lo) / (hi - lo)


class Retriever:
    def __init__(self, settings=None) -> None:
        self.s = settings or SETTINGS
        self.index, self.chunks, self.bm = load_index()
        self.client = LLMClient(self.s)
        self._pages = self._load_pages()
        # 索引用了哪个 embedding 模型？换了模型（或换了后端）后向量空间不通用，
        # 这里做一致性校验，不匹配就自动降级为 BM25，避免"检索出莫名其妙的页"。
        self.vector_ok = self.s.retrieval_mode != "bm25"
        meta_path = INDEX_DIR / "index_meta.json"
        if self.vector_ok and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            built_with = meta.get("embedding_model")
            if built_with and built_with != self.s.embedding_model:
                print(
                    f"[warn] 索引是用 {built_with} 建的，当前配置是 {self.s.embedding_model}，"
                    f"向量空间不一致，本次改用 BM25 召回；如需向量召回请重跑 python main.py index --force"
                )
                self.vector_ok = False

    # ------------------------------------------------------------------ 页面
    @staticmethod
    def _load_pages() -> Dict[str, Dict[int, Dict]]:
        """doc_id → {页码: {text, doc_name, broker}}，用于父页面检索。"""
        pages: Dict[str, Dict[int, Dict]] = {}
        for path in sorted(PARSED_DIR.glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            meta = doc["metainfo"]
            pages[meta["sha1"]] = {
                p["page"]: {
                    "text": p["text"],
                    "doc_name": meta["doc_title"],
                    "broker": meta.get("broker", ""),
                    "company_name": meta.get("company_name", ""),
                }
                for p in doc["content"]["pages"]
            }
        return pages

    # ------------------------------------------------------------------ 召回
    def recall(self, query: str, k: int = None) -> List[Tuple[int, float]]:
        """向量 + BM25 混合召回，返回 [(chunk 行号, 融合分数), ...]。"""
        k = k or self.s.recall_k
        vec_scores: Dict[int, float] = {}
        if not self.vector_ok:
            pass  # 显式 BM25 模式，或索引与当前 embedding 模型不匹配
        else:
            try:
                qvec = np.asarray(self.client.embed_query(query), dtype=np.float32).reshape(1, -1)
                norm = float(np.linalg.norm(qvec))
                if norm > 0:
                    qvec = qvec / norm
                sims, idx = self.index.search(qvec, min(k, self.index.ntotal))
                vec_scores = {int(i): float(s) for i, s in zip(idx[0], sims[0])}
            except Exception as exc:
                # embedding 服务不可用（额度/网络）时降级为纯 BM25，保证问答链路不中断
                self.vector_ok = False
                print(f"[warn] 向量召回不可用，降级为 BM25 单路召回：{type(exc).__name__}")

        bm = self.bm["bm25"]
        bm_scores = np.asarray(bm.get_scores(tokenize_zh(query)), dtype=np.float32)
        top_bm = np.argsort(bm_scores)[::-1][: max(k, 50)]
        bm_top = {int(i): float(bm_scores[i]) for i in top_bm if bm_scores[i] > 0}

        rows = sorted(set(vec_scores) | set(bm_top))
        if not rows:
            return []
        # 向量分数直接用余弦（截断到 0 以上），BM25 按本批最大值归一化到 [0,1]，
        # 这样融合分数仍可解释，也不会把边缘候选强行提到 1.0。
        v = np.array([max(0.0, vec_scores.get(r, 0.0)) for r in rows], dtype=np.float32)
        braw = np.array([bm_top.get(r, 0.0) for r in rows], dtype=np.float32)
        b = braw / braw.max() if braw.max() > 0 else braw
        fused = self.s.vector_weight * v + self.s.bm25_weight * b
        order = np.argsort(fused)[::-1][:k]
        return [(rows[i], float(fused[i])) for i in order]

    # -------------------------------------------------------------- 页级聚合
    def aggregate_pages(self, candidates: List[Tuple[int, float]], limit: int = None) -> List[Dict]:
        limit = limit or self.s.page_candidates
        best: Dict[Tuple[str, int], Dict] = {}
        for row, score in candidates:
            c = self.chunks[row]
            key = (c["doc_id"], c["page"])
            if key not in best or score > best[key]["recall_score"]:
                page_info = self._pages.get(c["doc_id"], {}).get(c["page"], {})
                best[key] = {
                    "doc_id": c["doc_id"],
                    "doc_name": c.get("doc_name") or page_info.get("doc_name", ""),
                    "broker": c.get("broker") or page_info.get("broker", ""),
                    "page": c["page"],
                    "chunk_text": c["text"],
                    "page_text": page_info.get("text", c["text"]),
                    "recall_score": score,
                }
        pages = sorted(best.values(), key=lambda x: x["recall_score"], reverse=True)
        return pages[:limit]

    # ------------------------------------------------------------------ 重排
    def rerank(self, query: str, pages: List[Dict]) -> List[Dict]:
        if not self.s.force_rerank or len(pages) <= 1:
            for p in pages:
                p["llm_score"] = None
                p["final_score"] = p["recall_score"]
            return pages

        blocks = []
        for i, p in enumerate(pages, 1):
            body = p["page_text"][:2400]
            blocks.append(f"C{i}\n[《{p['doc_name']}》第{p['page']}页]\n{body}")
        user = "查询：{q}\n\n候选文本块：\n\n{blocks}".format(q=query, blocks="\n\n".join(blocks))

        scores: Dict[int, float] = {}
        try:
            data = self.client.chat_json(RERANK_SYSTEM, user, model=self.s.rerank_model)
            for item in data.get("block_rankings", []):
                sid = str(item.get("id", "")).upper().lstrip("C")
                if sid.isdigit():
                    idx = int(sid) - 1
                    if 0 <= idx < len(pages):
                        scores[idx] = max(0.0, min(1.0, float(item.get("relevance_score", 0) or 0)))
        except Exception as exc:  # 重排失败不应阻断问答
            print(f"[warn] LLM 重排失败，退回召回排序：{exc}")

        for i, p in enumerate(pages):
            llm_score = scores.get(i)
            p["llm_score"] = llm_score
            p["final_score"] = (
                p["recall_score"]
                if llm_score is None
                else self.s.llm_rerank_weight * llm_score + self.s.retrieval_weight * p["recall_score"]
            )
        return sorted(pages, key=lambda x: x["final_score"], reverse=True)

    # ------------------------------------------------------------------ 对外
    def retrieve(self, query: str, top_n: int = None, verbose: bool = True) -> Dict:
        t0 = time.time()
        top_n = top_n or self.s.top_n_context
        candidates = self.recall(query)
        t1 = time.time()
        pages = self.aggregate_pages(candidates)
        t2 = time.time()
        pages = self.rerank(query, pages)
        t3 = time.time()
        result = pages[:top_n]
        for p in result:
            p["page_text"] = p["page_text"][:2600]  # 控制 prompt 体积
        timings = {"召回": t1 - t0, "页面聚合": t2 - t1, "LLM重排": t3 - t2}
        if verbose:
            print(
                f"[检索] 候选 {len(candidates)} 块 → {len(pages)} 页 → 取前 {len(result)} 页；"
                f"耗时 召回{t1-t0:.1f}s / 重排{t3-t2:.1f}s"
            )
        return {"pages": result, "candidates": len(candidates), "timings": timings}
