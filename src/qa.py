"""问答链路：检索 → 拼装上下文 → 结构化生成 → 证据回填。

answer() 是单一入口，CLI 与 Streamlit 界面都调用它，保证两条路径行为一致
（原项目里 pipeline.process_questions 和界面各写了一套解析逻辑，容易走偏）。
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src import prompts
from src.config import ANSWER_DIR, SETTINGS
from src.llm import USAGE
from src.retrieval import Retriever


def _snippet(text: str, limit: int = 160) -> str:
    """证据片段：MinerU 的表格是 HTML，界面上直接显示 <table><tr> 很难看，
    这里剥掉标签只留可读文本（送给模型的上下文仍然保留 HTML，不影响表格理解）。"""
    plain = re.sub(r"<[^>]+>", " ", text)
    plain = plain.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    plain = re.sub(r"\s+", " ", plain).strip()
    return plain[:limit]


class QAEngine:
    def __init__(self, settings=None) -> None:
        self.s = settings or SETTINGS
        self.retriever = Retriever(self.s)
        self.client = self.retriever.client

    # ---------------------------------------------------------------- 证据回填
    @staticmethod
    def _resolve_citations(raw_citations, pages: List[Dict]) -> List[Dict]:
        refs: List[Dict] = []
        seen = set()
        for item in raw_citations or []:
            m = re.search(r"C?\s*(\d+)", str(item), re.IGNORECASE)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if not (0 <= idx < len(pages)) or idx in seen:
                continue
            seen.add(idx)
            p = pages[idx]
            refs.append(
                {
                    "citation": f"C{idx + 1}",
                    "doc_name": p["doc_name"],
                    "broker": p.get("broker", ""),
                    "page": p["page"],
                    "snippet": _snippet(p["page_text"]),
                    "score": round(float(p.get("final_score", 0.0)), 4),
                }
            )
        if not refs and pages:  # 模型没给引用时，回填分数最高的页面，保证可溯源
            p = pages[0]
            refs.append(
                {
                    "citation": "C1",
                    "doc_name": p["doc_name"],
                    "broker": p.get("broker", ""),
                    "page": p["page"],
                    "snippet": _snippet(p["page_text"]),
                    "score": round(float(p.get("final_score", 0.0)), 4),
                }
            )
        return refs

    # ------------------------------------------------------------------- 单问
    def answer(self, question: str, kind: str = "string", top_n: Optional[int] = None, verbose: bool = True) -> Dict:
        t0 = time.time()
        retrieved = self.retriever.retrieve(question, top_n=top_n, verbose=verbose)
        pages = retrieved["pages"]
        context = prompts.format_context(pages)
        t1 = time.time()

        spec = prompts.build_prompts(kind)
        raw = self.client.chat_json(
            spec["system"],
            spec["user_template"].format(context=context, question=question),
            model=self.s.answer_model,
        )
        t2 = time.time()

        answer_text = raw.get("final_answer", "N/A")
        result = {
            "question": question,
            "kind": spec["kind"],
            "final_answer": answer_text,
            "reasoning_summary": raw.get("reasoning_summary", ""),
            "step_by_step_analysis": raw.get("step_by_step_analysis", ""),
            "references": self._resolve_citations(raw.get("relevant_citations"), pages),
            "timings": {
                **retrieved["timings"],
                "生成": t2 - t1,
                "总计": t2 - t0,
            },
            "usage": USAGE.summary(),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        if verbose:
            print(f"[回答] 类型={spec['kind']} 总耗时 {t2 - t0:.1f}s")
        return result

    # ------------------------------------------------------------------- 批量
    def answer_batch(self, questions_path: Path, out_path: Optional[Path] = None, kind_default: str = "string") -> Path:
        questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
        out_path = Path(out_path) if out_path else ANSWER_DIR / "answers.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        for i, q in enumerate(questions, 1):
            text = q["text"] if isinstance(q, dict) else str(q)
            kind = (q.get("kind") if isinstance(q, dict) else None) or kind_default
            print(f"\n===== 问题 {i}/{len(questions)}（kind={kind}）=====\n{text}")
            try:
                results.append(self.answer(text, kind=kind))
            except Exception as exc:
                print(f"[error] 该问题失败：{exc}")
                results.append({"question": text, "kind": kind, "final_answer": "N/A", "error": str(exc)})

        payload = {
            "meta": {
                "embedding_model": self.s.embedding_model,
                "answer_model": self.s.answer_model,
                "rerank_model": self.s.rerank_model,
                "chunk_tokens": self.s.chunk_tokens,
                "top_n_context": self.s.top_n_context,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "answers": results,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n答案已保存：{out_path}")
        return out_path
