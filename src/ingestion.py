"""建库：DashScope 向量化 + FAISS 统一索引 + BM25 关键词索引。

对原方案的改动：
- text-embedding-v1 → text-embedding-v3（1024 维，语义更强）；
- 原方案「一家公司一个库」，本知识库 9 份报告同属中芯国际，等于只查一个库；
  这里改为**全局统一索引**，文档名/页码写进 chunk 元数据，检索时可跨研报召回；
- embedding 结果按文本哈希落盘缓存，重复建库不再重复调用 API。
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.chunking import n_tokens
from src.config import CHUNK_DIR, INDEX_DIR, SETTINGS
from src.llm import LLMClient

def _cache_path(settings) -> Path:
    """缓存按「模型 + 维度」隔离：换 embedding 模型后不能复用旧向量。"""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", settings.embedding_model)
    return INDEX_DIR / f"embed_cache_{slug}_{settings.embedding_dim}.npz"


def _faiss_write(index: faiss.Index, path: Path) -> None:
    """faiss 的 C++ 接口用窄字符 fopen，Windows 下遇到中文路径会写失败，
    所以这里序列化成 numpy 数组再用 Python 的文件对象落盘。"""
    np.save(path, faiss.serialize_index(index))


def _faiss_read(path: Path) -> faiss.Index:
    return faiss.deserialize_index(np.load(path))


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def tokenize_zh(text: str) -> List[str]:
    """中文按单字 + 二元组切分，英文/数字按词切分，兼顾关键词召回。"""
    low = text.lower()
    latin = re.findall(r"[a-z][a-z0-9\.\-]*|\d+(?:\.\d+)?", low)
    cjk = re.findall(r"[\u4e00-\u9fff]", low)
    bigrams = ["".join(cjk[i : i + 2]) for i in range(len(cjk) - 1)]
    return latin + cjk + bigrams


class _Cache:
    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self.data: Dict[str, np.ndarray] = {}
        self.dirty = False
        if enabled and path.exists():
            with np.load(path, allow_pickle=True) as z:
                for k in z.files:
                    self.data[k] = z[k]

    def get(self, text: str):
        return self.data.get(_hash(text))

    def put(self, text: str, vec) -> None:
        if self.enabled:
            self.data[_hash(text)] = np.asarray(vec, dtype=np.float32)
            self.dirty = True

    def save(self) -> None:
        if self.enabled and self.dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.path, **self.data)


def load_chunks(chunk_dir: Path = CHUNK_DIR) -> List[Dict]:
    chunks: List[Dict] = []
    for path in sorted(chunk_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(data.get("chunks", []))
    if not chunks:
        raise FileNotFoundError(f"{chunk_dir} 下没有分块结果，请先执行 chunk")
    # 重新编号，保证与向量矩阵行号一致
    for i, c in enumerate(chunks):
        c["row"] = i
    return chunks


def embed_chunks(chunks: List[Dict], client: LLMClient, settings=None) -> np.ndarray:
    s = settings or SETTINGS
    cache = _Cache(_cache_path(s), s.use_cache)
    todo, todo_idx = [], []
    vectors: List[np.ndarray] = []
    hit = 0

    for i, c in enumerate(chunks):
        vec = cache.get(c["text"])
        if vec is not None:
            hit += 1
            vectors.append(vec)
        else:
            vectors.append(None)  # type: ignore[arg-type]
            todo.append(c["text"])
            todo_idx.append(i)

    print(f"embedding 缓存命中 {hit}/{len(chunks)}")
    if todo:
        bs = 64
        for i in tqdm(range(0, len(todo), bs), desc="embedding", unit="batch"):
            batch = todo[i : i + bs]
            embs = client.embed(batch)
            for j, emb in enumerate(embs):
                vectors[todo_idx[i + j]] = np.asarray(emb, dtype=np.float32)
                cache.put(batch[j], emb)
    cache.save()

    arr = np.vstack([v for v in vectors]).astype(np.float32)
    faiss.normalize_L2(arr)  # 归一化后内积等价于余弦相似度
    return arr


def build_index(force: bool = False, settings=None) -> Tuple[faiss.Index, List[Dict]]:
    s = settings or SETTINGS
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = INDEX_DIR / "chunks.json"
    faiss_path = INDEX_DIR / "faiss.index.npy"
    bm25_path = INDEX_DIR / "bm25.pkl"

    if not force and meta_path.exists() and faiss_path.exists() and bm25_path.exists():
        chunks = json.loads(meta_path.read_text(encoding="utf-8"))
        index = _faiss_read(faiss_path)
        print(f"跳过建库（已存在）：{len(chunks)} 块 / {index.ntotal} 向量")
        return index, chunks

    chunks = load_chunks()
    client = LLMClient(s)
    print(f"待向量化文本块：{len(chunks)}")
    arr = embed_chunks(chunks, client, s)

    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    _faiss_write(index, faiss_path)

    # BM25：需要把语料一起序列化，rank_bm25 本身不保存 corpus
    corpus = [tokenize_zh(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus}, f)

    meta_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "index_meta.json").write_text(
        json.dumps({"embedding_backend": s.embedding_backend, "embedding_model": s.embedding_model,
                    "dim": int(arr.shape[1]), "chunks": len(chunks)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"索引已保存：{faiss_path}（{index.ntotal} 条向量，维度 {arr.shape[1]}）")
    return index, chunks


def load_index() -> Tuple[faiss.Index, List[Dict], Dict]:
    faiss_path = INDEX_DIR / "faiss.index.npy"
    meta_path = INDEX_DIR / "chunks.json"
    bm25_path = INDEX_DIR / "bm25.pkl"
    if not (faiss_path.exists() and meta_path.exists() and bm25_path.exists()):
        raise FileNotFoundError("索引不存在，请先执行 python main.py index")
    index = _faiss_read(faiss_path)
    chunks = json.loads(meta_path.read_text(encoding="utf-8"))
    with open(bm25_path, "rb") as f:
        bm = pickle.load(f)
    return index, chunks, bm
