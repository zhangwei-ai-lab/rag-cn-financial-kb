"""全局配置：路径、模型、检索参数。

所有可调参数集中在 Settings 中，既可用环境变量覆盖，也可在代码里直接修改，
便于在不改动业务代码的前提下做参数对比实验。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录下的 .env

# 项目根目录（src 的上一级）
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "stock_data"

PDF_DIR = DATA_DIR / "pdf_reports"      # 原始 PDF 知识库
PARSED_DIR = DATA_DIR / "parsed"        # 01 解析结果：逐页 Markdown JSON
CHUNK_DIR = DATA_DIR / "chunks"         # 02 分块结果：带页码的 chunk JSON
INDEX_DIR = DATA_DIR / "index"          # 03 索引：faiss / bm25 / chunk 元数据
ANSWER_DIR = DATA_DIR / "answers"       # 04 问答结果
MINERU_DIR = DATA_DIR / "mineru"        # MinerU 本地解析产物（可选）


@dataclass
class Settings:
    """一次运行的全部超参数。改这里就能做消融实验。"""

    # ---- 模型（DashScope / 本地 Ollama）----
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    # 对话后端：dashscope | ollama | openai
    llm_backend: str = field(default_factory=lambda: (os.getenv("LLM_BACKEND") or "dashscope").lower())
    # 向量化后端：dashscope | ollama（本地 OpenAI 兼容端点）
    embedding_backend: str = field(default_factory=lambda: (os.getenv("EMBEDDING_BACKEND") or "dashscope").lower())
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"))
    # Qwen3 系列的思维链开关：关掉后单次响应从 ~16s 降到 ~1s
    ollama_think: bool = field(default_factory=lambda: os.getenv("OLLAMA_THINK", "0") == "1")

    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-v3"))
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    embed_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBED_BATCH_SIZE", "8")))
    answer_model: str = field(default_factory=lambda: os.getenv("ANSWER_MODEL", "qwen-plus"))
    rerank_model: str = field(default_factory=lambda: os.getenv("RERANK_MODEL", "qwen-turbo-latest"))

    # ---- 解析 ----
    parser: str = field(default_factory=lambda: os.getenv("PARSER", "auto"))  # auto / pymupdf / mineru
    mineru_api_key: str = field(default_factory=lambda: os.getenv("MINERU_API_KEY", ""))
    mineru_ocr: bool = field(default_factory=lambda: os.getenv("MINERU_OCR", "0") == "1")
    mineru_language: str = field(default_factory=lambda: os.getenv("MINERU_LANGUAGE", "ch"))

    # ---- 分块 ----
    # bge-small-zh 最长只吃 512 token，中文按字切分，故本地模型下建议 300
    chunk_tokens: int = field(default_factory=lambda: int(os.getenv("CHUNK_TOKENS", "350")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "60")))

    # ---- 检索 ----
    # hybrid：向量 + BM25 混合召回；bm25：只用关键词召回（无需 embedding，可完全离线）
    retrieval_mode: str = field(default_factory=lambda: (os.getenv("RETRIEVAL_MODE") or "hybrid").lower())
    recall_k: int = field(default_factory=lambda: int(os.getenv("RECALL_K", "40")))
    page_candidates: int = field(default_factory=lambda: int(os.getenv("PAGE_CANDIDATES", "12")))
    rerank_sample: int = 8
    top_n_context: int = field(default_factory=lambda: int(os.getenv("TOP_N_CONTEXT", "6")))
    vector_weight: float = 0.7  # 混合检索中向量分数权重
    bm25_weight: float = 0.3    # 混合检索中 BM25 分数权重
    llm_rerank_weight: float = 0.7  # 重排时 LLM 分数权重（原方案 0.7）
    retrieval_weight: float = 0.3   # 重排时原始召回分数权重

    # ---- 其他 ----
    temperature: float = 0.2
    use_cache: bool = True      # embedding 结果本地缓存，重复建库不重复花钱
    force_rerank: bool = True   # 是否启用 LLM 重排

    def require_key(self) -> None:
        """只有真正用到云端服务时才校验 Key；全本地部署可完全不填。"""
        needs_cloud = "dashscope" in (self.llm_backend, self.embedding_backend)
        if needs_cloud and not self.api_key:
            raise RuntimeError(
                "当前配置需要 DashScope：请复制 .env.example 为 .env 并填入百炼 API Key，"
                "或设置 LLM_BACKEND=ollama / EMBEDDING_BACKEND=ollama 改用本地模型。"
            )


SETTINGS = Settings()
