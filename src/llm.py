"""DashScope（阿里云百炼）统一封装：对话、结构化 JSON 输出、文本向量化。

原项目直接调用 openai / dashscope 两套 SDK，错误处理分散在各模块；
这里统一收口，并加入重试、限速与 token 统计，便于观察调用成本。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import dashscope
import requests
from dashscope import Generation, TextEmbedding
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import SETTINGS

log = logging.getLogger(__name__)


class Usage:
    """粗略统计调用量与耗时，供界面展示。"""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.embed_tokens = 0
        self.calls = 0

    def add_chat(self, usage: Dict[str, Any]) -> None:
        self.calls += 1
        # DashScope 用 input/output_tokens，OpenAI 兼容端点（含 Ollama）用 prompt/completion_tokens
        self.prompt_tokens += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        self.completion_tokens += int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)

    def add_embed(self, usage: Dict[str, Any]) -> None:
        self.calls += 1
        self.embed_tokens += int(usage.get("total_tokens", 0) or 0)

    def summary(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embed_tokens": self.embed_tokens,
        }


USAGE = Usage()


class LLMClient:
    """对话 + 向量化客户端，对话与向量化可分别选后端。

    对话后端（LLM_BACKEND）：
      dashscope —— 百炼原生 SDK（qwen-plus / qwen-turbo）
      openai    —— 任意 OpenAI 兼容端点（DeepSeek、百炼 compatible-mode、vLLM…）
      ollama    —— 本地 Ollama 的 OpenAI 兼容接口（http://127.0.0.1:11434/v1）
    向量化后端（EMBEDDING_BACKEND）：
      dashscope —— text-embedding-v3
      ollama    —— 本地 Ollama 的 /v1/embeddings（如 qwen3-embedding:0.6b、bge-m3）
    """

    def __init__(self, settings=None) -> None:
        self.s = settings or SETTINGS
        self.backend = self.s.llm_backend
        self.embed_backend = self.s.embedding_backend

        if self.backend == "ollama":
            self.base_url = self.s.ollama_base_url.rstrip("/")
            self.api_key = "ollama"
        elif self.backend == "openai":
            self.base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
            self.api_key = os.getenv("OPENAI_API_KEY", "")
            if not self.base_url or not self.api_key:
                raise RuntimeError("LLM_BACKEND=openai 时需要设置 OPENAI_BASE_URL 与 OPENAI_API_KEY")
        else:
            self.base_url = ""
            self.api_key = ""

        if self.embed_backend == "ollama":
            self.embed_base_url = self.s.ollama_base_url.rstrip("/")
            self.embed_api_key = "ollama"
        elif self.embed_backend == "openai":
            self.embed_base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
            self.embed_api_key = os.getenv("OPENAI_API_KEY", "")
        else:
            self.embed_base_url = ""
            self.embed_api_key = ""
        self._st_model = None  # sentence-transformers 懒加载

        self.s.require_key()
        dashscope.api_key = self.s.api_key
        # embedding 服务故障后的冷却期，避免每次检索都白等 4 次重试
        self._embed_down_until = 0.0

    # ------------------------------------------------------------------ chat
    @retry(wait=wait_exponential(multiplier=3, min=4, max=60), stop=stop_after_attempt(6))
    def chat(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """发起一次对话，返回纯文本内容。json_mode 时请求 JSON 对象格式。"""
        model = model or self.s.answer_model
        temp = self.s.temperature if temperature is None else temperature
        if self.backend == "dashscope":
            content, usage = self._chat_dashscope(system, user, model, json_mode, temp)
        else:
            content, usage = self._chat_openai_compatible(system, user, model, json_mode, temp)
        USAGE.add_chat(usage)
        if not content.strip():
            raise RuntimeError("模型返回空内容")
        return content

    def _chat_dashscope(self, system, user, model, json_mode, temperature):
        kwargs: Dict[str, Any] = {
            "api_key": self.s.api_key,
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "result_format": "message",
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = Generation.call(**kwargs)
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope 调用失败: {resp.status_code} {resp.code} {resp.message}")
        msg = resp["output"]["choices"][0]["message"]
        content = msg.get("content") or ""
        # 部分模型在 json_mode 下会把内容放到 reasoning_content
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content, resp.get("usage", {}) or {}

    def _chat_openai_compatible(self, system, user, model, json_mode, temperature):
        # Qwen3 系列默认会先输出思维链（实测单次 16s→关掉后 1s），推理/抽取任务用 /no_think 足够
        if self.backend == "ollama" and not self.s.ollama_think and "/no_think" not in system:
            system = system.rstrip() + "\n/no_think"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"{self.backend} 调用失败: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        content = data["choices"][0]["message"].get("content") or ""
        # 兜底：把内联的思考块去掉（不同 Ollama 版本有的放 reasoning 字段，有的内联）
        content = re.sub(r"</?think>.*?(?=(</think>|$))", "", content, flags=re.S).replace("</think>", "")
        if not content.strip():
            content = data["choices"][0]["message"].get("reasoning") or ""
        return content, data.get("usage", {}) or {}

    def chat_json(self, system: str, user: str, model: Optional[str] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        """要求模型返回 JSON 并解析；解析失败用 json_repair 兜底。"""
        raw = self.chat(system, user, model=model, json_mode=True, temperature=temperature)
        return parse_json(raw)

    # --------------------------------------------------------------- embedding
    @retry(wait=wait_exponential(multiplier=2, min=2, max=20), stop=stop_after_attempt(4))
    def _embed_dashscope(self, texts: List[str]) -> List[List[float]]:
        resp = TextEmbedding.call(
            api_key=self.s.api_key,
            model=self.s.embedding_model,
            input=texts,
            dimension=self.s.embedding_dim,
        )
        if resp.status_code != 200:
            # 兼容部分模型不支持 dimension 参数
            resp = TextEmbedding.call(api_key=self.s.api_key, model=self.s.embedding_model, input=texts)
        if resp.status_code != 200:
            raise RuntimeError(f"embedding 失败: {resp.status_code} {resp.code} {resp.message}")
        USAGE.add_embed(resp.get("usage", {}) or {})
        embs = resp["output"]["embeddings"]
        embs.sort(key=lambda e: e.get("text_index", 0))
        out = []
        for e in embs:
            vec = e.get("embedding")
            if not vec:
                raise RuntimeError("DashScope 返回的 embedding 为空")
            out.append(vec)
        if len(out) != len(texts):
            raise RuntimeError(f"embedding 数量不匹配: 期望 {len(texts)}，实际 {len(out)}")
        return out

    @retry(wait=wait_exponential(multiplier=2, min=2, max=20), stop=stop_after_attempt(4))
    def _embed_openai_compatible(self, texts: List[str]) -> List[List[float]]:
        """本地 Ollama / 任意 OpenAI 兼容 /v1/embeddings。"""
        resp = requests.post(
            f"{self.embed_base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.embed_api_key}", "Content-Type": "application/json"},
            json={"model": self.s.embedding_model, "input": texts},
            timeout=600,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"本地 embedding 失败: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        items = data.get("data", [])
        items.sort(key=lambda d: d.get("index", 0))
        out = []
        for item in items:
            vec = item.get("embedding") or []
            if not vec:
                raise RuntimeError("本地 embedding 返回空向量")
            out.append(vec)
        if len(out) != len(texts):
            raise RuntimeError(f"embedding 数量不匹配: 期望 {len(texts)}，实际 {len(out)}")
        USAGE.add_embed(data.get("usage", {}) or {"total_tokens": 0})
        return out

    def _embed_st(self, texts: List[str]) -> List[List[float]]:
        """本地 sentence-transformers（如 BAAI/bge-base-zh-v1.5），纯 CPU、无需联网。"""
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.s.embedding_model, device="cpu")
        vecs = self._st_model.encode(
            texts, batch_size=16, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False
        )
        USAGE.calls += 1
        return [v.tolist() for v in vecs]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """分批向量化，自动跳过空串输入。"""
        if time.time() < self._embed_down_until:
            raise RuntimeError("embedding 服务暂不可用（冷却中，稍后自动恢复）")
        results: List[List[float]] = []
        bs = max(1, self.s.embed_batch_size)
        fn = {
            "dashscope": self._embed_dashscope,
            "ollama": self._embed_openai_compatible,
            "openai": self._embed_openai_compatible,
            "st": self._embed_st,
        }.get(self.embed_backend)
        if fn is None:
            raise RuntimeError(f"未知的 EMBEDDING_BACKEND: {self.embed_backend}")
        try:
            for i in range(0, len(texts), bs):
                batch = [t if t.strip() else "（空）" for t in texts[i : i + bs]]
                results.extend(fn(batch))
                if i + bs < len(texts) and self.embed_backend == "dashscope":
                    time.sleep(0.2)  # 云端轻微限速；本地不需要
        except Exception:
            self._embed_down_until = time.time() + 120
            raise
        return results

    def embed_query(self, query: str) -> List[float]:
        """查询向量。bge 系列中文模型建议给查询加指令前缀，文档侧不加。"""
        if self.embed_backend == "st" and "bge" in self.s.embedding_model.lower():
            return self.embed([f"为这个句子生成表示以用于检索相关文章：{query}"])[0]
        return self.embed([query])[0]


def parse_json(raw: str) -> Dict[str, Any]:
    """从模型输出中稳健地取出 JSON 对象。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        # 去掉 ```json ... ``` 包裹
        text = text.split("```")[1] if text.count("```") >= 2 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return json.loads(repair_json(candidate))
    return json.loads(repair_json(text))
