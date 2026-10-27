"""页码感知的文本分块。

原改造版 split_markdown_file 只按「30 行」切分，chunk 里没有 page 字段，
导致后续「相关页面」全部退化成 0，引用不可溯源。这里改为：

1. 以「页」为最小单位遍历，chunk 必然记录 page；
2. 页内按 Markdown 结构（空行 / 标题）切块，表格整体保留不拆；
3. 按 token 预算（默认 350）贪心打包，块间保留 60 token 重叠；
4. 每块记录 doc_id / doc_name / broker / page / n_tokens，供检索与界面展示。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import tiktoken

from src.config import CHUNK_DIR, PARSED_DIR, SETTINGS

_ENC = tiktoken.get_encoding("cl100k_base")

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_HTML_TABLE = re.compile(r"<table[^>]*>.*?</table>", re.S)


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _tail_tokens(text: str, limit: int) -> str:
    ids = _ENC.encode(text, disallowed_special=())
    if len(ids) <= limit:
        return text
    return _ENC.decode(ids[-limit:])


def _split_blocks(page_text: str) -> List[str]:
    """把一页切成语义块：连续表格行合并为一块，标题与其后正文合并。"""
    lines = page_text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []
    in_table = False

    def flush() -> None:
        if buf:
            blocks.append("\n".join(buf).strip())
            buf.clear()

    for line in lines:
        is_table = line.strip().startswith("|")
        is_blank = not line.strip()
        if is_table:
            if not in_table:
                flush()
                in_table = True
            buf.append(line)
            continue
        if in_table:
            flush()
            in_table = False
        if is_blank:
            flush()
            continue
        if _HEADING.match(line):
            flush()
        buf.append(line)
    flush()
    return [b for b in blocks if b.strip()]


def _split_oversize(block: str, budget: int) -> List[str]:
    """超长块切到预算内。

    MinerU 把整张表输出成**一行 HTML**（`<table><tr>…</tr></table>` 没有换行），
    按行切根本切不开——实测有 12709 token 的单块。这类块必须按 `<tr>` 行切开，
    并且每段都重复表头，否则表格语义断裂、embedding 也会被模型截断。
    """
    if n_tokens(block) <= budget:
        return [block]

    table = _HTML_TABLE.search(block)
    if table and table.group(0).count("<tr") >= 3:
        prefix, body, suffix = block[: table.start()], table.group(0), block[ table.end() :]
        rows = re.findall(r"<tr[^>]*>.*?</tr>", body, flags=re.S)
        if len(rows) >= 3:
            header, rest = rows[0], rows[1:]
            base = n_tokens(prefix + header)
            out, buf, size = [], [], base
            for row in rest:
                t = n_tokens(row)
                if size + t > budget and buf:
                    out.append(f"{prefix}\n<table>{header}{''.join(buf)}</table>")
                    buf, size = [], base
                buf.append(row)
                size += t
            if buf:
                out.append(f"{prefix}\n<table>{header}{''.join(buf)}</table>")
            if suffix.strip():
                out.append(suffix.strip())
            return [c for c in out if c.strip()]

    # 普通长文本：按行贪心切
    out, buf, size = [], [], 0
    for line in block.splitlines():
        t = n_tokens(line) + 1
        if size + t > budget and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += t
    if buf:
        out.append("\n".join(buf))
    # 兜底：某些页（旋转水印、竖排免责声明）会被解析成一行几万字的乱码，
    # 按行切不开，这里按 token 窗口硬切，避免超出 embedding 模型长度后向量失真。
    final: List[str] = []
    for piece in out:
        if n_tokens(piece) <= budget * 2:
            final.append(piece)
        else:
            final.extend(_hard_split(piece, budget))
    return final


def _hard_split(text: str, budget: int) -> List[str]:
    ids = _ENC.encode(text, disallowed_special=())
    step = max(1, int(budget))
    return [_ENC.decode(ids[i : i + step]) for i in range(0, len(ids), step)]


def chunk_document(doc: Dict, chunk_tokens: int = None, overlap: int = None) -> List[Dict]:
    chunk_tokens = chunk_tokens or SETTINGS.chunk_tokens
    overlap = SETTINGS.chunk_overlap if overlap is None else overlap
    meta = doc["metainfo"]
    chunks: List[Dict] = []
    cid = 0

    for page in doc["content"]["pages"]:
        blocks: List[str] = []
        for b in _split_blocks(page["text"]):
            blocks.extend(_split_oversize(b, chunk_tokens))

        buf: List[str] = []
        size = 0
        carry = ""  # 上一块尾部，作为本块前缀形成重叠

        def emit() -> None:
            nonlocal buf, size, carry, cid
            if not buf:
                return
            body = "\n\n".join(buf).strip()
            text = (carry + "\n" + body).strip() if carry else body
            chunks.append(
                {
                    "chunk_id": cid,
                    "doc_id": meta["sha1"],
                    "doc_name": meta["doc_title"],
                    "broker": meta.get("broker", ""),
                    "company_name": meta.get("company_name", ""),
                    "page": page["page"],
                    "text": text,
                    "n_tokens": n_tokens(text),
                }
            )
            cid += 1
            carry = _tail_tokens(body, overlap)

        for block in blocks:
            t = n_tokens(block)
            if size + t > chunk_tokens and buf:
                emit()
                buf, size = [], 0
            buf.append(block)
            size += t
        emit()

    return chunks


def chunk_all(parsed_dir: Path = PARSED_DIR, out_dir: Path = CHUNK_DIR, chunk_tokens: int = None) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_chunks: List[Dict] = []
    parsed_files = sorted(parsed_dir.glob("*.json"))
    if not parsed_files:
        raise FileNotFoundError(f"{parsed_dir} 下没有解析结果，请先执行 parse")
    for path in parsed_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_document(doc, chunk_tokens=chunk_tokens)
        all_chunks.extend(chunks)
        (out_dir / path.name).write_text(
            json.dumps({"metainfo": doc["metainfo"], "chunks": chunks}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  {doc['metainfo']['doc_title'][:40]:<40} {len(doc['content']['pages']):>3} 页 → {len(chunks):>3} 块")
    print(f"共 {len(parsed_files)} 份文档，{len(all_chunks)} 个 chunk")
    return all_chunks
