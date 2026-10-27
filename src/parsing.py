"""PDF → 逐页 Markdown 解析。

原始实现直接调用 MinerU 云端 API，但仓库里内置的 token 是 2025-06 就过期的 JWT，
云端不可用，因此这里做成可插拔的三档解析后端，并保证**每页都带页码**
（原实现丢掉页码，导致引用失效）：

1. mineru：读取 MinerU 本地/云端解析产物（content_list.json），质量最好；
2. pymupdf：pymupdf4llm 逐页转 Markdown，纯 CPU、无需下载模型，作为默认后备；
3. auto：先找 MinerU 产物，找不到就用 pymupdf。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.config import MINERU_DIR, PARSED_DIR, PDF_DIR, SETTINGS

log = logging.getLogger(__name__)

# 研报里每页都重复的免责声明、页脚等噪声行
_NOISE_PATTERNS = [
    re.compile(r"^请(务必)?阅读(本报告)?(最后|末)一?页?(的)?(重要)?(声明|免责声明)"),
    re.compile(r"^敬请(参阅|关注)(本报告)?(最后|末)一?页"),
    re.compile(r"^第?\s*\d+\s*页(\s*/\s*共?\s*\d+\s*页)?$"),
    re.compile(r"^\**\s*\d+\s*/\s*\d+\s*\**$"),   # 形如 “3 / 222” 的页码
    re.compile(r"^\s*[-—–]?\s*\d+\s*[-—–]?\s*$"),
]


def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def guess_company(file_name: str) -> str:
    """从文件名推断公司名（本知识库围绕中芯国际，其余按文件名前缀兜底）。"""
    if "中芯国际" in file_name:
        return "中芯国际"
    m = re.match(r"^【[^】]+】([^：:（(]+)", Path(file_name).stem)
    return m.group(1).strip() if m else Path(file_name).stem


def guess_broker(file_name: str) -> str:
    m = re.match(r"^【([^】]+)】", Path(file_name).stem)
    return m.group(1).strip() if m else "未标注"


# --------------------------------------------------------------------- 清洗
def _clean_page_text(text: str, repeated_lines: set[str]) -> str:
    """删除跨页重复的页眉页脚、免责声明与孤立页码行。"""
    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if stripped in repeated_lines:
            continue
        if len(stripped) <= 60 and any(p.match(stripped) for p in _NOISE_PATTERNS):
            continue
        if stripped.startswith("![](data:image") or stripped.startswith("![]("):
            continue  # 图片占位，对文本检索无意义
        kept.append(line.rstrip())
    # 折叠 3 个以上连续空行
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(kept))
    return out.strip()


def _find_repeated_lines(pages: List[str]) -> set[str]:
    if len(pages) < 4:
        return set()
    counter: Dict[str, int] = {}
    for text in pages:
        seen = {ln.strip() for ln in text.splitlines() if 0 < len(ln.strip()) <= 80}
        for ln in seen:
            counter[ln] = counter.get(ln, 0) + 1
    threshold = max(3, int(len(pages) * 0.6))
    return {ln for ln, n in counter.items() if n >= threshold}


# --------------------------------------------------------------- 后端：pymupdf
def parse_with_pymupdf(pdf_path: Path) -> List[Dict]:
    import pymupdf4llm

    chunks = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        write_images=False,
        embed_images=False,
        show_progress=False,
    )
    pages: List[Dict] = []
    for idx, page in enumerate(chunks):
        meta = page.get("metadata") or {}
        page_no = meta.get("page")
        # pymupdf4llm 的 metadata['page'] 是 1 基页码；缺失时按顺序编号
        page_no = int(page_no) if isinstance(page_no, int) and page_no >= 1 else idx + 1
        pages.append({"page": page_no, "text": page.get("text", "")})
    return pages


# ---------------------------------------------------------------- 后端：MinerU
def parse_with_mineru(pdf_path: Path) -> Optional[List[Dict]]:
    """读取 MinerU 解析产物（`*_content_list.json`），按 page_idx 还原成逐页文本。

    使用方式：
      - 云端：`python main.py mineru-fetch`（走 src/mineru_cloud.py 的批量直传）
      - 本地：`mineru -p 报告.pdf -o data/stock_data/mineru` 后同样能被识别
    表格保留 MinerU 输出的 HTML（对 LLM 来说比 Markdown 表格更易对齐表头），标题按层级转成 #。
    """
    doc_dir = MINERU_DIR / pdf_path.stem
    if not doc_dir.exists():
        return None
    merged = doc_dir / "merged_content_list.json"
    if merged.exists():
        hits = [merged]
    else:
        # 兼容：本地 mineru CLI 直接产出的原始结构
        hits = sorted(doc_dir.glob("auto/*_content_list.json")) or sorted(
            doc_dir.glob("*_content_list.json")
        ) or sorted(doc_dir.glob("**/*_content_list.json"))
    if not hits:
        return None

    blocks = json.loads(hits[0].read_text(encoding="utf-8"))
    by_page: Dict[int, List[str]] = {}
    for block in blocks:
        page_no = int(block.get("page_idx", 0)) + 1
        btype = block.get("type")
        piece = ""
        if btype in {"text", "list", "equation"}:
            piece = (block.get("text") or "").strip()
            if btype == "equation" and piece:
                piece = f"$$\n{piece}\n$$"
        elif btype == "title":
            piece = (block.get("text") or "").strip()
            if piece:
                level = int(block.get("text_level") or 2)
                piece = f"{'#' * min(level, 6)} {piece}"
        elif btype == "table":
            body = (block.get("table_body") or "").strip()
            caption = " ".join(block.get("table_caption") or []).strip()
            footnote = " ".join(block.get("table_footnote") or []).strip()
            parts = [p for p in (caption, body, footnote) if p]
            piece = "\n\n".join(parts)
        elif btype in {"image", "figure"}:
            # 图片本身对文本检索无用，只保留图注
            piece = " ".join(block.get("img_caption") or []).strip()
        if piece:
            by_page.setdefault(page_no, []).append(piece)

    if not by_page:
        return None
    return [{"page": p, "text": "\n\n".join(by_page[p])} for p in sorted(by_page)]


# ------------------------------------------------------------------- 主入口
def parse_pdf(pdf_path: Path, parser: Optional[str] = None) -> Dict:
    parser = (parser or SETTINGS.parser).lower()
    raw_pages: List[Dict] = []
    used = "pymupdf"
    fallback_pages: List[int] = []
    total_pages = 0

    if parser in {"auto", "mineru"}:
        pages = parse_with_mineru(pdf_path)
        if pages:
            raw_pages, used = pages, "mineru"
        elif parser == "mineru":
            log.warning("未找到 MinerU 产物，回退到 pymupdf4llm：%s", pdf_path.name)

    if not raw_pages:
        raw_pages = parse_with_pymupdf(pdf_path)
    elif used == "mineru":
        # MinerU 会漏掉「识别成表格但表格体为空」的页（实测年报丢了 6 页财务报表），
        # 用 PyMuPDF 逐页补回来，保证页码完整可溯源。
        have = {p["page"] for p in raw_pages}
        import pymupdf

        with pymupdf.open(str(pdf_path)) as doc:
            total_pages = doc.page_count
        missing = [p for p in range(1, total_pages + 1) if p not in have]
        if missing:
            filled, _ = _pymupdf_texts_for_pages(pdf_path, missing)
            for p, text in sorted(filled.items()):
                raw_pages.append({"page": p, "text": text})
            raw_pages.sort(key=lambda x: x["page"])
            fallback_pages = sorted(filled)
            if fallback_pages:
                used = "mineru+pymupdf"
                print(f"    ↳ MinerU 漏页 {len(missing)} 页，已用 PyMuPDF 补 {len(fallback_pages)} 页：{fallback_pages}")

    repeated = _find_repeated_lines([p["text"] for p in raw_pages])
    pages = [
        {"page": p["page"], "text": _clean_page_text(p["text"], repeated)}
        for p in raw_pages
    ]
    pages = [p for p in pages if p["text"]]

    return {
        "metainfo": {
            "sha1": sha1_of_file(pdf_path),
            "file_name": pdf_path.name,
            "doc_title": pdf_path.stem,
            "company_name": guess_company(pdf_path.name),
            "broker": guess_broker(pdf_path.name),
            "n_pages": total_pages or len(raw_pages),
            "parser": used,
            "fallback_pages": fallback_pages,
        },
        "content": {"pages": pages},
    }


def parse_all(pdf_dir: Path = PDF_DIR, out_dir: Path = PARSED_DIR, parser: Optional[str] = None, force: bool = False) -> List[Dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"{pdf_dir} 下没有 PDF")
    results = []
    for i, pdf in enumerate(pdfs, 1):
        out_path = out_dir / f"{pdf.stem}.json"
        if out_path.exists() and not force:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"[{i}/{len(pdfs)}] 跳过（已有解析结果）：{pdf.name}")
        else:
            data = parse_pdf(pdf, parser=parser)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[{i}/{len(pdfs)}] {pdf.name} → {len(data['content']['pages'])} 页 "
                f"(parser={data['metainfo']['parser']})"
            )
        results.append(data)
    _write_subset(results, out_dir.parent / "subset.csv")
    return results


def _pymupdf_texts_for_pages(pdf_path: Path, pages: List[int]):
    """用 PyMuPDF 直接抓指定页的纯文本，作为 MinerU 漏页的补充。

    实测年报里 6 页财务报表被 MinerU 识别成 `table` 块但 `table_body` 为空
    （复杂/横排表格识别失败），如果不补，这些页会整页丢失。
    返回 (页码→文本, PDF 总页数)。
    """
    import pymupdf

    out: Dict[int, str] = {}
    doc = pymupdf.open(str(pdf_path))
    total = doc.page_count
    for p in pages:
        if 1 <= p <= total:
            text = doc.load_page(p - 1).get_text("text")
            if text and text.strip():
                out[p] = text
    doc.close()
    return out, total


def _write_subset(reports: List[Dict], path: Path) -> None:
    """重建 subset.csv（UTF-8），修复原文件 GBK 编码乱码导致的公司名/哈希映射失败。"""
    import pandas as pd

    rows = [
        {
            "sha1": r["metainfo"]["sha1"],
            "file_name": r["metainfo"]["file_name"],
            "company_name": r["metainfo"]["company_name"],
            "broker": r["metainfo"]["broker"],
        }
        for r in reports
    ]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"subset.csv 已重建：{path}（{len(rows)} 条）")
