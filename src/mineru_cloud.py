"""MinerU 云端解析（v4 批量接口）。

早期实现走的是「先把 PDF 传到某个对象存储的公共地址，再让 MinerU 按 URL 抓取」的老路子，
那条链路早已失效（内置 token 也是 2025-06 到期的 JWT）。现在的官方流程是**预签名直传**：

    1) POST /api/v4/file-urls/batch   → 拿到 batch_id 与每个文件的 OSS 预签名 PUT 地址
    2) PUT  <预签名地址>              → 把本地 PDF 字节直传上去（不带 Authorization）
    3) GET  /api/v4/extract-results/batch/{batch_id} → 轮询 state，done 后拿 full_zip_url
    4) 下载 zip 解压到 data/stock_data/mineru/<文件名>/，里面有 content_list.json / full.md

`content_list.json` 是逐块带 `page_idx` 的结构化结果，正好满足本项目「必须保留页码」的要求。
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.config import MINERU_DIR

API_BASE = "https://mineru.net/api/v4"
MAX_PAGES_PER_FILE = 200  # 官方限制：单文件不超过 200 页，超了要拆


class MineruError(RuntimeError):
    pass


class MineruClient:
    def __init__(self, api_key: str, timeout: int = 60) -> None:
        if not api_key:
            raise MineruError("未配置 MINERU_API_KEY")
        self.api_key = api_key
        self.timeout = timeout
        self.h = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # ---------------------------------------------------------------- 提交批次
    def submit(self, pdf_paths: List[Path], ocr: bool = False, language: str = "ch") -> Dict[str, str]:
        """返回 {文件名: 预签名上传地址}。"""
        payload = {
            "enable_formula": False,
            "language": language,
            "files": [{"name": p.name, "is_ocr": ocr} for p in pdf_paths],
        }
        r = requests.post(f"{API_BASE}/file-urls/batch", headers=self.h, json=payload, timeout=self.timeout)
        data = self._unwrap(r, "申请上传地址")
        batch_id = data["batch_id"]
        urls = data["file_urls"]
        if len(urls) != len(pdf_paths):
            raise MineruError(f"返回的上传地址数量不符：{len(urls)} vs {len(pdf_paths)}")
        self.batch_id = batch_id
        print(f"MinerU 批次已创建：{batch_id}（{len(pdf_paths)} 个文件，可 GET /extract-results/batch/{batch_id} 复查）")
        return {p.name: u for p, u in zip(pdf_paths, urls)}

    def upload(self, pdf_paths: List[Path], ocr: bool = False, language: str = "ch") -> str:
        """整批上传，返回 batch_id。"""
        mapping = self.submit(pdf_paths, ocr=ocr, language=language)
        for p in pdf_paths:
            url = mapping[p.name]
            with open(p, "rb") as f:
                resp = requests.put(url, data=f, timeout=300)
            if resp.status_code not in (200, 201, 204):
                raise MineruError(f"上传失败 {p.name}: HTTP {resp.status_code} {resp.text[:160]}")
            print(f"  已上传 {p.name}（{p.stat().st_size/1e6:.1f}MB）")
        return self.batch_id

    # ---------------------------------------------------------------- 轮询结果
    def wait(self, batch_id: str, timeout: int = 3600, interval: int = 8) -> Dict[str, str]:
        """轮询到全部「有结论」为止，返回 {文件名: full_zip_url}。

        注意：单个文件失败（超页数、解析异常）不应拖垮整批，这里把失败信息记在
        self.failed 里，只返回成功的部分，由调用方决定怎么汇报。
        """
        t0 = time.time()
        last = ""
        self.failed: Dict[str, str] = {}
        while time.time() - t0 < timeout:
            r = requests.get(f"{API_BASE}/extract-results/batch/{batch_id}", headers=self.h, timeout=self.timeout)
            data = self._unwrap(r, "查询进度")
            results = data.get("extract_result", [])
            done, states = {}, []
            for item in results:
                name, state = item.get("file_name"), item.get("state")
                states.append(f"{name[:18]}:{state}")
                if state == "done" and item.get("full_zip_url"):
                    done[name] = item["full_zip_url"]
                elif state == "failed":
                    self.failed[name] = item.get("err_msg") or "unknown"
            msg = " | ".join(states)
            if msg != last:
                print(f"  [{int(time.time()-t0):>3}s] {msg[:320]}")
                last = msg
            settled = len(done) + len(self.failed)
            if settled == len(results) and results:
                if self.failed:
                    print(f"  ⚠ {len(self.failed)} 个文件失败：" + "; ".join(
                        f"{k} -> {v}" for k, v in list(self.failed.items())[:3]))
                print(f"  批次结束，用时 {int(time.time()-t0)}s")
                return done
            time.sleep(interval)
        raise MineruError(f"轮询超时（{timeout}s），批次 {batch_id}")

    # ------------------------------------------------------------ 下载与解压
    @staticmethod
    def download_and_extract(zip_url: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        r = requests.get(zip_url, timeout=300)
        if r.status_code != 200:
            raise MineruError(f"下载结果包失败: HTTP {r.status_code}")
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(dest)
        # zip 内可能多一层同名目录，统一拍平，方便后续按文件名找产物
        entries = [p for p in dest.iterdir() if p.is_dir()]
        if len(entries) == 1 and not any(dest.glob("*.json")):
            inner = entries[0]
            for child in inner.iterdir():
                child.rename(dest / child.name)
            inner.rmdir()
        return dest

    # ------------------------------------------------------------ 分页与合并
    @staticmethod
    def _split_pdf(pdf: Path, part_dir: Path, max_pages: int) -> List[tuple]:
        """把超过页数上限的 PDF 拆成若干段，返回 [(分段路径, 起始页偏移0基), ...]。"""
        import pymupdf

        doc = pymupdf.open(str(pdf))
        total = doc.page_count
        if total <= max_pages:
            doc.close()
            return [(pdf, 0)]
        part_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        for idx, start in enumerate(range(0, total, max_pages), 1):
            end = min(start + max_pages, total) - 1
            out = part_dir / f"{pdf.stem}__p{idx}.pdf"
            new = pymupdf.open()
            new.insert_pdf(doc, from_page=start, to_page=end)
            new.save(str(out))
            new.close()
            parts.append((out, start))
            print(f"  拆分 {pdf.name}：第 {start+1}-{end+1} 页 → {out.name}")
        doc.close()
        return parts

    @staticmethod
    def _lookup(mapping: Dict[str, str], name: str) -> Optional[str]:
        """MinerU 返回的 file_name 不含扩展名，这里按 stem 兼容两种写法。"""
        if name in mapping:
            return mapping[name]
        stem = Path(name).stem
        for k, v in mapping.items():
            if Path(k).stem == stem:
                return v
        return None

    @staticmethod
    def _merge_content_lists(part_dirs: List[tuple], target: Path) -> int:
        """按页偏移合并各分段产物，输出统一的 merged_content_list.json，返回总块数。"""
        merged: List[dict] = []
        for d, offset in part_dirs:
            hits = sorted(d.glob("**/*_content_list.json"))
            if not hits:
                continue
            for block in json.loads(hits[0].read_text(encoding="utf-8")):
                block = dict(block)
                block["page_idx"] = int(block.get("page_idx", 0)) + offset
                merged.append(block)
        merged.sort(key=lambda b: (b["page_idx"], 0))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
        return len(merged)

    def parse(self, pdf_paths: List[Path], out_dir: Path = MINERU_DIR, ocr: bool = False,
              skip_existing: bool = True) -> List[Path]:
        """完整流程：拆超长文件 → 上传 → 等结果 → 下载解压 → 按页合并 → merged_content_list.json。"""
        # 1) 规划：哪些要跑、哪些要拆
        groups: Dict[str, dict] = {}
        jobs: List[tuple] = []  # (提交文件名, 分段路径, 偏移, 原文件名)
        for p in pdf_paths:
            dest = out_dir / p.stem
            if skip_existing and (dest / "merged_content_list.json").exists():
                print(f"  跳过（已有合并产物）：{p.name}")
                continue
            parts = self._split_pdf(p, dest / "_parts", MAX_PAGES_PER_FILE)
            if len(parts) == 1:
                # 没超页数：产物直接落在 <文档名>/ 下，不要多套一层 _parts
                part_dirs = [(dest, 0)]
            else:
                part_dirs = [(dest / "_parts" / pp.stem, off) for pp, off in parts]
            groups[p.stem] = {"pdf": p, "dest": dest, "parts": part_dirs}
            for part_path, offset in parts:
                jobs.append((part_path.name, part_path, offset, p.stem))

        if not jobs:
            return [out_dir / p.stem for p in pdf_paths]

        # 2) 整批上传 + 轮询
        batch_id = self.upload([j[1] for j in jobs], ocr=ocr)
        done = self.wait(batch_id)
        failed = getattr(self, "failed", {})

        # 3) 下载解压 + 合并
        ok_docs = []
        for stem, info in groups.items():
            part_dirs = []
            missing = []
            for part_path, offset in info["parts"]:
                url = self._lookup(done, part_path.name)
                if not url:
                    missing.append(part_path.name)
                    continue
                cache = Path(part_path)  # 上传时的临时分段文件，产物解压到同名目录
                cache.mkdir(parents=True, exist_ok=True)
                self.download_and_extract(url, cache)
                part_dirs.append((cache, offset))
            if part_dirs:
                n = self._merge_content_lists(part_dirs, info["dest"] / "merged_content_list.json")
                status = "完整" if not missing else f"缺少 {len(missing)} 段"
                print(f"  合并 {info['pdf'].name}：{n} 个块（{status}）")
                ok_docs.append(info["dest"])
            else:
                print(f"  ✗ {info['pdf'].name} 无可用产物：{missing}")

        if failed:
            print("失败清单：" + "; ".join(f"{k} -> {v}" for k, v in failed.items()))
        return ok_docs

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _unwrap(resp: requests.Response, action: str) -> dict:
        if resp.status_code != 200:
            raise MineruError(f"{action}失败: HTTP {resp.status_code} {resp.text[:300]}")
        body = resp.json()
        if body.get("code") not in (0, 200):
            raise MineruError(f"{action}失败: code={body.get('code')} msg={body.get('msg')}")
        return body.get("data") or {}


def find_content_list(doc_dir: Path) -> Optional[Path]:
    hits = sorted(doc_dir.glob("**/*_content_list.json")) or sorted(doc_dir.glob("**/content_list.json"))
    return hits[0] if hits else None


def read_blocks(content_list: Path) -> List[dict]:
    return json.loads(content_list.read_text(encoding="utf-8"))
