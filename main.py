"""RAG-财报企业知识库命令行入口。

用法：
    python main.py all                      # 解析→分块→建库→批量作答（一条龙）
    python main.py parse --force            # 只重跑解析
    python main.py chunk                    # 只重跑分块
    python main.py index --force            # 只重建索引
    python main.py ask -q "中芯国际2024年营收是多少？" -k string
    python main.py batch -o data/stock_data/answers/answers.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from src.config import ANSWER_DIR, CHUNK_DIR, DATA_DIR, INDEX_DIR, PDF_DIR
from src.ingestion import build_index
from src.llm import USAGE

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


@click.group()
def cli() -> None:
    """RAG-财报企业知识库：面向中芯国际投研报告的企业知识库系统。"""


@cli.command(name="mineru-fetch")
@click.option("--ocr", is_flag=True, help="强制走 OCR（扫描件才需要，速度慢很多）")
@click.option("--force", is_flag=True, help="忽略已有产物，重新提交解析")
@click.option("--file", "file_name", default=None, help="只处理某一个 PDF（文件名，可模糊匹配）")
def mineru_fetch(ocr: bool, force: bool, file_name: str) -> None:
    """①' 调用 MinerU 云端解析（预签名直传 → 轮询 → 下载解压 → content_list.json）。"""
    from src.config import SETTINGS
    from src.mineru_cloud import MineruClient

    if not SETTINGS.mineru_api_key:
        raise click.ClickException("未配置 MINERU_API_KEY（写入 .env 或设置环境变量）")

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if file_name:
        pdfs = [p for p in pdfs if file_name in p.name]
        if not pdfs:
            raise click.ClickException(f"没有匹配 {file_name} 的 PDF")

    click.echo(f"提交 MinerU 解析：{len(pdfs)} 个文件（ocr={ocr or SETTINGS.mineru_ocr}）")
    client = MineruClient(SETTINGS.mineru_api_key)
    client.parse(pdfs, ocr=ocr or SETTINGS.mineru_ocr, skip_existing=not force)
    click.echo("完成。接着执行：python main.py parse --parser mineru --force")


@cli.command()
@click.option("--force", is_flag=True, help="忽略已有结果，强制重新解析")
@click.option("--parser", type=click.Choice(["auto", "pymupdf", "mineru"]), default="auto", show_default=True)
def parse(force: bool, parser: str) -> None:
    """① 解析 PDF 为带页码的逐页 Markdown。"""
    from src.parsing import parse_all

    click.echo(f"解析 {PDF_DIR}")
    parse_all(parser=parser, force=force)


@cli.command()
@click.option("--chunk-tokens", default=None, type=int, help="每块 token 上限（默认取配置 350）")
def chunk(chunk_tokens: int) -> None:
    """② 页码感知分块。"""
    from src.chunking import chunk_all

    chunk_all(chunk_tokens=chunk_tokens)


@cli.command()
@click.option("--force", is_flag=True, help="强制重建索引（忽略缓存）")
def index(force: bool) -> None:
    """③ 向量化并建立 FAISS + BM25 索引。"""
    build_index(force=force)
    click.echo(f"索引目录：{INDEX_DIR}")


@cli.command()
@click.option("-q", "--question", required=True, help="要检索的问题")
@click.option("-n", "--top-n", default=8, show_default=True, type=int)
def retrieve(question: str, top_n: int) -> None:
    """④' 仅检索（不调用对话模型），用于验证召回质量。"""
    from src.retrieval import Retriever

    r = Retriever(SETTINGS_FOR_RETRIEVE(False))
    out = r.retrieve(question, top_n=top_n)
    click.echo("\n" + "=" * 70)
    for i, p in enumerate(out["pages"], 1):
        llm = p["llm_score"]
        llm_txt = "-" if llm is None else "{:.2f}".format(llm)
        click.echo(
            "[C{}]《{}》第{}页  召回={:.3f} 重排={}".format(
                i, p["doc_name"], p["page"], p["recall_score"], llm_txt
            )
        )
        click.echo("     " + p["page_text"][:120].replace("\n", " "))


def SETTINGS_FOR_RETRIEVE(use_rerank: bool):
    """检索演示默认关闭 LLM 重排，避免消耗对话额度。"""
    from src.config import SETTINGS

    SETTINGS.force_rerank = use_rerank
    return SETTINGS


@cli.command()
@click.option("-k", "--kind", default="string", type=click.Choice(["string", "number", "boolean", "name", "names"]))
def smoke(kind: str) -> None:
    """⑥ 冒烟测试：不调用对话模型，检查索引、召回与提示词拼装是否正常。"""
    from src import prompts
    from src.retrieval import Retriever

    r = Retriever(SETTINGS_FOR_RETRIEVE(False))
    res = r.retrieve("中芯国际2024年营业收入是多少？", top_n=3)
    ctx = prompts.format_context(res["pages"])
    spec = prompts.build_prompts(kind)
    click.echo(f"索引块数：{len(r.chunks)}；召回页数：{len(res['pages'])}")
    click.echo(f"上下文长度：{len(ctx)} 字；提示词长度：{len(spec['system'])} 字")
    click.echo("上下文开头：" + ctx[:120].replace("\n", " "))


@cli.command()
@click.option("-q", "--question", required=True, help="要提问的问题")
@click.option("-k", "--kind", default="string", type=click.Choice(["string", "number", "boolean", "name", "names"]))
@click.option("-n", "--top-n", default=None, type=int, help="送入 prompt 的页面数")
def ask(question: str, kind: str, top_n: int) -> None:
    """④ 单问题问答。"""
    from src.qa import QAEngine

    result = QAEngine().answer(question, kind=kind, top_n=top_n)
    click.echo("\n" + "=" * 60)
    click.echo(f"问题：{result['question']}")
    click.echo(f"答案：{result['final_answer']}")
    click.echo(f"摘要：{result['reasoning_summary']}")
    click.echo("引用：")
    for r in result["references"]:
        click.echo(f"  [{r['citation']}]《{r['doc_name']}》第{r['page']}页  {r['snippet'][:60]}...")
    click.echo(f"耗时：{ {k: round(v, 1) for k, v in result['timings'].items()} }")
    click.echo(f"用量：{USAGE.summary()}")


@cli.command()
@click.option("-f", "--questions-file", default=str(DATA_DIR / "questions.json"), show_default=True)
@click.option("-o", "--out", default=str(ANSWER_DIR / "answers.json"), show_default=True)
def batch(questions_file: str, out: str) -> None:
    """⑤ 批量回答问题清单。"""
    from src.qa import QAEngine

    QAEngine().answer_batch(Path(questions_file), Path(out))
    click.echo(f"累计用量：{USAGE.summary()}")


@cli.command(name="all")
@click.option("--force", is_flag=True, help="解析与建库都强制重跑")
def run_all(force: bool) -> None:
    """一条龙：解析 → 分块 → 建库 → 批量作答。"""
    from src.chunking import chunk_all
    from src.parsing import parse_all
    from src.qa import QAEngine

    click.echo("=== ① 解析 PDF ===")
    parse_all(force=force)
    click.echo("=== ② 分块 ===")
    chunk_all()
    click.echo("=== ③ 建库 ===")
    build_index(force=force)
    click.echo("=== ④ 批量作答 ===")
    QAEngine().answer_batch(DATA_DIR / "questions.json", ANSWER_DIR / "answers.json")
    click.echo(f"累计用量：{USAGE.summary()}")


if __name__ == "__main__":
    sys.exit(cli())
