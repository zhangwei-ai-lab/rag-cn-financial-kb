"""RAG-财报企业知识库 · Streamlit 界面。

相比原版界面的增强：
- 展示每条答案的**来源文档 + 页码 + 片段**（原版只显示页码，且页码恒为 0）；
- 分段耗时（召回 / 重排 / 生成）与 token 用量可视化；
- 支持切换问题类型（string / number / boolean / names）；
- 侧栏保留历史提问，便于对照。
启动：streamlit run app_streamlit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import INDEX_DIR, SETTINGS  # noqa: E402
from src.qa import QAEngine  # noqa: E402

st.set_page_config(page_title="RAG-财报企业知识库", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="正在加载索引与模型…")
def get_engine() -> QAEngine:
    return QAEngine()


def index_ready() -> bool:
    return all((INDEX_DIR / f).exists() for f in ("faiss.index.npy", "chunks.json", "bm25.pkl"))


st.markdown(
    """
<div style='background: linear-gradient(100deg, #1f4fd8 0%, #6a3df0 55%, #d24ad0 100%);
            padding: 22px 28px; border-radius: 14px; color: #fff;'>
  <div style='font-size: 26px; font-weight: 700; letter-spacing: .5px;'>📈 RAG-财报企业知识库</div>
  <div style='margin-top: 6px; font-size: 14px; opacity: .92;'>
    多份中芯国际财报与研报 · 向量 + BM25 混合召回 · LLM 重排 · 页码可溯源
  </div>
</div>
""",
    unsafe_allow_html=True,
)

if not index_ready():
    st.error("尚未建立索引。请先在项目根目录执行：`python main.py all`（或 parse → chunk → index）。")
    st.stop()

with st.sidebar:
    st.subheader("查询设置")
    question = st.text_area(
        "输入问题",
        "请简要总结中芯国际2024年主营业务的主要内容。",
        height=96,
    )
    kind = st.selectbox("答案类型 (kind)", ["string", "number", "boolean", "names", "name"], index=0)
    top_n = st.slider("送入上下文的页数", 3, 10, SETTINGS.top_n_context)
    use_rerank = st.toggle("启用 LLM 重排", value=SETTINGS.force_rerank)
    ask_btn = st.button("生成答案", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        f"embedding: {SETTINGS.embedding_model}\n\n"
        f"answer: {SETTINGS.answer_model}\n\n"
        f"rerank: {SETTINGS.rerank_model}"
    )

if "history" not in st.session_state:
    st.session_state.history = []

if ask_btn and question.strip():
    SETTINGS.force_rerank = use_rerank
    engine = get_engine()
    with st.spinner("检索 → 重排 → 生成中…"):
        try:
            result = engine.answer(question, kind=kind, top_n=top_n, verbose=False)
        except Exception as exc:  # 界面要给出可读错误，而不是整页崩掉
            st.error(f"生成失败：{exc}")
            st.stop()
    st.session_state.history.insert(0, result)

if st.session_state.history:
    result = st.session_state.history[0]

    st.markdown("#### 最终答案")
    st.success(result.get("final_answer", "N/A"))

    st.markdown("#### 证据来源")
    refs = result.get("references", [])
    if refs:
        st.dataframe(
            [
                {
                    "编号": r["citation"],
                    "报告": r["doc_name"],
                    "机构": r.get("broker", ""),
                    "页码": r["page"],
                    "相关度": r["score"],
                    "片段": r["snippet"],
                }
                for r in refs
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("本次回答未引用任何上下文（可能上下文中确实没有相关信息）。")

    timings = result.get("timings", {})
    cols = st.columns(4)
    for col, (name, val) in zip(cols, timings.items()):
        col.metric(name, f"{val:.1f}s")

    with st.expander("推理摘要与分步推理"):
        st.markdown("**推理摘要**")
        st.write(result.get("reasoning_summary", "-"))
        st.markdown("**分步推理**")
        st.write(result.get("step_by_step_analysis", "-"))

    with st.expander("历史提问"):
        for h in st.session_state.history[1:]:
            st.markdown(f"- **{h['question']}** → {str(h.get('final_answer'))[:120]}")
else:
    st.info("在左侧输入问题后点击「生成答案」。")
