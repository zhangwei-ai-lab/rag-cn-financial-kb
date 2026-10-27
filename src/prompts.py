"""提示词：系统指令 + 结构化输出约束 + 示例 + 上下文模板。

沿用原项目 prompts.py 的四段式（instruction / schema / example / template），
改动点：把「relevant_pages: List[int]」换成「relevant_citations: List[str]」，
用 C1、C2 这样的上下文编号来定位证据，从而支持跨多份研报引用（多文档场景下
单纯给页码会分不清是哪份报告的第几页）。
"""
from __future__ import annotations

from typing import Dict

SHARED_INSTRUCTION = """你是一个面向企业投研知识库的 RAG（检索增强生成）问答系统。
你的任务是**仅依据**下面提供的检索上下文回答问题，不得引入上下文之外的先验知识。

推理要求：
1. 先判断上下文是否真的包含回答问题所需的信息，注意问题措辞与指标口径；
2. 若上下文中只有相近但不完全对应的指标（如“归母净利润”与“净利润”），要明确指出差异；
3. 若上下文不足以回答，直接在 final_answer 中说明「上下文中未找到相关信息」，不要编造；
4. 引用证据时使用上下文编号（如 C1、C3），不要自己编造页码。
"""

CONTEXT_TEMPLATE = """以下是检索到的上下文（每段前的 [C?] 为引用编号）：

{context}

---

以下是问题：
"{question}"
"""

# 通用 JSON 结构说明（回答 schema 的字段约束）
_SCHEMA_COMMON = """
输出必须是一个 JSON 对象，包含且仅包含以下字段：
{{
  "step_by_step_analysis": "详细分步推理过程，至少 5 步、150 字以上，说明每步依据的是哪个上下文编号",
  "reasoning_summary": "对推理过程的简要总结，约 50 字",
  "relevant_citations": ["直接支撑答案的上下文编号，如 C2、C5；没有则为空列表"],
  "final_answer": {answer_desc}
}}
"""

_KINDS: Dict[str, Dict[str, str]] = {
    "string": {
        "answer_desc": '"一段完整、连贯的中文回答，可包含要点罗列；信息不足时说明未找到"',
        "extra": """最终答案是开放式文本，需要条理清晰：
- 先给出直接结论，再补充依据（数据、时间、口径）；
- 若上下文包含多个报告的不同说法，可分别说明并指出差异；
- 不要输出与问题无关的背景铺垫。""",
        "example": """示例：
问题："请简要总结中芯国际2024年主营业务的主要内容。"
答案：
{
  "step_by_step_analysis": "1. 问题要求总结主营业务构成。2. C1 提到公司主要从事集成电路晶圆代工。3. C3 列出营收按应用领域拆分：智能手机、电脑与平板、消费电子、互联与可穿戴、工业与汽车。4. C4 说明 12 英寸与 8 英寸产能布局。5. 综合归纳为主线业务+应用结构+产能布局。",
  "reasoning_summary": "依据 C1、C3、C4 归纳晶圆代工主业及下游应用结构。",
  "relevant_citations": ["C1", "C3", "C4"],
  "final_answer": "中芯国际主营集成电路晶圆代工，提供 0.35 微米至 FinFET 等多技术节点的代工与配套服务；收入按应用分为智能手机、电脑与平板、消费电子、互联与可穿戴、工业与汽车五大板块，并以 8 英寸与 12 英寸产线覆盖不同制程需求。"
}""",
    },
    "number": {
        "answer_desc": '"精确的数值（int 或 float），无法确定时输出字符串 \\"N/A\\""',
        "extra": """提取数值时的严格要求：
1. 明确问题中指标的精确定义；
2. 只接受与目标指标*完全等价*的数值，同义词可以，概念不同不行；
3. 单位换算要在推理中写清楚（如“亿元”与“百万元”）；
4. 需要计算、推导或聚合才能得到的，一律输出 "N/A"；
5. 百分比只输出数值，不带 % 号。""",
        "example": """示例：
问题："中芯国际 2024 年营业收入是多少亿元？"
答案：
{"step_by_step_analysis": "...", "reasoning_summary": "...", "relevant_citations": ["C2"], "final_answer": 577.96}""",
    },
    "boolean": {
        "answer_desc": "布尔值 true 或 false",
        "extra": """判断题要求：
1. 注意区分「政策变化」和「金额变化」这类易混淆表述；
2. 上下文明确说明未发生 → false；上下文完全没有相关信息 → 仍按最保守判断输出 false 并在推理中说明；
3. final_answer 只能是 true 或 false，不能是字符串。""",
        "example": """示例：
问题："中芯国际 2024 年报是否宣布了分红政策变更？"
答案：
{"step_by_step_analysis": "...", "reasoning_summary": "...", "relevant_citations": ["C4"], "final_answer": false}""",
    },
    "names": {
        "answer_desc": '字符串数组，如 ["7nm", "FinFET"]；无信息时输出 "N/A"',
        "extra": """列表类答案要求：
- 每一项都必须与上下文表述完全一致，不得改写、翻译或补充；
- 同一实体只出现一次；
- 若问题问的是职位，只返回职位名；问产品，只返回产品名。""",
        "example": """示例：
问题："报告中提到了哪些主要制程节点？"
答案：
{"step_by_step_analysis": "...", "reasoning_summary": "...", "relevant_citations": ["C2", "C5"], "final_answer": ["28nm", "14nm", "FinFET"]}""",
    },
    "name": {
        "answer_desc": '"单个实体名，需与上下文完全一致；无信息时输出 \\"N/A\\""',
        "extra": """实体类答案要求：
- 人名用全名，公司名与上下文完全一致；
- 不得包含多余解释、括号注释或标点。""",
        "example": """示例：
问题："报告披露的公司董事长是谁？"
答案：
{"step_by_step_analysis": "...", "reasoning_summary": "...", "relevant_citations": ["C1"], "final_answer": "刘训峰"}""",
    },
}


def build_prompts(kind: str = "string") -> Dict[str, str]:
    """按问题类型返回 {system, user_template}。"""
    kind = kind if kind in _KINDS else "string"
    spec = _KINDS[kind]
    system = SHARED_INSTRUCTION + "\n" + spec["extra"] + _SCHEMA_COMMON.format(answer_desc=spec["answer_desc"]) + "\n" + spec["example"]
    return {"system": system, "user_template": CONTEXT_TEMPLATE, "kind": kind}


def format_context(pages) -> str:
    """把检索到的页面拼成带引用编号的上下文。"""
    parts = []
    for i, p in enumerate(pages, 1):
        header = f"[C{i}]《{p['doc_name']}》第 {p['page']} 页"
        if p.get("broker"):
            header += f"（{p['broker']}）"
        parts.append(f"{header}\n{p['page_text']}")
    return "\n\n".join(parts)
