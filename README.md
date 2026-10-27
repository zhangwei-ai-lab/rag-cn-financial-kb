# RAG-财报企业知识库（基于RAG-Challenge-2自建 RAG 系统）

| 原项目区别 | 本项目做法 | 状态 |
| --- | --- | --- |
| 用 DashScope 替换 OpenAI，LLM 用 qwen，embedding 用 text_embedding | 对话与向量化都做成可切换后端：`dashscope`（qwen-plus + text-embedding-v3）或**本地 Ollama**（qwen3-8b）或 `st`（本地 bge-small-zh）。当前交付配置为**全本地**：`qwen3-8b-local`（生成/重排）+ `qwen3-embedding:0.6b`（向量） | ✅ 已跑通 |
| 用 MinerU 替换 docling 解析 | `src/mineru_cloud.py` 云端预签名直传 + 轮询 + 合并；`src/parsing.py` 三档后端；MinerU 漏页自动用 PyMuPDF 补齐 | ✅ 已跑通（9 份 / 312 页 / 337 张表） |
| 更新中文知识库（多份投研报告） | `data/stock_data/pdf_reports/` 下 9 份中芯国际研报（1 份年报 + 7 份券商研报 + 1 份调研纪要），共 312 页 | ✅ |
| 设置问题清单 questions.json | `data/stock_data/questions.json`（5 个开放问题，kind=string） | ✅ |
| 支持 kind=string 开放性问答 | `src/prompts.py` 内置 string/number/boolean/name/names 五类结构化模板 | ✅ |
| 搭建前端页面（streamlit / gradio） | `app_streamlit.py`：答案 + 证据来源表 + 分段耗时 + 历史记录 | ✅ |

## 〇、仓库内容边界（重要）

本仓库**只包含代码与文档**。以下内容因体积或版权原因不入库，需本地自行准备：

| 未入库内容 | 原因 | 如何获取 |
| --- | --- | --- |
| `.env` | 含 `MINERU_API_KEY` / `DASHSCOPE_API_KEY` | 复制 `.env.example`，填入自己的密钥 |
| `data/stock_data/pdf_reports/` | 券商研报与年报属第三方版权材料（约 14MB / 9 份） | 自行将中芯国际研报 PDF 放入该目录 |
| `models/` | 本地向量模型约 92MB | `ollama pull qwen3-embedding:0.6b` 或 `python scripts/fetch_embedding_model.py` |
| `data/stock_data/{parsed,chunks,index,answers,mineru}/` | 中间产物，可由脚本重建 | `python main.py all` |
| `.venv/` | 虚拟环境 | `pip install -r requirements.txt` |

因此完整复现顺序是：装依赖 → 配 `.env` → 放 PDF → `main.py all`。

## 一、整体流程

```
PDF(9份)
  │  src/parsing.py      逐页 Markdown + 页码 + 元信息，清洗重复页眉页脚
  ▼
pages(315 页，MinerU 逐块带 page_idx)
  │  src/chunking.py     页码感知 + Markdown 结构感知，350 token/块，60 token 重叠
  ▼
chunks(2105 块，300 token/块)
  │  src/ingestion.py    text-embedding-v3（带缓存）→ FAISS(IndexFlatIP) + BM25
  ▼
index
  │  src/retrieval.py    向量+BM25 混合召回 40 块 → 页级去重 → LLM 重排 → Top6 页
  ▼
context(带 [C1]…[Cn] 编号)
  │  src/prompts.py      kind → 系统指令 + JSON Schema + few-shot + 上下文模板
  ▼
answer + 引用(文档/页码/片段)
```

## 二、目录结构

```
RAG-财报企业知识库/
├─ main.py                 CLI 入口（parse / chunk / index / retrieve / smoke / ask / batch / all）
├─ app_streamlit.py        Streamlit 界面
├─ requirements.txt
├─ .env / .env.example     密钥与模型配置
├─ src/
│  ├─ config.py            全部超参数（分块大小、召回数、权重…）
│  ├─ llm.py               DashScope / OpenAI 兼容 / Ollama 三种后端 + 重试 + 用量统计
│  ├─ parsing.py           PDF → 逐页 Markdown（MinerU / pymupdf 双后端）
│  ├─ chunking.py          页码感知分块
│  ├─ ingestion.py         向量化（缓存）+ FAISS + BM25
│  ├─ retrieval.py         混合召回 + 页级聚合 + LLM 重排
│  ├─ prompts.py           五类结构化提示词
│  └─ qa.py                单问/批量问答，证据回填
└─ data/stock_data/
   ├─ pdf_reports/         原始 PDF
   ├─ parsed/              ① 解析产物
   ├─ chunks/              ② 分块产物
   ├─ index/               ③ faiss.index.npy / chunks.json / bm25.pkl / embed_cache.npz
   ├─ answers/             ④ 问答结果
   └─ questions.json       问题清单
```

## 三、快速开始

```bash
cd RAG-财报企业知识库
python -m venv venv && venv\Scripts\activate        # Windows
pip install -r requirements.txt

copy .env.example .env                              # 按需选择本地 / 百炼 / 其他端点

# 【方式一】全本地：先确认 Ollama 里有对话模型，再下载向量模型
ollama list                          # 需要 qwen3-8b-local:latest 之类的模型
ollama pull qwen3-embedding:0.6b                   # 本地向量模型（1024 维）
# 备用：python scripts/fetch_embedding_model.py    # bge-small-zh-v1.5（512 维，96MB，走 hf-mirror）

python main.py all                   # 解析 → 分块 → 建库 → 批量作答
streamlit run app_streamlit.py       # 打开界面
```

```bash
# 【MinerU 云端解析（推荐）】需要 MINERU_API_KEY
python main.py mineru-fetch          # 分批上传 → 轮询 → 下载解压 → 按页合并
python main.py parse --parser mineru --force
```

```bash
# 【方式二】用百炼：.env 里改成
#   LLM_BACKEND=dashscope  EMBEDDING_BACKEND=dashscope
#   EMBEDDING_MODEL=text-embedding-v3  ANSWER_MODEL=qwen-plus
```

## 三·补充、MinerU 云端解析的三个坑（已解决）

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| 单文件 200 页上限 | 222 页年报直接 `failed: number of pages exceeds limit` | `_split_pdf` 用 PyMuPDF 切段上传，`_merge_content_lists` 按 `page_idx` 偏移合并 |
| 返回的 `file_name` 不带 `.pdf` | `done.get(name)` 全部错配，9 份都判成失败 | `_lookup` 按 stem 兼容匹配 |
| 表格识别失败导致整页丢失 | 年报 6 页财务报表被标成 `table` 但 `table_body` 为空，该页无文本 | 解析后对比 PDF 总页数，缺页用 PyMuPDF 逐页补回（`metainfo.parser = mineru+pymupdf`，`fallback_pages` 记录补了哪些页） |

MinerU 产物（`data/stock_data/mineru/<文档名>/`）里含 `merged_content_list.json`（逐块 + `page_idx`）、
`full.md`、`images/`、`layout.json`，`content_list` 把页眉页脚单独标成 `header/footer` 块，
本项目解析时直接丢弃，等于免费拿到一层版式清洗。

常用命令：

```bash
python main.py parse --force            # 只重跑解析
python main.py chunk                    # 只重跑分块
python main.py mineru-fetch --file 年报   # 只重跑某一篇的 MinerU 解析
python main.py index --force            # 只重建索引（换 embedding 模型后必须重跑）
python main.py retrieve -q "问题" -n 8   # 只检索（不调对话模型，省时间）
python main.py smoke                    # 冒烟测试：索引/召回/提示词是否正常
python main.py ask -q "问题" -k string   # 单问答
python main.py batch -o data/stock_data/answers/answers.json
```

## 四、关键配置（`src/config.py` 或 `.env`）

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_BACKEND` | `dashscope` | `dashscope` / `ollama` / `openai`（任意兼容端点） |
| `EMBEDDING_BACKEND` | `dashscope` | `dashscope` / `ollama` / `st`（本地 sentence-transformers） |
| `ANSWER_MODEL` | `qwen-plus` | 生成模型；本地用 `qwen3-8b-local:latest` |
| `RERANK_MODEL` | `qwen-turbo-latest` | 重排要跑十几次调用，本地与生成共用同一个 8B 模型 |
| `OLLAMA_THINK` | `0` | Qwen3 思维链开关；关掉后单次响应 ~16s → ~1s（实测） |
| `EMBEDDING_MODEL` | `text-embedding-v3` | 本地：`qwen3-embedding:0.6b`（Ollama）或 `models/bge-small-zh-v1.5`（st） |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP` | 300 / 50 | 中文研报；bge 上限 512，故不要再放大 |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`（向量+BM25）/ `bm25`（纯关键词、无需向量模型） |
| `chunk_tokens` / `chunk_overlap` | 350 / 60 | 中文研报信息密度高于英文年报，比原方案 300 略大 |
| `recall_k` / `page_candidates` / `top_n_context` | 40 / 12 / 6 | 召回块 → 候选页 → 进 prompt 的页数（本地 8B 建议 40/8/5） |
| `vector_weight` / `bm25_weight` | 0.7 / 0.3 | 混合召回权重 |
| `llm_rerank_weight` / `retrieval_weight` | 0.7 / 0.3 | 重排加权（沿用冠军方案） |

## 五、相对原始代码（RAG-Challenge-2初版）的改动

### 5.1 修复的缺陷

| 问题 | 原代码表现 | 修复方式 |
| --- | --- | --- |
| **页码丢失** | `split_markdown_file` 只按 30 行切块，chunk 里没有 `page`；`retrieval` 用 `chunk.get("page", 0)` → **页码恒为 0**，`relevant_pages` 全部失效 | 解析阶段保留逐页边界，分块时把 `page` 写进每个 chunk；检索按 (文档, 页) 聚合 |
| **知识库只索引了 1 篇** | `subset.csv` 是 GBK 编码，`file_name` 与 md 文件名对不上 → `sha1` 为空 → faiss 文件名变成 `.faiss`；实际只产出 1 个 chunk json + 2 个 faiss | 解析后自动重建 UTF-8 的 `subset.csv`；`sha1` 直接由 PDF 字节算，不再依赖外部映射 |
| **9 份报告只查 1 份** | 检索按「公司名 → 单库」路由，9 份报告同属中芯国际，等于每次只查一个库 | 改为**全局统一索引**，文档名/页码写在 chunk 元数据里，可跨研报召回 |
| **引用无法区分文档** | `relevant_pages: List[int]`，多文档下只给页码分不清是哪份 | `relevant_citations: List[str]`，用 [C1]…[Cn] 上下文编号，回填成「文档 + 页码 + 片段」 |
| **faiss 写盘失败** | faiss 内部用窄字符 `fopen`，Windows 下路径含中文直接抛 `RuntimeError` | 用 `faiss.serialize_index` + `np.save` 落盘，绕开 C++ 文件接口 |
| **界面解析链路脆弱** | 界面自己解析 `answer.content.final_answer` 字符串，字段变动就报错 | CLI 与界面共用 `QAEngine.answer()` 单一入口，直接拿结构化 dict |
| **依赖过重** | docling + torch + google-generativeai + openai + langchain 全套，Python 3.13 装不上 | 精简为 12 个纯 CPU 依赖，`pip install -r requirements.txt` 一次装完 |

### 5.2 新增的能力

- **混合检索**：向量 + BM25（中文按字 + 二元组切分），融合分数可解释（余弦原值×0.7 + BM25 归一×0.3）；
- **LLM 重排**：对候选页打分，`0.7×LLM + 0.3×召回`，失败自动退回召回排序，不阻断问答；
- **embedding 缓存**：按文本哈希 + 模型名落盘（`embed_cache_<model>_<dim>.npz`），重复建库只算新增；
- **页级上下文**：命中块只作指针，实际喂给模型的是整页（父页面检索）；
- **引用回填**：模型漏给引用时，自动回填分数最高的页面，保证每条答案都能翻回原文；
- **可切换后端**：`LLM_BACKEND=dashscope|openai|ollama`，一行配置换模型；
- **可观测性**：分段耗时（召回/聚合/重排/生成）、token 用量、每个问题的引用与得分。

## 六、本地全链路实测（2026-10-25，MinerU + 全本地模型）

环境：Windows 11、CPU（无 GPU）、Ollama `qwen3-8b-local:latest`（8.2B Q4_K_M）+ `qwen3-embedding:0.6b`（1024 维）、
解析走 MinerU 云端（9 份 / 312 页）。

| 指标 | 数值 |
| --- | --- |
| 解析 | MinerU 批次 130s（10 个分段，含 222 页年报切成 200+22）+ 下载合并 ~60s |
| 解析产物 | 315 页 / 702,737 字 / **337 张表格**（pymupdf 版为 312 页 / 401,375 字 / 0 张表） |
| 分块 | 2105 块，平均 292 token，最大 704 token |
| 建库 | 2105×1024 向量，**137s**（qwen3-embedding 首次调用要加载模型 ~40s） |
| 单问题 | 召回 0.1s｜LLM 重排 10–17s｜生成 10–15s｜合计 **22–31s** |
| 5 问批量 | **2 分 22 秒** |
| 关闭思维链前/后 | 同一 prompt：16.0s → **0.6s**（`/no_think`） |
| 抽样核对 | Q4 的 83.65%/12.22%/57.86%/30.96% 均在引用页（上海证券 p7）原文；Q1「全球第二、中国大陆第一」出自年报 p14 |

完整问答见 `../docs/本地运行结果.md`，本次发现的问题见 `../docs/复现问题清单.md`。

## 七、已知限制

1. **MinerU 表格识别失败会丢页**：年报 6 页、国信 2 页、中原 1 页被标成 `table` 但表格体为空，
   这些页现在由 PyMuPDF 补**纯文本**（数字在、行列关系没了）。要更准可对这 9 页单独开 `--ocr` 重跑。
2. **图表数字不可检索**：MinerU 对图片只给图注，研报里营收/产能趋势图上的数字进不了库，这是纯文本 RAG 的天花板。
3. **跨研报口径冲突**：同名指标不同报告口径不同（如归母净利润 1.88 亿 vs 1.09 亿美元），
   当前模型会并列给出并标注来源，需要单一数字时得人工定口径。
4. **本地 8B 模型的能力边界**：个别问题会漏次要要点；换 `qwen-plus` 或更大的本地模型会更好。
5. **MinerU 免费额度 2000 页/天**：一次全量 312 页，约 6 次/天；产物已缓存在 `data/stock_data/mineru/`，默认跳过已解析。
6. **无重排时的排序**：只跑 `retrieve` 时按关键词+语义融合排序，个别页面会因关键词密集排前面
   （例如「工艺平台明细」压过「营收毛利率」），这正是需要 LLM 重排的原因。
7. **百炼账号仍欠费**：`DASHSCOPE_API_KEY` 可用作后备，但当前返回 `400 Arrearage`。
