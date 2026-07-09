# Local RAG Agent

完全本地运行的 RAG（检索增强生成）知识库 AI 助手。

让你的文档会说话——用自然语言与自己的 Markdown、PDF、Word、TXT 文档对话，**一切在本地完成，无需联网**。

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://github.com/86orange/LocalRAG/actions/workflows/test.yml/badge.svg" alt="Test">
  <img src="https://img.shields.io/badge/tests-125%20passed-brightgreen.svg" alt="Tests">
</p>

## ✨ 特性

### 数据治理
- **多格式加载** — PDF（页级分页 + 页码映射）、DOCX（段落/表格/页眉页脚）、Markdown（Front Matter 清洗）、TXT（多编码自适应）
- **统一清洗管线** — 10 步可插拔规则链，内置中文 OCR 形近字纠错，覆盖控制字符过滤、全半角统一、标点规范化
- **三层去重** — 文件级 MD5 → SimHash 近重复检测 → 向量相似度二次校验
- **版本管理** — 索引归档、版本回滚、变更日志，旧版本数据保留不丢失
- **完整元数据** — 每个 Chunk 携带 doc_id / source / 章节 / 页码 / 版本号 / 生效状态

### 混合检索
- **BM25 关键词检索** — 纯 Python 实现，中文 2-gram 分词，零外部依赖
- **向量语义检索** — ChromaDB + BGE-M3 Embedding 1024 维
- **RRF 融合排序** — Reciprocal Rank Fusion 交叉互补，解决纯向量检索对专有名词召回不足
- **检索后处理** — 相似度阈值过滤、去重、动态补齐、上下文智能压缩

### 证据驱动生成
- **严格约束** — 禁止使用外部知识，证据不足直接拒答，关键结论必须带 `[来源:n]` 标注
- **流式输出** — Ollama Chat API 流式生成，回答逐字显示
- **双模式交互** — CLI 命令行 + Streamlit Web 可视化界面

### 评估体系
- **忠实度评估** — LLM-as-Judge 逐条拆解回答主张，验证证据支持度
- **召回评估** — Recall@1/3/5 + MRR，支持自定义 JSON 数据集
- **错误归因** — 未命中项详情输出，快速定位检索瓶颈

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| LLM 推理 | [Ollama](https://ollama.com)（支持 Qwen / DeepSeek / Llama 等） |
| 向量数据库 | [ChromaDB](https://www.trychroma.com) |
| Embedding | BGE-M3（中文语义检索，1024 维） |
| 关键词检索 | BM25（自研，中文 2-gram） |
| 文档解析 | PyMuPDF / python-docx |
| 文本切片 | 字符切片 + 语义感知切片（标题边界） |
| 文本清洗 | 10 步规则链 + OCR 形近字纠错 |
| 评估 | LLM-as-Judge 忠实度 + Recall@K/MRR |
| Web 前端 | [Streamlit](https://streamlit.io) |

## ⚡ 快速开始

### 1. 环境要求

- Python 3.12+
- [Ollama](https://ollama.com/download)（已安装并运行）

### 2. 安装

```bash
git clone https://github.com/86orange/LocalRAG.git
cd LocalRAG
pip install -e .
```

### 3. 拉取模型

```bash
# LLM 对话模型（必选其一）
ollama pull qwen3:4b      # 推荐，轻量高效 (~2.5GB)

# Embedding 模型（必选）
ollama pull bge-m3         # 中文语义检索 (~1.2GB)
```

### 4. 开始使用

#### CLI 命令行

```bash
# 将文档放入 documents/ 目录，然后索引
rag index

# 向知识库提问
rag query "什么是 RAG？"

# 查看统计
rag stats
```

#### Web 可视化界面

```bash
streamlit run local_rag/web/app.py
```

浏览器打开 `http://localhost:8501`，支持拖拽上传、对话问答、来源标注。

## 📁 项目结构

```
local_rag/
├── cli.py                      # 命令行入口（index/query/stats/eval/versions/rollback）
├── config.py                   # 全局配置（模型、参数、阈值）
├── loader/                     # 文档加载
│   ├── pdf_loader.py           # PDF（页级分页 + 页码映射）
│   ├── docx_loader.py          # Word（段落/表格/页眉页脚提取）
│   ├── markdown_loader.py      # Markdown（Front Matter 清洗）
│   └── txt_loader.py           # TXT（多编码自适应）
├── cleaner/                    # 文本清洗
│   └── __init__.py             # 10 步规则链 + OCR 纠错
├── chunker/                    # 文本切片
│   ├── text_chunker.py         # 固定字符切片 + overlap
│   └── semantic_chunker.py     # 语义感知切片（标题边界 + 章节归属）
├── retrieval/                  # 检索系统
│   ├── bm25_retriever.py       # BM25 关键词检索器
│   └── hybrid_retriever.py     # 混合检索器（RRF 融合 + 去重 + 阈值过滤）
├── vector_store/               # 向量存储
│   └── chroma_store.py         # ChromaDB 封装（增删改查 + 归档/激活）
├── qa/                         # RAG 问答链路
│   ├── chain.py                # QA Chain（Chat/Generate API + 流式输出）
│   └── prompt.py               # 证据驱动 Prompt 约束
├── eval/                       # 评估系统
│   ├── faithfulness.py         # LLM-as-Judge 忠实度评分
│   └── recall.py               # Recall@K/MRR 召回命中率
├── web/                        # Web 界面
│   ├── app.py                  # Streamlit 主页面
│   └── components/             # UI 组件（上传/聊天/侧边栏）
├── utils/                      # 工具
│   ├── dedup.py                # 三层去重（MD5 + SimHash + 向量）
│   ├── file_utils.py           # 文件扫描 / 哈希 / 校验
│   └── logger.py               # 统一日志
└── version_manager.py          # 索引版本管理
```

## 🎯 命令一览

| 命令 | 说明 |
|------|------|
| `rag index` | 扫描 documents/ 并构建索引 |
| `rag index --semantic` | 使用语义感知切片 |
| `rag index --force` | 强制重建全部索引 |
| `rag query "问题"` | 混合检索 + 生成回答 |
| `rag query -k 10 -t 0.5 "问题"` | 指定 top_k 和 RRF 阈值 |
| `rag stats` | 查看知识库统计（含版本信息） |
| `rag versions ./文件` | 查看文件版本历史 |
| `rag rollback ./文件 2` | 回滚到版本 2 |
| `rag delete ./文件` | 删除文件的所有索引 |
| `rag eval faithfulness "问题"` | 评估答案忠实度 |
| `rag eval recall dataset.json` | 评估召回命中率 |

## 🔧 配置

可通过环境变量覆盖默认配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_RAG_LLM_MODEL` | `qwen3:4b` | LLM 对话模型 |
| `LOCAL_RAG_EMBEDDING_MODEL` | `bge-m3` | Embedding 模型 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `LOCAL_RAG_CHUNK_SIZE` | `512` | 切片最大字符数 |
| `LOCAL_RAG_CHUNK_OVERLAP` | `128` | 切片重叠字符数 |
| `LOCAL_RAG_TOP_K` | `5` | 检索返回片段数 |
| `LOCAL_RAG_SIMILARITY_THRESHOLD` | `0.35` | 检索相似度阈值 |
| `LOCAL_RAG_TEMPERATURE` | `0.1` | LLM 生成温度 |
| `LOCAL_RAG_MAX_CONTEXT_TOKENS` | `4096` | 最大上下文 token 数 |

## 🧪 运行测试

```bash
# 运行所有测试（125 个）
uv run pytest tests/ --ignore=tests/test_vector_store.py --ignore=tests/test_qa.py -m "not integration" -v

# 运行集成测试（需要 Ollama 运行且已拉取模型）
uv run pytest tests/test_integration.py -v -m integration
```

## 📄 License

[MIT](LICENSE)
