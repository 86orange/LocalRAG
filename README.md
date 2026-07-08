# LocalRAG

完全本地运行的 RAG（检索增强生成）知识库 AI 助手。

让你的文档会说话——用自然语言与自己的 Markdown、PDF、Word、TXT 文档对话，**一切在本地完成，无需联网**。

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://github.com/86orange/LocalRAG/actions/workflows/test.yml/badge.svg" alt="Test">
</p>

## ✨ 特性

- **隐私优先** — 所有数据在本地处理，文档不上传任何云端
- **多格式支持** — Markdown、PDF、Word (.docx)、TXT
- **双模式交互** — CLI 命令行 + Streamlit Web 可视化界面
- **即用即问** — `rag index` 索引，`rag query "问题"` 回答
- **增量索引** — 新文件自动处理，已索引文件不重复处理
- **来源标注** — 每个回答附带回应的文档来源和相似度
- **多会话管理** — Web 界面支持新建/切换/重命名/删除会话，刷新后自动恢复
- **流式输出** — 回答逐字显示，体验接近 ChatGPT
- **轻量部署** — 基于 Ollama + ChromaDB，无需 GPU 也能运行

## 🛠 技术栈

| 组件 | 技术 |
|------|------|
| LLM 推理 | [Ollama](https://ollama.com)（支持 Qwen / DeepSeek / Llama 等） |
| 向量数据库 | [ChromaDB](https://www.trychroma.com) |
| Embedding | BGE-M3（中文语义检索，1024 维） |
| 文档解析 | PyMuPDF / python-docx |
| 文本切片 | 字符切片 + 语义感知切片 |
| Web 前端 | [Streamlit](https://streamlit.io) |

## ⚡ 快速开始

### 1. 环境要求

- Python 3.12+
- [Ollama](https://ollama.com/download)（已安装并运行）

### 2. 安装

```bash
git clone https://github.com/86orange/LocalRAG.git
cd LocalRAG
uv pip install -e .
```

### 3. 拉取模型

```bash
# LLM 对话模型（必选其一）
ollama pull qwen3:4b      # 推荐，轻量高效 (~2.5GB)
# ollama pull qwen3:8b    # 更大参数量，更好效果 (~5GB)

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

浏览器会自动打开 `http://localhost:8501`，支持拖拽上传、对话问答、来源标注等完整功能。

## 📁 项目结构

```
local_rag/
├── cli.py                    # 命令行入口
├── config.py                 # 全局配置（模型、参数）
├── loader/                   # 文档加载
│   ├── markdown_loader.py    # Markdown（支持 Front Matter 清洗）
│   ├── pdf_loader.py         # PDF（页眉页脚移除 / 跨页表格合并）
│   ├── docx_loader.py        # Word（表格转 pipe 格式）
│   └── txt_loader.py         # TXT（自动编码检测）
├── chunker/                  # 文本切片
│   ├── text_chunker.py       # 固定字符切片 + overlap
│   └── semantic_chunker.py   # 语义感知（标题边界）切片
├── vector_store/             # 向量存储
│   └── chroma_store.py       # ChromaDB 增删改查封装
├── qa/                       # RAG 问答链路
│   ├── chain.py              # QA Chain（支持流式输出）
│   └── prompt.py             # Prompt 模板（防幻觉）
├── web/                      # Web 界面
│   ├── app.py                # Streamlit 主页面
│   └── components/           # UI 组件
│       ├── upload.py         # 文件拖拽上传 + 自动索引
│       ├── sidebar.py        # 侧边栏知识库管理
│       └── chat.py           # 对话窗口（流式输出 + 多会话）
└── utils/                    # 工具
    ├── logger.py             # 统一日志（控制台 + 文件）
    └── file_utils.py         # 文件扫描 / 哈希 / 校验
tests/                        # 单元测试 + 集成测试
├── test_loader.py
├── test_chunker.py
├── test_vector_store.py
├── test_qa.py
├── test_web.py
└── test_integration.py       # 端到端流水线测试
```

## 🎯 命令一览

### CLI 命令

| 命令 | 说明 |
|------|------|
| `rag index` | 扫描 documents/ 并构建向量索引 |
| `rag index --semantic` | 使用语义感知切片策略 |
| `rag query "问题"` | 检索并生成回答（附来源标注） |
| `rag query -k 10 "问题"` | 指定检索返回片段数 |
| `rag stats` | 查看知识库统计（片段数 / 来源文件数） |
| `rag delete ./文件路径` | 删除指定文件的索引 |

### Web 界面操作

| 功能 | 操作方式 |
|------|---------|
| 上传文档 | 拖拽文件至上传区域 |
| 对话问答 | 在底部输入框提问，流式显示回答 |
| 查看来源 | 点击回答下方的 📎 参考来源 |
| 管理会话 | 点击 ➕ 新建 / 📋 切换 / ✏ 重命名 / 🗑 删除 |
| 检索片段数 | 左下角数字选择器，悬停 ? 查看详细说明 |
| 知识库管理 | 左侧边栏查看文件列表、删除索引、重建全部 |

## 🔧 配置

可通过环境变量覆盖默认配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOCAL_RAG_LLM_MODEL` | `qwen3:4b` | LLM 对话模型名称 |
| `LOCAL_RAG_EMBEDDING_MODEL` | `bge-m3` | Embedding 向量化模型名称 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `LOCAL_RAG_CHUNK_SIZE` | `512` | 文本切片最大字符数 |
| `LOCAL_RAG_CHUNK_OVERLAP` | `128` | 相邻切片重叠字符数 |
| `LOCAL_RAG_TOP_K` | `5` | 检索返回的文档片段数 |
| `LOCAL_RAG_TEMPERATURE` | `0.1` | LLM 生成温度（越低越稳定） |
| `LOCAL_RAG_MAX_CONTEXT_TOKENS` | `4096` | 输入 LLM 的最大上下文 token 数 |

### 更换模型

默认使用 `qwen3:4b`（2.5GB），追求更好效果可升级为 8B 版本：

```bash
# 拉取新模型
ollama pull qwen3:8b

# macOS / Linux
export LOCAL_RAG_LLM_MODEL=qwen3:8b

# Windows (CMD)
set LOCAL_RAG_LLM_MODEL=qwen3:8b

# Windows (PowerShell)
$env:LOCAL_RAG_LLM_MODEL="qwen3:8b"
```

## 🧪 运行测试

```bash
# 运行所有测试（排除需 Ollama 的集成测试）
python -m pytest tests/ -v --ignore=tests/test_vector_store.py --ignore=tests/test_qa.py -m "not integration"

# 运行集成测试（需要 Ollama 运行且已拉取模型）
python -m pytest tests/test_integration.py -v -m integration
```

## 📄 License

[MIT](LICENSE)
