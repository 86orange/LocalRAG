# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

## [v0.1.0] - 待发布

### 新增

**CLI 命令行工具**
- `rag index` — 扫描 documents/ 目录并构建向量索引
- `rag query "问题"` — 语义检索 + LLM 生成带来源的回答
- `rag stats` — 查看知识库统计信息
- `rag delete` — 删除指定文件的索引

**文档加载**
- PDF 加载器（PyMuPDF）：文本提取 / 页眉页脚自动识别与移除 / 跨页表格检测与合并 / 扫描件 OCR 支持
- Markdown 加载器：YAML Front Matter 去除 / HTML 标签剥离 / 图片链接清洗 / 代码块保留
- DOCX 加载器（python-docx）：段落提取 / 表格转 pipe 格式 / 页眉页脚过滤
- TXT 加载器：自动编码检测（UTF-8 → GBK → Latin-1）/ 控制字符过滤 / 全角半角空白统一

**文本切片**
- 字符切片：固定 chunk_size + overlap 段落感知切分，超长段自动拆解
- 语义切片：识别标题边界分组，无标题时退化为长度切分

**向量存储**
- ChromaDB 封装：入库 / 语义检索 / 按来源删除 / 全量清空 / 统计
- Ollama Embedding 集成（BGE-M3 默认，支持 nomic-embed-text）
- 增量索引：同一文件重新入库时自动覆盖旧数据

**RAG 问答**
- Prompt 模板：严格基于文档回答，防幻觉，标注来源
- Ollama chat API 优先，generate API 回退
- 流式生成（逐 token 输出）

**Web 界面（Streamlit）**
- 文件拖拽上传 + 自动索引进度条
- 侧边栏知识库管理：文件列表、删除、重建索引
- 聊天窗口：对话式问答 + 流式输出 + 来源标注
- 多会话管理：新建 / 切换 / 重命名 / 删除，刷新后自动恢复
- 检索数量可调节 + 问号悬浮提示

**工程化**
- 全局配置管理（环境变量可覆盖）
- 统一日志系统（控制台 + 滚动日志文件）
- 单元测试：loader / chunker / vector_store / QA / Web 组件
- 集成测试：端到端索引 → 检索 → LLM 回答流程
- GitHub Actions CI：push / PR 自动运行测试

### 依赖

- Python >= 3.12
- Ollama（LLM 推理 + Embedding）
- ChromaDB >= 1.5.0（向量数据库）
- PyMuPDF >= 1.28.0（PDF 解析）
- python-docx >= 1.1.0（Word 解析）
- Streamlit >= 1.59.0（Web 界面）
