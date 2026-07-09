"""
Prompt 模板模块

定义 RAG 问答系统使用的 Prompt 模板，引导 LLM：
1. 严格基于提供的上下文回答问题
2. 关键结论必须标注引用来源
3. 证据不足时明确拒答，禁止编造
4. 不知道时明确说不知道，避免编造（防幻觉）
"""

# 默认 RAG 系统 Prompt
RAG_SYSTEM_PROMPT = """你是一个严谨的知识库助手，**只**能根据提供的参考文档内容回答问题。

=== 核心约束 ===
1. **禁止使用外部知识**：你不得使用自己的训练数据、常识或任何文档之外的知识来回答。即使你"知道"答案，如果文档中没有，也必须拒答。
2. **证据驱动回答**：你的每一句回答都必须能在参考文档中找到直接依据。不得推测、猜测或延伸。
3. **必须标注引用**：关键的事实、数据、结论必须在对应句末标注来源编号，格式为 [来源:编号]。例如："RAG 技术能提升回答准确性 [来源:1][来源:3]。"
4. **证据不足直接拒答**：如果参考文档中的信息不足以回答问题，你必须明确告知用户"根据提供的文档，无法回答该问题"，并说明缺少什么信息。禁止用知识硬编补全。
5. **不要编造**：不得编造文档中不存在的事实、数字、日期、人名或结论。
6. **区分确定性与不确定性**：如果文档中的信息互相矛盾，应指出矛盾而非只选一方。如果信息不完整，应说明"文档中仅提到...，未涉及..."

=== 回答格式 ===
- 使用中文，简洁清晰
- 先给出直接回答，再提供支撑细节
- 结尾列出所有引用的来源列表：**参考来源**: [来源:1] 文件名, [来源:2] 文件名"""

# 带上下文的问题模板
RAG_QA_TEMPLATE = """【参考文档】

{context}

【用户问题】

{question}

请根据以上参考文档，严格遵守系统约束回答问题："""


def build_qa_prompt(
    context: str,
    question: str,
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """构建完整的 QA Prompt。

    返回 (system_prompt, user_prompt) 二元组，
    适配 Ollama chat API 的 messages 格式。

    Args:
        context: 检索到的参考文档片段（已拼接格式化）
        question: 用户提出的问题
        system_prompt: 自定义系统提示词，默认使用 RAG_SYSTEM_PROMPT

    Returns:
        (system_prompt, user_prompt) 元组
    """
    system = system_prompt or RAG_SYSTEM_PROMPT
    user = RAG_QA_TEMPLATE.format(context=context, question=question)
    return system, user


def build_qa_prompt_text(
    context: str,
    question: str,
    system_prompt: str | None = None,
) -> str:
    """构建单文本格式的完整 Prompt（用于 generate API）。

    Args:
        context: 检索到的参考文档片段
        question: 用户问题
        system_prompt: 自定义系统提示词

    Returns:
        拼接好的完整 Prompt 文本
    """
    system = system_prompt or RAG_SYSTEM_PROMPT
    return f"""{system}

{RAG_QA_TEMPLATE.format(context=context, question=question)}"""
