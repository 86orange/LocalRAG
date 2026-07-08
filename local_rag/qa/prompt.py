"""
Prompt 模板模块

定义 RAG 问答系统使用的 Prompt 模板，引导 LLM：
1. 严格基于提供的上下文回答问题
2. 不知道时明确说不知道，避免编造（防幻觉）
3. 回答后标注引用的来源序号
"""

# 默认 RAG 系统 Prompt
RAG_SYSTEM_PROMPT = """你是一个专业的知识库助手，你的任务是根据提供的文档上下文来回答用户问题。

请严格遵守以下规则：
1. 只能基于【参考文档】中的内容回答问题，不得使用你自己的知识
2. 如果文档中没有相关信息，请明确回答"根据提供的文档，未找到相关信息"
3. 回答要简洁清晰，使用中文
4. 在回答末尾，列出你所引用的文档来源编号，格式为 [来源: 编号]
5. 不要编造文档中不存在的事实、数据或结论"""

# 带上下文的问题模板
RAG_QA_TEMPLATE = """【参考文档】

{context}

【用户问题】

{question}

请根据以上参考文档回答问题："""


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
