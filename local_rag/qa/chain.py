"""
QA 链路模块

组装检索结果与 LLM，构建完整的 RAG 问答链路。
支持 Ollama generate 和 chat 两种 API 模式。
"""

import ollama
from typing import Generator

from local_rag.config import LLM_MODEL, OLLAMA_HOST, TEMPERATURE, MAX_CONTEXT_TOKENS
from local_rag.qa.prompt import build_qa_prompt, build_qa_prompt_text
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)


def build_qa_chain():
    """构建并返回可调用的 QA 链路函数。

    返回一个 callable，接受 {"context": str, "question": str} 并返回回答文本。

    Usage:
        chain = build_qa_chain()
        answer = chain.invoke({"context": "...", "question": "什么是RAG？"})
    """
    return _QAChain()


class _QAChain:
    """RAG 问答链路的轻量封装。

    提供 invoke() 方法供 CLI 和后续 Web 界面调用。
    """

    def invoke(self, inputs: dict) -> str:
        """执行问答。

        Args:
            inputs: {"context": str, "question": str}

        Returns:
            LLM 生成的回答文本
        """
        context = inputs.get("context", "")
        question = inputs.get("question", "")

        if not question.strip():
            return "请提供一个有效的问题。"

        if not context.strip():
            return "未找到相关文档，无法回答该问题。请先运行 `rag index` 构建知识库索引。"

        return generate_answer(context, question)


def generate_answer(
    context: str,
    question: str,
    model: str | None = None,
) -> str:
    """根据上下文和问题生成回答。

    优先使用 Ollama chat API，失败后回退到 generate API。

    Args:
        context: 检索到的参考文档上下文
        question: 用户问题
        model: LLM 模型名，默认使用全局配置

    Returns:
        LLM 生成的回答
    """
    model = model or LLM_MODEL

    trimmed_context = _trim_context(context)
    system_prompt, user_prompt = build_qa_prompt(trimmed_context, question)

    logger.debug("调用 LLM: model=%s, context_len=%d", model, len(trimmed_context))

    # 优先尝试 chat API（支持 system message）
    try:
        return _generate_via_chat(model, system_prompt, user_prompt)
    except Exception as e:
        logger.warning("Ollama chat API 调用失败，回退到 generate API: %s", e)

    # 回退到 generate API
    return _generate_via_generate(model, trimmed_context, question)


def generate_answer_stream(
    context: str,
    question: str,
    model: str | None = None,
) -> Generator[str, None, None]:
    """流式生成回答，逐 token yield。

    优先使用 Ollama chat stream API，失败后回退到 generate stream API。

    Args:
        context: 检索到的参考文档上下文
        question: 用户问题
        model: LLM 模型名，默认使用全局配置

    Yields:
        每次 yield 一个或几个 token 组成的小文本块
    """
    model = model or LLM_MODEL

    trimmed_context = _trim_context(context)
    system_prompt, user_prompt = build_qa_prompt(trimmed_context, question)

    logger.debug("流式调用 LLM: model=%s, context_len=%d", model, len(trimmed_context))

    try:
        yield from _stream_via_chat(model, system_prompt, user_prompt)
    except Exception as e:
        logger.warning("Ollama chat stream 失败，回退到 generate stream: %s", e)
        yield from _stream_via_generate(model, trimmed_context, question)


def _stream_via_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Generator[str, None, None]:
    """通过 Ollama chat stream API 逐 token 生成回答。"""
    client = ollama.Client(host=OLLAMA_HOST)
    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": TEMPERATURE},
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content


def _stream_via_generate(
    model: str,
    context: str,
    question: str,
) -> Generator[str, None, None]:
    """通过 Ollama generate stream API 逐 token 生成回答（回退）。"""
    client = ollama.Client(host=OLLAMA_HOST)
    full_prompt = build_qa_prompt_text(context, question)
    stream = client.generate(
        model=model,
        prompt=full_prompt,
        options={"temperature": TEMPERATURE},
        stream=True,
    )
    for chunk in stream:
        content = chunk.get("response", "")
        if content:
            yield content


def _generate_via_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """通过 Ollama chat API 生成回答。"""
    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": TEMPERATURE,
        },
    )
    return response["message"]["content"].strip()


def _generate_via_generate(
    model: str,
    context: str,
    question: str,
) -> str:
    """通过 Ollama generate API 生成回答（回退方案）。"""
    client = ollama.Client(host=OLLAMA_HOST)
    full_prompt = build_qa_prompt_text(context, question)
    response = client.generate(
        model=model,
        prompt=full_prompt,
        options={
            "temperature": TEMPERATURE,
        },
    )
    return response["response"].strip()


def _trim_context(context: str) -> str:
    """按最大 token 数智能裁剪上下文。

    策略：
    1. 将上下文按 "---" 分隔恢复为独立片段
    2. 按片段在原文中的顺序保留（前面分数高、更相关）
    3. 累计长度不超过 max_chars 时截断，优先保留完整片段

    Returns:
        裁剪后的上下文文本
    """
    max_chars = MAX_CONTEXT_TOKENS * 2 // 3
    if len(context) <= max_chars:
        return context

    parts = context.split("\n\n---\n\n")
    if len(parts) <= 1:
        return context[:max_chars]

    kept: list[str] = []
    used_chars = 0
    sep_len = len("\n\n---\n\n")

    for part in parts:
        part_len = len(part)
        added = sep_len + part_len if kept else part_len
        if used_chars + added <= max_chars:
            kept.append(part)
            used_chars += added
        else:
            # 剩余空间不够完整片段，尝试截断该片段填入
            remaining = max_chars - used_chars - (sep_len if kept else 0)
            if remaining > 100 and part:
                kept.append(part[:remaining])
            break

    logger.debug("上下文压缩: %d 个片段 → %d 个 (%d 字符)", len(parts), len(kept), len(context))
    return "\n\n---\n\n".join(kept) if kept else context[:max_chars]
