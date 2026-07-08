"""
QA 模块测试

验证 Prompt 模板和 QA 链路的正确性。
LLM 集成测试仅在 Ollama 可用时执行。
"""

import pytest

from local_rag.qa.prompt import (
    RAG_SYSTEM_PROMPT,
    RAG_QA_TEMPLATE,
    build_qa_prompt,
    build_qa_prompt_text,
)
from local_rag.qa.chain import build_qa_chain, generate_answer
from local_rag.config import LLM_MODEL

# 测试用上下文（模拟检索结果拼接后的格式）
SAMPLE_CONTEXT = (
    "[来源 1 - RAG概述.md]\n"
    "RAG（Retrieval-Augmented Generation）是一种结合信息检索与文本生成的技术。\n\n"
    "[来源 2 - Python简介.md]\n"
    "Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 于 1991 年发布。"
)

SAMPLE_QUESTION = "什么是RAG？"


# ==================== Prompt 模板测试 ====================

def test_rag_system_prompt_not_empty():
    """系统 Prompt 应为非空字符串。"""
    assert len(RAG_SYSTEM_PROMPT) > 50
    assert "参考文档" in RAG_SYSTEM_PROMPT


def test_rag_qa_template_has_placeholders():
    """QA 模板应包含 {context} 和 {question} 占位符。"""
    assert "{context}" in RAG_QA_TEMPLATE
    assert "{question}" in RAG_QA_TEMPLATE


def test_build_qa_prompt_returns_tuple():
    """build_qa_prompt 应返回 (system, user) 二元组。"""
    system, user = build_qa_prompt(SAMPLE_CONTEXT, SAMPLE_QUESTION)
    assert isinstance(system, str)
    assert isinstance(user, str)
    assert len(system) > 0
    assert len(user) > 0


def test_build_qa_prompt_includes_context():
    """user prompt 中应包含输入的上下文文本。"""
    _, user = build_qa_prompt(SAMPLE_CONTEXT, SAMPLE_QUESTION)
    assert "RAG概述" in user
    assert "Python 是一种" in user


def test_build_qa_prompt_includes_question():
    """user prompt 中应包含输入的问题文本。"""
    _, user = build_qa_prompt(SAMPLE_CONTEXT, SAMPLE_QUESTION)
    assert SAMPLE_QUESTION in user


def test_build_qa_prompt_custom_system():
    """自定义 system prompt 应生效。"""
    custom = "你是一个测试助手。"
    system, user = build_qa_prompt(SAMPLE_CONTEXT, SAMPLE_QUESTION, system_prompt=custom)
    assert system == custom


def test_build_qa_prompt_text_format():
    """build_qa_prompt_text 应合并 system + template 为单个字符串。"""
    text = build_qa_prompt_text(SAMPLE_CONTEXT, SAMPLE_QUESTION)
    assert "专业的知识库助手" in text
    assert SAMPLE_QUESTION in text
    assert "RAG概述" in text


def test_template_empty_context():
    """空 context 不应导致异常。"""
    system, user = build_qa_prompt("", SAMPLE_QUESTION)
    assert isinstance(user, str)
    assert SAMPLE_QUESTION in user


def test_template_empty_question():
    """空 question 不应导致异常。"""
    system, user = build_qa_prompt(SAMPLE_CONTEXT, "")
    assert isinstance(user, str)


# ==================== QA Chain 测试（无 LLM） ====================

def test_build_qa_chain_returns_callable():
    """build_qa_chain 应返回有 invoke 方法的对象。"""
    chain = build_qa_chain()
    assert hasattr(chain, "invoke")
    assert callable(chain.invoke)


def test_qa_chain_empty_question():
    """空问题应返回提示信息，不报异常。"""
    chain = build_qa_chain()
    result = chain.invoke({"context": SAMPLE_CONTEXT, "question": ""})
    assert isinstance(result, str)
    assert len(result) > 0


def test_qa_chain_empty_context():
    """空上下文应返回提示信息。"""
    chain = build_qa_chain()
    result = chain.invoke({"context": "", "question": SAMPLE_QUESTION})
    assert isinstance(result, str)
    assert len(result) > 0


def test_qa_chain_missing_keys():
    """缺少 context 和 question 不应崩溃。"""
    chain = build_qa_chain()
    result = chain.invoke({})
    assert isinstance(result, str)


# ==================== LLM 集成测试（需 Ollama） ====================

@pytest.mark.integration
def test_generate_answer_with_ollama():
    """使用真实 Ollama 生成回答（需要 Ollama 运行并已拉取模型）。

    标记为 integration，CI 环境可跳过。
    """
    try:
        answer = generate_answer(SAMPLE_CONTEXT, SAMPLE_QUESTION)
    except Exception as e:
        pytest.skip(f"Ollama 不可用: {e}")

    assert isinstance(answer, str)
    assert len(answer) > 10


@pytest.mark.integration
def test_generate_answer_custom_model():
    """使用指定模型名调用（仅验证不报错）。"""
    try:
        # 使用当前默认模型（不额外拉取）
        answer = generate_answer(SAMPLE_CONTEXT, SAMPLE_QUESTION, model=LLM_MODEL)
    except Exception as e:
        pytest.skip(f"Ollama 不可用: {e}")

    assert isinstance(answer, str)
    assert len(answer) > 10
