"""
问答引擎模块

集成检索与 LLM 生成，构建 RAG QA 链路。
"""

from local_rag.qa.chain import (
    build_qa_chain,
    generate_answer,
    generate_answer_stream,
)
from local_rag.qa.prompt import build_qa_prompt, build_qa_prompt_text
