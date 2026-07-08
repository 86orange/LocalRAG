"""
文本切片模块

将长文档切分为适合向量化和检索的文本片段。
支持基础字符切片和语义感知切片两种策略。
"""

from local_rag.chunker.text_chunker import chunk_by_size
from local_rag.chunker.semantic_chunker import chunk_by_semantic
