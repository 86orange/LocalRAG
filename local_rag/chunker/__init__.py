"""
文本切片模块

提供固定字符切片和语义感知切片两种策略。
"""

from local_rag.chunker.text_chunker import chunk_by_size, chunk_by_size_with_metadata
from local_rag.chunker.semantic_chunker import chunk_by_semantic, chunk_by_semantic_with_metadata
