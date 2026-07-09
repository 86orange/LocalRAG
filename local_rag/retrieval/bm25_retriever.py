"""
BM25 关键词检索器

纯 Python 实现，零外部依赖。
支持中英混合分词（中文 2-gram + 英文/数字空格切分）。
"""

import math
from typing import Generator

from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# BM25 超参数
K1 = 1.5   # 词频饱和度参数
B = 0.75   # 文档长度归一化参数


class BM25Retriever:
    """BM25 关键词检索器。

    索引阶段构建倒排索引，检索时计算 BM25 分数并返回 top-k 结果。

    Usage:
        bm25 = BM25Retriever()
        bm25.index(chunks)
        results = bm25.search("关键词", top_k=5)
    """

    def __init__(self, k1: float = K1, b: float = B) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[str] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._inverted_index: dict[str, dict[int, int]] = {}  # token → {doc_id: tf}
        self._idf_cache: dict[str, float] = {}
        self._doc_count: int = 0

    def index(self, chunks: list[str]) -> None:
        """构建 BM25 倒排索引。

        Args:
            chunks: 待索引的文本块列表
        """
        self._chunks = list(chunks)
        self._doc_lens = [len(_tokenize(c)) for c in chunks]
        self._doc_count = len(chunks)
        self._avgdl = sum(self._doc_lens) / max(1, self._doc_count)
        self._inverted_index.clear()
        self._idf_cache.clear()

        for doc_id, chunk in enumerate(chunks):
            tokens = _tokenize(chunk)
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            for token, tf in term_freq.items():
                if token not in self._inverted_index:
                    self._inverted_index[token] = {}
                self._inverted_index[token][doc_id] = tf

        logger.info(
            "BM25 索引构建完成: %d 文档, %d 词条, avg_len=%.1f",
            self._doc_count,
            len(self._inverted_index),
            self._avgdl,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """BM25 关键词检索。

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            [(doc_id, bm25_score), ...]，按分数降序排列
        """
        if self._doc_count == 0:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * self._doc_count

        for token in set(query_tokens):
            postings = self._inverted_index.get(token, {})
            if not postings:
                continue

            idf = self._get_idf(token, len(postings))
            for doc_id, tf in postings.items():
                doc_len = self._doc_lens[doc_id]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avgdl)
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(
            [(i, round(s, 4)) for i, s in enumerate(scores) if s > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def get_document(self, doc_id: int) -> str:
        """获取指定 ID 的文档文本。"""
        return self._chunks[doc_id] if 0 <= doc_id < len(self._chunks) else ""

    @property
    def doc_count(self) -> int:
        return self._doc_count

    def _get_idf(self, token: str, df: int) -> float:
        """计算 IDF（逆文档频率），带缓存。"""
        if token in self._idf_cache:
            return self._idf_cache[token]

        idf = math.log(1 + (self._doc_count - df + 0.5) / (df + 0.5))
        self._idf_cache[token] = idf
        return idf


# ==================== 分词器 ====================


def _tokenize(text: str) -> list[str]:
    """中英混合分词。

    中文：2-gram 滑动窗口（双字）
    英文/数字：空格 + 标点边界切分

    Args:
        text: 输入文本

    Returns:
        token 列表
    """
    tokens: list[str] = []
    normalized = text.lower().strip()
    if not normalized:
        return tokens

    i = 0
    while i < len(normalized):
        ch = normalized[i]

        if '\u4e00' <= ch <= '\u9fff':
            # 中文 2-gram
            if i + 1 < len(normalized) and '\u4e00' <= normalized[i + 1] <= '\u9fff':
                tokens.append(normalized[i:i + 2])
                i += 1
            else:
                tokens.append(ch)
                i += 1
        elif ch.isalnum():
            start = i
            while i < len(normalized) and normalized[i].isalnum():
                i += 1
            tokens.append(normalized[start:i])
        else:
            i += 1

    return tokens
