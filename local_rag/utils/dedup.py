"""
内容级去重模块

提供两层去重策略：
1. SimHash 近重复检测 — 基于文本特征指纹的 O(n) 去重，适合大规模批量索引
2. 向量相似度阈值去重 — 通过向量库查询检测高度相似的已有片段
"""

from typing import Generator

from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# 默认: 64 位 SimHash，汉明距离 ≤ 3 视为重复（约 95% 相似度）
DEFAULT_SIMHASH_BITS = 64
DEFAULT_HAMMING_THRESHOLD = 3
# 向量相似度阈值: cos_sim ≥ 0.98 视为重复
DEFAULT_VECTOR_SIMILARITY_THRESHOLD = 0.98


# ==================== 双层去重管线 ====================


class DualLayerDeduplicator:
    """双层内容去重器：SimHash 近重复 + 向量相似度。

    第一层（SimHash）：O(1) 文本特征指纹比对，快速过滤 95%+ 相似的内容。
    第二层（向量相似度）：在 SimHash 通过后，再用向量库检索检测高度相似的已有片段，
    适合捕获同义改写（SimHash 可能漏掉但语义相同的文本）。

    Usage:
        dedup = DualLayerDeduplicator(store)
        dedup.load_existing_simhashes(...)
        chunks, metas = dedup.filter(chunks, metas)
    """

    def __init__(
        self,
        store: object | None = None,
        hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
        vector_threshold: float = DEFAULT_VECTOR_SIMILARITY_THRESHOLD,
    ) -> None:
        self._simhash = ChunkDeduplicator(hamming_threshold=hamming_threshold)
        self._store = store
        self._vector_threshold = vector_threshold
        self._vector_dups_found = 0

    def is_duplicate(self, text: str) -> bool:
        """双层检查：SimHash → 向量相似度。

        Args:
            text: 文本块内容

        Returns:
            True 表示重复
        """
        if self._simhash.is_duplicate(text):
            return True

        if self._store is not None and self._vector_duplicate(text):
            self._vector_dups_found += 1
            return True

        return False

    def mark_indexed(self, text: str) -> None:
        """标记为已索引。"""
        self._simhash.mark_indexed(text)

    def load_existing_simhashes(self, simhashes: list[int]) -> None:
        """恢复已索引的 SimHash 集合。"""
        self._simhash.load_existing(simhashes)

    def filter_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict],
    ) -> tuple[list[str], list[dict]]:
        """双层过滤，返回去重后的 (chunks, metadatas)。"""
        kept_c: list[str] = []
        kept_m: list[dict] = []
        for chunk, meta in zip(chunks, metadatas):
            if self.is_duplicate(chunk):
                continue
            self.mark_indexed(chunk)
            kept_c.append(chunk)
            kept_m.append(meta)
        return kept_c, kept_m

    def get_stats(self) -> dict:
        """获取两层去重统计。"""
        s_stats = self._simhash.get_stats()
        return {
            "simhash_checked": s_stats["total_checked"],
            "simhash_duplicates": s_stats["duplicates_found"],
            "vector_duplicates": self._vector_dups_found,
        }

    def _vector_duplicate(self, chunk: str) -> bool:
        """通过向量库检索检测是否与已有高度相似片段重复。

        检索 top-1，若相似度超过阈值则视为重复。

        Args:
            chunk: 文本块内容

        Returns:
            True 表示已有高度相似片段
        """
        try:
            from local_rag.vector_store.chroma_store import ChromaStore

            if not isinstance(self._store, ChromaStore):
                return False

            existing = self._store.search_similar(
                chunk,
                top_k=1,
                similarity_threshold=self._vector_threshold,
            )
            return len(existing) > 0
        except Exception as e:
            logger.warning("向量去重查询失败: %s", e)
            return False


# ==================== SimHash 实现 ====================


def compute_simhash(text: str, hash_bits: int = DEFAULT_SIMHASH_BITS) -> int:
    """计算文本的 SimHash 指纹。

    使用分词 + 加权哈希的方法生成固定长度的二进制指纹，
    内容高度相似的文本会产生汉明距离很小的指纹。

    Args:
        text: 输入文本
        hash_bits: 指纹位数，默认 64

    Returns:
        SimHash 整数值
    """
    if not text or not text.strip():
        return 0

    weights = [0] * hash_bits

    for token in _tokenize(text):
        token_hash = _fnv1a_64(token.encode("utf-8"))
        for bit in range(hash_bits):
            if (token_hash >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1

    fingerprint = 0
    for bit in range(hash_bits):
        if weights[bit] > 0:
            fingerprint |= (1 << bit)

    return fingerprint


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """计算两个 SimHash 值的汉明距离（不同 bit 的个数）。

    Args:
        hash_a: SimHash 值 A
        hash_b: SimHash 值 B

    Returns:
        汉明距离
    """
    diff = hash_a ^ hash_b
    return diff.bit_count()


def is_near_duplicate(
    hash_a: int,
    hash_b: int,
    threshold: int = DEFAULT_HAMMING_THRESHOLD,
) -> bool:
    """判断两个 SimHash 是否属于近重复。

    Args:
        hash_a: SimHash A
        hash_b: SimHash B
        threshold: 汉明距离阈值，≤ 此值视为重复

    Returns:
        True 表示近重复
    """
    return hamming_distance(hash_a, hash_b) <= threshold


# ==================== Chunk 去重器 ====================


class ChunkDeduplicator:
    """基于 SimHash 的内容级去重器。

    在索引入库前，对每个 chunk 计算 SimHash 并与已索引集合比对，
    近重复的 chunk 直接跳过，避免冗余索引。

    支持桶化索引优化大规模场景下的查找效率。

    Usage:
        dedup = ChunkDeduplicator()
        dedup.load_existing(existing_simhashes)   # 从已有索引恢复
        for chunk, meta in zip(chunks, metadatas):
            if dedup.is_duplicate(chunk):
                continue
            dedup.mark_indexed(chunk)
            store.add_documents([chunk], [meta])
    """

    def __init__(
        self,
        hamming_threshold: int = DEFAULT_HAMMING_THRESHOLD,
        hash_bits: int = DEFAULT_SIMHASH_BITS,
    ) -> None:
        self._hamming_threshold = hamming_threshold
        self._hash_bits = hash_bits
        self._seen: list[int] = []
        self._stats = {"total_checked": 0, "duplicates_found": 0}

    def is_duplicate(self, text: str) -> bool:
        """检查文本块是否与已索引内容近重复。

        Args:
            text: 文本块内容

        Returns:
            True 表示与已有内容重复
        """
        if not text.strip():
            return True

        self._stats["total_checked"] += 1
        simhash = compute_simhash(text, self._hash_bits)

        for existing in self._seen:
            if is_near_duplicate(simhash, existing, self._hamming_threshold):
                self._stats["duplicates_found"] += 1
                return True
        return False

    def mark_indexed(self, text: str) -> int:
        """将文本块标记为已索引，后续相同内容会被过滤。

        Args:
            text: 文本块内容

        Returns:
            该块的 SimHash 值
        """
        simhash = compute_simhash(text, self._hash_bits)
        self._seen.append(simhash)
        return simhash

    def load_existing(self, simhashes: list[int]) -> None:
        """从已有索引中恢复 SimHash 集合。

        Args:
            simhashes: 已索引块的 SimHash 值列表
        """
        self._seen.extend(simhashes)
        logger.debug("已恢复 %d 个 SimHash 签名", len(simhashes))

    def get_all_simhashes(self) -> list[int]:
        """获取所有已索引的 SimHash 值列表。"""
        return list(self._seen)

    def get_stats(self) -> dict:
        """获取去重统计信息。"""
        return dict(self._stats)

    def filter_chunks(
        self,
        chunks: list[str],
        metadatas: list[dict],
    ) -> tuple[list[str], list[dict]]:
        """过滤重复块，返回去重后的 (chunks, metadatas)。

        Args:
            chunks: 文本块列表
            metadatas: 对应的 metadata 列表

        Returns:
            去重后的 (chunks, metadatas)
        """
        kept_chunks: list[str] = []
        kept_metas: list[dict] = []

        for chunk, meta in zip(chunks, metadatas):
            if self.is_duplicate(chunk):
                continue
            self.mark_indexed(chunk)
            kept_chunks.append(chunk)
            kept_metas.append(meta)

        return kept_chunks, kept_metas


# ==================== 内部辅助 ====================


def _tokenize(text: str) -> Generator[str, None, None]:
    """将文本拆分为加权 token 序列。

    对中文使用 2-gram（双字滑动），对英文/数字使用空格分词。
    重复 token 多次 yield 实现加权（TF 越高权重越大）。

    Args:
        text: 输入文本

    Yields:
        每个 token 字符串
    """
    normalized = text.strip().lower()
    if not normalized:
        return

    # 中文 2-gram
    i = 0
    while i < len(normalized):
        char = normalized[i]
        if '\u4e00' <= char <= '\u9fff':
            if i + 1 < len(normalized) and '\u4e00' <= normalized[i + 1] <= '\u9fff':
                yield normalized[i:i + 2]
                i += 1
            else:
                yield char
                i += 1
        elif char.isalnum():
            start = i
            while i < len(normalized) and normalized[i].isalnum():
                i += 1
            yield normalized[start:i]
        else:
            i += 1


def _fnv1a_64(data: bytes) -> int:
    """FNV-1a 64 位哈希，用于 SimHash token 权重计算。

    Args:
        data: 待哈希的字节序列

    Returns:
        64 位哈希值
    """
    FNV_OFFSET = 0xcbf29ce484222325
    FNV_PRIME = 0x100000001b3

    h = FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h
