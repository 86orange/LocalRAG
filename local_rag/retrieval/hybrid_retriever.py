"""
混合检索融合器

使用 RRF (Reciprocal Rank Fusion) 融合 BM25 关键词检索与 ChromaDB 向量检索结果。
支持相似度阈值过滤与动态补齐。
"""

from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# RRF 平滑参数
RRF_K = 60
# 默认相似度阈值（RRF 分数无上限，设为 0 表示不过滤）
DEFAULT_SIMILARITY_THRESHOLD = 0.0


def rrf_fusion(
    keyword_results: list[tuple[int, float]],
    vector_results: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """RRF 融合两组排序结果。

    公式: RRF_score(d) = Σ(1 / (k + rank_i(d)))

    对两路结果分别按原始分数排名，同一文档在任意一路排名越靠前，
    融合后分数越高。实现"交叉互补"：能被任意一路命中的文档都有机会。

    Args:
        keyword_results: [(doc_id, bm25_score), ...] BM25 结果
        vector_results: [(doc_id, vector_score), ...] 向量结果
        k: RRF 平滑参数，默认 60

    Returns:
        [(doc_id, rrf_score), ...] 按 RRF 分数降序排列
    """
    rrf_scores: dict[int, float] = {}

    for rank, (doc_id, _score) in enumerate(keyword_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, (doc_id, _score) in enumerate(vector_results):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    ranked = sorted(
        [(doc_id, round(rrf_scores[doc_id], 6)) for doc_id in rrf_scores],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


def linear_fusion(
    keyword_results: list[tuple[int, float]],
    vector_results: list[tuple[int, float]],
    keyword_weight: float = 0.3,
    vector_weight: float = 0.7,
) -> list[tuple[int, float]]:
    """线性加权融合两组结果。

    适用场景：需要显式控制关键词 vs 语义的权重。

    Args:
        keyword_results: [(doc_id, bm25_score), ...]
        vector_results: [(doc_id, vector_score), ...]
        keyword_weight: 关键词权重
        vector_weight: 语义权重

    Returns:
        [(doc_id, weighted_score), ...] 按加权分数降序
    """
    scores: dict[int, float] = {}

    # 归一化向量分数 (0~1 float)
    if vector_results:
        v_min = min(s for _, s in vector_results)
        v_max = max(s for _, s in vector_results)
        v_range = v_max - v_min or 1.0
        for doc_id, score in vector_results:
            scores[doc_id] = (score - v_min) / v_range * vector_weight

    # 归一化关键词分数 (整数 rank)
    if keyword_results:
        k_max = max(s for _, s in keyword_results)
        k_range = k_max or 1.0
        for doc_id, score in keyword_results:
            norm = (score / k_range) * keyword_weight
            scores[doc_id] = scores.get(doc_id, 0.0) + norm

    ranked = sorted(
        [(doc_id, round(scores[doc_id], 6)) for doc_id in scores],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked


class HybridRetriever:
    """混合检索器：BM25 + 向量 → RRF/线性融合。

    Usage:
        store = ChromaStore()
        hr = HybridRetriever(store)
        hr.build_index()               # 从向量库构建 BM25 索引
        results = hr.search("问题", top_k=5)
    """

    def __init__(
        self,
        store: object,
        rrf_k: int = RRF_K,
        keyword_top_k_multiplier: int = 3,
    ) -> None:
        from local_rag.vector_store.chroma_store import ChromaStore
        from local_rag.retrieval.bm25_retriever import BM25Retriever

        self._store = store
        self._bm25 = BM25Retriever()
        self._rrf_k = rrf_k
        self._keyword_mult = keyword_top_k_multiplier
        self._is_built = False

    def build_index(self) -> None:
        """从向量库加载全部文档并构建 BM25 索引。"""
        self._ensure_store()

        all_docs = self._store.get_all_chunks()
        if not all_docs:
            logger.warning("向量库为空，跳过 BM25 索引构建")
            return

        self._bm25.index(all_docs)
        self._is_built = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        fusion: str = "rrf",
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        auto_backfill: bool = True,
    ) -> list[dict]:
        """混合检索，含相似度阈值过滤与动态补齐。

        流程：BM25 + 向量 → RRF/线性融合 → 去重 → 阈值过滤 → 补齐

        Args:
            query: 查询文本
            top_k: 期望返回结果数
            fusion: 融合方式 "rrf" 或 "linear"
            similarity_threshold: RRF 分数阈值，低于此值的结果被过滤
            auto_backfill: 阈值过滤后不足 top_k 时是否自动补齐

        Returns:
            融合后的结果列表，格式与 ChromaStore.search 兼容
        """
        self._ensure_store()

        if not self._is_built:
            self.build_index()

        # 宽检索：扩大候选池以便后续过滤+补齐有余地
        pool_size = max(top_k * self._keyword_mult * 2, 20)

        kw_raw = self._bm25.search(query, top_k=pool_size)
        vec_raw = self._store.search(query, top_k=pool_size)
        vec_indices = self._store.map_results_to_indices(vec_raw, self._bm25)

        if fusion == "linear":
            fused = linear_fusion(kw_raw, vec_indices)
        else:
            fused = rrf_fusion(kw_raw, vec_indices, self._rrf_k)

        # 组装全部候选
        candidates: list[dict] = []
        for doc_id, fused_score in fused:
            candidates.append({
                "id": f"hybrid_{doc_id}",
                "document": self._bm25.get_document(doc_id),
                "metadata": self._store.get_chunk_metadata(doc_id) or {},
                "score": fused_score,
            })

        # 去重
        candidates = _dedup_results(candidates)

        # 阈值过滤
        passed = [
            c for c in candidates
            if c["score"] >= similarity_threshold
        ]

        # 动态补齐：过滤后不足 top_k 时从候选池补足
        if auto_backfill and len(passed) < top_k:
            seen_docs = {c["id"] for c in passed}
            for c in candidates:
                if len(passed) >= top_k:
                    break
                if c["id"] not in seen_docs and c["score"] >= 0:
                    passed.append(c)
                    seen_docs.add(c["id"])

        result = passed[:top_k]

        if not result:
            logger.debug("检索结果为空 (threshold=%.3f)", similarity_threshold)

        return result

    def _ensure_store(self) -> None:
        if self._store is None:
            raise RuntimeError("HybridRetriever 未关联 ChromaStore")


def hybrid_search(
    store: object,
    query: str,
    top_k: int = 5,
    fusion: str = "rrf",
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """一行调用混合检索的便捷函数。

    Args:
        store: ChromaStore 实例
        query: 查询文本
        top_k: 返回结果数
        fusion: 融合方式 "rrf" 或 "linear"
        similarity_threshold: RRF 分数阈值

    Returns:
        混合检索结果列表
    """
    hr = HybridRetriever(store)
    return hr.search(query, top_k=top_k, fusion=fusion, similarity_threshold=similarity_threshold)


# ==================== 检索结果去重 ====================


def _dedup_results(results: list[dict]) -> list[dict]:
    """去除检索结果中的重复/高度相似项。

    策略：
    1. 完全相同的 document 文本 → 直接去重
    2. 前 80 字符相同且来自不同文件 → 视为近似重复，保留 score 最高的
       （同一文件的不同 chunk 允许共存）

    保留原始排序（score 高的在前）。

    Args:
        results: 检索结果列表

    Returns:
        去重后的结果列表
    """
    if len(results) <= 1:
        return list(results)

    seen_contents: set[str] = set()
    seen_prefixes: dict[str, int] = {}  # prefix → index in kept
    kept: list[dict] = []

    for result in results:
        doc = result.get("document", "")

        # 1. 完全匹配
        if doc in seen_contents:
            continue

        # 2. 前缀匹配（跨文件才去重，同文件的不同 chunk 保留）
        prefix = doc[:80].strip()
        if prefix and prefix in seen_prefixes:
            existing_idx = seen_prefixes[prefix]
            exist_source = kept[existing_idx].get("metadata", {}).get("source", "")
            cur_source = result.get("metadata", {}).get("source", "")
            same_file = bool(exist_source and cur_source and exist_source == cur_source)
            if not same_file:
                if result.get("score", 0) > kept[existing_idx].get("score", 0):
                    kept[existing_idx] = result
                continue

        seen_contents.add(doc)
        if prefix:
            seen_prefixes[prefix] = len(kept)
        kept.append(result)

    return kept
