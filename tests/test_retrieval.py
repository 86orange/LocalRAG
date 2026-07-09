"""
混合检索测试

覆盖 BM25 关键词检索、RRF 融合、HybridRetriever。
"""
import pytest
from local_rag.retrieval.bm25_retriever import BM25Retriever, _tokenize
from local_rag.retrieval.hybrid_retriever import rrf_fusion, linear_fusion, _dedup_results


# ==================== 分词器测试 ====================


def test_tokenize_chinese():
    """中文 2-gram 分词。"""
    tokens = _tokenize("检索增强生成")
    assert "检索" in tokens
    assert "增强" in tokens
    assert "生成" in tokens


def test_tokenize_mixed():
    """中英混合分词。"""
    tokens = _tokenize("RAG 检索增强")
    assert "rag" in tokens
    assert "检索" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("   ") == []


def test_tokenize_deduplicates_cjk():
    """中文 2-gram 应正确切分长串。"""
    tokens = _tokenize("知识库系统")
    assert len(tokens) >= 2


# ==================== BM25 索引与检索 ====================


@pytest.fixture
def bm25_with_docs():
    """预构建的 BM25 索引。"""
    docs = [
        "RAG 系统结合了信息检索与文本生成技术",
        "知识库是 RAG 系统的核心组件用于存储文档",
        "向量数据库用于高效存储和检索高维向量",
        "今天天气晴朗适合户外运动",
        "混合检索结合关键词匹配和语义搜索提高准确性",
    ]
    bm25 = BM25Retriever()
    bm25.index(docs)
    return bm25, docs


def test_bm25_index(bm25_with_docs):
    bm25, docs = bm25_with_docs
    assert bm25.doc_count == 5


def test_bm25_search_returns_results(bm25_with_docs):
    bm25, docs = bm25_with_docs
    results = bm25.search("RAG 检索", top_k=3)
    assert len(results) > 0


def test_bm25_search_relevant_first(bm25_with_docs):
    bm25, docs = bm25_with_docs
    results = bm25.search("RAG 系统", top_k=3)
    top_doc = bm25.get_document(results[0][0])
    assert "RAG" in top_doc or "检索" in top_doc


def test_bm25_search_empty_query(bm25_with_docs):
    bm25, docs = bm25_with_docs
    results = bm25.search("", top_k=3)
    assert results == []


def test_bm25_search_no_match(bm25_with_docs):
    bm25, docs = bm25_with_docs
    results = bm25.search("zzzXYZnotexist", top_k=3)
    assert results == []  # 无匹配 token


def test_bm25_empty_index():
    bm25 = BM25Retriever()
    results = bm25.search("test")
    assert results == []


def test_bm25_get_document(bm25_with_docs):
    bm25, docs = bm25_with_docs
    assert bm25.get_document(0) == docs[0]
    assert bm25.get_document(999) == ""


# ==================== RRF 融合测试 ====================


def test_rrf_fusion_basic():
    """RRF 应融合两路结果。"""
    kw = [(0, 3.5), (1, 2.1)]
    vec = [(1, 0.95), (2, 0.80)]
    fused = rrf_fusion(kw, vec)
    assert len(fused) == 3
    # doc 1 在两路都有 → RRF 分数最高
    assert fused[0][0] == 1


def test_rrf_fusion_single_source():
    """仅一路有结果时也应正常工作。"""
    kw = [(0, 5.0), (1, 3.0)]
    vec = []
    fused = rrf_fusion(kw, vec)
    assert len(fused) == 2
    assert fused[0][0] == 0


def test_linear_fusion():
    """线性加权融合测试。"""
    kw = [(0, 3.0), (1, 2.0)]
    vec = [(1, 0.9), (2, 0.8)]
    fused = linear_fusion(kw, vec, keyword_weight=0.3, vector_weight=0.7)
    assert len(fused) == 3


def test_rrf_k_param():
    """不同 k 值影响排序。"""
    kw = [(0, 1.0)]
    vec = [(0, 1.0), (1, 0.5)]
    fused_k60 = rrf_fusion(kw, vec, k=60)
    fused_k5 = rrf_fusion(kw, vec, k=5)
    assert fused_k60[0][1] != fused_k5[0][1]


# ==================== 检索结果去重测试 ====================


def _make_result(doc: str, score: float = 0.9, source: str = "/tmp/a.txt", chunk_index: int = 0) -> dict:
    return {
        "id": "test",
        "document": doc,
        "metadata": {"source": source, "chunk_index": chunk_index},
        "score": score,
    }


def test_dedup_identical_text():
    """完全相同的文本→去重。"""
    results = [
        _make_result("相同的文本内容A", score=0.9),
        _make_result("相同的文本内容A", score=0.8),
        _make_result("不同的文本内容B", score=0.7),
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2


def test_dedup_same_prefix():
    """前 80 字符相同→去重，保留 score 最高的（跨文件场景）。"""
    shared = "A" * 80 + "unique_tail"
    results = [
        _make_result(shared + "1", score=0.7, source="/tmp/a.txt"),
        _make_result(shared + "2", score=0.9, source="/tmp/b.txt"),
        _make_result("完全不同的内容", score=0.5, source="/tmp/c.txt"),
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2
    assert deduped[0]["score"] == 0.9


def test_dedup_same_file_multiple_chunks():
    """同一文件不同 chunk 的前缀不同 → 全部保留。"""
    results = [
        _make_result("A" * 100, score=0.9, source="/tmp/a.txt", chunk_index=0),
        _make_result("B" * 100, score=0.7, source="/tmp/a.txt", chunk_index=1),
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 2


def test_dedup_cross_file_same_prefix():
    """不同文件相同前缀 → 去重，保留 score 最高的。"""
    shared = "X" * 80
    results = [
        _make_result(shared + "a", score=0.6, source="/tmp/a.txt"),
        _make_result(shared + "b", score=0.9, source="/tmp/b.txt"),
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 1
    assert deduped[0]["score"] == 0.9


def test_dedup_empty():
    assert _dedup_results([]) == []


def test_dedup_single():
    results = [_make_result("唯一一条")]
    assert len(_dedup_results(results)) == 1


def test_dedup_no_duplicates():
    """无重复时保持原样。"""
    results = [
        _make_result("内容A", source="/tmp/a.txt", chunk_index=0),
        _make_result("内容B", source="/tmp/b.txt", chunk_index=0),
        _make_result("内容C", source="/tmp/c.txt", chunk_index=0),
    ]
    deduped = _dedup_results(results)
    assert len(deduped) == 3
