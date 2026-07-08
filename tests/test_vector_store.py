"""
向量存储模块测试

验证 ChromaStore 的增删改查功能。
需要 Ollama 正常启动且已拉取配置的 embedding 模型。
"""

import tempfile
from pathlib import Path

import pytest

from local_rag.vector_store.chroma_store import ChromaStore

# 测试用示例文本块
SAMPLE_CHUNKS = [
    "Python 是一种解释型、面向对象的高级编程语言。",
    "RAG（检索增强生成）结合了信息检索与文本生成技术。",
    "ChromaDB 是一个开源的向量数据库，专为 AI 应用设计。",
    "Ollama 允许在本地运行大型语言模型，无需云端依赖。",
    "向量化是将文本转换为高维数值表示的过程。",
]

SAMPLE_METADATAS = [
    {"source": "test/python_intro.md", "chunk_index": 0},
    {"source": "test/rag_overview.md", "chunk_index": 0},
    {"source": "test/chromadb_intro.md", "chunk_index": 0},
    {"source": "test/ollama_intro.md", "chunk_index": 0},
    {"source": "test/embeddings.md", "chunk_index": 0},
]


@pytest.fixture
def store() -> ChromaStore:
    """创建临时目录的 ChromaStore 实例，测试后自动清理。

    使用固定临时目录避免 Windows 文件锁导致的 PermissionError。
    """
    import shutil

    tmpdir = Path(__file__).parent / "test_chroma_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    s = ChromaStore(persist_dir=tmpdir)
    s.delete_all()
    yield s
    s.delete_all()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ==================== 初始化测试 ====================

def test_store_initialization(store: ChromaStore):
    """ChromaStore 应正常初始化且 stats 可调用。"""
    stats = store.get_stats()
    assert stats["total_chunks"] >= 0
    assert stats["total_sources"] >= 0
    assert "persist_dir" in stats
    assert "embedding_model" in stats


# ==================== 文档入库测试 ====================

def test_add_documents_success(store: ChromaStore):
    """文档入库应返回对应数量的 ID。"""
    ids = store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    assert len(ids) == len(SAMPLE_CHUNKS)
    for id_ in ids:
        assert isinstance(id_, str) and len(id_) > 0


def test_add_documents_no_metadata(store: ChromaStore):
    """省略 metadata 应自动填充空 dict。"""
    ids = store.add_documents(SAMPLE_CHUNKS)
    assert len(ids) == len(SAMPLE_CHUNKS)


def test_add_documents_auto_generated_ids(store: ChromaStore):
    """未指定 IDs 时应自动生成 UUID。"""
    ids = store.add_documents(SAMPLE_CHUNKS[:2])
    assert len(ids) == 2
    assert ids[0] != ids[1]
    # UUID 格式: 8-4-4-4-12
    for id_ in ids:
        parts = id_.split("-")
        assert len(parts) == 5


def test_add_duplicate_chunks(store: ChromaStore):
    """重复入库相同内容应正常执行（不同 ID）。"""
    ids1 = store.add_documents(SAMPLE_CHUNKS[:2])
    ids2 = store.add_documents(SAMPLE_CHUNKS[:2])
    assert len(ids1) == 2
    assert len(ids2) == 2
    assert ids1 != ids2


def test_add_documents_mismatched_metadata_raises(store: ChromaStore):
    """metadata 长度与 chunks 不匹配应报 ValueError。"""
    with pytest.raises(ValueError):
        store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS[:2])


# ==================== 检索测试 ====================

def test_search_returns_results(store: ChromaStore):
    """入库后检索应返回结果。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    results = store.search("什么是 Python？", top_k=2)
    assert len(results) <= 2
    assert len(results) >= 1


def test_search_result_structure(store: ChromaStore):
    """每条结果应包含 id / document / metadata / score。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    results = store.search("RAG 技术", top_k=1)
    assert len(results) == 1
    item = results[0]
    assert "id" in item
    assert "document" in item
    assert "metadata" in item
    assert "score" in item
    assert isinstance(item["score"], float)


def test_search_score_range(store: ChromaStore):
    """score 应在 0~1 之间（余弦相似度转换）。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    results = store.search("向量数据库", top_k=3)
    for item in results:
        assert 0.0 <= item["score"] <= 1.0, f"score={item['score']} 超出范围"


def test_search_empty_collection(store: ChromaStore):
    """空集合检索应返回空列表。"""
    results = store.search("任何问题")
    assert results == []


def test_search_with_where_filter(store: ChromaStore):
    """按 source 过滤应仅返回匹配来源的结果。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    results = store.search("Python", top_k=3, where={"source": "test/python_intro.md"})
    for item in results:
        assert item["metadata"].get("source") == "test/python_intro.md"


def test_search_top_k_larger_than_collection(store: ChromaStore):
    """请求数超过库中总数时返回实际数量。"""
    store.add_documents(SAMPLE_CHUNKS[:3], SAMPLE_METADATAS[:3])
    results = store.search("测试", top_k=100)
    assert len(results) == 3


# ==================== 删除测试 ====================

def test_delete_by_source(store: ChromaStore):
    """按 source 删除后检索该来源应为空。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    deleted = store.delete_by_source("test/python_intro.md")
    assert deleted == 1

    results = store.search("Python", where={"source": "test/python_intro.md"})
    assert len(results) == 0


def test_delete_nonexistent_source(store: ChromaStore):
    """删除不存在的 source 应返回 0。"""
    deleted = store.delete_by_source("nonexistent/file.md")
    assert deleted == 0


def test_delete_all(store: ChromaStore):
    """delete_all 后 stats 应归零。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    count_before = store.get_stats()["total_chunks"]
    assert count_before == len(SAMPLE_CHUNKS)

    deleted = store.delete_all()
    assert deleted == len(SAMPLE_CHUNKS)

    count_after = store.get_stats()["total_chunks"]
    assert count_after == 0


# ==================== 统计测试 ====================

def test_get_stats_after_add(store: ChromaStore):
    """入库后 stats 应反映正确的文档数和来源数。"""
    store.add_documents(SAMPLE_CHUNKS, SAMPLE_METADATAS)
    stats = store.get_stats()
    assert stats["total_chunks"] == len(SAMPLE_CHUNKS)
    # 每个 chunk 有不同 source
    assert stats["total_sources"] == len(SAMPLE_CHUNKS)


def test_get_stats_shared_source(store: ChromaStore):
    """同一来源的多个 chunk 应只计为一个 source。"""
    chunks = SAMPLE_CHUNKS[:3]
    metas = [{"source": "test/shared.md", "chunk_index": i} for i in range(3)]
    store.add_documents(chunks, metas)
    stats = store.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["total_sources"] == 1
