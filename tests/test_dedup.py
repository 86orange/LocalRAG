"""
内容去重模块测试

覆盖 SimHash 计算、汉明距离、ChunkDeduplicator、DualLayerDeduplicator。
"""
import pytest

from local_rag.utils.dedup import (
    compute_simhash,
    hamming_distance,
    is_near_duplicate,
    ChunkDeduplicator,
    DualLayerDeduplicator,
)


# ==================== SimHash 测试 ====================


def test_compute_simhash_nonzero():
    """非空文本应产生非零指纹。"""
    h = compute_simhash("这是测试文本")
    assert h != 0


def test_compute_simhash_empty():
    """空文本应返回 0。"""
    assert compute_simhash("") == 0
    assert compute_simhash("   ") == 0


def test_compute_simhash_deterministic():
    """相同文本应产生相同指纹。"""
    text = "知识库是RAG系统的核心组件，用于存储和检索文档内容"
    assert compute_simhash(text) == compute_simhash(text)


def test_compute_simhash_same_text_zero_distance():
    """完全相同文本汉明距离应为 0。"""
    text_a = "RAG系统结合了检索和生成两个步骤来提高回答准确性" * 3
    text_b = "RAG系统结合了检索和生成两个步骤来提高回答准确性" * 3
    dist = hamming_distance(compute_simhash(text_a), compute_simhash(text_b))
    assert dist == 0


def test_compute_simhash_different_texts():
    """完全不同的文本应产生大汉明距离。"""
    text_a = "今天天气晴朗适合出游" * 10
    text_b = "人工智能模型训练需要大量GPU计算资源" * 10
    dist = hamming_distance(compute_simhash(text_a), compute_simhash(text_b))
    assert dist > 10, f"不同文本汉明距离应较大，实际 {dist}"


def test_hamming_distance_self_zero():
    """同一指纹的汉明距离应为 0。"""
    h = compute_simhash("测试文本")
    assert hamming_distance(h, h) == 0


def test_is_near_duplicate_true():
    """相同内容应判定为近重复。"""
    text = "RAG 是一种检索增强生成技术，它结合了信息检索与文本生成" * 3
    h1 = compute_simhash(text)
    h2 = compute_simhash(text)
    assert is_near_duplicate(h1, h2)


def test_is_near_duplicate_false():
    """完全不同内容不应判定为重复。"""
    h1 = compute_simhash("今天吃火锅" * 10)
    h2 = compute_simhash("深度学习的核心是反向传播算法" * 10)
    assert not is_near_duplicate(h1, h2)


# ==================== ChunkDeduplicator 测试 ====================


def test_dedup_first_chunk_not_duplicate():
    """第一个 chunk 不应被判定为重复。"""
    dedup = ChunkDeduplicator()
    assert not dedup.is_duplicate("全新的内容")


def test_dedup_same_chunk_is_duplicate():
    """完全相同内容第二次应被判定为重复。"""
    dedup = ChunkDeduplicator()
    chunk = "文档去重是数据治理的关键步骤重复信息会浪费存储空间并降低检索精度" * 2
    assert not dedup.is_duplicate(chunk)
    dedup.mark_indexed(chunk)
    assert dedup.is_duplicate(chunk)


def test_dedup_empty_text_is_duplicate():
    """空文本应判定为重复。"""
    dedup = ChunkDeduplicator()
    assert dedup.is_duplicate("")


def test_dedup_filter_chunks():
    """filter_chunks 应过滤掉重复块。"""
    dedup = ChunkDeduplicator(hamming_threshold=3)
    chunks = [
        "唯一的内容块主要用于验证去重功能是否正常工作中长远内容块足够多数据",
        "唯一的内容块主要用于验证去重功能是否正常工作中长远内容块足够多数据",
        "另一段完全不同内容的文本块B包含不同信息",
    ]
    metas = [{"idx": i} for i in range(3)]
    kept_c, kept_m = dedup.filter_chunks(chunks, metas)
    assert len(kept_c) == 2


def test_dedup_similar_long_chunks_filtered():
    """共享大量相同内容的两段文本应被检测为近重复。"""
    dedup = ChunkDeduplicator(hamming_threshold=3)
    common = "企业知识库系统的核心价值在于将分散在各个部门的文档资料进行统一管理" * 3
    base = common + "结尾段落A收尾句子不同"
    dedup.mark_indexed(base)
    similar = common + "结尾段落B收尾句略有差异"
    assert dedup.is_duplicate(similar)


def test_dedup_stats():
    """统计信息应正确计数。"""
    dedup = ChunkDeduplicator()
    dedup.is_duplicate("这个是一个测试文本内容A")
    dedup.mark_indexed("这个是一个测试文本内容A")
    dedup.is_duplicate("这个是一个测试文本内容A")  # 重复
    dedup.is_duplicate("这个是一个完全不同的文本")
    stats = dedup.get_stats()
    assert stats["total_checked"] == 3
    assert stats["duplicates_found"] == 1


# ==================== DualLayerDeduplicator 测试 ====================


def test_dual_layer_no_store():
    """无 store 时只做 SimHash 层。"""
    dedup = DualLayerDeduplicator(store=None)
    chunk = "一段测试文本内容用于验证双层去重器基本功能" * 2
    assert not dedup.is_duplicate(chunk)
    dedup.mark_indexed(chunk)
    assert dedup.is_duplicate(chunk)


def test_dual_layer_filter_chunks():
    """双层过滤应正确返回去重结果。"""
    dedup = DualLayerDeduplicator(store=None)
    chunks = [
        "文本块A用于测试去重功能验证是否正确工作包含足够长度",
        "文本块A用于测试去重功能验证是否正确工作包含足够长度",
        "文本块B完全不同的内容用于验证去重后保留不同块",
    ]
    metas = [{"i": i} for i in range(3)]
    kept_c, kept_m = dedup.filter_chunks(chunks, metas)
    assert len(kept_c) == 2


def test_dual_layer_stats():
    """统计信息应包含两层数据。"""
    dedup = DualLayerDeduplicator(store=None)
    chunk = "统计功能测试文本用于验证去重统计信息的正确性"
    dedup.is_duplicate(chunk)
    dedup.mark_indexed(chunk)
    dedup.is_duplicate(chunk)
    stats = dedup.get_stats()
    assert "simhash_checked" in stats
    assert "simhash_duplicates" in stats
    assert "vector_duplicates" in stats


# ==================== 中英文混合场景 ====================


def test_simhash_mixed_language():
    """中英混合文本也应有稳定指纹。"""
    text_a = "RAG (Retrieval-Augmented Generation) 结合了检索和生成" * 2
    text_b = "RAG (Retrieval-Augmented Generation) 结合了检索和生成" * 2
    assert compute_simhash(text_a) == compute_simhash(text_b)


def test_dedup_full_paragraph_similar():
    """段落级别的近重复文本应被检测到。"""
    dedup = ChunkDeduplicator(hamming_threshold=6)
    para = (
        "企业知识库系统的核心价值在于将分散在各个部门"
        "和员工电脑中的文档资料进行统一管理和检索"
        "通过引入RAG技术可以实现自然语言问答大幅提升知识获取效率"
    )
    dedup.mark_indexed(para)
    similar = (
        "企业知识库系统的核心价值在于将分散在各个部门"
        "和员工电脑中的文档资料进行统一管理与检索"
        "通过引入RAG技术可以实现自然语言问答大幅提升知识获取效率"
    )
    assert dedup.is_duplicate(similar)
