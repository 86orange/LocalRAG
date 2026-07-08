"""
端到端集成测试

验证完整 RAG 流水线：
文档扫描 → 加载 → 切片 → 向量化入库 → 检索 → LLM 回答

需要 Ollama 正在运行且已拉取 LLM + Embedding 模型。
"""

from pathlib import Path

import pytest

from local_rag.utils.file_utils import scan_documents, validate_file, get_file_type
from local_rag.loader import LOADER_MAP
from local_rag.chunker.text_chunker import chunk_by_size
from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.qa.chain import generate_answer


@pytest.fixture
def temp_store() -> ChromaStore:
    """创建临时向量库，测试后自动清理。"""
    import shutil

    tmpdir = Path(__file__).parent / "test_integration_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    store = ChromaStore(persist_dir=tmpdir)
    store.delete_all()
    yield store
    store.delete_all()
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.integration
def test_full_pipeline_index_and_search(temp_store: ChromaStore):
    """完整流程：加载 → 切片 → 入库 → 检索。

    如果 documents/ 中有文件，则对其执行完整索引入库与检索验证。
    如果为空则跳过。
    """
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 目录为空，跳过集成测试")

    total_chunks = 0

    for file_path in files[:2]:  # 最多测 2 个文件以节省时间
        if not validate_file(file_path):
            continue

        ftype = get_file_type(file_path)
        loader = LOADER_MAP.get(ftype)
        if loader is None:
            continue

        text = loader(file_path)
        if not text.strip():
            continue

        chunks = chunk_by_size(text)
        if not chunks:
            continue

        source = str(file_path.resolve())
        metadatas = [
            {"source": source, "chunk_index": j, "file_type": ftype}
            for j in range(len(chunks))
        ]
        temp_store.add_documents(chunks, metadatas)
        total_chunks += len(chunks)

    assert total_chunks > 0, "未成功索引任何文档"

    # 检索验证
    results = temp_store.search("测试查询", top_k=3)
    assert isinstance(results, list)
    if results:
        assert "document" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]


@pytest.mark.integration
def test_full_pipeline_with_llm(temp_store: ChromaStore):
    """完整流程含 LLM：检索 + 生成回答。

    需要 Ollama 运行且模型已拉取。
    """
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 目录为空，跳过集成测试")

    # 索引第一个文件
    file_path = files[0]
    if not validate_file(file_path):
        pytest.skip("文件校验失败")

    ftype = get_file_type(file_path)
    loader = LOADER_MAP.get(ftype)
    if loader is None:
        pytest.skip(f"未找到 loader: {ftype}")

    text = loader(file_path)
    if not text.strip():
        pytest.skip("文件为空")

    chunks = chunk_by_size(text)
    if not chunks:
        pytest.skip("切片后无内容")

    source = str(file_path.resolve())
    metadatas = [
        {"source": source, "chunk_index": j, "file_type": ftype}
        for j in range(len(chunks))
    ]
    temp_store.add_documents(chunks, metadatas)

    # 检索
    question = "这篇文档的主要内容是什么？"
    results = temp_store.search(question, top_k=3)
    if not results:
        pytest.skip("检索无结果")

    # 构建上下文
    contexts = []
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "未知")
        contexts.append(f"[来源 {i} - {Path(src).name}]\n{r['document']}")
    context_text = "\n\n---\n\n".join(contexts)

    # LLM 生成回答
    try:
        answer = generate_answer(context_text, question)
    except Exception as e:
        pytest.skip(f"Ollama 不可用: {e}")

    assert isinstance(answer, str)
    assert len(answer) > 10


@pytest.mark.integration
def test_index_update_flow(temp_store: ChromaStore):
    """索引覆盖流程：入库 → 重新索引同文件 → 旧数据应被替换。"""
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 目录为空")

    file_path = files[0]
    if not validate_file(file_path):
        pytest.skip("文件校验失败")

    ftype = get_file_type(file_path)
    loader = LOADER_MAP.get(ftype)
    if loader is None:
        pytest.skip(f"未找到 loader: {ftype}")

    text = loader(file_path)
    chunks = chunk_by_size(text)
    source = str(file_path.resolve())

    # 第一次入库
    metadatas = [{"source": source, "chunk_index": i, "file_type": ftype} for i in range(len(chunks))]
    ids1 = temp_store.add_documents(chunks, metadatas)
    assert len(ids1) == len(chunks)

    # 删除后重新入库（模拟更新）
    temp_store.delete_by_source(source)
    ids2 = temp_store.add_documents(chunks, metadatas)
    assert len(ids2) == len(chunks)

    stats = temp_store.get_stats()
    assert stats["total_chunks"] == len(chunks)
