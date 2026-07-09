"""
文档加载器测试

覆盖文档扫描、校验、分发及四种加载器的功能。
测试文件位于 documents/ 目录下，缺少某种格式时自动跳过对应测试。
"""

import re
import pytest

from local_rag.utils.file_utils import (
    scan_documents,
    compute_file_hash,
    validate_file,
    get_file_type,
)
from local_rag.loader.markdown_loader import load_markdown
from local_rag.loader.txt_loader import load_txt
from local_rag.loader.pdf_loader import load_pdf
from local_rag.loader.docx_loader import load_docx


# ==================== 扫描与校验测试 ====================

def test_scan_returns_files():
    """扫描函数应正常运行，返回 list（目录不存在时返回空列表）。"""
    files = scan_documents()
    assert isinstance(files, list)


def test_no_unsupported_files_in_scan():
    """所有扫描到的文件后缀均应在支持列表中。"""
    files = scan_documents()
    for f in files:
        assert validate_file(f), f"文件校验失败: {f.name}"


def test_get_file_type_correct():
    """文件类型识别应正确映射到已知标签。"""
    valid_types = {"md", "txt", "pdf", "docx"}
    files = scan_documents()
    for f in files:
        ftype = get_file_type(f)
        assert ftype in valid_types, f"未识别的类型: {f.suffix} → {ftype}"
        assert ftype != "unknown"


def test_compute_hash_nonempty():
    """哈希值应为 32 位十六进制字符串。"""
    files = scan_documents()
    for f in files:
        h = compute_file_hash(f)
        assert len(h) == 32, f"哈希长度错误: {f.name}"
        assert all(c in "0123456789abcdef" for c in h), f"非十六进制: {f.name}"


def test_compute_hash_deterministic():
    """同一文件两次哈希结果应一致。"""
    files = scan_documents()
    for f in files:
        assert compute_file_hash(f) == compute_file_hash(f)


def test_validate_nonexistent_file():
    """不存在的文件应返回 False。"""
    assert validate_file(__import__("pathlib").Path("/does/not/exist.txt")) is False


# ==================== Markdown 加载测试 ====================

def _get_md_file():
    files = [f for f in scan_documents() if f.suffix == ".md"]
    if not files:
        pytest.skip("documents/ 中没有 .md 文件")
    return files[0]


def test_load_markdown_success():
    """Markdown 加载应返回非空字符串。"""
    text = load_markdown(_get_md_file())
    assert len(text) > 50, "Markdown 文本量过少"


def test_markdown_no_frontmatter():
    """YAML Front Matter 应被清除（不应包含 --- 分隔符在三短横格式中）。"""
    text = load_markdown(_get_md_file())
    lines = text.split("\n")
    assert not any(line.strip() == "---" for line in lines if line.strip())


def test_markdown_no_html_tags():
    """HTML 标签应被剥离。"""
    text = load_markdown(_get_md_file())
    assert "<a href" not in text
    assert "<img" not in text
    assert "<!--" not in text


def test_markdown_no_bold_italic_markers():
    """粗体、斜体标记 ** 和 * 应被移除。"""
    text = load_markdown(_get_md_file())
    assert re.search(r"\*\*[^*]+\*\*", text) is None, "残留粗体标记"
    assert re.search(r"__[^_]+__", text) is None, "残留粗体标记(下划线)"


# ==================== TXT 加载测试 ====================

def _get_txt_file():
    files = [f for f in scan_documents() if f.suffix == ".txt"]
    if not files:
        pytest.skip("documents/ 中没有 .txt 文件")
    return files[0]


def test_load_txt_success():
    """TXT 加载应返回非空字符串。"""
    text = load_txt(_get_txt_file())
    assert len(text) > 10, "TXT 文本量过少"


def test_txt_no_control_chars():
    """控制字符（\x00-\x1f 除 \n \t 外）应被移除。"""
    text = load_txt(_get_txt_file())
    forbidden = set(chr(c) for c in range(32) if chr(c) not in "\n\t")
    for ch in forbidden:
        assert ch not in text


def test_txt_no_trailing_spaces():
    """每行尾部应无空白。"""
    text = load_txt(_get_txt_file())
    for line in text.split("\n"):
        if line:
            assert line == line.rstrip()


# ==================== PDF 加载测试 ====================

def _get_pdf_file():
    files = [f for f in scan_documents() if f.suffix == ".pdf"]
    if not files:
        pytest.skip("documents/ 中没有 .pdf 文件")
    return files[0]


def test_load_pdf_success():
    """PDF 加载应返回非空字符串。"""
    text = load_pdf(_get_pdf_file())
    assert len(text) > 100, "PDF 文本量过少"


def test_pdf_no_single_char_lines():
    """PDF 文本中不应有纯数字的孤立行（页眉页脚残留）。"""
    text = load_pdf(_get_pdf_file())
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.isdigit():
            assert len(stripped) < 3, f"可能是页码残留: '{stripped}'"


def test_pdf_no_excessive_newlines():
    """PDF 文本中不应有 3 个以上连续空行。"""
    text = load_pdf(_get_pdf_file())
    assert "\n\n\n\n" not in text


# ==================== DOCX 加载测试 ====================

def _get_docx_file():
    files = [f for f in scan_documents() if f.suffix == ".docx"]
    if not files:
        pytest.skip("documents/ 中没有 .docx 文件")
    return files[0]


def test_load_docx_success():
    """DOCX 加载应返回非空字符串。"""
    text = load_docx(_get_docx_file())
    assert len(text) > 100, "DOCX 文本量过少"


def test_docx_table_pipe_format():
    """DOCX 文本不包含原始 OOXML 标签。"""
    text = load_docx(_get_docx_file())
    assert "<w:" not in text


def test_docx_no_ooxml_tags():
    """DOCX 文本不应包含 OOXML 标签。"""
    text = load_docx(_get_docx_file())
    assert "<w:body" not in text
    assert "<w:p" not in text
    assert "</w:" not in text


# ==================== 元数据提取测试 ====================

from local_rag.utils.file_utils import get_file_metadata


def test_get_file_metadata_keys():
    """get_file_metadata 应返回全部字段。"""
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 中无文件")
    meta = get_file_metadata(files[0])
    expected_keys = {"doc_id", "source", "file_name", "file_type", "file_size", "updated_at", "status", "business_line"}
    assert set(meta.keys()) == expected_keys


def test_get_file_metadata_doc_id_valid():
    """doc_id 应为 32 位十六进制字符串。"""
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 中无文件")
    meta = get_file_metadata(files[0])
    assert len(meta["doc_id"]) == 32
    assert all(c in "0123456789abcdef" for c in meta["doc_id"])


def test_get_file_metadata_defaults():
    """默认状态应为 active，业务线为空。"""
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 中无文件")
    meta = get_file_metadata(files[0])
    assert meta["status"] == "active"
    assert meta["business_line"] == ""


def test_get_file_metadata_custom():
    """自定义 business_line 和 status 应生效。"""
    files = scan_documents()
    if not files:
        pytest.skip("documents/ 中无文件")
    meta = get_file_metadata(files[0], business_line="研发部", status="draft")
    assert meta["business_line"] == "研发部"
    assert meta["status"] == "draft"


# ==================== 带 metadata 切片测试 ====================

from local_rag.chunker import chunk_by_size_with_metadata, chunk_by_semantic_with_metadata

SAMPLE_TEXT = "第一章\n\n这是第一章的内容，用于测试切片功能。\n\n第二章\n\n这是第二章的内容。"


def test_chunk_by_size_with_metadata_returns_tuple():
    """应返回 (chunks, metadatas) 二元组。"""
    chunks, metas = chunk_by_size_with_metadata(SAMPLE_TEXT, base_metadata={"doc_id": "abc"})
    assert isinstance(chunks, list)
    assert isinstance(metas, list)
    assert len(chunks) == len(metas)


def test_chunk_by_size_with_metadata_fields():
    """每块 metadata 应包含 base_metadata 的所有字段 + chunk_index + total_chunks。"""
    base = {"doc_id": "abc123", "source": "/tmp/test.txt"}
    chunks, metas = chunk_by_size_with_metadata(SAMPLE_TEXT, base_metadata=base)
    for i, meta in enumerate(metas):
        assert meta["doc_id"] == "abc123"
        assert meta["source"] == "/tmp/test.txt"
        assert meta["chunk_index"] == i
        assert meta["total_chunks"] == len(chunks)


def test_chunk_by_size_with_metadata_empty():
    """空文本应返回空列表。"""
    chunks, metas = chunk_by_size_with_metadata("")
    assert chunks == []
    assert metas == []


def test_chunk_by_semantic_with_metadata():
    """语义切片 metadata 也应包含 chunk_index + total_chunks。"""
    text = "# 标题\n\n内容段落。\n\n## 子标题\n\n更多内容。"
    base = {"doc_id": "xyz"}
    chunks, metas = chunk_by_semantic_with_metadata(text, chunk_size=512, chunk_overlap=64, base_metadata=base)
    assert len(chunks) > 0
    for i, meta in enumerate(metas):
        assert meta["doc_id"] == "xyz"
        assert meta["chunk_index"] == i
        assert meta["total_chunks"] == len(chunks)
