"""
版本管理模块测试

覆盖 VersionManager 的 record / list / rollback / get_active 功能。
"""
import json
import tempfile
from pathlib import Path

import pytest

from local_rag.version_manager import VersionManager


@pytest.fixture
def vm():
    """创建临时目录下的 VersionManager 实例。"""
    import local_rag.version_manager as vm_module
    origin = vm_module.VERSION_DIR
    with tempfile.TemporaryDirectory() as tmp:
        vm_module.VERSION_DIR = Path(tmp)
        mgr = VersionManager()
        yield mgr
        vm_module.VERSION_DIR = origin


def test_record_first_version(vm):
    """首次记录应返回版本 1。"""
    v = vm.record("abc123", "测试.docx", 5)
    assert v == 1


def test_record_creates_json(vm):
    """记录后应生成 JSON 文件。"""
    vm.record("abc123", "测试.docx", 5)
    path = vm._log_path("abc123")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["doc_id"] == "abc123"
    assert len(data["versions"]) == 1
    assert data["active_version"] == 1


def test_record_has_all_fields(vm):
    """版本记录应包含所有字段。"""
    vm.record("abc123", "文档.docx", 8, metadata={"doc_id": "abc123"})
    versions = vm.list_versions("abc123")
    v = versions[0]
    assert v["version"] == 1
    assert v["file_name"] == "文档.docx"
    assert v["chunk_count"] == 8
    assert v["previous_chunk_count"] == 0
    assert v["delta"] == 8
    assert "created_at" in v
    assert "metadata" in v


def test_record_increments_version(vm):
    """多次记录应递增版本号。"""
    vm.record("abc123", "doc.docx", 5)
    vm.record("abc123", "doc.docx", 8)
    vm.record("abc123", "doc.docx", 12)
    versions = vm.list_versions("abc123")
    assert len(versions) == 3
    assert versions[0]["version"] == 3
    assert versions[1]["version"] == 2
    assert versions[2]["version"] == 1


def test_record_delta_tracks_change(vm):
    """delta 应正确反映 chunk 数量变化。"""
    vm.record("abc123", "doc.docx", 10)
    vm.record("abc123", "doc.docx", 15)
    vm.record("abc123", "doc.docx", 7)
    versions = vm.list_versions("abc123")
    assert versions[0]["delta"] == -8   # v3: 7-15 = -8
    assert versions[1]["delta"] == 5    # v2: 15-10 = 5
    assert versions[2]["delta"] == 10   # v1: 10-0 = 10


def test_get_active_version(vm):
    """新建时 active 应为最新版本号。"""
    vm.record("abc123", "doc.docx", 5)
    vm.record("abc123", "doc.docx", 8)
    assert vm.get_active_version("abc123") == 2


def test_rollback_to(vm):
    """回滚应更改 active_version。"""
    vm.record("abc123", "doc.docx", 5)
    vm.record("abc123", "doc.docx", 8)
    vm.record("abc123", "doc.docx", 12)
    assert vm.rollback_to("abc123", 2)
    assert vm.get_active_version("abc123") == 2


def test_rollback_invalid(vm):
    """无效版本号应返回 False。"""
    vm.record("abc123", "doc.docx", 5)
    assert not vm.rollback_to("abc123", 0)
    assert not vm.rollback_to("abc123", 99)


def test_no_versions_return_zero(vm):
    """无记录的文档应返回 active_version=0。"""
    assert vm.get_active_version("nonexistent") == 0
    assert vm.list_versions("nonexistent") == []


def test_get_all_docs(vm):
    """get_all_docs 应返回所有记录的 doc_id。"""
    vm.record("aaa", "a.docx", 5)
    vm.record("bbb", "b.docx", 3)
    docs = vm.get_all_docs()
    assert "aaa" in docs
    assert "bbb" in docs


def test_multiple_docs_independent(vm):
    """不同文档的版本号应独立。"""
    vm.record("aaa", "a.docx", 5)
    vm.record("aaa", "a.docx", 10)
    vm.record("bbb", "b.docx", 3)
    assert vm.get_active_version("aaa") == 2
    assert vm.get_active_version("bbb") == 1
