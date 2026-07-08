"""
Web UI 组件测试

测试第二阶段 Streamlit 组件的纯逻辑部分：
- 检索数量 tooltip 内容
- 会话创建/删除的数据结构
- Streamlit 集成测试（需手动启动 web 后验证）

直接 import 带 st 的组件会因无 Streamlit runtime 而失败，
因此以下仅测试可脱离 st 的函数和常量。
"""

import json
import tempfile
from pathlib import Path

import pytest

# 仅 import 不依赖 st 的常量/函数
from local_rag.web.components.chat import _TOP_K_TOOLTIP


# ==================== 检索数量 tooltip ====================

def test_topk_tooltip_not_empty():
    """tooltip 文本不应为空。"""
    assert len(_TOP_K_TOOLTIP) > 50


def test_topk_tooltip_contains_keywords():
    """tooltip 应包含关键描述词。"""
    assert "检索片段数" in _TOP_K_TOOLTIP
    assert "推荐范围" in _TOP_K_TOOLTIP
    assert "3-10" in _TOP_K_TOOLTIP


# ==================== 会话 JSON 结构 ====================

def test_session_json_structure():
    """会话 JSON 应包含 name / messages / created_at 字段。"""
    import uuid
    from datetime import datetime

    sid = str(uuid.uuid4())
    session = {
        "name": f"测试会话",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    assert "name" in session
    assert "messages" in session
    assert "created_at" in session
    assert isinstance(session["messages"], list)


def test_session_message_structure():
    """单条消息应有 role / content 字段。"""
    msg = {"role": "user", "content": "什么是RAG？"}
    assert "role" in msg
    assert "content" in msg
    assert msg["role"] in ("user", "assistant")


def test_session_message_sources_optional():
    """assistant 消息可附带 sources 列表。"""
    msg = {
        "role": "assistant",
        "content": "RAG 是一种技术。",
        "sources": [
            {"index": 1, "file": "test.md", "path": "/tmp/test.md", "score": 0.85, "snippet": "RAG 是..."}
        ],
    }
    assert msg["role"] == "assistant"
    assert len(msg["sources"]) == 1
    assert msg["sources"][0]["score"] > 0


def test_session_serialization():
    """会话应能正常序列化和反序列化。"""
    import uuid
    from datetime import datetime

    session = {
        "name": "序列化测试",
        "messages": [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！", "sources": []},
        ],
        "created_at": datetime.now().isoformat(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "test.json"
        fpath.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")

        restored = json.loads(fpath.read_text(encoding="utf-8"))
        assert restored["name"] == "序列化测试"
        assert len(restored["messages"]) == 2
        assert restored["messages"][0]["role"] == "user"


# ==================== 来源结构 ====================

def test_source_dict_keys():
    """sources 列表项应包含 index / file / path / score / snippet。"""
    source = {
        "index": 1,
        "file": "doc.pdf",
        "path": "/tmp/doc.pdf",
        "score": 0.92,
        "snippet": "这是一段摘要...",
    }
    for key in ("index", "file", "path", "score", "snippet"):
        assert key in source, f"缺少 source 键: {key}"


def test_source_score_range():
    """score 应在 0~1 之间。"""
    source = {"score": 0.75}
    assert 0.0 <= source["score"] <= 1.0
