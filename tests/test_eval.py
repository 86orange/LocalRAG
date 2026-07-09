"""
答案忠实度评估模块测试

覆盖 JSON 解析、FaithfulnessResult 数据结构、边界情况。
"""

import pytest
from local_rag.eval.faithfulness import (
    _parse_response,
    FaithfulnessResult,
    ClaimResult,
    FAITHFULNESS_PROMPT,
)


# ==================== JSON 解析测试 ====================


def test_parse_valid_response():
    """正常 JSON 应正确解析。"""
    raw = '''```json
{
  "claims": [
    {
      "text": "RAG 系统能提升回答准确性",
      "verdict": "支持",
      "evidence": "文档中提到 RAG 可提升准确性",
      "citation_check": {"citations_found": ["来源:1"], "accuracy": "正确"}
    },
    {
      "text": "该技术由 OpenAI 发明",
      "verdict": "无依据",
      "evidence": "无",
      "citation_check": {"citations_found": [], "accuracy": "无引用"}
    }
  ],
  "summary": {
    "total_claims": 2,
    "supported": 1,
    "contradicted": 0,
    "unsupported": 1,
    "partial": 0,
    "citation_correct": 1,
    "citation_wrong": 0,
    "citation_fabricated": 0,
    "faithfulness_score": 0.5
  }
}
```'''
    result = _parse_response(raw)
    assert result.total_claims == 2
    assert result.supported == 1
    assert result.unsupported == 1
    assert result.faithfulness_score == 0.5
    assert len(result.claims) == 2
    assert result.claims[0].verdict == "支持"
    assert result.claims[1].verdict == "无依据"


def test_parse_partial_support():
    """部分支持的场景。"""
    raw = '''```json
{
  "claims": [
    {"text": "某事", "verdict": "部分支持", "evidence": "部分", "citation_check": {}}
  ],
  "summary": {
    "total_claims": 1, "supported": 0, "contradicted": 0,
    "unsupported": 0, "partial": 1,
    "citation_correct": 0, "citation_wrong": 0, "citation_fabricated": 0,
    "faithfulness_score": 0.5
  }
}
```'''
    result = _parse_response(raw)
    assert result.partial == 1
    assert result.faithfulness_score == 0.5


def test_parse_contradiction():
    """矛盾的场景。"""
    raw = '''```json
{
  "claims": [
    {"text": "某事", "verdict": "矛盾", "evidence": "文档说相反", "citation_check": {}}
  ],
  "summary": {
    "total_claims": 1, "supported": 0, "contradicted": 1,
    "unsupported": 0, "partial": 0,
    "citation_correct": 0, "citation_wrong": 0, "citation_fabricated": 0,
    "faithfulness_score": 0.0
  }
}
```'''
    result = _parse_response(raw)
    assert result.contradicted == 1
    assert result.faithfulness_score == 0.0


def test_parse_no_json():
    """无效输入应返回 error。"""
    result = _parse_response("这不是 JSON")
    assert result.error != ""
    assert result.total_claims == 0


def test_parse_empty_response():
    """空字符串应返回 error。"""
    result = _parse_response("")
    assert result.error != ""


def test_parse_claims_override_total():
    """claims 数量和 summary 不一致时以 summary 为准。"""
    raw = '''{
  "claims": [
    {"text": "A", "verdict": "支持", "evidence": "e", "citation_check": {}},
    {"text": "B", "verdict": "支持", "evidence": "e", "citation_check": {}},
    {"text": "C", "verdict": "支持", "evidence": "e", "citation_check": {}}
  ],
  "summary": {
    "total_claims": 3, "supported": 3, "contradicted": 0,
    "unsupported": 0, "partial": 0,
    "citation_correct": 2, "citation_wrong": 0, "citation_fabricated": 1,
    "faithfulness_score": 1.0
  }
}'''
    result = _parse_response(raw)
    assert result.total_claims == 3
    assert result.citation_correct == 2
    assert result.citation_fabricated == 1


# ==================== FaithfulnessResult 数据结构测试 ====================


def test_result_defaults():
    """默认值应正确。"""
    r = FaithfulnessResult()
    assert r.faithfulness_score == 0.0
    assert r.claims == []
    assert r.error == ""


def test_result_with_error():
    """带 error 的结果。"""
    r = FaithfulnessResult(unsupported=1, total_claims=1, faithfulness_score=0.0, error="TEST")
    assert r.error == "TEST"
    assert r.unsupported == 1


# ==================== Prompt 模板测试 ====================


def test_faithfulness_prompt_contains_key_sections():
    """Prompt 应包含关键部分。"""
    p = FAITHFULNESS_PROMPT
    assert "faithfulness_score" in p
    assert "参考文档片段" in p
    assert "拆解主张" in p
    assert "逐条验证" in p
    assert "引用验证" in p


def test_faithfulness_prompt_format():
    """Prompt 应支持 format 填充。"""
    p = FAITHFULNESS_PROMPT.format(
        context="测试上下文",
        question="测试问题",
        answer="测试回答",
    )
    assert "测试上下文" in p
    assert "测试问题" in p
    assert "测试回答" in p


# ==================== Recall 评估测试 ====================


def test_is_hit_exact():
    """精确匹配应命中。"""
    from local_rag.eval.recall import _is_hit
    assert _is_hit(["文档内容包含目标关键词"], ["目标关键词"])


def test_is_hit_partial():
    """部分匹配应命中（子串）。"""
    from local_rag.eval.recall import _is_hit
    assert _is_hit(["很长的文档文本"], ["文档"])


def test_is_hit_across_chunks():
    """跨多个 chunk 拼接后应命中。"""
    from local_rag.eval.recall import _is_hit
    assert _is_hit(["前半段", "后半段包含目标"], ["目标"])


def test_is_hit_miss():
    """不匹配应返回 False。"""
    from local_rag.eval.recall import _is_hit
    assert not _is_hit(["完全不相关的内容"], ["目标关键词"])


def test_is_hit_empty_relevant():
    """空 relevant_texts 应返回 False。"""
    from local_rag.eval.recall import _is_hit
    assert not _is_hit(["内容"], [])


def test_is_hit_multiple_relevant_any():
    """只要有一个 relevant_text 命中即可。"""
    from local_rag.eval.recall import _is_hit
    assert _is_hit(["只有关键词B"], ["关键词A", "关键词B"])


def test_recall_eval_result_defaults():
    """RecallEvalResult 默认值。"""
    from local_rag.eval.recall import RecallEvalResult
    r = RecallEvalResult(total=10, hit_count=8, recall_at_1=0.5, recall_at_3=0.7, recall_at_5=0.8, mrr=0.6)
    assert r.recall_at_1 == 0.5
    assert r.mrr == 0.6


def test_single_recall_first_rank_calculation():
    """first_rank 应正确记录第一个命中位置。"""
    from local_rag.eval.recall import _is_hit

    docs = ["无关", "命中文档", "其他"]
    relevant = ["命中"]
    for k, expected in [(1, False), (2, True), (3, True)]:
        assert _is_hit(docs[:k], relevant) == expected


def test_builtin_dataset_structure():
    """内置数据集应包含必要字段。"""
    from local_rag.eval.recall import builtin_dataset
    ds = builtin_dataset()
    assert len(ds) > 0
    for item in ds:
        assert "question" in item
        assert "relevant_texts" in item
        assert isinstance(item["relevant_texts"], list)
        assert len(item["relevant_texts"]) > 0


def test_builtin_dataset_json_roundtrip():
    """内置数据集应可 JSON 序列化/反序列化。"""
    import json, tempfile
    from pathlib import Path
    from local_rag.eval.recall import builtin_dataset, load_dataset

    ds = builtin_dataset()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(ds, f, ensure_ascii=False)
        tmp_path = f.name

    try:
        loaded = load_dataset(tmp_path)
        assert len(loaded) == len(ds)
        assert loaded[0]["question"] == ds[0]["question"]
    finally:
        Path(tmp_path).unlink(missing_ok=True)
