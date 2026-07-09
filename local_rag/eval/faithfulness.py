"""
答案忠实度评估（LLM-as-Judge）

评估 LLM 生成的回答是否忠实于检索到的参考文档：
1. 逐句拆解回答中的主张
2. 对每条主张检查是否在上下文中找到直接证据
3. 计算忠实度分数 = 有证据的主张数 / 总主张数
4. 验证引用标注的准确性
"""

import json
import re
from dataclasses import dataclass, field

import ollama

from local_rag.config import LLM_MODEL, OLLAMA_HOST
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# 默认评估用轻量模型（速度快、成本低）
EVAL_LLM_MODEL = LLM_MODEL

# ==================== Prompt 模板 ====================

FAITHFULNESS_PROMPT = """你是一个严格的 QA 质量审核员。你的任务是评估 AI 助手的回答是否忠实于提供的参考文档。

## 任务

给定：
1. 参考文档片段（唯一的信息来源）
2. 用户问题
3. AI 助手的回答

请完成以下评估：

### 步骤 1：拆解主张
将 AI 回答拆解为独立的主张（claim），每条主张是一个可验证的陈述句。
- 忽略纯过渡性语句（如"根据文档"、"综上所述"）
- 每个事实、数据、结论各算一条主张

### 步骤 2：逐条验证
对每条主张，在参考文档中查找直接证据。
- "支持"：文档中明确包含该主张的内容
- "矛盾"：文档中有内容与该主张冲突
- "无依据"：文档中完全没有相关信息（即 AI 编造的）
- "部分支持"：文档中提到相关内容但不完全一致

### 步骤 3：引用验证
检查 AI 回答中的 [来源:X] 引用标注。
- "正确"：引用指向的文档片段确实包含该主张
- "错误"：引用指向的文档片段不包含该主张
- "虚构"：引用的来源编号在文档中不存在

## 参考文档片段

{context}

## 用户问题

{question}

## AI 回答

{answer}

## 输出格式

请严格按以下 JSON 格式输出，不要包含任何其他内容：

```json
{{
  "claims": [
    {{
      "text": "主张原文",
      "verdict": "支持|矛盾|无依据|部分支持",
      "evidence": "证据片段（无依据时填'无'）",
      "citation_check": {{
        "citations_found": ["来源:1"],
        "accuracy": "正确|错误|虚构|无引用"
      }}
    }}
  ],
  "summary": {{
    "total_claims": 0,
    "supported": 0,
    "contradicted": 0,
    "unsupported": 0,
    "partial": 0,
    "citation_correct": 0,
    "citation_wrong": 0,
    "citation_fabricated": 0,
    "faithfulness_score": 0.0
  }}
}}
```

注意：faithfulness_score = (supported + 0.5 * partial) / total_claims，取值范围 0-1。"""


# ==================== 数据结构 ====================


@dataclass
class ClaimResult:
    text: str
    verdict: str         # 支持 / 矛盾 / 无依据 / 部分支持
    evidence: str
    citation_check: dict = field(default_factory=dict)


@dataclass
class FaithfulnessResult:
    claims: list[ClaimResult] = field(default_factory=list)
    total_claims: int = 0
    supported: int = 0
    contradicted: int = 0
    unsupported: int = 0
    partial: int = 0
    citation_correct: int = 0
    citation_wrong: int = 0
    citation_fabricated: int = 0
    faithfulness_score: float = 0.0
    raw_response: str = ""
    error: str = ""


# ==================== 核心评估函数 ====================


def evaluate_faithfulness(
    question: str,
    context: str,
    answer: str,
    model: str | None = None,
) -> FaithfulnessResult:
    """评估 LLM 回答的忠实度。

    Args:
        question: 用户问题
        context: 检索到的参考文档上下文
        answer: LLM 生成的回答
        model: 用于评估的模型名，默认使用全局 LLM 模型

    Returns:
        FaithfulnessResult 包含逐条主张的评估结果及总分
    """
    model = model or EVAL_LLM_MODEL

    if not answer.strip():
        return FaithfulnessResult(
            total_claims=0,
            faithfulness_score=1.0,
            error="回答为空",
        )

    if not context.strip():
        return FaithfulnessResult(
            total_claims=1,
            unsupported=1,
            faithfulness_score=0.0,
            error="无参考文档上下文",
        )

    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
    )

    raw_response = ""
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"temperature": 0.0},
        )
        raw_response = response["response"].strip()
    except Exception as e:
        logger.error("忠实度评估 LLM 调用失败: %s", e)
        return FaithfulnessResult(
            total_claims=0,
            error=f"LLM 调用失败: {e}",
        )

    result = _parse_response(raw_response)
    result.raw_response = raw_response
    return result


# ==================== 内部解析 ====================


def _parse_response(raw: str) -> FaithfulnessResult:
    """从 LLM 返回的 JSON 中解析忠实度结果。"""
    try:
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            raise ValueError("未找到有效 JSON")

        data = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("忠实度结果 JSON 解析失败: %s, raw=%s", e, raw[:200])
        return FaithfulnessResult(error=f"JSON 解析失败: {e}", raw_response=raw)

    claims_raw = data.get("claims", [])
    summary = data.get("summary", {})

    claims = []
    for c in claims_raw:
        claims.append(ClaimResult(
            text=c.get("text", ""),
            verdict=c.get("verdict", "无依据"),
            evidence=c.get("evidence", ""),
            citation_check=c.get("citation_check", {}),
        ))

    return FaithfulnessResult(
        claims=claims,
        total_claims=summary.get("total_claims", len(claims)),
        supported=summary.get("supported", 0),
        contradicted=summary.get("contradicted", 0),
        unsupported=summary.get("unsupported", 0),
        partial=summary.get("partial", 0),
        citation_correct=summary.get("citation_correct", 0),
        citation_wrong=summary.get("citation_wrong", 0),
        citation_fabricated=summary.get("citation_fabricated", 0),
        faithfulness_score=summary.get("faithfulness_score", 0.0),
    )
