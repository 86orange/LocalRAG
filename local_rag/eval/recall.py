"""
召回命中率评估（Recall@K）

评估检索系统是否能将正确答案所在的文档片段召回，并计算：
- Recall@1 / Recall@3 / Recall@5：分别以 top-1/3/5 为窗口的召回率
- MRR (Mean Reciprocal Rank)：第一个正确答案排名的倒数均值
- Hit Rate：有没有至少命中一次

数据集格式（JSON）：
[
    {
        "question": "用户问题",
        "relevant_texts": ["答案必须包含的关键词/短语1", "短语2"]
    }
]

判断逻辑：检索结果中任一文档片段包含任意 relevant_texts → 命中。
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_rag.utils.logger import get_logger

logger = get_logger(__name__)


# ==================== 数据结构 ====================


@dataclass
class SingleRecallResult:
    question: str
    relevant_texts: list[str] = field(default_factory=list)
    recalled: bool = False
    first_rank: int = -1         # 第一个命中的排名，未命中为 -1
    top_k_hits: dict[int, bool] = field(default_factory=dict)  # k → 是否命中
    retrieved_docs: list[str] = field(default_factory=list)


@dataclass
class RecallEvalResult:
    total: int = 0
    hit_count: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    details: list[SingleRecallResult] = field(default_factory=list)


# ==================== 核心评估 ====================


def evaluate_recall(
    store: object,
    dataset: list[dict],
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> RecallEvalResult:
    """评估多路召回命中率。

    对数据集中的每个问题，调混合检索 → 检查 relevant_texts 是否在 top-K 结果中出现。

    Args:
        store: ChromaStore 实例
        dataset: 数据集列表，每项含 question / relevant_texts
        top_k: 检索窗口大小
        similarity_threshold: 检索阈值

    Returns:
        RecallEvalResult 含各 K 值的召回率、MRR 和逐题详情
    """
    from local_rag.retrieval import HybridRetriever

    hybrid = HybridRetriever(store)

    details: list[SingleRecallResult] = []

    for item in dataset:
        question = item.get("question", "")
        relevant_texts = item.get("relevant_texts", [])

        if not question or not relevant_texts:
            logger.warning("跳过无效数据项: %s", question)
            continue

        results = hybrid.search(
            question,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        docs = [r.get("document", "") for r in results]
        single = SingleRecallResult(
            question=question,
            relevant_texts=list(relevant_texts),
            retrieved_docs=docs,
        )

        single.top_k_hits = {}
        single.first_rank = -1

        for k in range(1, top_k + 1):
            hit = _is_hit(docs[:k], relevant_texts)
            single.top_k_hits[k] = hit
            if hit and single.first_rank == -1:
                single.first_rank = k

        single.recalled = single.top_k_hits.get(top_k, False)
        details.append(single)

    total = len(details)
    if total == 0:
        return RecallEvalResult(total=0)

    hit_count = sum(1 for d in details if d.recalled)
    recall_1 = sum(1 for d in details if d.top_k_hits.get(1, False)) / total
    recall_3 = sum(1 for d in details if d.top_k_hits.get(3, False)) / total if top_k >= 3 else 0
    recall_5 = sum(1 for d in details if d.top_k_hits.get(5, False)) / total if top_k >= 5 else 0

    # MRR
    reciprocal_sum = 0.0
    for d in details:
        if d.first_rank > 0:
            reciprocal_sum += 1.0 / d.first_rank
    mrr = reciprocal_sum / total

    logger.info(
        "Recall 评估完成: %d 题, Recall@1=%.2%%, Recall@3=%.2%%, Recall@5=%.2%%, MRR=%.4f",
        total, recall_1, recall_3, recall_5, mrr,
    )

    return RecallEvalResult(
        total=total,
        hit_count=hit_count,
        recall_at_1=round(recall_1, 4),
        recall_at_3=round(recall_3, 4) if top_k >= 3 else 0,
        recall_at_5=round(recall_5, 4) if top_k >= 5 else 0,
        mrr=round(mrr, 4),
        details=details,
    )


def load_dataset(path: str | Path) -> list[dict]:
    """从 JSON 文件加载评估数据集。

    Args:
        path: 数据集文件路径

    Returns:
        数据集列表
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("数据集必须是 JSON 数组格式")

    return data


# ==================== 内部 ====================


def _is_hit(docs: list[str], relevant_texts: list[str]) -> bool:
    """检查检索结果中是否包含任一相关文本。

    Args:
        docs: 检索到的文档片段列表
        relevant_texts: 必须命中的关键词/短语列表

    Returns:
        只要有一个 relevant_text 出现在任一个文档片段中即为命中
    """
    if not relevant_texts:
        return False

    combined = " ".join(docs)
    for text in relevant_texts:
        if text in combined:
            return True
    return False


# ==================== 内置示例数据集 ====================


def builtin_dataset() -> list[dict]:
    """返回内置示例数据集模板。

    用户可根据自己的知识库内容修改 relevant_texts 后使用。
    """
    return [
        {
            "question": "什么是 RAG 技术？",
            "relevant_texts": ["检索增强生成", "Retrieval-Augmented Generation"],
        },
        {
            "question": "RAG 系统的主要组成部分有哪些？",
            "relevant_texts": ["检索", "生成", "索引"],
        },
        {
            "question": "如何评估 RAG 系统的效果？",
            "relevant_texts": ["评估", "指标"],
        },
    ]
