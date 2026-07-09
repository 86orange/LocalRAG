"""
评估模块

提供 RAG 系统质量评估功能：
- 答案忠实度评估（LLM-as-Judge）
- 召回命中率评估（Recall@K）
- 更多维度陆续添加中
"""

from local_rag.eval.faithfulness import (
    evaluate_faithfulness,
    FaithfulnessResult,
    ClaimResult,
)
from local_rag.eval.recall import (
    evaluate_recall,
    load_dataset,
    builtin_dataset,
    RecallEvalResult,
    SingleRecallResult,
)
