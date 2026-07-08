"""
全局配置模块

集中管理项目中所有可配置参数，支持通过环境变量覆盖默认值。
"""

import os
from pathlib import Path

# ==================== 路径配置 ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = PROJECT_ROOT / "documents"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"
LOG_DIR = PROJECT_ROOT / "logs"

# ==================== Ollama 模型配置 ====================

# LLM 对话模型 — 用于 RAG 问答生成
LLM_MODEL = os.getenv("LOCAL_RAG_LLM_MODEL", "qwen3:4b")

# Embedding 模型 — 用于文档向量化（支持中文：bge-m3 / nomic-embed-text）
EMBEDDING_MODEL = os.getenv("LOCAL_RAG_EMBEDDING_MODEL", "bge-m3")

# Ollama 服务地址
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ==================== 文本切片配置 ====================

CHUNK_SIZE = int(os.getenv("LOCAL_RAG_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("LOCAL_RAG_CHUNK_OVERLAP", "128"))

# ==================== 检索与问答配置 ====================

# 检索返回的最相关文档片段数
TOP_K = int(os.getenv("LOCAL_RAG_TOP_K", "5"))

# 输入 LLM 的最大上下文 token 数
MAX_CONTEXT_TOKENS = int(os.getenv("LOCAL_RAG_MAX_CONTEXT_TOKENS", "4096"))

# LLM 生成温度（越低越稳定，越高越有创造性）
TEMPERATURE = float(os.getenv("LOCAL_RAG_TEMPERATURE", "0.1"))


def init_dirs() -> None:
    """创建项目运行所需的目录（data、documents、logs 等）。"""
    for directory in [DATA_DIR, DOCUMENTS_DIR, VECTOR_DB_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
