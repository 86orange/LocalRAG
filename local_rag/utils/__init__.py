"""
工具模块

提供日志、文件操作等通用工具函数。
"""

from local_rag.utils.logger import get_logger
from local_rag.utils.file_utils import (
    SUPPORTED_SUFFIXES,
    scan_documents,
    compute_file_hash,
    validate_file,
    get_file_type,
)
