"""
通用工具模块

提供日志、文件扫描、哈希计算、格式校验和文档元数据提取等功能。
"""

from local_rag.utils.logger import get_logger
from local_rag.utils.file_utils import (
    SUPPORTED_SUFFIXES,
    scan_documents,
    compute_file_hash,
    validate_file,
    get_file_type,
    get_file_metadata,
)
from local_rag.utils.dedup import (
    compute_simhash,
    hamming_distance,
    is_near_duplicate,
    ChunkDeduplicator,
    DualLayerDeduplicator,
)
