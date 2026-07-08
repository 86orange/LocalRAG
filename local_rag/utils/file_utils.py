"""
文件操作工具模块

提供文档目录扫描、文件哈希计算、格式校验等功能，
供 loader 模块调用，实现增量索引和文件筛选。
"""

import hashlib
from pathlib import Path


from local_rag.config import DOCUMENTS_DIR
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# ==================== 支持的文档格式 ====================

SUPPORTED_SUFFIXES: set[str] = {".md", ".txt", ".pdf", ".docx"}
# 100MB，超过该大小的文件会在日志中警告但仍可处理
MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024


def scan_documents(root: Path | None = None) -> list[Path]:
    """递归扫描文档目录，返回所有支持格式的文件路径列表。

    Args:
        root: 扫描根目录，默认使用 config.DOCUMENTS_DIR

    Returns:
        按文件名排序的文档路径列表
    """
    root = root or DOCUMENTS_DIR

    if not root.exists():
        logger.warning("文档目录不存在: %s", root)
        return []

    files: list[Path] = []
    for file_path in root.rglob("*"):
        if file_path.is_file() and _is_supported(file_path.suffix):
            files.append(file_path.resolve())

    files.sort(key=lambda p: p.name.lower())
    logger.info("扫描 %s 完成，发现 %d 个支持的文档", root, len(files))
    return files


def compute_file_hash(file_path: Path) -> str:
    """计算文件的 MD5 哈希值，用于检测文件变更实现增量索引。

    Args:
        file_path: 文件路径

    Returns:
        十六进制 MD5 字符串，读取失败返回空字符串
    """
    md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
    except (OSError, PermissionError) as e:
        logger.error("无法读取文件 %s: %s", file_path.name, e)
        return ""
    return md5.hexdigest()


def validate_file(file_path: Path) -> bool:
    """校验文件是否适合加载。

    检查项：
    1. 后缀在支持列表中
    2. 文件存在且可读
    3. 文件大小在合理范围内

    Args:
        file_path: 待校验的文件路径

    Returns:
        True 表示文件可以安全加载
    """
    if not file_path.exists():
        logger.warning("文件不存在: %s", file_path)
        return False

    if not file_path.is_file():
        logger.warning("路径不是普通文件: %s", file_path)
        return False

    suffix = file_path.suffix.lower()
    if not _is_supported(suffix):
        logger.warning("不支持的文件格式: %s (%s)", file_path.name, suffix)
        return False

    file_size = file_path.stat().st_size
    if file_size == 0:
        logger.warning("文件为空，跳过: %s", file_path.name)
        return False

    if file_size > MAX_FILE_SIZE_BYTES:
        logger.warning(
            "文件过大 (%d MB)，处理可能较慢: %s",
            file_size // (1024 * 1024),
            file_path.name,
        )

    return True


def get_file_type(file_path: Path) -> str:
    """根据后缀返回文件类型标签，用于 loader 分发。

    Args:
        file_path: 文件路径

    Returns:
        'md' / 'txt' / 'pdf' / 'docx'，未知格式返回 'unknown'
    """
    suffix = file_path.suffix.lower()
    mapping = {
        ".md": "md",
        ".txt": "txt",
        ".pdf": "pdf",
        ".docx": "docx",
    }
    return mapping.get(suffix, "unknown")


def _is_supported(suffix: str) -> bool:
    """判断后缀是否在支持列表中（内部使用）。"""
    return suffix.lower() in SUPPORTED_SUFFIXES
