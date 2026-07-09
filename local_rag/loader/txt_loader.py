"""
纯文本文档加载器

加载 .txt 文件，处理：
1. 自动编码检测（UTF-8 → GBK → Latin-1 回退）
2. BOM 头移除
3. 统一清洗管线（控制字符、空白、OCR 纠错等）
"""

import re
from pathlib import Path

from local_rag.utils.logger import get_logger
from local_rag.cleaner import clean

logger = get_logger(__name__)


def load_txt(file_path: str | Path) -> str:
    """加载纯文本文件，自动检测编码并清洗内容。

    Args:
        file_path: .txt 文件路径

    Returns:
        清洗后的纯文本
    """
    file_path = Path(file_path)
    logger.info("开始加载 TXT: %s", file_path.name)

    raw = _read_with_encoding(file_path)
    if raw is None:
        return ""

    text = clean(raw)

    logger.info("TXT 加载完成: %s (%d 字符)", file_path.name, len(text))
    return text


def _read_with_encoding(file_path: Path) -> str | None:
    """按优先级尝试多种编码读取文件。

    顺序：UTF-8 → UTF-8-SIG (BOM) → GBK → Latin-1

    Returns:
        文件文本内容，全部失败返回 None
    """
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin-1"]
    for enc in encodings:
        try:
            return file_path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except (OSError, PermissionError) as e:
            logger.error("无法读取 TXT 文件 %s: %s", file_path.name, e)
            return None
    logger.error("所有编码均失败: %s", file_path.name)
    return None
