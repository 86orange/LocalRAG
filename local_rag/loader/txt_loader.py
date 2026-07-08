"""
纯文本文档加载器

加载 .txt 文件，处理：
1. 自动编码检测（UTF-8 → GBK → Latin-1 回退）
2. BOM 头移除
3. 空白行压缩与首尾修剪
4. 不可见控制字符过滤
5. 全角/半角混合空白统一
"""

import re
from pathlib import Path

from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# 需要从文本中移除的控制字符（保留 \n \t）
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# 3 个及以上连续空行 → 2 个空行
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


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

    text = _clean_txt(raw)

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


def _clean_txt(text: str) -> str:
    """对纯文本执行清洗流水线。

    处理步骤：
    1. 移除不可见控制字符（保留换行和制表符）
    2. 统一全角空格为半角
    3. 合并多余空行
    4. 首尾空白修剪
    """
    # 移除控制字符
    text = _CONTROL_CHARS_RE.sub("", text)

    # 全角空格 → 半角空格
    text = text.replace("\u3000", " ")

    # 合并 3 个及以上连续换行为双换行
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 修剪每行尾部空白
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()
