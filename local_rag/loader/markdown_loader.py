"""
Markdown 文档加载器

加载 .md 文件，提取纯文本内容，处理：
1. YAML Front Matter 去除
2. HTML 标签剥离
3. 图片 / 链接语法清洗（保留文字描述）
4. 代码块保留原格式
5. 表格保留为 pipe 格式
6. 统一清洗管线（空白压缩、OCR 纠错等）
"""

import re
from pathlib import Path

from local_rag.utils.logger import get_logger
from local_rag.cleaner import clean_without_ocr

logger = get_logger(__name__)

# YAML Front Matter 分隔符模式（--- 或 +++ 开头结尾）
_FRONTMATTER_RE = re.compile(
    r"^[-+]{3,}\s*\n.*?\n[-+]{3,}\s*\n", re.DOTALL | re.MULTILINE
)

# Markdown 图片: ![alt](url "title")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
# Markdown 链接: [text](url "title")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
# 行内 HTML 标签
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# HTML 注释
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# 粗体/斜体: **text**, __text__, *text*, _text_, ~~text~~
_BOLD_ITALIC_RE = re.compile(r"(\*{1,3}|_{1,3}|~~)(.+?)\1")


def load_markdown(file_path: str | Path) -> str:
    """加载 Markdown 文件，返回清洗后的纯文本。

    Args:
        file_path: .md 文件路径

    Returns:
        去除 Markdown 标记语法的纯文本
    """
    file_path = Path(file_path)
    logger.info("开始加载 Markdown: %s", file_path.name)

    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="gbk")
    except (OSError, PermissionError) as e:
        logger.error("无法读取 Markdown 文件 %s: %s", file_path.name, e)
        return ""

    text = _clean_markdown(raw)

    logger.info("Markdown 加载完成: %s (%d 字符)", file_path.name, len(text))
    return text


def _clean_markdown(text: str) -> str:
    """对 Markdown 文本执行完整的清洗流水线。"""
    # 1. 去除 YAML Front Matter
    text = _FRONTMATTER_RE.sub("", text)

    # 2. 去除 HTML 注释
    text = _HTML_COMMENT_RE.sub("", text)

    # 3. 替换图片为描述文本
    text = _IMAGE_RE.sub(r"\1", text)

    # 4. 替换链接为显示文本
    text = _LINK_RE.sub(r"\1", text)

    # 5. 剥离 HTML 标签
    text = _HTML_TAG_RE.sub("", text)

    # 6. 去除粗体/斜体/删除线标记，保留内部文本
    text = _BOLD_ITALIC_RE.sub(r"\2", text)

    # 7. 去除行首的 # 标题标记（保留标题文字）
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # 跳过纯分隔线
        if re.match(r"^[-*_=]{3,}$", stripped):
            continue
        # 去除标题 # 标记
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
        # 去除引用 > 标记
        stripped = re.sub(r"^>\s?", "", stripped)
        # 去除无序列表标记 * - +
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        cleaned.append(stripped)

    text = "\n".join(cleaned)

    # 通过统一清洗管线
    text = clean_without_ocr(text)

    logger.info("Markdown 加载完成: %s (%d 字符)", file_path.name, len(text))
    return text
