"""
统一文本清洗管线

所有 loader 加载完成后统一过一遍清洗管线，保证输出质量一致。
清洗规则按优先级组成管道，可插拔、可配置。

规则链：
1. 控制字符过滤（保留 \n \t）
2. 全角/半角空白统一
3. 不可见 Unicode 字符移除
4. 多余空行压缩
5. 行尾空白修剪
6. URL / 电子邮件清理
7. OCR 常见纠错
8. 标点符号规范化
"""

import re
from typing import Callable

from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# ==================== 基础规则 ====================

# 控制字符（保留换行和制表符）
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# 3 个及以上连续空行 → 双空行
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
# URL pattern
_URL_RE = re.compile(r"https?://\S+|ftp://\S+|www\.\S+")
# 电子邮件
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
# 纯数字长串（10+ 位，可能是条码/ID）
_LONG_NUMBER_RE = re.compile(r"\b\d{15,}\b")
# 2 个以上连续空格
_MULTI_SPACE_RE = re.compile(r" {2,}")
# 零宽字符
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2028-\u202f\ufeff]")


# ==================== OCR 常见错误对照表（中文） ====================

OCR_CHINESE_FIXES: dict[str, str] = {
    # 形近字混淆
    "己经": "已经", "巳经": "已经",
    "末来": "未来", "未末": "未来",
    "大阳": "太阳", "太阴": "太阳",
    "干万": "千万",
    "白勺": "白的", "因力": "因为",
    "下牛": "下午", "下年": "下午",
    "木文": "本文",
    "昨大": "昨天",
    # OCR 常见标点错误
    "．": "。",
}
# 正则版本：更泛化的替换
OCR_REGEX_FIXES: list[tuple[re.Pattern, str]] = [
    # 英文单词内数字混入 (l→1, O→0 等只改明显的含义词)
    (re.compile(r"(\d)l([a-z])"), r"\1I\2"),
    # 中文破折号 "一一一" → "——"
    (re.compile(r"一{2,}"), "——"),
]


# ==================== 清洗管线 ====================


class TextCleaner:
    """可插拔的统一文本清洗管线。

    内置默认规则链，支持运行时增删规则。

    Usage:
        cleaner = TextCleaner()
        text = cleaner.clean(raw_text)

        cleaner.add_rule(lambda t: t.replace("xxx", "yyy"))
    """

    def __init__(self, enable_ocr_fixes: bool = True) -> None:
        self._rules: list[tuple[str, Callable[[str], str]]] = []
        self._enable_ocr_fixes = enable_ocr_fixes
        self._register_default_rules()

    def clean(self, text: str) -> str:
        """执行全部清洗规则。

        Args:
            text: 原始文本

        Returns:
            清洗后的文本
        """
        if not text or not text.strip():
            return ""

        for name, rule in self._rules:
            original_len = len(text)
            text = rule(text)
            trimmed = len(text) - original_len
            if trimmed != 0:
                logger.debug(
                    "清洗规则 [%s]: %s%d 字符",
                    name,
                    "-" if trimmed < 0 else "+",
                    abs(trimmed),
                )

        return text.strip()

    def add_rule(self, name: str, rule: Callable[[str], str]) -> None:
        """在管线末尾追加一条自定义规则。"""
        self._rules.append((name, rule))

    def _register_default_rules(self) -> None:
        """注册默认清洗规则链。"""
        rules: list[tuple[str, Callable[[str], str]]] = [
            ("控制字符过滤", lambda t: _CONTROL_CHARS_RE.sub("", t)),
            ("零宽字符移除", lambda t: _ZERO_WIDTH_RE.sub("", t)),
            ("全角空格转半角", lambda t: t.replace("\u3000", " ")),
            ("URL 移除", lambda t: _URL_RE.sub("[链接]", t)),
            ("Email 移除", lambda t: _EMAIL_RE.sub("[邮箱]", t)),
            ("长数字移除", lambda t: _LONG_NUMBER_RE.sub("[ID]", t)),
            ("多空格合并", lambda t: _MULTI_SPACE_RE.sub(" ", t)),
            ("多余空行压缩", lambda t: _MULTI_NEWLINE_RE.sub("\n\n", t)),
            ("行尾空白修剪", _trim_line_endings),
            ("中文引号统一", _normalize_quotes),
        ]
        if self._enable_ocr_fixes:
            rules.append(("OCR 汉字纠错", _fix_ocr_chinese))
            rules.append(("OCR 标点纠错", _fix_ocr_patterns))

        self._rules = rules


# ==================== 便捷函数 ====================


def clean(text: str, enable_ocr_fixes: bool = True) -> str:
    """一行调用清洗管线。

    Args:
        text: 原始文本
        enable_ocr_fixes: 是否启用 OCR 纠错（默认开启）

    Returns:
        清洗后的文本
    """
    return TextCleaner(enable_ocr_fixes=enable_ocr_fixes).clean(text)


def clean_without_ocr(text: str) -> str:
    """不含 OCR 纠错的清洗（适用于数字化文档如 Markdown/DOCX）。"""
    return clean(text, enable_ocr_fixes=False)


# ==================== 内部清洗函数 ====================


def _trim_line_endings(text: str) -> str:
    """修剪每行尾部空白。"""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _normalize_quotes(text: str) -> str:
    """统一中文引号格式。"""
    text = text.replace('"', '\u201c', 1)
    cursor = 0
    result: list[str] = []
    while cursor < len(text):
        idx = text.find('"', cursor)
        if idx == -1:
            result.append(text[cursor:])
            break
        result.append(text[cursor:idx])
        result.append('\u201d')
        cursor = idx + 1
    return "".join(result) if result else text


def _fix_ocr_chinese(text: str) -> str:
    """基于对照表修复 OCR 常见形近字错误。"""
    for wrong, correct in OCR_CHINESE_FIXES.items():
        if wrong in text:
            text = text.replace(wrong, correct)
    return text


def _fix_ocr_patterns(text: str) -> str:
    """基于正则修复 OCR 常见模式错误。"""
    for pattern, replacement in OCR_REGEX_FIXES:
        text = pattern.sub(replacement, text)
    return text
