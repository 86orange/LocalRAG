"""
统一清洗管线测试

覆盖 TextCleaner 各规则及 OCR 纠错。
"""
import pytest
from local_rag.cleaner import clean, clean_without_ocr, TextCleaner, OCR_CHINESE_FIXES


# ==================== 基础清洗 ====================

def test_clean_empty():
    assert clean("") == ""
    assert clean("   \n\n  ") == ""


def test_clean_removes_control_chars():
    """控制字符（如 \x00-\x08）应被移除。"""
    result = clean("正常\x00文本\x08内容")
    assert "\x00" not in result
    assert "\x08" not in result


def test_clean_preserves_newlines_and_tabs():
    """应保留换行符和制表符。"""
    result = clean("hello\tworld\n你好    世界")
    assert "\t" in result
    assert "\n" in result


def test_clean_fullwidth_space():
    """全角空格应转为半角。"""
    result = clean("你好\u3000世界")
    assert "\u3000" not in result
    assert " " in result


def test_clean_multi_newline():
    """3 个以上连续空行应压缩为 2 个。"""
    result = clean("段落1\n\n\n\n段落2")
    assert result.count("\n") <= 3  # 最多 "段落1\n\n段落2"


def test_clean_trailing_spaces():
    """每行尾部空白应被修剪。"""
    result = clean("行尾空格   \n  有前置空格的  ")
    lines = result.split("\n")
    for line in lines:
        assert line == line.rstrip()


def test_clean_multi_space():
    """2 个以上连续空格应合并为 1 个。"""
    result = clean("多个    空格  之间")
    assert "  " not in result


def test_clean_url_removal():
    """URL 应被替换为 [链接]。"""
    result = clean("参考 https://example.com/page 了解更多")
    assert "https://" not in result
    assert "[链接]" in result


def test_clean_email_removal():
    """电子邮件应被替换为 [邮箱]。"""
    result = clean("联系 test@example.com 获取帮助")
    assert "test@example.com" not in result
    assert "[邮箱]" in result


def test_clean_long_number():
    """15 位以上数字串应被替换为 [ID]。"""
    result = clean("订单号 1234567890123456 已确认")
    assert "1234567890123456" not in result
    assert "[ID]" in result


def test_clean_preserves_normal_numbers():
    """普通短数字不应被移除。"""
    result = clean("第 3 页，金额 5000 元")
    assert "3" in result
    assert "5000" in result


# ==================== OCR 纠错 ====================

def test_ocr_fix_common():
    """常见形近字错误应被修复。"""
    result = clean("己经完成了任务")
    assert "已经" in result
    assert "己经" not in result


def test_ocr_fix_multiple():
    """多个错误同时修复。"""
    result = clean("己经完成了末来的规划下牛开会")
    assert "已经" in result
    assert "未来" in result
    assert "下午" in result


def test_ocr_fix_punctuation():
    """中文破折号应被规范化。"""
    result = clean("这是一一一测试")
    assert "一一一" not in result
    assert "——" in result


def test_ocr_fix_fullstop():
    """全角句号错误应被修复。"""
    result = clean("完成的．任务")
    assert "．" not in result


def test_without_ocr_skips_fixes():
    """clean_without_ocr 不应触 OCR 纠错。"""
    result = clean_without_ocr("己经完成")
    assert "己经" in result  # 保留原文
    assert result != clean("己经完成")


# ==================== 中文引号 ====================

def test_normalize_quotes():
    """英文引号在中文字符中应被替换为中文引号。"""
    result = clean('说"你好"世界')
    assert '\u201c' in result


# ==================== TextCleaner 可插拔 ====================

def test_custom_rule():
    """可追加自定义清洗规则。"""
    tc = TextCleaner(enable_ocr_fixes=False)
    tc.add_rule("自定义替换", lambda t: t.replace("ABC", "XYZ"))
    result = tc.clean("包含ABC的文本")
    assert "XYZ" in result
    assert "ABC" not in result


def test_ocr_fixes_coverage():
    """确认 OCR 对照表非空。"""
    assert len(OCR_CHINESE_FIXES) > 5
    for wrong, correct in OCR_CHINESE_FIXES.items():
        assert wrong != correct
