"""
PDF 文档加载器

基于 PyMuPDF (fitz) 实现 PDF 文本提取，包含：
1. 页眉页脚自动识别与移除
2. 跨页表格检测与合并
3. 扫描版 PDF OCR 支持（可选）
"""

from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

from local_rag.utils.logger import get_logger
from local_rag.cleaner import clean

logger = get_logger(__name__)

# ==================== 页眉页脚参数 ====================
HEADER_RATIO = 0.15   # 页面顶部 15% 视为页眉候选区
FOOTER_RATIO = 0.15   # 页面底部 15% 视为页脚候选区
MIN_REPEAT_COUNT = 3  # 同一位置文本至少出现在 3 页才认定为页眉/页脚

# ==================== 表格检测参数 ====================
TABLE_MAX_GAP_LINES = 2   # 表格跨页时中间允许的最大空行数
TABLE_MIN_ROWS = 2        # 表格最少行数


def load_pdf(file_path: str | Path) -> str:
    """加载 PDF 文件，返回清洗后的纯文本内容。

    Args:
        file_path: PDF 文件路径

    Returns:
        去除了页眉页脚并合并了跨页表格的文本，段落间以双换行分隔
    """
    file_path = Path(file_path)
    logger.info("开始加载 PDF: %s", file_path.name)

    if not file_path.exists():
        logger.error("PDF 文件不存在: %s", file_path)
        return ""

    doc = fitz.open(str(file_path))
    pages_text: list[str] = []

    for page_num, page in enumerate(doc):
        text = _extract_page_text(page)
        if text.strip():
            pages_text.append(text)
        else:
            # 文本为空可能是扫描件，尝试 OCR
            ocr_text = _ocr_page(page)
            pages_text.append(ocr_text)

    if not pages_text:
        doc.close()
        logger.warning("PDF 未提取到任何文本: %s", file_path.name)
        return ""

    # 识别并标记页眉页脚（需要 doc 保持打开状态以读取页面坐标）
    header_lines, footer_lines = _detect_headers_footers(doc, pages_text)

    doc.close()

    # 逐页清洗并拼接
    cleaned_pages: list[str] = []
    for page_num, text in enumerate(pages_text):
        cleaned = _clean_page(text, page_num, header_lines, footer_lines)
        if cleaned.strip():
            cleaned_pages.append(cleaned.strip())

    # 跨页表格合并
    merged = _merge_split_tables(cleaned_pages)

    result = "\n\n".join(merged)
    result = clean(result)
    logger.info("PDF 加载完成: %s (%d 页, %d 字符)", file_path.name, len(cleaned_pages), len(result))
    return result


# ---- 带页码信息的加载（供 metadata 管线使用） ----


def load_pdf_page_by_page(file_path: str | Path) -> list[tuple[int, str]]:
    """逐页加载 PDF，返回 (页码, 页面文本) 列表。

    与 load_pdf 使用相同的清洗逻辑（页眉页脚移除、跨页表格合并），
    但保留页面边界信息，供切片后标注页码 metadata。

    Args:
        file_path: PDF 文件路径

    Returns:
        [(page_number, page_text), ...] 列表，页码从 1 开始
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.error("PDF 文件不存在: %s", file_path)
        return []

    doc = fitz.open(str(file_path))
    pages_text: list[str] = []

    for page_num, page in enumerate(doc):
        text = _extract_page_text(page)
        if text.strip():
            pages_text.append(text)
        else:
            ocr_text = _ocr_page(page)
            pages_text.append(ocr_text)

    if not pages_text:
        doc.close()
        return []

    header_lines, footer_lines = _detect_headers_footers(doc, pages_text)

    doc.close()

    cleaned: list[str] = []
    for page_num, text in enumerate(pages_text):
        c = _clean_page(text, page_num, header_lines, footer_lines)
        cleaned.append(c.strip())

    merged_text = _merge_split_tables(cleaned)

    return [(i + 1, t) for i, t in enumerate(merged_text) if t.strip()]


def compute_page_offsets(pages: list[tuple[int, str]]) -> list[dict]:
    """计算页面在全文中对应的字符偏移区间。

    用于在切片后确定每个 chunk 属于哪些页。

    Args:
        pages: load_pdf_page_by_page 的返回结果 [(page_num, text), ...]

    Returns:
        [{"page": int, "start": int, "end": int}, ...]
        其中 start/end 是全文拼接后的字符索引
    """
    offsets: list[dict] = []
    cursor = 0
    sep = "\n\n"
    for page_num, text in pages:
        offset = cursor + len(sep) if offsets else 0
        cursor = offset
        offsets.append({
            "page": page_num,
            "start": cursor,
            "end": cursor + len(text),
        })
        cursor += len(text)
    return offsets


def map_chunks_to_pages(
    chunks: list[str],
    full_text: str,
    page_offsets: list[dict],
) -> list[dict]:
    """将切分后的文本块映射回其所在的页面范围。

    对每个 chunk 在全文中的字符位置查找，确定其落在哪些页面上。

    Args:
        chunks: 切片后的文本块列表
        full_text: 拼接后的全文
        page_offsets: compute_page_offsets 的返回结果

    Returns:
        [{"page_start": int, "page_end": int}, ...]
        每个元素对应一个 chunk，page_start 和 page_end 均从 1 开始
    """
    results: list[dict] = []
    offset_map = sorted(page_offsets, key=lambda x: x["start"])

    for chunk in chunks:
        pos = full_text.find(chunk)
        if pos == -1:
            # 回退：chunk 可能在全文中有微小差异，从开头找
            pos = max(0, full_text.find(chunk[:min(50, len(chunk))]))
        chunk_end = pos + len(chunk)

        page_start = 1
        page_end = 1
        for off in offset_map:
            if off["start"] <= pos < off["end"] + 2:
                page_start = off["page"]
                break
        for off in reversed(offset_map):
            if off["start"] < chunk_end:
                page_end = off["page"]
                break

        results.append({
            "page_start": page_start,
            "page_end": page_end,
        })
    return results


def load_and_chunk_pdf(
    file_path: str | Path,
    chunk_fn,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    base_metadata: dict | None = None,
) -> tuple[list[str], list[dict]]:
    """加载 PDF 并进行带页码信息的切片，一次性完成全流程。

    相比分别调用 load_pdf + chunk：
    - 使用 load_pdf_page_by_page 保留页面边界
    - 每个 chunk metadata 自动注入 page_start / page_end

    Args:
        file_path: PDF 文件路径
        chunk_fn: 切片函数，需接受 (text, chunk_size, chunk_overlap, base_metadata)
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数
        base_metadata: 文档基础元数据

    Returns:
        (chunks, metadatas) — metadatas 已含 page_start / page_end
    """
    pages = load_pdf_page_by_page(file_path)
    if not pages:
        return [], []

    offsets = compute_page_offsets(pages)
    full_text = "\n\n".join(text for _, text in pages)

    chunks, metadatas = chunk_fn(full_text, chunk_size, chunk_overlap, base_metadata)
    if not chunks:
        return [], []

    page_maps = map_chunks_to_pages(chunks, full_text, offsets)

    for meta, pmap in zip(metadatas, page_maps):
        meta["page_start"] = pmap["page_start"]
        meta["page_end"] = pmap["page_end"]

    return chunks, metadatas


def _extract_page_text(page: fitz.Page) -> str:
    """从单个页面提取文本块，按行排序拼接。

    使用 dict 方式获取文本并按 y 坐标排序，保持阅读顺序。
    """
    blocks = page.get_text("dict")["blocks"]
    lines: list[tuple[float, str]] = []

    for block in blocks:
        if block.get("type") != 0:  # 仅处理文本块，跳过图片
            continue
        for line in block.get("lines", []):
            text = "".join(
                span["text"]
                for span in line["spans"]
                if span["text"].strip()
            )
            if text.strip():
                y = line["bbox"][1]  # 行的顶部 y 坐标
                lines.append((y, text.strip()))

    lines.sort(key=lambda x: x[0])
    return "\n".join(line for _, line in lines)


def _detect_headers_footers(
    doc: fitz.Document,
    pages_text: list[str],
) -> tuple[set[str], set[str]]:
    """通过多页位置对比检测页眉和页脚文本。

    策略：
    1. 页面顶部 / 底部 15% 区域内的文本标记为候选
    2. 同一文本至少在 MIN_REPEAT_COUNT 页中出现才确认

    Args:
        doc: PDF 文档对象
        pages_text: 已提取的每页文本

    Returns:
        (页眉文本集合, 页脚文本集合)
    """
    if len(pages_text) < MIN_REPEAT_COUNT:
        return set(), set()

    header_candidates: dict[str, int] = defaultdict(int)
    footer_candidates: dict[str, int] = defaultdict(int)

    for page_num, page in enumerate(doc):
        rect = page.rect
        header_y_limit = rect.height * HEADER_RATIO
        footer_y_start = rect.height * (1 - FOOTER_RATIO)

        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            bbox = block["bbox"]
            for line in block.get("lines", []):
                text = "".join(
                    s["text"] for s in line["spans"] if s["text"].strip()
                ).strip()
                if not text:
                    continue

                y0 = line["bbox"][1]
                if y0 < header_y_limit:
                    header_candidates[text] += 1
                elif y0 > footer_y_start:
                    footer_candidates[text] += 1

    headers = {t for t, c in header_candidates.items() if c >= MIN_REPEAT_COUNT}
    footers = {t for t, c in footer_candidates.items() if c >= MIN_REPEAT_COUNT}

    if headers:
        logger.debug("检测到 %d 条页眉文本", len(headers))
    if footers:
        logger.debug("检测到 %d 条页脚文本", len(footers))

    return headers, footers


def _clean_page(
    text: str,
    page_num: int,
    headers: set[str],
    footers: set[str],
) -> str:
    """移除页面中的页眉页脚行。

    Args:
        text: 单页文本（行拼接）
        page_num: 页码（0 起始）
        headers: 已确认的页眉文本集合
        footers: 已确认的页脚文本集合

    Returns:
        清洗后的文本
    """
    lines = text.split("\n")
    kept: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped in headers or stripped in footers:
            continue
        kept.append(line)

    return "\n".join(kept)


def _merge_split_tables(pages: list[str]) -> list[str]:
    """检测并合并跨页表格。

    策略：
    1. 扫描每页最后几行和下一页最前几行
    2. 如果相邻都是表格结构（每行含制表符或列数一致），则合并

    Args:
        pages: 清洗后的每页文本列表

    Returns:
        合并跨页表格后的文本列表
    """
    if len(pages) < 2:
        return pages

    result: list[str] = []
    buffer = pages[0]

    for i in range(1, len(pages)):
        prev_tail = _get_tail_lines(buffer, TABLE_MAX_GAP_LINES)
        curr_head = _get_head_lines(pages[i], TABLE_MAX_GAP_LINES)

        if _is_table_continuation(prev_tail, curr_head):
            # 跨页表格：去重表头后合并
            merged_body = _merge_table_body(buffer, pages[i])
            buffer = merged_body
        else:
            result.append(buffer)
            buffer = pages[i]

    result.append(buffer)
    return result


def _get_tail_lines(text: str, count: int) -> list[str]:
    """获取文本最后 N 行（非空）。"""
    lines = [l for l in text.split("\n") if l.strip()]
    return lines[-count:] if len(lines) >= count else lines


def _get_head_lines(text: str, count: int) -> list[str]:
    """获取文本前 N 行（非空）。"""
    lines = [l for l in text.split("\n") if l.strip()]
    return lines[:count]


def _is_table_continuation(prev_lines: list[str], curr_lines: list[str]) -> bool:
    """判断两段文本是否为同一表格的连续部分。

    判定条件：
    - 两边的行数至少各有一行
    - 分离列的方式一致（都用 tab / 都用多个空格 / 都用 |）
    """
    if not prev_lines or not curr_lines:
        return False

    prev_cols = _detect_column_count(prev_lines[-1])
    curr_cols = _detect_column_count(curr_lines[0])
    if prev_cols is None or curr_cols is None:
        return False

    return prev_cols == curr_cols


def _detect_column_count(line: str) -> int | None:
    """检测一行文本的列数，通过分隔符推断。

    支持制表符、竖线、多空格分隔。非表格文本返回 None。
    """
    if "\t" in line:
        return len(line.split("\t"))
    if " | " in line or line.startswith("|"):
        return len([c for c in line.split("|") if c.strip() != ""])
    # 2+ 个连续空格视为列分隔
    parts = line.split("  ")
    parts = [p for p in parts if p.strip()]
    if len(parts) >= TABLE_MIN_ROWS:
        return len(parts)
    return None


def _merge_table_body(prev_page: str, curr_page: str) -> str:
    """合并跨页表格，保留前一页的表头，拼接后续行。

    Args:
        prev_page: 前一页文本
        curr_page: 当前页文本

    Returns:
        合并后的文本
    """
    prev_lines = prev_page.split("\n")
    curr_lines = curr_page.split("\n")

    # 找到前一页表格起始行的参考列数
    tail = _get_tail_lines(prev_page, TABLE_MAX_GAP_LINES)
    if not tail:
        return prev_page + "\n" + curr_page
    ref_cols = _detect_column_count(tail[-1])
    if ref_cols is None:
        return prev_page + "\n" + curr_page

    # 在上一页中找到表格的起始行
    table_start = len(prev_lines)
    for idx in range(len(prev_lines) - 1, -1, -1):
        cols = _detect_column_count(prev_lines[idx])
        if cols != ref_cols:
            table_start = idx + 1
            break
    if table_start >= len(prev_lines):
        table_start = max(0, len(prev_lines) - TABLE_MIN_ROWS)

    # 在下一页中找到表格的结束行
    table_end = 0
    for idx, line in enumerate(curr_lines):
        cols = _detect_column_count(line)
        if cols == ref_cols:
            table_end = idx + 1
        else:
            break
    if table_end == 0:
        table_end = min(len(curr_lines), TABLE_MIN_ROWS)

    # 拼接：非表格前文 + 表格行 + 非表格后文
    before_table = prev_lines[:table_start]
    table_lines = prev_lines[table_start:] + curr_lines[:table_end]
    after_table = curr_lines[table_end:]

    merged_lines = before_table + table_lines + after_table
    return "\n".join(merged_lines)


def _ocr_page(page: fitz.Page) -> str:
    """对扫描版 PDF 页面进行 OCR 文字识别。

    尝试导入 pytesseract，若不可用则返回空字符串。
    使用 PIL 将页面渲染为图片后送入 OCR。

    Args:
        page: 单页 PDF 页面对象

    Returns:
        OCR 识别的文本，失败返回空字符串
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        logger.debug("pytesseract 未安装，跳过 OCR")
        return ""

    try:
        mat = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [mat.width, mat.height], mat.samples)  # type: ignore[arg-type]
        text_bytes: bytes = pytesseract.image_to_string(img, lang="chi_sim+eng")
        text = text_bytes if isinstance(text_bytes, str) else text_bytes.decode("utf-8")
        return text
    except Exception as e:
        logger.warning("OCR 失败 (第 %d 页): %s", page.number + 1, e)
        return ""
