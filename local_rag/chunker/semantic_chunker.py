"""
语义感知切片模块

识别文档中的语义边界（标题、段落），在自然分段处进行切分，
尽可能保持每个 chunk 内部的语义完整性。

适合 Markdown、DOCX 等结构化文档。
"""

import re

from local_rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# Markdown 标题行（行首 1-6 个 # + 至少一个空格）
_HEADING_RE = re.compile(r"^#{1,6}\s+")
# 水平分隔线
_HORIZONTAL_RULE_RE = re.compile(r"^[-*_]{3,}\s*$")
# 双换行以上视为段落边界
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def chunk_by_semantic(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """按语义边界对文本进行智能切片。

    策略：
    1. 识别标题行，标题作为新分块的起点
    2. 段落→块组→块：段落组的累计长度接近 chunk_size 时切分
    3. 下一个块回退 chunk_overlap 个段落，保证上下文连续

    Args:
        text: 输入文本
        chunk_size: 每块最大字符数，默认使用全局配置 CHUNK_SIZE
        chunk_overlap: 相邻块重叠字符数，默认使用全局配置 CHUNK_OVERLAP

    Returns:
        切分后的文本块列表
    """
    chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    if not text or not text.strip():
        logger.warning("输入文本为空，跳过切片")
        return []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数，当前值: {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
        )

    # 按双换行拆分为语义段，过滤空段和分隔线
    segments = [
        s.strip()
        for s in _PARAGRAPH_SPLIT_RE.split(text)
        if s.strip() and not _HORIZONTAL_RULE_RE.match(s.strip())
    ]

    total_len = sum(len(s) for s in segments)
    if total_len <= chunk_size:
        logger.debug("文本仅 %d 字符，无需切片", total_len)
        return ["\n\n".join(segments).strip()]

    # 按标题边界合并语义段为语义组
    groups = _group_by_semantic(segments, chunk_size)
    chunks, _chapters = _build_semantic_chunks(groups, chunk_size, chunk_overlap)

    logger.debug(
        "语义切片完成: %d 字符 → %d 组 → %d 块 (chunk_size=%d, overlap=%d)",
        total_len,
        len(groups),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _group_by_semantic(segments: list[str], chunk_size: int) -> list[str]:
    """将段落按标题边界合并为语义组。

    每个语义组由一个标题及其下属的段落组成。
    标题级联的保留在一起（# 标题 + ## 子标题 + 正文）。

    若文本无任何标题，则退化为按 chunk_size 强制分组，
    避免所有段落聚为一个超长语义组导致 ValueError。
    """
    groups: list[str] = []
    current_group: list[str] = []

    for seg in segments:
        is_heading = bool(_HEADING_RE.match(seg))
        if _HORIZONTAL_RULE_RE.match(seg):
            continue

        if is_heading and current_group:
            groups.append("\n\n".join(current_group))
            current_group = [seg]
        else:
            current_group.append(seg)

    if current_group:
        groups.append("\n\n".join(current_group))

    # 无标题时退化为长度切分，避免单一超大语义组
    has_headings = any(_HEADING_RE.match(seg) for seg in segments)
    if not has_headings:
        groups = _force_split_large_groups(groups, chunk_size)

    # 防御性检查：任何语义组仍超限则报错
    for group in groups:
        if len(group) > chunk_size:
            raise ValueError(
                f"语义组长度 ({len(group)}) 超过 chunk_size ({chunk_size})，"
                f"该组以 '{group[:80]}...' 开头，"
                f"请增大 chunk_size 或手动拆分该组内容。"
            )

    return groups


def _extract_chapter(group: str) -> str:
    """从语义组中提取章节标题。

    若语义组以 Markdown 标题行开头，返回该标题文本（去除 # 前缀）。
    否则返回空字符串。

    Args:
        group: 语义组文本

    Returns:
        章节标题，无标题时返回 ""
    """
    first_line = group.split("\n", 1)[0].strip()
    m = _HEADING_RE.match(first_line)
    if m:
        return first_line[m.end():].strip()
    return ""


def _force_split_large_groups(groups: list[str], max_size: int) -> list[str]:
    """将超过 max_size 的组按段落强制拆小。

    用于无标题文本在语义切分模式下不会因单一超长组而整体失败。
    超长段落按句子边界进一步切分。
    """
    result: list[str] = []
    for group in groups:
        if len(group) <= max_size:
            result.append(group)
            continue

        paragraphs = [p.strip() for p in group.split("\n\n") if p.strip()]
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para)
            if para_len > max_size:
                # 超长段落按句子拆
                sub_parts = _split_by_sentence(para, max_size)
                for sp in sub_parts:
                    sp_len = len(sp)
                    sep = 2 if current else 0
                    if current_len + sep + sp_len <= max_size:
                        current.append(sp)
                        current_len += sep + sp_len
                    else:
                        if current:
                            result.append("\n\n".join(current))
                        current = [sp]
                        current_len = sp_len
                continue

            sep = 2 if current else 0
            if current_len + sep + para_len <= max_size:
                current.append(para)
                current_len += sep + para_len
            else:
                if current:
                    result.append("\n\n".join(current))
                current = [para]
                current_len = para_len
        if current:
            result.append("\n\n".join(current))

    return result


def _split_by_sentence(text: str, max_size: int) -> list[str]:
    """按句子边界切分超长文本。"""
    import re

    parts = re.split(r"(?<=[。？！；\n])", text)
    result: list[str] = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) <= max_size:
            buf = (buf + " " + p).strip() if buf else p
        else:
            if buf:
                result.append(buf)
            buf = p
    if buf:
        result.append(buf)
    return result or [text]


def _build_semantic_chunks(
    groups: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[list[str], list[str]]:
    """将语义组合并成最终文本块，同时返回每块对应的章节标题。

    每组间以 "\n\n" 分隔。当前累计长度接近 chunk_size 时创建新块，
    新块回退 chunk_overlap 个语义组作为上下文延续。

    Returns:
        (chunks, chapters) — chapters[i] 是 chunks[i] 所属章节标题，
        无标题时为空字符串
    """
    chunks: list[str] = []
    chapters: list[str] = []
    current: list[str] = []
    current_chapter = ""
    current_len = 0

    for group in groups:
        chapter = _extract_chapter(group)
        group_len = len(group)
        separator_len = 2 if current else 0

        if current_len + separator_len + group_len <= chunk_size:
            if not current_chapter:
                current_chapter = chapter
            current.append(group)
            current_len += separator_len + group_len
        else:
            chunks.append("\n\n".join(current))
            chapters.append(current_chapter)

            current, current_len = _rebuild_prefix(current, chunk_overlap)
            current_chapter = chapter
            current.append(group)
            current_len += (2 if current else 0) + group_len

    if current:
        chunks.append("\n\n".join(current))
        chapters.append(current_chapter)

    return chunks, chapters


def _rebuild_prefix(
    prev_groups: list[str],
    overlap_size: int,
) -> tuple[list[str], int]:
    """从上一块的末尾语义组中选取重叠前缀。

    反向扫描，累计长度不超过 overlap_size。

    Returns:
        (重叠语义组列表, 重叠部分总字符数)
    """
    if not prev_groups or overlap_size <= 0:
        return [], 0

    prefix: list[str] = []
    prefix_len = 0

    for group in reversed(prev_groups):
        additional = len(group) + (2 if prefix else 0)
        if prefix_len + additional > overlap_size:
            break
        prefix.insert(0, group)
        prefix_len += additional

    return prefix, prefix_len


def chunk_by_semantic_with_metadata(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    base_metadata: dict | None = None,
) -> tuple[list[str], list[dict]]:
    """语义切片并返回每块的 metadata，含 chunk_index / total_chunks / chapter 及基础元数据。

    Args:
        text: 输入文本
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数
        base_metadata: 文档基础元数据，会复制到每块的 metadata 中

    Returns:
        (chunk_texts, chunk_metadatas) 二元组
    """
    chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP

    if not text or not text.strip():
        return [], []

    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正数，当前值: {chunk_size}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) 必须小于 chunk_size ({chunk_size})"
        )

    segments = [
        s.strip()
        for s in _PARAGRAPH_SPLIT_RE.split(text)
        if s.strip() and not _HORIZONTAL_RULE_RE.match(s.strip())
    ]

    total_len = sum(len(s) for s in segments)
    if total_len <= chunk_size:
        short_text = "\n\n".join(segments).strip()
        chapter = _extract_chapter(short_text)
        base = base_metadata or {}
        return [short_text], [{**base, "chunk_index": 0, "total_chunks": 1, "chapter": chapter}]

    groups = _group_by_semantic(segments, chunk_size)
    chunks, chapters = _build_semantic_chunks(groups, chunk_size, chunk_overlap)

    base = base_metadata or {}
    metadatas: list[dict] = []
    total = len(chunks)
    for i in range(total):
        metadatas.append({
            **base,
            "chunk_index": i,
            "total_chunks": total,
            "chapter": chapters[i] if i < len(chapters) else "",
        })
    return chunks, metadatas
