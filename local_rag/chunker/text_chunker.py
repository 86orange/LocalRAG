"""
基础文本切片模块

按固定字符数（chunk_size）切分长文本，相邻块之间保留重叠区域（chunk_overlap），
避免关键语义被截断在分片边界。

仅做物理切分，不感知文档语义结构。适合纯文本、PDF 提取正文等场景。
"""

import re

from local_rag.config import CHUNK_SIZE, CHUNK_OVERLAP
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# 段落分隔符模式（双换行以上视为段落边界）
_PARAGRAPH_SPLIT_RE = re.compile(r"\n{2,}")


def chunk_by_size(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """按固定字符数对文本进行切片。

    策略：
    1. 先按段落（双换行）拆分，避免在段落中间生硬切断
    2. 将段落重新拼接，每当累计长度接近 chunk_size 时生成一个 chunk
    3. 下一个 chunk 会回退 chunk_overlap 个字符，保证语义连续性

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

    # 按段落拆分，过滤空段
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]

    # 短文本直接返回
    total_len = sum(len(p) for p in paragraphs)
    if total_len <= chunk_size:
        logger.debug("文本仅 %d 字符，无需切片", total_len)
        return [text.strip()]

    chunks = _build_chunks(paragraphs, chunk_size, chunk_overlap)

    logger.debug(
        "切片完成: %d 字符 → %d 个块 (chunk_size=%d, overlap=%d)",
        total_len,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def _build_chunks(
    paragraphs: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """将段落列表拼接成固定大小的文本块。

    每块内的段落间以双换行连接，保持原有可读性。
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)

        # 超长段落自动拆解
        if para_len > chunk_size:
            sub_paras = _split_long_paragraph(para, chunk_size)
            # 将拆分后的子段逐个处理
            for sub in sub_paras:
                sub_len = len(sub)
                separator_len = 2 if current else 0
                if current_len + separator_len + sub_len <= chunk_size:
                    current.append(sub)
                    current_len += separator_len + sub_len
                else:
                    chunks.append("\n\n".join(current))
                    current, current_len = _build_overlap_prefix(current, chunk_overlap)
                    current.append(sub)
                    current_len += (2 if current else 0) + sub_len
            continue

        separator_len = 2 if current else 0  # 段落间 "\n\n"

        if current_len + separator_len + para_len <= chunk_size:
            current.append(para)
            current_len += separator_len + para_len
        else:
            # 当前块已满，保存并开始新块
            chunks.append("\n\n".join(current))

            # 回退 overlap：从上一块末尾取段落作为新块前缀
            current, current_len = _build_overlap_prefix(
                current, chunk_overlap
            )
            current.append(para)
            current_len += (2 if current else 0) + para_len

    # 不丢弃尾部
    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _split_long_paragraph(para: str, max_size: int) -> list[str]:
    """将超长段落按句子边界拆解为多个子段落。

    优先在标点符号（。？！；\n）后切分，保持语义完整。
    max_size 同时作为 chunk 上限。

    Args:
        para: 超长段落文本
        max_size: 每个子段最大字符数

    Returns:
        拆分后的子段落列表
    """
    import re

    # 先在换行处切一次
    lines = para.split("\n")
    result: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_len = len(line)

        if line_len > max_size:
            # 单行仍超长，按标点强制切分
            sub_parts = re.split(r"(?<=[。？！；])", line)
            for sp in sub_parts:
                sp = sp.strip()
                if not sp:
                    continue
                sp_len = len(sp)
                sep = 0 if not current_parts else 2
                if current_len + sep + sp_len <= max_size:
                    current_parts.append(sp)
                    current_len += sep + sp_len
                else:
                    if current_parts:
                        result.append("\n\n".join(current_parts))
                    current_parts = [sp]
                    current_len = sp_len
        else:
            sep = 0 if not current_parts else 2
            if current_len + sep + line_len <= max_size:
                current_parts.append(line)
                current_len += sep + line_len
            else:
                if current_parts:
                    result.append("\n\n".join(current_parts))
                current_parts = [line]
                current_len = line_len

    if current_parts:
        result.append("\n\n".join(current_parts))

    return result or [para]


def _build_overlap_prefix(
    prev_paras: list[str],
    overlap_size: int,
) -> tuple[list[str], int]:
    """从上一块的末尾段落中选取部分作为下一块的重叠前缀。

    反向扫描已切块的段落，累计长度尽量接近 overlap_size。

    Returns:
        (重叠段落列表, 重叠部分总字符数)
    """
    if not prev_paras or overlap_size <= 0:
        return [], 0

    prefix: list[str] = []
    prefix_len = 0

    for para in reversed(prev_paras):
        additional = len(para) + (2 if prefix else 0)  # 段落间 \n\n
        if prefix_len + additional > overlap_size:
            break
        prefix.insert(0, para)
        prefix_len += additional

    return prefix, prefix_len
