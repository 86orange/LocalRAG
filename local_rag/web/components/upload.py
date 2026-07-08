"""
文件上传组件

提供拖拽/点击上传区域，上传后自动触发文档加载、切片与向量化入库。
上传过程中显示进度条和实时日志。
"""

import time
from pathlib import Path

import streamlit as st

from local_rag.config import DOCUMENTS_DIR
from local_rag.utils.file_utils import validate_file, get_file_type
from local_rag.loader import LOADER_MAP
from local_rag.chunker.text_chunker import chunk_by_size
from local_rag.chunker.semantic_chunker import chunk_by_semantic
from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_FILE_SIZE_MB = 50


def render() -> None:
    """渲染文件上传组件。

    在页面中创建一个拖拽上传区域，用户上传文件后自动完成：
    保存到 documents/ → 文本加载 → 切片 → 向量化入库。
    """
    st.subheader("📤 上传文档")

    uploaded_files = st.file_uploader(
        label="拖拽或点击选择文件",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.caption("支持 PDF / Word / Markdown / TXT 格式，单文件上限 50MB")
        return

    _process_uploads(uploaded_files)


def _process_uploads(uploaded_files: list) -> None:
    """处理上传的文件列表，逐个保存并索引。

    流程：验证 → 保存 → 加载文本 → 切片 → 向量化入库。
    每个文件使用独立的进度步骤，整体进度也同步更新。
    """
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    store = ChromaStore()

    total = len(uploaded_files)
    progress_bar = st.progress(0, text="准备中...")
    status_container = st.empty()

    # 自动选择切片策略（MD / DOCX 用语义切片）
    auto_chunker = lambda ftype: chunk_by_semantic if ftype in ("md", "docx") else chunk_by_size

    success_count = 0
    skip_count = 0

    for i, uploaded in enumerate(uploaded_files):
        filename = uploaded.name
        current_progress = (i + 1) / total

        # ====== 1. 大小的验证 ======
        file_size_mb = uploaded.size / (1024 * 1024)
        if file_size_mb > _MAX_FILE_SIZE_MB:
            status_container.warning(
                f"⚠️ 跳过 {filename}（{file_size_mb:.1f}MB，超过 {_MAX_FILE_SIZE_MB}MB 上限）"
            )
            skip_count += 1
            progress_bar.progress(current_progress, text=f"({i+1}/{total})")
            continue

        # ====== 2. 保存到 documents/ ======
        save_path = DOCUMENTS_DIR / filename
        progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在保存 {filename}")
        save_path.write_bytes(uploaded.getbuffer())

        # ====== 3. 检验 + 加载 ======
        if not validate_file(save_path):
            status_container.warning(f"⚠️ {filename} 格式不支持，已跳过")
            skip_count += 1
            continue

        ftype = get_file_type(save_path)
        loader = LOADER_MAP.get(ftype)
        if loader is None:
            status_container.warning(f"⚠️ {filename} 未知文件类型，已跳过")
            skip_count += 1
            continue

        progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在解析 {filename}")
        status_container.info(f"📄 正在加载 {filename} ...")
        text = loader(save_path)

        if not text.strip():
            status_container.warning(f"⚠️ {filename} 未能提取到文本内容，已跳过")
            skip_count += 1
            continue

        # ====== 4. 切片 ======
        progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在切片 {filename}")
        chunk_fn = auto_chunker(ftype)
        chunks = chunk_fn(text)
        if not chunks:
            status_container.warning(f"⚠️ {filename} 切片后无内容，已跳过")
            skip_count += 1
            continue

        # ====== 5. 向量化入库 ======
        source = str(save_path.resolve())
        progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在索引 {filename}")

        try:
            store.delete_by_source(source)
        except Exception:
            pass

        metadatas = [
            {"source": source, "chunk_index": j, "file_type": ftype}
            for j in range(len(chunks))
        ]
        store.add_documents(chunks, metadatas)

        success_count += 1
        status_container.success(f"✅ {filename} — {len(chunks)} 个片段已入库")

        time.sleep(0.1)

    # 汇总
    progress_bar.progress(1.0, text="索引完成!")
    st.toast(f"上传完成: {success_count} 成功, {skip_count} 跳过", icon="✅")
