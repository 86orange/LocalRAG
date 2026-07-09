"""
文件上传组件

提供拖拽/点击上传区域，上传后自动触发文档加载、切片与向量化入库。
上传过程中显示进度条和实时日志。
"""

import time
from pathlib import Path

import streamlit as st

from local_rag.config import DOCUMENTS_DIR
from local_rag.utils.file_utils import validate_file, get_file_type, get_file_metadata
from local_rag.loader import LOADER_MAP
from local_rag.loader.pdf_loader import load_and_chunk_pdf
from local_rag.chunker import chunk_by_size_with_metadata, chunk_by_semantic_with_metadata
from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.utils.dedup import DualLayerDeduplicator
from local_rag.version_manager import VersionManager
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
    vm = VersionManager()

    total = len(uploaded_files)
    progress_bar = st.progress(0, text="准备中...")
    status_container = st.empty()

    success_count = 0
    skip_count = 0
    dup_chunk_count = 0

    # 内容级去重器（双层：SimHash + 向量相似度）
    dedup = DualLayerDeduplicator(store=store)

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

        # ====== 文件级去重：doc_id 已存在则跳过 ======
        base_meta = get_file_metadata(save_path)
        doc_id = base_meta["doc_id"]
        if store.doc_exists(doc_id):
            status_container.info(f"⏭ {filename} 已索引，跳过（内容未变）")
            skip_count += 1
            progress_bar.progress(current_progress, text=f"({i+1}/{total})")
            continue

        # ====== 版本管理：归档旧索引 ======
        store.archive_source(str(save_path.resolve()))

        # ====== 加载 + 切片（PDF 使用页面感知管线）======
        if ftype == "pdf":
            chunk_fn_wrapper = (
                chunk_by_semantic_with_metadata if ftype in ("md", "docx")
                else chunk_by_size_with_metadata
            )
            chunks, metadatas = load_and_chunk_pdf(
                save_path,
                chunk_fn=chunk_fn_wrapper,
                base_metadata=base_meta,
            )
        else:
            progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在解析 {filename}")
            status_container.info(f"📄 正在加载 {filename} ...")
            text = loader(save_path)

            if not text.strip():
                status_container.warning(f"⚠️ {filename} 未能提取到文本内容，已跳过")
                skip_count += 1
                continue

            # ====== 切片（带 metadata）======
            progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在切片 {filename}")
            if ftype in ("md", "docx"):
                chunks, metadatas = chunk_by_semantic_with_metadata(text, base_metadata=base_meta)
            else:
                chunks, metadatas = chunk_by_size_with_metadata(text, base_metadata=base_meta)
        if not chunks:
            status_container.warning(f"⚠️ {filename} 切片后无内容，已跳过")
            skip_count += 1
            continue

        # ====== 4.5. 内容级去重（SimHash + 向量相似度）======
        orig_count = len(chunks)
        chunks, metadatas = dedup.filter_chunks(chunks, metadatas)
        dup_chunk_count += orig_count - len(chunks)

        if not chunks:
            status_container.info(f"⏭ {filename} ({orig_count} 个片段全部重复，已跳过)")
            skip_count += 1
            continue

        # ====== 5. 向量化入库 ======
        progress_bar.progress(current_progress, text=f"({i+1}/{total}) 正在索引 {filename}")

        try:
            store.delete_by_source(str(save_path.resolve()))
        except Exception:
            pass

        # 记录版本 + 注入 version / is_active
        new_version = vm.record(base_meta["doc_id"], filename, len(chunks), metadata=base_meta)
        for meta in metadatas:
            meta["version"] = str(new_version)
            meta["is_active"] = True

        store.add_documents(chunks, metadatas)

        success_count += 1
        status_container.success(f"✅ {filename} — {len(chunks)} 个片段已入库")

        time.sleep(0.1)

    # 汇总
    progress_bar.progress(1.0, text="索引完成!")
    summary = f"上传完成: {success_count} 成功, {skip_count} 跳过"
    if dup_chunk_count > 0:
        summary += f" (过滤 {dup_chunk_count} 个重复片段)"
    st.toast(summary, icon="✅")
