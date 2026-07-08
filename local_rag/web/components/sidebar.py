"""
侧边栏组件

展示知识库文件列表、索引状态，支持删除单个文件索引和全量重建。
"""

from pathlib import Path

import streamlit as st

from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)


def render() -> None:
    """渲染侧边栏：统计信息 + 知识库文件列表 + 管理操作。"""
    with st.sidebar:
        st.header("📊 知识库管理")

        store = ChromaStore()
        stats = store.get_stats()

        # ====== 统计卡片 ======
        col1, col2 = st.columns(2)
        col1.metric("片段数", stats["total_chunks"])
        col2.metric("来源文件", stats["total_sources"])

        with st.expander("📋 详细统计"):
            st.caption(f"存储目录: {stats['persist_dir']}")
            st.caption(f"Embedding: {stats['embedding_model']}")

        st.divider()

        # ====== 已索引文件列表 ======
        st.subheader("📁 已索引文件")

        sources = _get_sources(store)
        if not sources:
            st.caption("暂无已索引文件，请先上传文档。")
            return

        st.caption(f"共 {len(sources)} 个文件")

        for source_path in sources:
            _render_file_row(store, source_path)

        st.divider()

        # ====== 批量操作 ======
        st.subheader("🔧 批量操作")

        if st.button("🔄 重建全部索引", use_container_width=True, type="secondary"):
            if st.session_state.get("confirm_rebuild"):
                _rebuild_all(store)
                st.rerun()
            else:
                st.session_state["confirm_rebuild"] = True
                st.warning("再次点击确认重建全部索引（此操作不可撤回）")
        else:
            st.session_state.pop("confirm_rebuild", None)


def _get_sources(store: ChromaStore) -> list[str]:
    """从 ChromaDB 中提取所有来源文件路径，按文件名排序。

    Returns:
        去重后的来源路径列表
    """
    try:
        all_data = store._collection.get()
        metas = all_data.get("metadatas", [])
        sources: set[str] = set()
        for meta in metas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources, key=lambda s: Path(s).name.lower())
    except Exception:
        return []


def _render_file_row(store: ChromaStore, source_path: str) -> None:
    """渲染单行文件信息，含文件名、路径和删除按钮。

    Args:
        store: ChromaStore 实例
        source_path: 文件完整路径
    """
    file_name = Path(source_path).name

    col_info, col_btn = st.columns([4, 1])

    with col_info:
        st.write(f"📄 {file_name}")
        st.caption(source_path)

    with col_btn:
        if st.button("🗑", key=f"del_{source_path}", help=f"删除 {file_name} 的索引"):
            deleted = store.delete_by_source(source_path)
            if deleted > 0:
                st.toast(f"已删除 {file_name} 的 {deleted} 个片段", icon="🗑")
            else:
                st.toast(f"未找到 {file_name} 的索引", icon="ℹ")
            st.rerun()


def _rebuild_all(store: ChromaStore) -> None:
    """清空向量库，重新扫描 documents/ 并索引所有文件。"""
    deleted = store.delete_all()
    st.toast(f"已清空 {deleted} 个片段，请重新上传文件进行索引", icon="🔄")
