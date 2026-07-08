"""
对话窗口组件

聊天界面：顶部消息区（流式输出）→ 检索数量选择器 → 底部输入栏。
支持多会话管理：新建、切换、删除会话，刷新后自动恢复。
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from local_rag.config import TOP_K, DATA_DIR
from local_rag.qa.chain import generate_answer_stream
from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

_CHATS_DIR = DATA_DIR / "chats"

_TOP_K_TOOLTIP = (
    "检索片段数决定了每次提问时，从知识库中拉取多少个最相关的文档片段作为参考上下文。\n\n"
    "• 数值越小（1-3）：回答更聚焦，速度更快，但可能遗漏关键信息\n"
    "• 数值越大（8-20）：覆盖更全面，但可能引入噪声，速度较慢\n"
    "• 推荐范围：3-10，默认 5"
)


def render() -> None:
    """渲染对话区域。"""
    st.divider()

    _init_session_state()
    _ensure_default_session()

    # ====== 顶部：标题 + 会话操作按钮 ======
    _render_header()

    # ====== 消息列表 ======
    active = st.session_state["active_session"]
    msgs = st.session_state["chat_sessions"][active]["messages"]
    for msg in msgs:
        _render_message(msg)

    # ====== 检索数量选择器（固定位置） ======
    top_k = _render_topk_row()

    # ====== 聊天输入框（最底部） ======
    user_input = st.chat_input(placeholder="输入你的问题，按 Enter 发送...")

    if user_input and user_input.strip():
        _handle_question(user_input.strip(), top_k)


# ==================== 会话管理 ====================


def _init_session_state() -> None:
    """初始化或从文件恢复所有会话数据。"""
    if "chat_sessions" in st.session_state:
        return

    st.session_state["chat_sessions"] = {}
    st.session_state["active_session"] = None

    _CHATS_DIR.mkdir(parents=True, exist_ok=True)

    restored = 0
    for fpath in sorted(_CHATS_DIR.glob("*.json")):
        try:
            session = json.loads(fpath.read_text(encoding="utf-8"))
            sid = fpath.stem
            st.session_state["chat_sessions"][sid] = session
            restored += 1
        except Exception as e:
            logger.warning("无法恢复会话 %s: %s", fpath.name, e)

    if restored > 0:
        logger.info("已恢复 %d 个历史会话", restored)


def _ensure_default_session() -> None:
    """如果没有活跃会话且存在历史会话，自动选中最后活跃的。"""
    sessions = st.session_state["chat_sessions"]
    if st.session_state["active_session"] is None and sessions:
        st.session_state["active_session"] = _last_modified_session()
    elif st.session_state["active_session"] is None:
        _create_session()


def _last_modified_session() -> str:
    """返回最后修改的会话 ID。"""
    fpath = max(_CHATS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, default=None)
    if fpath:
        return fpath.stem
    # fallback: pick first from state
    sessions = st.session_state["chat_sessions"]
    return next(iter(sessions))


def _create_session(name: str | None = None) -> None:
    """创建新会话并设为活跃。"""
    sid = str(uuid.uuid4())
    session = {
        "name": name or f"对话 {datetime.now().strftime('%m-%d %H:%M')}",
        "messages": [],
        "created_at": datetime.now().isoformat(),
    }
    st.session_state["chat_sessions"][sid] = session
    st.session_state["active_session"] = sid
    _save_session(sid)


def _save_session(sid: str | None = None) -> None:
    """将会话持久化到文件。"""
    sid = sid or st.session_state.get("active_session")
    if not sid:
        return
    sessions = st.session_state["chat_sessions"]
    if sid not in sessions:
        return
    _CHATS_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _CHATS_DIR / f"{sid}.json"
    fpath.write_text(
        json.dumps(sessions[sid], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _delete_session(sid: str) -> None:
    """删除会话及其文件。"""
    if sid in st.session_state["chat_sessions"]:
        del st.session_state["chat_sessions"][sid]
    (_CHATS_DIR / f"{sid}.json").unlink(missing_ok=True)

    # 如果删的是当前活跃的，切换到其他会话或新建
    sessions = st.session_state["chat_sessions"]
    if st.session_state["active_session"] == sid:
        if sessions:
            st.session_state["active_session"] = _last_modified_session()
        else:
            _create_session()
    st.rerun()


def _render_header() -> None:
    """渲染标题行：会话名称 + 操作按钮。"""
    sessions = st.session_state["chat_sessions"]
    active = st.session_state["active_session"]
    active_data = sessions.get(active, {})

    c_title, c_rename, c_new, c_list = st.columns([10, 1, 1, 1])

    with c_title:
        st.subheader(f"💬 {active_data.get('name', '知识库问答')}")

    with c_rename:
        if st.button("✏", help="重命名会话", use_container_width=True):
            st.session_state["rename_prompt"] = active
            st.rerun()

    with c_new:
        if st.button("➕", help="新建会话", use_container_width=True):
            _create_session()
            st.rerun()

    with c_list:
        if st.button("📋", help="会话列表", use_container_width=True):
            st.session_state["show_session_list"] = not st.session_state.get(
                "show_session_list", False
            )

    # 重命名弹窗（不 pop，保留直到用户确认或取消）
    rename_target = st.session_state.get("rename_prompt")
    if rename_target and rename_target in sessions:
        with st.expander("✏ 重命名当前会话", expanded=True):
            new_name = st.text_input(
                "新名称",
                value=sessions[rename_target]["name"],
                key="rename_input",
                on_change=_on_rename_confirm,
            )
            c_confirm, c_cancel = st.columns([1, 1])
            with c_confirm:
                if st.button("✅ 确认", use_container_width=True):
                    _on_rename_confirm()
            with c_cancel:
                if st.button("❌ 取消", use_container_width=True):
                    st.session_state.pop("rename_prompt", None)
                    st.session_state.pop("rename_input", None)
                    st.rerun()

    # 会话列表
    if st.session_state.get("show_session_list"):
        _render_session_list(sessions, active)


def _render_session_list(sessions: dict, active: str) -> None:
    """渲染会话切换/删除列表。"""
    with st.expander("📋 所有会话", expanded=True):
        for sid in list(sessions.keys()):
            sdata = sessions[sid]
            c_name, c_switch, c_del = st.columns([7, 1, 1])

            with c_name:
                marker = "🟢 " if sid == active else "   "
                msg_count = len(sdata.get("messages", [])) // 2
                st.write(f"{marker}{sdata['name']} ({msg_count} 轮)")

            with c_switch:
                if sid != active and st.button("↗", key=f"sw_{sid}", help="切换到此会话"):
                    st.session_state["active_session"] = sid
                    st.session_state["show_session_list"] = False
                    st.rerun()

            with c_del:
                if st.button("🗑", key=f"dels_{sid}", help="删除此会话"):
                    _delete_session(sid)


def _on_rename_confirm() -> None:
    """回车或点击确认后执行重命名。"""
    rename_target = st.session_state.get("rename_prompt")
    new_name = st.session_state.get("rename_input", "")
    if not rename_target or not new_name.strip():
        return
    sessions = st.session_state["chat_sessions"]
    if rename_target in sessions:
        sessions[rename_target]["name"] = new_name.strip()
        _save_session(rename_target)
    st.session_state.pop("rename_prompt", None)
    st.session_state.pop("rename_input", None)
    st.rerun()


# ==================== 检索数量选择器 ====================


def _render_topk_row() -> int:
    """渲染检索数量选择器行。"""
    c1, c2, c3, c4 = st.columns([0.8, 1.2, 18, 1])

    with c1:
        top_k = st.number_input(
            "检索片段数",
            min_value=1, max_value=20, value=TOP_K, step=1,
            label_visibility="collapsed",
        )

    with c2:
        escaped_tip = _TOP_K_TOOLTIP.replace('"', '&quot;').replace('\n', '&#10;')
        st.markdown(
            f'<span style="font-size:13px;color:#888;white-space:nowrap;">检索数量'
            f' <span title="{escaped_tip}" '
            f'style="display:inline-flex;align-items:center;justify-content:center;'
            f'width:16px;height:16px;border-radius:50%;border:1px solid #aaa;color:#aaa;'
            f'font-size:11px;font-weight:bold;cursor:help;">?</span></span>',
            unsafe_allow_html=True,
        )

    with c4:
        if st.button("🗑", use_container_width=True, help="清空当前对话历史"):
            active = st.session_state["active_session"]
            st.session_state["chat_sessions"][active]["messages"] = []
            _save_session()
            st.rerun()

    return top_k


# ==================== 问答处理 ====================


def _handle_question(question: str, top_k: int) -> None:
    """处理用户提问：检索 → 流式生成 → 保存到历史。"""
    active = st.session_state["active_session"]
    msgs = st.session_state["chat_sessions"][active]["messages"]

    msgs.append({"role": "user", "content": question})

    store = ChromaStore()
    results = store.search(question, top_k=top_k)

    if not results:
        msgs.append({
            "role": "assistant",
            "content": "未找到相关内容，请先上传文档并构建索引。",
            "sources": [],
        })
        _save_session()
        st.rerun()

    contexts = []
    sources = []
    for i, result in enumerate(results, 1):
        source_path = result["metadata"].get("source", "未知来源")
        contexts.append(f"[来源 {i} - {Path(source_path).name}]\n{result['document']}")
        sources.append({
            "index": i,
            "file": Path(source_path).name,
            "path": source_path,
            "score": result["score"],
            "snippet": result["document"][:200],
        })

    context_text = "\n\n---\n\n".join(contexts)

    try:
        stream = generate_answer_stream(context_text, question)
    except Exception as e:
        logger.error("LLM 流式调用失败: %s", e)
        msgs.append({
            "role": "assistant",
            "content": f"回答生成失败: {e}",
            "sources": sources,
        })
        _save_session()
        st.rerun()

    full_answer = st.write_stream(stream)

    msgs.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
    _save_session()

    st.rerun()


# ==================== 消息渲染 ====================


def _render_message(msg: dict) -> None:
    """渲染单条对话消息。"""
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])

        if role == "assistant" and msg.get("sources"):
            with st.expander("📎 参考来源"):
                for src in msg["sources"]:
                    score_pct = src["score"] * 100
                    st.markdown(
                        f"**[{src['index']}] {src['file']}** "
                        f"*(相似度 {score_pct:.1f}%)*\n\n"
                        f"> {src['snippet']}..."
                    )
