"""
Streamlit Web 主页面。

启动命令: streamlit run local_rag/web/app.py
"""

import streamlit as st

from local_rag.config import init_dirs
from local_rag.web.components import upload, sidebar, chat

st.set_page_config(
    page_title="LocalRAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 LocalRAG — 本地知识库 AI 助手")

init_dirs()

sidebar.render()

upload.render()

chat.render()
