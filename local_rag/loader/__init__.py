"""
文档加载模块

根据文件类型自动分发给对应的加载器，返回统一的文本内容。
支持格式：Markdown、TXT、PDF、DOCX。
"""

from local_rag.loader.markdown_loader import load_markdown
from local_rag.loader.txt_loader import load_txt
from local_rag.loader.pdf_loader import load_pdf, load_pdf_page_by_page, compute_page_offsets, map_chunks_to_pages, load_and_chunk_pdf
from local_rag.loader.docx_loader import load_docx

# 文件类型 → loader 映射表，供 CLI / Web / 测试统一使用
LOADER_MAP = {
    "md": load_markdown,
    "txt": load_txt,
    "pdf": load_pdf,
    "docx": load_docx,
}
