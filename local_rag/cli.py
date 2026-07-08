"""
命令行入口模块

提供 rag 命令行工具，支持索引构建、知识库查询、统计与删除操作。
"""

import argparse
import sys
from pathlib import Path

from local_rag.config import init_dirs, TOP_K, DOCUMENTS_DIR
from local_rag.utils.file_utils import scan_documents, validate_file, get_file_type
from local_rag.loader import LOADER_MAP
from local_rag.chunker.text_chunker import chunk_by_size
from local_rag.chunker.semantic_chunker import chunk_by_semantic
from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.utils.logger import get_logger, _init as init_logger

logger = get_logger(__name__)


def main() -> None:
    """CLI 主入口。"""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    init_dirs()
    init_logger()

    try:
        if args.command == "index":
            _cmd_index(args)
        elif args.command == "query":
            _cmd_query(args)
        elif args.command == "stats":
            _cmd_stats()
        elif args.command == "delete":
            _cmd_delete(args)
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        logger.error("命令执行失败: %s", e)
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


# ==================== 命令实现 ====================


def _cmd_index(args) -> None:
    """索引命令：扫描文档 → 加载 → 切片 → 向量化入库。"""
    chunk_fn = chunk_by_semantic if args.semantic else chunk_by_size
    chunk_label = "语义切片" if args.semantic else "字符切片"

    scan_root = Path(args.dir) if args.dir else DOCUMENTS_DIR
    files = scan_documents(root=scan_root)
    if not files:
        print(f"文档目录为空 ({scan_root})，请将文件放入后重试。")
        return

    print(f"开始索引 {len(files)} 个文件（{chunk_label}）...")

    store = ChromaStore()

    if args.force:
        deleted = store.delete_all()
        print(f"已清空旧索引 ({deleted} 个片段)")

    total_chunks = 0

    for i, file_path in enumerate(files, 1):
        if not validate_file(file_path):
            continue

        ftype = get_file_type(file_path)
        loader = LOADER_MAP.get(ftype)
        if loader is None:
            logger.warning("未知文件类型: %s", file_path.name)
            continue

        source = str(file_path.resolve())
        print(f"  [{i}/{len(files)}] {file_path.name} ...", end=" ")

        try:
            store.delete_by_source(source)
        except Exception:
            pass

        text = loader(file_path)
        if not text.strip():
            print("(空)")
            continue

        chunks = chunk_fn(text)
        if not chunks:
            print("(无内容)")
            continue

        metadatas = [
            {"source": source, "chunk_index": j, "file_type": ftype}
            for j in range(len(chunks))
        ]
        store.add_documents(chunks, metadatas)

        total_chunks += len(chunks)
        print(f"{len(chunks)} 块")

    stats = store.get_stats()
    print(f"\n索引完成: {stats['total_chunks']} 个片段, {stats['total_sources']} 个来源文件")


def _cmd_query(args) -> None:
    """查询命令：检索 + LLM 回答。"""
    store = ChromaStore()
    stats = store.get_stats()
    if stats["total_chunks"] == 0:
        print("知识库为空，请先运行 `rag index` 构建索引。")
        return

    print(f"检索中 (top_k={args.top_k})...")
    results = store.search(args.question, top_k=args.top_k)
    if not results:
        print("未找到相关内容。")
        return

    # 拼接上下文
    contexts: list[str] = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "未知来源")
        contexts.append(f"[来源 {i} - {Path(source).name}]\n{result['document']}")

    context_text = "\n\n---\n\n".join(contexts)

    # 调用 QA Chain 生成回答
    try:
        from local_rag.qa.chain import build_qa_chain

        qa_chain = build_qa_chain()
        answer = qa_chain.invoke({
            "context": context_text,
            "question": args.question,
        })

        print(f"\n{'='*60}")
        print(f"问题: {args.question}")
        print(f"{'='*60}")
        print(f"\n{answer}\n")
        print(f"{'='*60}")
        print(f"参考来源:")
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "未知来源")
            print(f"  [{i}] {Path(source).name} (相似度: {result['score']:.4f})")

    except ImportError:
        print("QA 模块尚未实现，以下是检索到的相关内容片段:\n")
        print(context_text)
    except Exception as e:
        logger.error("问答生成失败: %s", e)
        print("检索到的相关内容片段:\n")
        print(context_text)


def _cmd_stats() -> None:
    """统计命令：展示知识库状态。"""
    store = ChromaStore()
    stats = store.get_stats()

    print(f"知识库统计:")
    print(f"  文档片段数: {stats['total_chunks']}")
    print(f"  来源文件数: {stats['total_sources']}")
    print(f"  存储目录:   {stats['persist_dir']}")
    print(f"  Embedding:   {stats['embedding_model']}")

    if "error" in stats:
        print(f"  警告: {stats['error']}")


def _cmd_delete(args) -> None:
    """删除命令：移除指定文件的所有索引。"""
    store = ChromaStore()
    deleted = store.delete_by_source(args.source)

    if deleted > 0:
        print(f"已删除 {deleted} 个片段 (来源: {args.source})")
    else:
        print(f"未找到匹配的索引 (来源: {args.source})")


# ==================== 参数解析 ====================


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Local RAG Agent — 完全本地运行的知识库 AI 助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  rag index              # 索引 documents/ 下的所有文件\n"
            "  rag index --semantic   # 使用语义切片索引\n"
            "  rag query \"什么是RAG？\" # 向知识库提问\n"
            "  rag stats              # 查看知识库统计\n"
            "  rag delete ./doc.pdf   # 删除指定文件的索引"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # index
    idx = subparsers.add_parser("index", help="扫描并索引文档")
    idx.add_argument("--dir", type=str, default=None, help="自定义文档目录")
    idx.add_argument("--semantic", action="store_true", help="使用语义切片")
    idx.add_argument("--force", action="store_true", help="强制重建全部索引")

    # query
    q = subparsers.add_parser("query", help="向知识库提问")
    q.add_argument("question", type=str, help="问题文本")
    q.add_argument("-k", "--top-k", type=int, default=TOP_K, help=f"检索片段数 (默认 {TOP_K})")

    # stats
    subparsers.add_parser("stats", help="查看知识库统计信息")

    # delete
    d = subparsers.add_parser("delete", help="删除指定文件的索引")
    d.add_argument("source", type=str, help="文件路径标识（完整路径或文件名）")

    return parser
