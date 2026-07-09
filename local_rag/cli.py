"""
命令行入口模块

提供 rag 命令行工具，支持索引构建、知识库查询、统计与删除操作、版本管理。
"""

import argparse
import sys
from pathlib import Path

from local_rag.config import init_dirs, TOP_K, DOCUMENTS_DIR, SIMILARITY_THRESHOLD
from local_rag.utils.file_utils import scan_documents, validate_file, get_file_type, get_file_metadata
from local_rag.utils.dedup import DualLayerDeduplicator
from local_rag.loader import LOADER_MAP
from local_rag.loader.pdf_loader import load_and_chunk_pdf
from local_rag.chunker import chunk_by_size_with_metadata, chunk_by_semantic_with_metadata
from local_rag.vector_store.chroma_store import ChromaStore
from local_rag.version_manager import VersionManager
from local_rag.retrieval import HybridRetriever
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
        elif args.command == "versions":
            _cmd_versions(args)
        elif args.command == "rollback":
            _cmd_rollback(args)
        elif args.command == "eval":
            _cmd_eval(args)
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        logger.error("命令执行失败: %s", e)
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


# ==================== 命令实现 ====================


def _cmd_index(args) -> None:
    """索引命令：扫描文档 → 加载 → 切片 → 向量化入库（含版本管理）。"""
    chunk_label = "语义切片" if args.semantic else "字符切片"

    scan_root = Path(args.dir) if args.dir else DOCUMENTS_DIR
    files = scan_documents(root=scan_root)
    if not files:
        print(f"文档目录为空 ({scan_root})，请将文件放入后重试。")
        return

    print(f"开始索引 {len(files)} 个文件（{chunk_label}）...")

    store = ChromaStore()
    vm = VersionManager()

    if args.force:
        deleted = store.delete_all()
        print(f"已清空旧索引 ({deleted} 个片段)")

    total_chunks = 0
    skipped_by_dedup = 0
    dup_chunks = 0
    versioned_files = 0

    dedup = DualLayerDeduplicator(store=store)

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

        base_meta = get_file_metadata(file_path)
        doc_id = base_meta["doc_id"]

        if not args.force and store.doc_exists(doc_id):
            print("(已索引，跳过)")
            skipped_by_dedup += 1
            continue

        # ====== 版本管理：归档旧索引 + 记录新版本 ======
        archived = store.archive_source(source)
        if archived > 0:
            versioned_files += 1

        # ====== 加载 + 切片（PDF 使用页面感知管线）======
        if ftype == "pdf":
            chunk_fn_wrapper = (
                chunk_by_semantic_with_metadata if args.semantic
                else chunk_by_size_with_metadata
            )
            chunks, metadatas = load_and_chunk_pdf(
                file_path,
                chunk_fn=chunk_fn_wrapper,
                base_metadata=base_meta,
            )
        else:
            text = loader(file_path)
            if not text.strip():
                print("(空)")
                continue

            if args.semantic:
                chunks, metadatas = chunk_by_semantic_with_metadata(text, base_metadata=base_meta)
            else:
                chunks, metadatas = chunk_by_size_with_metadata(text, base_metadata=base_meta)

        if not chunks:
            print("(无内容)")
            continue

        # 内容级去重
        orig_count = len(chunks)
        chunks, metadatas = dedup.filter_chunks(chunks, metadatas)
        dup_chunks += orig_count - len(chunks)

        if not chunks:
            print(f"({orig_count} 块全部重复，跳过)")
            continue

        # 注入版本号 + 活跃标记
        new_version = vm.record(doc_id, file_path.name, len(chunks), metadata=base_meta)
        for meta in metadatas:
            meta["version"] = str(new_version)
            meta["is_active"] = True

        store.add_documents(chunks, metadatas)

        total_chunks += len(chunks)
        print(f"{len(chunks)} 块 (v{new_version})")

    stats = store.get_stats()
    msg = f"\n索引完成: {stats['total_chunks']} 个片段, {stats['total_sources']} 个来源文件"
    if skipped_by_dedup > 0:
        msg += f" (跳过 {skipped_by_dedup} 个重复文件)"
    if versioned_files > 0:
        msg += f" (版本更新 {versioned_files} 个)"
    if dup_chunks > 0:
        msg += f" (过滤 {dup_chunks} 个重复片段)"
    print(msg)


def _cmd_query(args) -> None:
    """查询命令：混合检索 + LLM 回答。"""
    store = ChromaStore()
    stats = store.get_stats()
    if stats["total_chunks"] == 0:
        print("知识库为空，请先运行 `rag index` 构建索引。")
        return

    print(f"混合检索中 (top_k={args.top_k}, threshold={args.threshold})...")
    hybrid = HybridRetriever(store)
    results = hybrid.search(
        args.question,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )
    if not results:
        print("未找到相关内容。")
        return

    contexts: list[str] = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("source", "未知来源")
        contexts.append(f"[来源 {i} - {Path(source).name}]\n{result['document']}")

    context_text = "\n\n---\n\n".join(contexts)

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

    vm = VersionManager()
    docs = vm.get_all_docs()
    if docs:
        print(f"  版本记录:   {len(docs)} 个文档")
        for doc_id in docs:
            versions = vm.list_versions(doc_id)
            active = vm.get_active_version(doc_id)
            name = versions[-1].get("file_name", doc_id) if versions else doc_id
            print(f"    {name}: {len(versions)} 个版本 (当前 v{active})")

    if "error" in stats:
        print(f"  警告: {stats['error']}")


def _cmd_delete(args) -> None:
    """删除命令：移除指定文件的所有索引（含所有版本）。"""
    store = ChromaStore()
    deleted = store.delete_by_source(args.source)

    if deleted > 0:
        print(f"已删除 {deleted} 个片段 (来源: {args.source})")
    else:
        print(f"未找到匹配的索引 (来源: {args.source})")


def _cmd_versions(args) -> None:
    """版本列表命令：显示指定文件的所有版本历史。"""
    vm = VersionManager()
    store = ChromaStore()

    source = str(Path(args.source).resolve())
    results = store.search("", top_k=1, where={"source": source})
    if not results:
        doc_id = args.source
    else:
        doc_id = results[0]["metadata"].get("doc_id", args.source)

    versions = vm.list_versions(doc_id)
    if not versions:
        print(f"未找到版本记录: {args.source}")
        return

    active = vm.get_active_version(doc_id)
    name = versions[-1].get("file_name", doc_id)

    print(f"\n版本历史: {name}")
    print(f"{'='*60}")
    print(f"{'版本':<6} {'chunks':<8} {'变化':<8} {'时间':<20} {'状态'}")
    print(f"{'-'*60}")
    for v in versions:
        marker = " ◀ 当前" if v["version"] == active else ""
        delta_str = f"{v['delta']:+d}"
        print(
            f"v{v['version']:<5}"
            f" {v['chunk_count']:<8}"
            f" {delta_str:<8}"
            f" {v['created_at'][:19]:<20}"
            f"{marker}"
        )


def _cmd_rollback(args) -> None:
    """回滚命令：将指定文件的索引恢复到目标版本。"""
    vm = VersionManager()
    store = ChromaStore()

    source = str(Path(args.source).resolve())

    # 查找 doc_id
    try:
        results = store.search(source, top_k=1)
        doc_id = results[0]["metadata"].get("doc_id", source) if results else source
    except Exception:
        doc_id = source

    target = args.version
    versions = vm.list_versions(doc_id)
    if not versions:
        print(f"未找到版本记录: {source}")
        return

    if target < 1 or target > len(versions):
        print(f"无效版本号: {target} (有效范围: 1-{len(versions)})")
        return

    # 归档当前活跃版本
    store.archive_source(source)

    # 回滚版本记录
    if not vm.rollback_to(doc_id, target):
        print("回滚失败")
        return

    # 重新激活目标版本的 chunks
    activated = store.activate_all(source)
    print(
        f"已回滚 → v{target} ({activated} 个片段)"
        f" | {versions[-target].get('file_name', '?')}"
        f" | {versions[-target].get('created_at', '')[:19]}"
    )


# ==================== eval 命令 ====================


def _cmd_eval(args) -> None:
    """评估命令：对答案忠实度 / 召回率进行评分。"""
    if args.eval_type == "faithfulness":
        _cmd_eval_faithfulness(args)
    elif args.eval_type == "recall":
        _cmd_eval_recall(args)


def _cmd_eval_recall(args) -> None:
    """召回命中率评估。"""
    from local_rag.eval import evaluate_recall, load_dataset

    store = ChromaStore()
    stats = store.get_stats()
    if stats["total_chunks"] == 0:
        print("知识库为空，请先运行 `rag index` 构建索引。")
        return

    print(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"共 {len(dataset)} 个测试问题\n")

    result = evaluate_recall(
        store,
        dataset,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )

    if result.total == 0:
        print("无有效数据项。")
        return

    print(f"{'=' * 60}")
    print(f"📊 召回命中率评估结果")
    print(f"{'=' * 60}")
    print(f"  总问题数:    {result.total}")
    print(f"  命中数:      {result.hit_count} ({result.hit_count/result.total:.1%})")
    print(f"  Recall@1:    {result.recall_at_1:.2%}")
    print(f"  Recall@3:    {result.recall_at_3:.2%}")
    print(f"  Recall@5:    {result.recall_at_5:.2%}")
    print(f"  MRR:         {result.mrr:.4f}")
    print(f"{'=' * 60}")

    # 打印未命中项
    missed = [d for d in result.details if not d.recalled]
    if missed:
        print(f"\n⚠️  未命中 ({len(missed)}/{result.total}):")
        for d in missed:
            print(f"   Q: {d.question}")
            print(f"      应命中: {d.relevant_texts}")
            print(f"      实际召回: 未找到")
            if d.retrieved_docs:
                snippet = d.retrieved_docs[0][:100]
                print(f"      top-1 片段: {snippet}...")
            print()

    print(f"\n💡 提示: 修改数据集文件 {args.dataset} 中的 relevant_texts 以适配你的知识库内容。")


def _cmd_eval_faithfulness(args) -> None:
    """答案忠实度评估。"""
    from local_rag.eval import evaluate_faithfulness

    store = ChromaStore()
    stats = store.get_stats()
    if stats["total_chunks"] == 0:
        print("知识库为空，请先运行 `rag index` 构建索引。")
        return

    print(f"混合检索评估问题: {args.question}")

    hybrid = HybridRetriever(store)
    results = hybrid.search(
        args.question,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )
    if not results:
        print("未检索到相关内容，无法评估。")
        return

    contexts = []
    for i, r in enumerate(results, 1):
        src = r["metadata"].get("source", "未知")
        contexts.append(f"[来源 {i} - {src}]\n{r['document']}")
    context_text = "\n\n---\n\n".join(contexts)

    from local_rag.qa.chain import generate_answer
    print("生成回答中...")
    answer = generate_answer(context_text, args.question)

    print("\n" + "=" * 60)
    print("回答:")
    print(answer)
    print("=" * 60)

    print("\n评估中 (LLM-as-Judge, 需要 Ollama 运行中)...")
    result = evaluate_faithfulness(
        question=args.question,
        context=context_text,
        answer=answer,
    )

    if result.error:
        print(f"\n评估出错: {result.error}")
        return

    print(f"\n📊 忠实度评估结果")
    print(f"{'=' * 60}")
    print(f"  忠实度分数:  {result.faithfulness_score:.2%}")
    print(f"  总主张数:    {result.total_claims}")
    print(f"  有证据支持:  {result.supported}")
    print(f"  部分支持:    {result.partial}")
    print(f"  矛盾:        {result.contradicted}")
    print(f"  无依据:      {result.unsupported}")
    print(f"  引用正确:    {result.citation_correct}")
    print(f"  引用错误:    {result.citation_wrong}")
    print(f"  引用虚构:    {result.citation_fabricated}")
    print(f"{'=' * 60}")

    if result.claims:
        print("\n逐条主张详情:")
        for i, claim in enumerate(result.claims, 1):
            icon = {"支持": "✅", "部分支持": "⚠️", "矛盾": "❌", "无依据": "🚫"}.get(claim.verdict, "❓")
            print(f"  [{i}] {icon} {claim.verdict}: {claim.text[:100]}")
            if claim.verdict in ("无依据", "矛盾"):
                print(f"       ⤷ 这是 LLM 编造/不忠实的内容")


# ==================== 参数解析 ====================


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Local RAG Agent — 完全本地运行的知识库 AI 助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  rag index                # 索引 documents/ 下的所有文件\n"
            "  rag index --semantic     # 使用语义切片索引\n"
            "  rag query \"什么是RAG？\"   # 向知识库提问\n"
            "  rag stats                # 查看知识库统计\n"
            "  rag versions ./doc.pdf   # 查看文件版本历史\n"
            "  rag rollback ./doc.pdf 2 # 回滚到版本 2\n"
            "  rag delete ./doc.pdf     # 删除指定文件的所有索引"
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
    q.add_argument("-t", "--threshold", type=float, default=SIMILARITY_THRESHOLD,
                   help=f"RRF 分数阈值，低于此值过滤 (默认 {SIMILARITY_THRESHOLD})")

    # stats
    subparsers.add_parser("stats", help="查看知识库统计信息")

    # delete
    d = subparsers.add_parser("delete", help="删除指定文件的索引")
    d.add_argument("source", type=str, help="文件路径标识（完整路径或文件名）")

    # versions
    v = subparsers.add_parser("versions", help="查看文件的版本历史")
    v.add_argument("source", type=str, help="文件路径")

    # rollback
    r = subparsers.add_parser("rollback", help="回滚文件索引到指定版本")
    r.add_argument("source", type=str, help="文件路径")
    r.add_argument("version", type=int, help="目标版本号")

    # eval
    ev = subparsers.add_parser("eval", help="评估 RAG 系统质量")
    ev_sub = ev.add_subparsers(dest="eval_type", help="评估类型")
    ef = ev_sub.add_parser("faithfulness", help="答案忠实度评估")
    ef.add_argument("question", type=str, help="评估用问题")
    ef.add_argument("-k", "--top-k", type=int, default=TOP_K, help=f"检索片段数 (默认 {TOP_K})")
    ef.add_argument("-t", "--threshold", type=float, default=SIMILARITY_THRESHOLD,
                    help=f"RRF 分数阈值 (默认 {SIMILARITY_THRESHOLD})")

    # eval recall
    er = ev_sub.add_parser("recall", help="召回命中率评估")
    er.add_argument("dataset", type=str, help="评估数据集 JSON 文件路径")
    er.add_argument("-k", "--top-k", type=int, default=TOP_K, help=f"召回窗口大小 (默认 {TOP_K})")
    er.add_argument("-t", "--threshold", type=float, default=0.0,
                    help="检索阈值 (默认 0，不过滤)")

    return parser
