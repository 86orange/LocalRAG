"""
ChromaDB 向量存储封装

基于 ChromaDB 实现文档向量化存储与语义检索。
Embedding 由 Ollama 本地模型提供，所有数据落盘存储。
"""

import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from local_rag.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    TOP_K,
)
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

# ChromaDB 集合名称
_COLLECTION_NAME = "local_rag_docs"


class ChromaStore:
    """ChromaDB 向量存储管理器。

    封装文档的向量化入库、语义检索、索引删除与统计功能。

    Usage:
        store = ChromaStore()
        store.add_documents(chunks, metadata_list)
        results = store.search("什么是 RAG？", top_k=3)
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        """初始化 ChromaDB 客户端与 Ollama Embedding 函数。

        Args:
            persist_dir: 向量数据库持久化目录，默认使用 config.VECTOR_DB_DIR
        """
        persist_dir = persist_dir or VECTOR_DB_DIR
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._persist_dir = persist_dir
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

        self._init_client()
        logger.info(
            "ChromaStore 初始化完成: %s (模型=%s)",
            persist_dir,
            EMBEDDING_MODEL,
        )

    # ==================== 公共接口 ====================

    def get_all_chunks(self) -> list[str]:
        """获取向量库中所有文档的文本内容。

        用于构建 BM25 关键词索引。

        Returns:
            所有文档文本列表（顺序与 ChromaDB 内部 ID 顺序一致）
        """
        self._ensure_collection()
        try:
            all_data = self._collection.get()
            docs = all_data.get("documents", [])
            return docs if docs else []
        except Exception as e:
            logger.warning("获取全量文档失败: %s", e)
            return []

    def get_chunk_metadata(self, index: int) -> dict | None:
        """按索引获取指定文档的 metadata。

        Args:
            index: 文档在 get_all_chunks() 返回列表中的位置

        Returns:
            metadata 字典，不存在返回 None
        """
        self._ensure_collection()
        try:
            all_data = self._collection.get()
            metas = all_data.get("metadatas", [])
            if 0 <= index < len(metas):
                return metas[index]
            return None
        except Exception as e:
            logger.warning("获取 chunk metadata 失败 (index=%d): %s", index, e)
            return None

    def map_results_to_indices(
        self,
        results: list[dict[str, Any]],
        bm25_retriever,
    ) -> list[tuple[int, float]]:
        """将 ChromaDB 检索结果映射到 BM25 全局文档索引。

        通过文档内容在 BM25 的 chunks 列表中定位对应位置。

        Args:
            results: ChromaStore.search 返回的结果列表
            bm25_retriever: BM25Retriever 实例

        Returns:
            [(doc_index, score), ...]
        """
        mapped: list[tuple[int, float]] = []
        bm25_docs = [bm25_retriever.get_document(i) for i in range(bm25_retriever.doc_count)]

        for result in results:
            doc = result.get("document", "")
            score = result.get("score", 0.0)
            try:
                idx = bm25_docs.index(doc)
                mapped.append((idx, score))
            except ValueError:
                mapped.append((-1, score))
        return mapped

    def add_documents(
        self,
        chunks: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """将文本块向量化并存入 ChromaDB。

        Args:
            chunks: 文本块列表
            metadatas: 每条文本的元数据（来源文件、块序号等），长度需与 chunks 一致
            ids: 自定义 ID 列表，未提供则自动生成 UUID

        Returns:
            入库的文档 ID 列表
        """
        self._ensure_collection()

        ids = ids or [str(uuid.uuid4()) for _ in chunks]

        if metadatas and len(metadatas) != len(chunks):
            raise ValueError(
                f"metadatas 长度 ({len(metadatas)}) 与 chunks ({len(chunks)}) 不一致"
            )

        if len(ids) != len(chunks):
            raise ValueError(
                f"ids 长度 ({len(ids)}) 与 chunks ({len(chunks)}) 不一致"
            )

        try:
            self._collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas if metadatas else None,
            )
        except Exception as e:
            logger.error("文档入库失败: %s", e)
            raise

        logger.info("已入库 %d 个文档片段", len(chunks))
        return ids

    def search(
        self,
        query: str,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """语义检索最相关的文档片段。

        Args:
            query: 查询文本
            top_k: 返回结果数，默认使用全局配置 TOP_K
            where: ChromaDB 过滤条件（如按来源文件过滤）

        Returns:
            结果列表，每项包含 id / document / metadata / distance
        """
        self._ensure_collection()

        top_k = top_k or TOP_K

        try:
            raw = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
            )
        except Exception as e:
            logger.error("检索失败: %s", e)
            return []

        return self._format_results(raw)

    def search_similar(
        self,
        query: str,
        top_k: int = 1,
        similarity_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """检索与查询文本高度相似的已有片段，用于内容去重。

        与 search 方法使用同一底层查询，但增加了相似度过滤：
        仅返回 cosine similarity >= threshold 的结果。

        Args:
            query: 查询文本
            top_k: 最大返回数
            similarity_threshold: 相似度阈值，0~1

        Returns:
            过滤后的结果列表（可能为空）
        """
        raw = self.search(query, top_k=top_k)
        return [
            r for r in raw
            if r.get("score", 0) >= similarity_threshold
        ]

    def doc_exists(self, doc_id: str) -> bool:
        """检查指定 doc_id 的文档是否已存在于向量库中。

        用于文件级去重：同一内容 MD5 的文件无需重复索引。

        Args:
            doc_id: 文档内容哈希（即 metadata 中的 doc_id 字段）

        Returns:
            True 表示已存在，False 表示可以安全入���
        """
        self._ensure_collection()

        try:
            results = self._collection.get(
                where={"doc_id": doc_id},
                limit=1,
            )
            ids = results.get("ids", [])
            return len(ids) > 0
        except Exception as e:
            logger.warning("doc_exists 查询失败 (doc_id=%s): %s", doc_id, e)
            return False

    def delete_by_source(self, source: str) -> int:
        """删除指定来源文件的所有文档片段。

        Args:
            source: 文件路径标识（即 metadata 中的 source 字段）

        Returns:
            删除的文档数量，集合未初始化返回 0
        """
        self._ensure_collection()

        try:
            results = self._collection.get(
                where={"source": source},
            )
            ids_to_delete = results.get("ids", [])
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                logger.info("已删除 %d 个片段 (来源: %s)", len(ids_to_delete), source)
            else:
                logger.debug("未找到需要删除的片段 (来源: %s)", source)
            return len(ids_to_delete)
        except Exception as e:
            logger.error("删除失败 (来源: %s): %s", source, e)
            return 0

    def archive_source(self, source: str) -> int:
        """归档指定来源文件的所有片段（标记 is_active=False）。

        用于版本管理：旧版本数据保留但检索时过滤掉。

        Args:
            source: 文件路径标识

        Returns:
            归档的文档数量
        """
        self._ensure_collection()

        try:
            results = self._collection.get(
                where={"source": source},
            )
            ids_to_update = results.get("ids", [])
            if ids_to_update:
                # ChromaDB v1 的 update 接口：逐条更新 metadata
                for chunk_id in ids_to_update:
                    self._collection.update(
                        ids=[chunk_id],
                        metadatas=[{"is_active": False}],
                    )
                logger.info("已归档 %d 个片段 (来源: %s)", len(ids_to_update), source)
            return len(ids_to_update)
        except Exception as e:
            logger.error("归档失败 (来源: %s): %s", source, e)
            return 0

    def activate_all(self, source: str) -> int:
        """重新激活指定来源的所有归档片段（is_active=True）。

        用于版本回滚：将指定版本的 chunks 重新标记为活跃。

        Args:
            source: 文件路径标识

        Returns:
            激活的文档数量
        """
        self._ensure_collection()

        try:
            results = self._collection.get(
                where={"source": source},
            )
            ids_to_update = results.get("ids", [])
            if ids_to_update:
                for chunk_id in ids_to_update:
                    self._collection.update(
                        ids=[chunk_id],
                        metadatas=[{"is_active": True}],
                    )
                logger.info("已激活 %d 个片段 (来源: %s)", len(ids_to_update), source)
            return len(ids_to_update)
        except Exception as e:
            logger.error("激活失败 (来源: %s): %s", source, e)
            return 0

    def get_source_versions(self, source: str) -> list[str]:
        """获取指定来源文件的所有版本号。

        Args:
            source: 文件路径标识

        Returns:
            去重后的版本号列表
        """
        self._ensure_collection()
        try:
            results = self._collection.get(
                where={"source": source},
            )
            metas = results.get("metadatas", [])
            versions: set[str] = set()
            for meta in metas:
                if meta and "version" in meta:
                    versions.add(str(meta["version"]))
            return sorted(versions, key=lambda v: int(v))
        except Exception as e:
            logger.warning("获取版本列表失败 (来源: %s): %s", source, e)
            return []

    def delete_all(self) -> int:
        """清空整个向量库。

        Returns:
            删除的文档数量
        """
        self._ensure_collection()

        try:
            count = self._collection.count()
            self._client.delete_collection(_COLLECTION_NAME)
            self._collection = None
            logger.info("已清空向量库，共删除 %d 个片段", count)
            return count
        except Exception as e:
            logger.error("清空向量库失败: %s", e)
            return 0

    def get_stats(self) -> dict[str, Any]:
        """获取向量库统计信息。

        Returns:
            包含文档总数、来源文件数等统计信息
        """
        self._ensure_collection()

        try:
            count = self._collection.count()
            all_data = self._collection.get()

            sources: set[str] = set()
            if all_data.get("metadatas"):
                for meta in all_data["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])

            return {
                "total_chunks": count,
                "total_sources": len(sources),
                "persist_dir": str(self._persist_dir),
                "embedding_model": EMBEDDING_MODEL,
            }
        except Exception as e:
            logger.error("获取统计信息失败: %s", e)
            return {
                "total_chunks": 0,
                "total_sources": 0,
                "persist_dir": str(self._persist_dir),
                "embedding_model": EMBEDDING_MODEL,
                "error": str(e),
            }

    # ==================== 内部实现 ====================

    def _init_client(self) -> None:
        """初始化 ChromaDB 持久化客户端。

        若 Ollama 不可用，chromadb 将使用默认（HuggingFace）Embedding 回退。
        """
        try:
            self._client = chromadb.PersistentClient(
                path=str(self._persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        except Exception as e:
            logger.error("无法连接 ChromaDB: %s", e)
            raise

    def _ensure_collection(self) -> None:
        """确保集合存在，首次调用时创建。

        使用自定义 Ollama Embedding 函数，若初始化失败则抛出异常。
        """
        if self._collection is not None:
            return

        embedding_fn = self._create_embedding_function()

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.debug("ChromaDB 集合已就绪: %s", _COLLECTION_NAME)

    def _create_embedding_function(self):
        """创建 Ollama Embedding 函数。

        ChromaDB v1.x 需要实现 __call__(self, input: list[str]) -> list[list[float]] 接口。

        Returns:
            可用的 Embedding 函数对象
        """
        # 优先使用 chromadb 内置的 Ollama 集成
        try:
            from chromadb.utils.embedding_functions import (
                OllamaEmbeddingFunction,
            )
            fn = OllamaEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                url=OLLAMA_HOST,
            )
            logger.info("使用 Ollama Embedding: %s @ %s", EMBEDDING_MODEL, OLLAMA_HOST)
            return fn
        except ImportError:
            logger.warning("chromadb 内置 Ollama EF 不可用，尝试自定义实现")

        # 回退：自定义 Ollama Embedding 调用
        import requests

        class _OllamaEmbedding:
            def __call__(self, input: list[str]) -> list[list[float]]:
                results: list[list[float]] = []
                for text in input:
                    try:
                        resp = requests.post(
                            f"{OLLAMA_HOST}/api/embeddings",
                            json={"model": EMBEDDING_MODEL, "prompt": text},
                            timeout=30,
                        )
                        resp.raise_for_status()
                        results.append(resp.json()["embedding"])
                    except Exception as e:
                        logger.error("Ollama Embedding 调用失败: %s", e)
                        raise
                return results

        logger.info("使用自定义 Ollama Embedding: %s @ %s", EMBEDDING_MODEL, OLLAMA_HOST)
        return _OllamaEmbedding()

    def _format_results(self, raw: dict) -> list[dict[str, Any]]:
        """将 ChromaDB 原生查询结果格式化为统一结构。

        Args:
            raw: chromadb Collection.query() 的返回值

        Returns:
            结构化结果列表，按相似度降序排列
        """
        results: list[dict[str, Any]] = []

        ids_list = raw.get("ids", [[]])
        docs_list = raw.get("documents", [[]])
        metas_list = raw.get("metadatas", [[]])
        dists_list = raw.get("distances", [[]])

        ids = ids_list[0] if ids_list else []
        docs = docs_list[0] if docs_list else []
        metas = metas_list[0] if metas_list else []
        dists = dists_list[0] if dists_list else []

        for idx, doc_id in enumerate(ids):
            item = {
                "id": doc_id,
                "document": docs[idx] if idx < len(docs) else "",
                "metadata": metas[idx] if idx < len(metas) else {},
                "score": max(0.0, min(1.0, round(1.0 - dists[idx], 4))) if idx < len(dists) else 0.0,
            }
            results.append(item)

        return results
