"""
版本管理模块

管理文档索引的版本历史，支持：
1. 版本记录存储（doc_id → [v1, v2, ...]）
2. 自动版本号递增
3. 变更日志（新旧 chunk 数对比、时间戳）
4. 版本列表查询
5. 回滚指针（当前激活版本）
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_rag.config import DATA_DIR
from local_rag.utils.logger import get_logger

logger = get_logger(__name__)

VERSION_DIR = DATA_DIR / "versions"


class VersionManager:
    """文档索引版本管理器。

    每个文件对应一个版本记录文件 {doc_id}.json，
    记录所有版本的元信息和变更日志。

    Usage:
        vm = VersionManager()
        vm.record(doc_id, "文档名.docx", chunk_count=12, metadata={})
        versions = vm.list_versions(doc_id)
        vm.rollback_to(doc_id, 1)
    """

    def __init__(self) -> None:
        VERSION_DIR.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        doc_id: str,
        file_name: str,
        chunk_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """记录一个新版本，返回版本号。

        Args:
            doc_id: 文档内容 MD5
            file_name: 文件名
            chunk_count: 本次索引的 chunk 数量
            metadata: 文档基础元数据，存入版本记录

        Returns:
            新版本号（从 1 开始递增）
        """
        log = self._load_log(doc_id)
        prev_versions = log.get("versions", [])

        new_version = len(prev_versions) + 1

        prev_count = 0
        if prev_versions:
            prev_count = prev_versions[-1].get("chunk_count", 0)

        entry = {
            "version": new_version,
            "file_name": file_name,
            "chunk_count": chunk_count,
            "previous_chunk_count": prev_count,
            "delta": chunk_count - prev_count,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        prev_versions.append(entry)

        log["doc_id"] = doc_id
        log["versions"] = prev_versions
        log["active_version"] = new_version
        log["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

        self._save_log(doc_id, log)

        logger.info(
            "版本记录: %s v%d (%d chunks, Δ=%+d)",
            file_name,
            new_version,
            chunk_count,
            chunk_count - prev_count,
        )
        return new_version

    def list_versions(self, doc_id: str) -> list[dict]:
        """列出指定文档的所有版本。

        Args:
            doc_id: 文档内容 MD5

        Returns:
            版本列表，最新在前
        """
        log = self._load_log(doc_id)
        versions = list(log.get("versions", []))
        versions.reverse()
        return versions

    def get_active_version(self, doc_id: str) -> int:
        """获取当前激活的版本号。

        Args:
            doc_id: 文档内容 MD5

        Returns:
            当前激活版本号，无记录时返回 0
        """
        log = self._load_log(doc_id)
        return log.get("active_version", 0)

    def rollback_to(self, doc_id: str, target_version: int) -> bool:
        """回滚到指定版本。

        Args:
            doc_id: 文档内容 MD5
            target_version: 目标版本号

        Returns:
            True 表示成功
        """
        log = self._load_log(doc_id)
        versions = log.get("versions", [])

        if target_version < 1 or target_version > len(versions):
            logger.warning("无效的版本号: %d (共 %d 个版本)", target_version, len(versions))
            return False

        log["active_version"] = target_version
        log["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._save_log(doc_id, log)

        logger.info(
            "回滚 %s → v%d",
            versions[target_version - 1].get("file_name", doc_id),
            target_version,
        )
        return True

    def get_version_dir(self) -> Path:
        """获取版本记录目录路径。"""
        return VERSION_DIR

    def get_all_docs(self) -> list[str]:
        """获取所有有版本记录的 doc_id 列表。"""
        ids: list[str] = []
        if VERSION_DIR.exists():
            for f in VERSION_DIR.glob("*.json"):
                ids.append(f.stem)
        return ids

    # ==== 内部 ====

    def _log_path(self, doc_id: str) -> Path:
        safe = doc_id.replace("/", "_").replace("\\", "_")
        return VERSION_DIR / f"{safe}.json"

    def _load_log(self, doc_id: str) -> dict:
        path = self._log_path(doc_id)
        if not path.exists():
            return {"doc_id": doc_id, "versions": [], "active_version": 0}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("版本记录读取失败: %s", e)
            return {"doc_id": doc_id, "versions": [], "active_version": 0}

    def _save_log(self, doc_id: str, log: dict) -> None:
        path = self._log_path(doc_id)
        path.write_text(
            json.dumps(log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
