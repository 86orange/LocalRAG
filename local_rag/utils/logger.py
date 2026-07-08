"""
日志配置模块

提供统一的日志格式和一个 get_logger 工厂函数，
各模块通过 `get_logger(__name__)` 获取专属 logger。
日志同时输出到控制台和 logs/ 目录下的文件。
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from local_rag.config import LOG_DIR

_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_log_initialized = False


def _init() -> None:
    """初始化根 logger，仅执行一次。"""
    global _log_initialized
    if _log_initialized:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # 控制台输出（INFO 级别）
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # 文件输出（DEBUG 级别，单文件最大 5MB，保留 3 个备份）
    file_handler = RotatingFileHandler(
        filename=LOG_DIR / "local_rag.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _log_initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger，首次调用自动初始化日志系统。

    Args:
        name: 模块名，通常传入 __name__

    Returns:
        配置完成的 Logger 实例
    """
    _init()
    return logging.getLogger(name)
