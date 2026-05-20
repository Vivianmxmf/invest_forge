"""Centralised logging configuration for InvestForge.

Importing this module once at process start (via ``get_logger``) configures
the root logger.  When running inside an existing orchestration (Airflow,
LangSmith, jupyter), we *augment* rather than reset the root handlers.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

LOG_FMT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FMT: Final = "%Y-%m-%d %H:%M:%S"


def _running_under_orchestrator() -> bool:
    return any(
        env in os.environ
        for env in ("AIRFLOW_HOME", "AIRFLOW__CORE__EXECUTOR", "LANGCHAIN_TRACING_V2")
    )


def configure_logging(
    level: str | None = None,
    log_file: str | os.PathLike[str] | None = None,
) -> None:
    """Idempotently configure the root logger."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if getattr(root, "_invest_forge_configured", False):
        return

    root.setLevel(log_level)
    formatter = logging.Formatter(LOG_FMT, datefmt=DATE_FMT)

    if not _running_under_orchestrator():
        root.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Quiet some noisy third-party loggers
    for noisy in ("urllib3", "httpx", "matplotlib.font_manager", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._invest_forge_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    if not getattr(logging.getLogger(), "_invest_forge_configured", False):
        configure_logging()
    return logging.getLogger(name)
