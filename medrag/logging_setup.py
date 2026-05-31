"""Centralized logging configuration for the MedRAG pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from medrag import config as cfg


_CONFIGURED = False


def setup_logging(level: str | None = None, log_to_file: bool | None = None) -> None:
    """Configure root logger for the entire pipeline.

    Safe to call multiple times – subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    log_cfg = cfg.get("logging", default={})
    effective_level = (level or log_cfg.get("level", "INFO")).upper()
    effective_file = log_to_file if log_to_file is not None else log_cfg.get("log_to_file", True)

    root = logging.getLogger()
    root.setLevel(effective_level)

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(effective_level)
    console_fmt = log_cfg.get("console_format", "%(levelname)-8s | %(message)s")
    console.setFormatter(logging.Formatter(console_fmt))
    root.addHandler(console)

    # File handler
    if effective_file:
        log_path = cfg.log_dir() / "medrag.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(effective_level)
        file_fmt = log_cfg.get(
            "log_format",
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        fh.setFormatter(logging.Formatter(file_fmt))
        root.addHandler(fh)

    logging.getLogger("medrag").info(
        "Logging initialized – level=%s, file=%s", effective_level, effective_file
    )
