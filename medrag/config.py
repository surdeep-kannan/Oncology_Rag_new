"""Configuration loader for the MedRAG pipeline.

Reads ``config.yaml`` and exposes typed helpers so every module can pull
settings without parsing YAML itself.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"

_CONFIG: dict[str, Any] | None = None


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load and cache the YAML configuration."""
    global _CONFIG
    if _CONFIG is not None and config_path is None:
        return _CONFIG

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning("Config file not found at %s – using defaults.", path)
        _CONFIG = {}
        return _CONFIG

    with open(path, "r", encoding="utf-8") as fh:
        _CONFIG = yaml.safe_load(fh) or {}
    logger.info("Loaded config from %s", path)
    return _CONFIG


def get(section: str, key: str | None = None, default: Any = None) -> Any:
    """Retrieve a config value by *section* and optional *key*."""
    cfg = load_config()
    section_data = cfg.get(section, {})
    if key is None:
        return section_data if section_data else default
    return section_data.get(key, default)


def project_root() -> Path:
    return _PROJECT_ROOT


def input_dir() -> Path:
    p = project_root() / get("paths", "input_dir", "input")
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir() -> Path:
    p = project_root() / get("paths", "output_dir", "output")
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = project_root() / get("paths", "log_dir", "logs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def vector_store_dir() -> Path:
    p = project_root() / get("paths", "vector_store_dir", "output/vector_store")
    p.mkdir(parents=True, exist_ok=True)
    return p


def index_dir() -> Path:
    p = project_root() / get("paths", "index_dir", "output/indices")
    p.mkdir(parents=True, exist_ok=True)
    return p
