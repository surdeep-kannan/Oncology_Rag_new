"""Parser factory – tries Docling first, falls back to PyMuPDF."""

from __future__ import annotations

import logging
from pathlib import Path

from medrag import config as cfg
from medrag.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)


def parse_pdf(pdf_path: Path, config_path: Path | str | None = None) -> ParsedDocument:
    """Parse a single PDF, respecting the configured parser priority.

    1. Try the *primary_parser* (default: docling).
    2. On failure, try the *fallback_parser* (default: pymupdf).
    3. If both fail, raise the last exception.
    """
    if config_path:
        cfg.load_config(config_path)

    primary = cfg.get("parsing", "primary_parser", "docling")
    fallback = cfg.get("parsing", "fallback_parser", "pymupdf")

    parser_order = [primary, fallback]
    last_error: Exception | None = None

    for parser_name in parser_order:
        try:
            parser = _get_parser(parser_name)
            doc = parser.parse(pdf_path)
            if doc.blocks:
                return doc
            logger.warning(
                "%s returned 0 blocks for %s – trying next parser.",
                parser_name, pdf_path.name,
            )
        except Exception as exc:
            logger.warning(
                "%s failed for %s: %s – trying next parser.",
                parser_name, pdf_path.name, exc,
            )
            last_error = exc

    if last_error:
        raise last_error
    raise RuntimeError(f"All parsers returned empty results for {pdf_path}")


def _get_parser(name: str):
    """Lazy-import the requested parser to avoid hard dependencies."""
    name = name.lower().strip()
    if name == "docling":
        from medrag.parsers.docling_parser import DoclingParser
        return DoclingParser()
    if name in ("pymupdf", "fitz"):
        from medrag.parsers.pymupdf_parser import PyMuPDFParser
        return PyMuPDFParser()
    raise ValueError(f"Unknown parser: {name!r}. Choose 'docling' or 'pymupdf'.")
