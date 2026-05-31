"""Parser sub-package – Docling-first with PyMuPDF fallback."""

from medrag.parsers.base import ParsedDocument, ParsedBlock, ParserBase
from medrag.parsers.parser_factory import parse_pdf

__all__ = ["ParsedDocument", "ParsedBlock", "ParserBase", "parse_pdf"]
