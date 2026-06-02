"""PyMuPDF (fitz) based PDF parser – robust fallback parser.

Extracts text blocks with full font metadata, bounding boxes, and
per-span typography information. Used as fallback when Docling fails.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from statistics import median
from typing import Any

import fitz

from medrag.parsers.base import ParsedBlock, ParsedDocument, ParserBase

logger = logging.getLogger(__name__)


def _normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


class PyMuPDFParser(ParserBase):
    """Parse PDFs using PyMuPDF (fitz)."""

    name = "pymupdf"

    def parse(self, pdf_path: Path) -> ParsedDocument:
        logger.info("Parsing %s with PyMuPDF ...", pdf_path.name)
        doc = fitz.open(pdf_path)

        blocks: list[ParsedBlock] = []
        page_heights: list[float] = []

        for page_index, page in enumerate(doc):
            page_heights.append(page.rect.height)
            page_dict = page.get_text("dict")

            for block_idx, raw_block in enumerate(page_dict.get("blocks", [])):
                if raw_block.get("type") != 0:
                    continue

                spans: list[dict[str, Any]] = []
                texts: list[str] = []

                for line in raw_block.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if span_text:
                            spans.append(span)
                            texts.append(span_text)

                if not texts:
                    continue

                text = _normalize_ws(" ".join(texts))
                if not text:
                    continue

                max_font = max(float(s["size"]) for s in spans)
                font_names = tuple(sorted({str(s["font"]) for s in spans}))
                is_bold = any("bold" in str(s.get("font", "")).lower() for s in spans)

                blocks.append(
                    ParsedBlock(
                        text=text,
                        page_index=page_index,
                        block_index=block_idx,
                        min_x=float(raw_block["bbox"][0]),
                        min_y=float(raw_block["bbox"][1]),
                        max_x=float(raw_block["bbox"][2]),
                        max_y=float(raw_block["bbox"][3]),
                        max_font_size=max_font,
                        font_names=font_names,
                        is_bold=is_bold,
                        doc_item_type="text",
                        heading_level=0,
                    )
                )

        doc.close()

        # Post-processing: infer heading levels from font size
        if blocks:
            body_fonts = [b.max_font_size for b in blocks if b.word_count > 8]
            body_font = median(body_fonts) if body_fonts else 11.0

            for block in blocks:
                if self._is_heading_candidate(block, body_font):
                    block.doc_item_type = "heading"
                    block.heading_level = self._infer_heading_level(block, body_font)

        logger.info(
            "PyMuPDF extracted %d blocks from %d pages in %s",
            len(blocks), len(page_heights), pdf_path.name,
        )

        return ParsedDocument(
            source_path=pdf_path,
            source_file=pdf_path.name,
            blocks=blocks,
            page_count=len(page_heights),
            page_heights=page_heights,
            parser_used="pymupdf",
        )

    @staticmethod
    def _is_heading_candidate(block: ParsedBlock, body_font: float) -> bool:
        """Heuristic heading detection based on font size and text properties."""
        text = block.text.strip()
        words = text.split()
        word_count = len(words)

        if word_count == 0 or word_count > 14:
            return False
        if text.endswith((".", "!", "?")):
            return False
        if "»" in text and block.min_y < 80:
            return False
        if re.fullmatch(r"\d+", text):
            return False
        if re.fullmatch(r'[""\"\'`\-]+', text):
            return False

        # Font size must exceed body by threshold, OR block must be bold
        font_delta = block.max_font_size - body_font
        if font_delta < 1.5 and not block.is_bold:
            return False

        return True

    @staticmethod
    def _infer_heading_level(block: ParsedBlock, body_font: float) -> int:
        """Map font size delta to heading levels 1-3."""
        delta = block.max_font_size - body_font
        if delta >= 10:
            return 1
        if delta >= 5:
            return 2
        return 3
