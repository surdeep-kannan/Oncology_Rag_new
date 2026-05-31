"""Docling-based PDF parser – primary parser for the pipeline.

Docling provides excellent structural parsing with heading detection,
table extraction, and markdown-level hierarchy for medical documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from medrag.parsers.base import ParsedBlock, ParsedDocument, ParserBase

logger = logging.getLogger(__name__)


class DoclingParser(ParserBase):
    """Parse PDFs using the Docling library."""

    name = "docling"

    def parse(self, pdf_path: Path) -> ParsedDocument:
        logger.info("Parsing %s with Docling ...", pdf_path.name)

        try:
            from docling.document_converter import DocumentConverter
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling_core.types.doc import DocItemLabel
        except ImportError as exc:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling docling-core"
            ) from exc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
        )

        result = converter.convert(pdf_path)
        doc = result.document

        blocks: list[ParsedBlock] = []
        block_index = 0

        # Map DocItemLabel to our types
        _LABEL_MAP = {
            DocItemLabel.TITLE: "heading",
            DocItemLabel.SECTION_HEADER: "heading",
            DocItemLabel.PAGE_HEADER: "header",
            DocItemLabel.PAGE_FOOTER: "footer",
            DocItemLabel.LIST_ITEM: "list-item",
            DocItemLabel.TABLE: "table",
            DocItemLabel.CAPTION: "caption",
            DocItemLabel.PICTURE: "image",
            DocItemLabel.FOOTNOTE: "footnote",
            DocItemLabel.FORMULA: "formula",
        }

        _HEADING_LEVELS = {
            DocItemLabel.TITLE: 1,
            DocItemLabel.SECTION_HEADER: 2,
        }

        for item, _level in doc.iterate_items():
            text = item.text if hasattr(item, "text") else ""
            if not text or not text.strip():
                continue

            label = item.label if hasattr(item, "label") else None
            doc_type = _LABEL_MAP.get(label, "text")
            heading_level = _HEADING_LEVELS.get(label, 0)

            # Extract page info from provenance
            page_index = 0
            min_x, min_y, max_x, max_y = 0.0, 0.0, 0.0, 0.0

            if hasattr(item, "prov") and item.prov:
                prov = item.prov[0]
                page_index = prov.page_no - 1 if hasattr(prov, "page_no") else 0
                if hasattr(prov, "bbox"):
                    bbox = prov.bbox
                    if hasattr(bbox, "l"):
                        min_x, min_y = bbox.l, bbox.t
                        max_x, max_y = bbox.r, bbox.b
                    elif hasattr(bbox, "x0"):
                        min_x, min_y = bbox.x0, bbox.y0
                        max_x, max_y = bbox.x1, bbox.y1

            blocks.append(
                ParsedBlock(
                    text=text.strip(),
                    page_index=page_index,
                    block_index=block_index,
                    min_x=min_x,
                    min_y=min_y,
                    max_x=max_x,
                    max_y=max_y,
                    doc_item_type=doc_type,
                    heading_level=heading_level,
                )
            )
            block_index += 1

        # Try to get page count
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)
        elif blocks:
            page_count = max(b.page_index for b in blocks) + 1

        logger.info(
            "Docling extracted %d blocks from %d pages in %s",
            len(blocks), page_count, pdf_path.name,
        )

        return ParsedDocument(
            source_path=pdf_path,
            source_file=pdf_path.name,
            blocks=blocks,
            page_count=page_count,
            page_heights=[],
            parser_used="docling",
        )
