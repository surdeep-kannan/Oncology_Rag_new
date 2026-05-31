"""Base types shared by all PDF parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedBlock:
    """One logical block of text extracted from a PDF page."""

    text: str
    page_index: int
    block_index: int = 0

    # Bounding box (PDF coordinates)
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0

    # Typography
    max_font_size: float = 0.0
    font_names: tuple[str, ...] = ()
    is_bold: bool = False

    # Structural hints from the parser
    doc_item_type: str = "text"  # text | heading | list-item | table | caption
    heading_level: int = 0       # 0 = not a heading, 1–6 = markdown heading level

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ParsedDocument:
    """A fully parsed PDF, ready for cleaning and chunking."""

    source_path: Path
    source_file: str
    blocks: list[ParsedBlock] = field(default_factory=list)
    page_count: int = 0
    page_heights: list[float] = field(default_factory=list)
    parser_used: str = "unknown"
    metadata: dict = field(default_factory=dict)


class ParserBase:
    """Interface that every parser must implement."""

    name: str = "base"

    def parse(self, pdf_path: Path) -> ParsedDocument:
        raise NotImplementedError
