"""Noise filter – removes headers, footers, page numbers, TOC, and artifacts."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from medrag import config as cfg
from medrag.parsers.base import ParsedBlock

logger = logging.getLogger(__name__)


@dataclass
class FilterStats:
    """Track what the noise filter removed."""

    headers_removed: int = 0
    footers_removed: int = 0
    page_numbers_removed: int = 0
    toc_lines_removed: int = 0
    artifacts_removed: int = 0
    navigation_removed: int = 0
    total_blocks_in: int = 0
    total_blocks_out: int = 0


def _normalize_for_match(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-").replace("»", " ")
    text = re.sub(r"[^\w\s/-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class NoiseFilter:
    """Filter out repeated headers/footers, page numbers, TOC, and artifacts."""

    def __init__(self, page_heights: list[float] | None = None) -> None:
        self._cfg = cfg.get("cleaning", default={})
        self.stats = FilterStats()
        self.page_heights = page_heights or []
        self.margin_top = self._cfg.get("margin_top_px", 55)
        self.margin_bottom = self._cfg.get("margin_bottom_px", 35)
        self.repeat_threshold = self._cfg.get("header_repeat_threshold", 3)

        self._known_artifacts = set(self._cfg.get("known_artifacts", []))
        self._noise_patterns = [
            re.compile(p)
            for p in self._cfg.get("noise_patterns", [])
        ]

        # These get populated during detect_repeated_content
        self._repeated_headers: set[str] = set()
        self._repeated_footers: set[str] = set()

    def reset_stats(self) -> None:
        self.stats = FilterStats()

    def detect_repeated_content(self, blocks: list[ParsedBlock]) -> None:
        """Scan blocks to identify repeated header/footer text across pages."""
        top_counts: Counter[str] = Counter()
        bottom_counts: Counter[str] = Counter()

        for block in blocks:
            norm = _normalize_for_match(block.text)
            if not norm or len(norm.split()) > 10:
                continue

            page_height = self._get_page_height(block.page_index)

            if block.min_y < self.margin_top:
                top_counts[norm] += 1
            if block.max_y > page_height - self.margin_bottom:
                bottom_counts[norm] += 1

        self._repeated_headers = {
            text for text, count in top_counts.items()
            if count >= self.repeat_threshold
        }
        self._repeated_footers = {
            text for text, count in bottom_counts.items()
            if count >= self.repeat_threshold
        }

        logger.info(
            "Detected %d repeated headers, %d repeated footers",
            len(self._repeated_headers), len(self._repeated_footers),
        )

    def filter_blocks(self, blocks: list[ParsedBlock]) -> list[ParsedBlock]:
        """Remove noise blocks, returning only content blocks."""
        self.stats.total_blocks_in = len(blocks)
        self.detect_repeated_content(blocks)

        filtered: list[ParsedBlock] = []
        for block in blocks:
            reason = self._should_remove(block)
            if reason:
                logger.debug("Removing block [%s]: %s", reason, block.text[:60])
                continue
            filtered.append(block)

        self.stats.total_blocks_out = len(filtered)
        logger.info(
            "Noise filter: %d -> %d blocks (%d removed)",
            self.stats.total_blocks_in,
            self.stats.total_blocks_out,
            self.stats.total_blocks_in - self.stats.total_blocks_out,
        )
        return filtered

    def _should_remove(self, block: ParsedBlock) -> str | None:
        """Return the removal reason, or None to keep."""
        text = block.text.strip()
        norm = _normalize_for_match(text)

        # 1. Repeated header
        if norm in self._repeated_headers:
            self.stats.headers_removed += 1
            return "repeated_header"

        # 2. Repeated footer
        if norm in self._repeated_footers:
            self.stats.footers_removed += 1
            return "repeated_footer"

        # 3. Bare page number
        if re.fullmatch(r"\d{1,4}", text):
            self.stats.page_numbers_removed += 1
            return "page_number"

        # 4. Bottom-of-page content (likely footer)
        page_height = self._get_page_height(block.page_index)
        if block.max_y > page_height - 28 and len(text.split()) <= 8:
            self.stats.footers_removed += 1
            return "bottom_footer"

        # 5. Known artifacts
        if text in self._known_artifacts:
            self.stats.artifacts_removed += 1
            return "known_artifact"

        # 6. Symbol-only noise
        if re.fullmatch(r"[®Ü•*=\-_/|]+", text):
            self.stats.artifacts_removed += 1
            return "symbol_noise"

        # 7. Noise patterns from config
        for pattern in self._noise_patterns:
            if pattern.fullmatch(text):
                self.stats.artifacts_removed += 1
                return "noise_pattern"

        # 8. Navigation / boilerplate
        if self._is_navigation(text):
            self.stats.navigation_removed += 1
            return "navigation"

        # 9. TOC lines
        if self._is_toc_line(text):
            self.stats.toc_lines_removed += 1
            return "toc_line"

        # 10. Parser-identified headers/footers
        if block.doc_item_type in ("header", "footer"):
            self.stats.headers_removed += 1
            return "parser_header_footer"

        return None

    def _get_page_height(self, page_index: int) -> float:
        if self.page_heights and page_index < len(self.page_heights):
            return self.page_heights[page_index]
        return 792.0  # default US Letter height

    @staticmethod
    def _is_navigation(text: str) -> bool:
        lowered = text.lower()
        nav_phrases = [
            "available online", "connect with us", "nccn.org",
            "patientguidelines", "please take a moment",
            "find an nccn cancer", "share with us",
        ]
        return any(phrase in lowered for phrase in nav_phrases)

    @staticmethod
    def _is_toc_line(text: str) -> bool:
        stripped = re.sub(r"\s+", " ", text).strip()
        # Pattern: "Chapter Title ... 42"
        if re.match(
            r"^[A-Z][A-Za-z'()/,& -]+\s+\d{1,3}(?:[-–]\d{1,3})?$",
            stripped,
        ):
            return True
        # Pattern: "42 Chapter Title"
        if re.match(r"^\d+\s+[A-Z].*", stripped) and len(stripped.split()) <= 6:
            return True
        return False
