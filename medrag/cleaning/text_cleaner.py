"""Text cleaning: OCR repair, unicode normalization, line merging, bullet preservation."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from medrag import config as cfg

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Tracks what the cleaner did for reporting."""

    ocr_merges_fixed: int = 0
    unicode_normalized: int = 0
    lines_merged: int = 0
    whitespace_cleaned: int = 0
    artifacts_removed: int = 0


# Patterns for OCR merge detection:
# "treatmentYour" → "treatment Your"
_OCR_MERGE_PATTERN = re.compile(
    r"([a-z])([A-Z][a-z])"
)

# Additional OCR merge patterns
_OCR_MERGE_PATTERNS = [
    # lowercase followed by uppercase word
    (re.compile(r"([a-z])([A-Z][a-z]{2,})"), r"\1 \2"),
    # missing space after period
    (re.compile(r"(\.)([A-Z][a-z])"), r"\1 \2"),
    # missing space after comma
    (re.compile(r"(\,)([A-Z])"), r"\1 \2"),
    # parenthesis merge: "cancer)Treatment" -> "cancer) Treatment"
    (re.compile(r"(\))([A-Z][a-z])"), r"\1 \2"),
]

# Unicode replacement map
_UNICODE_REPLACEMENTS = {
    "\u00ad": "",       # soft hyphen
    "\u2010": "-",      # hyphen
    "\u2011": "-",      # non-breaking hyphen
    "\u2012": "-",      # figure dash
    "\u2013": "–",      # en dash
    "\u2014": "—",      # em dash
    "\u2018": "'",      # left single quote
    "\u2019": "'",      # right single quote
    "\u201c": '"',      # left double quote
    "\u201d": '"',      # right double quote
    "\u2022": "•",      # bullet
    "\u2026": "...",    # ellipsis
    "\u00a0": " ",      # non-breaking space
    "\ufeff": "",       # BOM
    "\u200b": "",       # zero-width space
    "\u200c": "",       # zero-width non-joiner
    "\u200d": "",       # zero-width joiner
    "\uf0b7": "•",      # Wingdings bullet
    "\uf0a7": "•",      # Wingdings bullet variant
}

# Bullet patterns to preserve
_BULLET_PATTERN = re.compile(
    r"^(\s*)([-*•●◦▪▸►‣⁃]|\d+[.)]\s|[a-zA-Z][.)]\s|h\s)"
)


class TextCleaner:
    """Production text cleaner for medical PDF content."""

    def __init__(self) -> None:
        self.stats = CleaningStats()
        self._cleaning_cfg = cfg.get("cleaning", default={})
        self._known_artifacts = set(self._cleaning_cfg.get("known_artifacts", []))
        self._noise_patterns = [
            re.compile(p)
            for p in self._cleaning_cfg.get("noise_patterns", [])
        ]

    def reset_stats(self) -> None:
        self.stats = CleaningStats()

    # ── Public API ──────────────────────────────────────────

    def clean_text(self, text: str) -> str:
        """Full cleaning pipeline for a block of text."""
        if not text or not text.strip():
            return ""

        text = self._normalize_unicode(text)
        text = self._fix_ocr_merges(text)
        text = self._normalize_whitespace(text)
        text = self._remove_isolated_artifacts(text)

        return text.strip()

    def clean_paragraph(self, paragraph: str) -> str:
        """Clean a single paragraph, preserving bullet structure."""
        if not paragraph.strip():
            return ""

        lines = paragraph.split("\n")
        cleaned_lines: list[str] = []

        for line in lines:
            cleaned = self.clean_text(line)
            if cleaned:
                cleaned_lines.append(cleaned)

        if not cleaned_lines:
            return ""

        # Merge wrapped lines (but preserve bullets)
        merged = self._merge_wrapped_lines(cleaned_lines)
        return "\n".join(merged)

    def merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Clean and merge a list of paragraphs."""
        result: list[str] = []
        for para in paragraphs:
            cleaned = self.clean_paragraph(para)
            if cleaned:
                result.append(cleaned)
        return result

    def is_artifact(self, text: str) -> bool:
        """Check if text is a known artifact to remove."""
        stripped = text.strip()
        if stripped in self._known_artifacts:
            return True
        for pattern in self._noise_patterns:
            if pattern.fullmatch(stripped):
                return True
        return False

    # ── Internal methods ────────────────────────────────────

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters to ASCII-compatible equivalents."""
        original = text
        for char, replacement in _UNICODE_REPLACEMENTS.items():
            text = text.replace(char, replacement)

        # NFKC normalization for remaining characters
        text = unicodedata.normalize("NFKC", text)

        if text != original:
            self.stats.unicode_normalized += 1

        return text

    def _fix_ocr_merges(self, text: str) -> str:
        """Fix OCR merge issues like 'treatmentYour' -> 'treatment Your'."""
        if not self._cleaning_cfg.get("fix_ocr_merges", True):
            return text

        original = text
        for pattern, replacement in _OCR_MERGE_PATTERNS:
            text = pattern.sub(replacement, text)

        if text != original:
            self.stats.ocr_merges_fixed += 1

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs, trim."""
        original = text
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" \.", ".", text)
        text = text.strip()

        if text != original.strip():
            self.stats.whitespace_cleaned += 1

        return text

    def _remove_isolated_artifacts(self, text: str) -> str:
        """Remove isolated symbols and known artifacts."""
        stripped = text.strip()
        if self.is_artifact(stripped):
            self.stats.artifacts_removed += 1
            return ""
        return text

    def _merge_wrapped_lines(self, lines: list[str]) -> list[str]:
        """Merge lines that were wrapped mid-sentence, preserving bullets."""
        if not lines:
            return []

        merged: list[str] = [lines[0]]

        for line in lines[1:]:
            # Preserve bullet lines as separate
            if _BULLET_PATTERN.match(line):
                merged.append(line)
                continue

            # Preserve headings (lines starting with #)
            if line.startswith("#"):
                merged.append(line)
                continue

            prev = merged[-1]

            # If previous line ends mid-word/sentence, merge
            if (
                re.search(r"[a-z,;:\-]$", prev)
                and re.match(r"^[a-z(]", line)
            ):
                merged[-1] = f"{prev} {line}"
                self.stats.lines_merged += 1
                continue

            # If previous ends without terminal punctuation and next is lowercase
            if (
                not prev.endswith((".", "!", "?", ":", '"', "'"))
                and re.match(r"^[a-z]", line)
            ):
                merged[-1] = f"{prev} {line}"
                self.stats.lines_merged += 1
                continue

            merged.append(line)

        return merged
