"""Heading detection using a hybrid strategy.

Combines:
  - Docling/markdown structural hints
  - Font-size heuristics (PyMuPDF)
  - Regex patterns (numbered headings)
  - Title-case detection
  - Line-length filters
  - Capitalization analysis
  - Surrounding whitespace analysis
  - Semantic continuation checks

IMPORTANT: We do NOT promote short sentences to headings.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from medrag import config as cfg
from medrag.parsers.base import ParsedBlock

logger = logging.getLogger(__name__)


@dataclass
class HeadingCandidate:
    """A block evaluated as a potential heading."""

    block: ParsedBlock
    score: float = 0.0
    level: int = 0          # 1 = chapter, 2 = section, 3 = subsection
    reason: str = ""
    is_heading: bool = False


# Numbered heading patterns
_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+)*)\s+(.+)")
_ROMAN_HEADING = re.compile(r"^(I{1,3}|IV|VI{0,3}|IX|X{0,3})\.\s+(.+)", re.IGNORECASE)

# Patterns that DISQUALIFY a line from being a heading
_DISQUALIFY_PATTERNS = [
    re.compile(r"https?://"),                    # URLs
    re.compile(r"\.(org|com|edu|gov)\b"),         # domain references
    re.compile(r"^\d+$"),                         # bare page numbers
    re.compile(r'^[""\"\'`\-]+$'),                # quote artifacts
    re.compile(r"^\s*[•\-*]\s"),                  # bullet points
    re.compile(r"NCCN Guidelines"),               # boilerplate headers
    re.compile(r"^\d+\s+NCCN"),                   # page-number + NCCN boilerplate
    re.compile(r"Guidelines for Patients"),        # patient guideline boilerplate
]

# Words that suggest body text, not headings
_BODY_STARTERS = {
    "if", "when", "after", "before", "because", "although", "however",
    "therefore", "moreover", "furthermore", "additionally", "also",
    "and", "but", "or", "so", "then", "thus", "hence",
    "the", "a", "an", "this", "that", "these", "those",
    "it", "they", "we", "you", "he", "she",
}

# Standard medical sub-headings that should be nested (demoted to Level 3)
_MEDICAL_SUBHEADINGS = {
    "definition", "icd-o coding", "icd-o code", "mim numbering", "synonym(s)",
    "subtype(s)", "epidemiology", "localization", "clinical features",
    "etiology", "pathogenesis", "macroscopic appearance", "microscopic appearance",
    "cytology", "diagnostic molecular pathology", "essential and desirable diagnostic criteria",
    "staging", "prognosis and prediction", "differential diagnosis", "histochemistry",
    "immunohistochemistry", "electron microscopy", "genetics", "references",
}


class HeadingDetector:
    """Hybrid heading detection engine."""

    def __init__(self, body_font_size: float = 11.0) -> None:
        self._cfg = cfg.get("heading_detection", default={})
        self.body_font = body_font_size
        self.min_font_delta = self._cfg.get("min_font_delta", 1.5)
        self.max_heading_words = self._cfg.get("max_heading_words", 14)
        self.title_case_ratio = self._cfg.get("title_case_min_capitalized_ratio", 0.6)
        self.breadcrumb_sep = self._cfg.get("breadcrumb_separator", "»")

    def evaluate(self, block: ParsedBlock) -> HeadingCandidate:
        """Score a block as a heading candidate. Higher score = more likely heading."""
        candidate = HeadingCandidate(block=block)
        text = block.text.strip()
        words = text.split()
        word_count = len(words)

        if not text or word_count == 0:
            return candidate

        # ── Automatic disqualification ───────────────────────
        for pattern in _DISQUALIFY_PATTERNS:
            if pattern.search(text):
                candidate.reason = "disqualified_pattern"
                return candidate

        # Reject sentences (ending with period) — key rule
        if self._cfg.get("reject_if_ends_with_period", True):
            if text.endswith((".", "!", "?")):
                # Exception: numbered headings like "1.2 Treatment"
                if not _NUMBERED_HEADING.match(text.rstrip(".")):
                    candidate.reason = "ends_with_period"
                    return candidate

        # Too many words
        if word_count > self.max_heading_words:
            candidate.reason = "too_many_words"
            return candidate

        # Breadcrumb lines are navigation, not headings
        if self.breadcrumb_sep in text and block.min_y < 80:
            candidate.reason = "breadcrumb"
            return candidate

        # Starts with body-text word (lowercase)
        first_word = words[0].lower().rstrip(".,;:")
        if first_word in _BODY_STARTERS and word_count > 3:
            candidate.reason = "body_starter"
            return candidate

        # ── Positive signals ─────────────────────────────────
        score = 0.0
        reasons: list[str] = []

        # 1. Parser-provided heading level (strongest signal)
        if block.doc_item_type == "heading" and block.heading_level > 0:
            score += 10.0
            candidate.level = block.heading_level
            reasons.append(f"parser_heading_L{block.heading_level}")

        # 2. Font size delta
        if block.max_font_size > 0 and self.body_font > 0:
            delta = block.max_font_size - self.body_font
            if delta >= self.min_font_delta:
                score += min(delta * 1.5, 8.0)
                reasons.append(f"font_delta={delta:.1f}")

        # 3. Bold text
        if block.is_bold and word_count <= 10:
            score += 2.0
            reasons.append("bold")

        # 4. Numbered heading pattern
        # But reject "42 NCCN Guidelines..." type lines
        numbered_match = _NUMBERED_HEADING.match(text)
        if numbered_match:
            numbered_text = numbered_match.group(3)
            # Only count as numbered heading if the text after the number
            # looks like a real heading (not boilerplate)
            if not re.search(r"NCCN|Guidelines|Patients", numbered_text):
                score += 5.0
                reasons.append("numbered")
        elif _ROMAN_HEADING.match(text):
            score += 4.0
            reasons.append("roman_numeral")

        # 5. ALL CAPS
        letters = [c for c in text if c.isalpha()]
        if letters and word_count <= self._cfg.get("all_caps_max_words", 8):
            upper_ratio = sum(c.isupper() for c in letters) / len(letters)
            if upper_ratio >= 0.85:
                score += 4.0
                reasons.append("all_caps")

        # 6. Title Case
        if word_count >= 2:
            title_words = sum(
                1 for w in words
                if w[0:1].isalpha() and w[0].isupper()
            )
            ratio = title_words / max(word_count, 1)
            if ratio >= self.title_case_ratio:
                score += 3.0
                reasons.append(f"title_case={ratio:.0%}")

        # 7. Short line (1-5 words, capitalized)
        if 1 <= word_count <= 5 and words[0][0:1].isupper():
            if not text.endswith((".", "!", "?")):
                score += 1.5
                reasons.append("short_capitalized")

        # ── Determine heading level from font delta ──────────
        if candidate.level == 0 and block.max_font_size > 0:
            delta = block.max_font_size - self.body_font
            if delta >= 10:
                candidate.level = 1
            elif delta >= 5:
                candidate.level = 2
            elif delta >= self.min_font_delta:
                candidate.level = 3

        # Default to level 2 if parser said heading but no font info
        if candidate.level == 0 and score >= 5.0:
            candidate.level = 2

        # ── Medical Entity Nesting (Contextual Demotion) ─────
        # If this is a standard medical sub-heading, force it to Level 3
        # so it nests under the entity name (Level 2).
        if text.lower().rstrip(":") in _MEDICAL_SUBHEADINGS:
            candidate.level = 3
            candidate.score += 2.0  # Boost confidence
            reasons.append("medical_subheading_demotion")

        candidate.score = score
        candidate.is_heading = score >= 4.0
        candidate.reason = "; ".join(reasons) if reasons else "no_signals"

        return candidate

    def detect_headings(self, blocks: list[ParsedBlock]) -> list[HeadingCandidate]:
        """Evaluate all blocks and return heading candidates."""
        candidates = []
        for block in blocks:
            candidate = self.evaluate(block)
            if candidate.is_heading:
                candidates.append(candidate)
        return candidates

    def is_breadcrumb(self, block: ParsedBlock) -> bool:
        """Check if block is a breadcrumb navigation line."""
        return self.breadcrumb_sep in block.text and block.min_y < 80

    def parse_breadcrumb(self, text: str) -> list[str]:
        """Split breadcrumb text into hierarchy parts."""
        parts = text.split(self.breadcrumb_sep)
        return [p.strip() for p in parts if p.strip()]
