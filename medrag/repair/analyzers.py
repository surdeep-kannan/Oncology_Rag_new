"""Analyzers for identifying headers, footers, and broken headings."""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

from rapidfuzz import fuzz
from medrag import config as cfg

logger = logging.getLogger(__name__)

class HeaderFooterLearner:
    """Automatically learns recurring headers and footers across pages."""

    def __init__(self):
        self._cfg = cfg.get("repair", default={})
        self.freq_threshold = self._cfg.get("min_pattern_frequency", 0.05)
        self.fuzzy_threshold = self._cfg.get("fuzzy_match_threshold", 90)
        self.patterns: list[str] = []

    def learn(self, chunks: list[dict[str, Any]]) -> None:
        """Analyze first and last lines of chunks to find repeated patterns."""
        first_lines = []
        last_lines = []
        page_count = len(set(c.get("page_start", 0) for c in chunks))

        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if not content: continue
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            if lines:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])

        # Identify frequent lines
        all_candidates = first_lines + last_lines
        counts = Counter(all_candidates)
        
        # Filter by frequency (e.g. appearing on >3% of pages)
        min_count = max(2, int(page_count * self.freq_threshold))
        self.patterns = [line for line, count in counts.items() if count >= min_count]
        
        # Add common NCCN patterns manually just in case
        self.patterns.extend([
            "NCCN Guidelines for Patients",
            "Adolescent and Young Adult Cancer",
            "NCCN Cancer Centers",
            "NCCN Contributors",
        ])
        
        # Remove duplicates and very short lines
        self.patterns = list(set(p for p in self.patterns if len(p) > 8))
        
        logger.info("Learned %d header/footer patterns from %d pages", len(self.patterns), page_count)

    def is_artifact(self, text: str) -> bool:
        """Check if a line matches a learned boilerplate pattern."""
        t = text.strip()
        if not t: return False
        
        for p in self.patterns:
            if fuzz.ratio(t, p) >= self.fuzzy_threshold:
                return True
        return False

class HeadingHealer:
    """Detects and repairs broken or split headings."""

    def __init__(self):
        self._cfg = cfg.get("repair", default={})
        self.max_frag_words = self._cfg.get("max_heading_fragment_words", 6)
        # List of common medical heading starters that are often split
        self.suspicious_starters = [
            "What options are", "AYAs with", "Seek help to", "Questions to ask",
            "Treatment for", "Staging of", "Follow-up", "Key points",
        ]

    def is_broken_heading(self, heading: str, content: str) -> bool:
        """Identify if a heading looks like an incomplete fragment."""
        h = heading.strip()
        if not h: return True
        
        words = h.split()
        # Case 1: Incomplete phrase (e.g. "What options are")
        if h.lower().rstrip().endswith(("are", "with", "the", "of", "to", "for", "at", "and", "is")):
            return True
            
        # Case 2: Very short and generic
        if len(words) < 2 and not h[0].isdigit():
            return True
            
        # Case 3: Suspicious prefix but too short
        for starter in self.suspicious_starters:
            if h.lower().startswith(starter.lower()) and len(words) < 5:
                # Check if content starts with something that completes it
                return True
                
        # Case 4: Content starts with lowercase (heading likely split)
        c = content.strip()
        if c and c[0].islower():
            return True

        return False

    def reconstruct(self, heading: str, next_text: str) -> tuple[str, str]:
        """Attempt to merge a heading fragment with following text."""
        h = heading.strip()
        t = next_text.strip()
        if not t: return h, t
        
        lines = [l for l in t.split("\n") if l.strip()]
        if not lines: return h, t
        first_line = lines[0]
        
        # Check if the heading fragment is completed by the first line of content
        # Rules: h ends mid-phrase, h is short, first_line starts lowercase or is short
        h_words = h.split()
        is_fragment = h.lower().endswith(("with", "of", "to", "for", "at", "and", "the", "are", "is"))
        
        if is_fragment or len(h_words) < 4:
            # Only merge if it's not a full sentence already
            if not h.endswith((".", "!", "?")):
                new_h = f"{h} {first_line}".strip()
                # If the merge created a valid looking heading (max 15 words)
                if len(new_h.split()) <= 15:
                    # Remove the merged part from text by replacing the first occurrence
                    new_t = t.replace(first_line, "", 1).strip()
                    return new_h, new_t
                
        return h, t
