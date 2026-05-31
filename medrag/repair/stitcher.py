"""Stitcher for merging chunks split across page boundaries."""

from __future__ import annotations

import logging
from typing import Any

from medrag import config as cfg

logger = logging.getLogger(__name__)

class ChunkStitcher:
    """Intelligently merges chunks based on grammar, hierarchy, and context."""

    def __init__(self):
        self._cfg = cfg.get("repair", default={})
        self.merge_mid_sentence = self._cfg.get("merge_mid_sentence_splits", True)

    def should_merge(self, chunk_a: dict[str, Any], chunk_b: dict[str, Any]) -> bool:
        """Decide if Chunk B should be appended to Chunk A."""
        # Must be same source file and hierarchy
        if chunk_a.get("source_file") != chunk_b.get("source_file"):
            return False
        
        # Check hierarchy alignment (level1 and level2 must match)
        if chunk_a.get("level1") != chunk_b.get("level1") or \
           chunk_a.get("level2") != chunk_b.get("level2"):
            return False

        content_a = chunk_a.get("content", "").strip()
        content_b = chunk_b.get("content", "").strip()
        
        if not content_a or not content_b: return False

        # Case 1: Mid-sentence split
        # Chunk A does not end with sentence-ending punctuation
        # AND Chunk B starts with a lowercase letter
        if self.merge_mid_sentence:
            ends_punc = content_a[-1] in ".!?:;\"')] "
            starts_lower = content_b[0].islower()
            if not ends_punc and starts_lower:
                return True

        # Case 2: Broken heading continuation
        # (Already partially handled by healer, but if headings are identical and small, merge)
        if chunk_a.get("heading") == chunk_b.get("heading"):
            # If they are very small or clearly continuations
            if len(content_a.split()) < 100 or len(content_b.split()) < 100:
                return True

        # Case 3: Same level3 and adjacent pages
        if chunk_a.get("level3") == chunk_b.get("level3") and chunk_a.get("level3"):
             # If B starts with a lowercase letter, it's almost certainly a continuation
             if content_b[0].islower():
                 return True

        return False

    def merge(self, chunk_a: dict[str, Any], chunk_b: dict[str, Any]) -> dict[str, Any]:
        """Combine two chunks into one, updating metadata."""
        merged = chunk_a.copy()
        
        # Join content
        text_a = chunk_a.get("content", "").strip()
        text_b = chunk_b.get("content", "").strip()
        
        # Determine joiner (space if mid-word, newline if paragraph)
        joiner = " " if text_a and text_a[-1].isalpha() and text_b and text_b[0].islower() else "\n\n"
        merged["content"] = text_a + joiner + text_b
        
        # Update page range
        pages = [
            chunk_a.get("page_start"), chunk_a.get("page_end"),
            chunk_b.get("page_start"), chunk_b.get("page_end")
        ]
        valid_pages = [p for p in pages if p is not None]
        if valid_pages:
            merged["page_start"] = min(valid_pages)
            merged["page_end"] = max(valid_pages)
            
        # Update token count
        merged["token_count"] = chunk_a.get("token_count", 0) + chunk_b.get("token_count", 0)
        
        # Append repair log
        merged.setdefault("_repair_actions", []).append(f"Merged with chunk {chunk_b.get('chunk_id')}")
        
        return merged
