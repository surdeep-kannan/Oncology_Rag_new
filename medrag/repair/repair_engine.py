"""Engine to coordinate the chunk repair and reconstruction process."""

from __future__ import annotations

import logging
from typing import Any

from medrag.repair.analyzers import HeaderFooterLearner, HeadingHealer
from medrag.repair.stitcher import ChunkStitcher

logger = logging.getLogger(__name__)

class RepairEngine:
    """Orchestrates header/footer removal, heading healing, and chunk stitching."""

    def __init__(self):
        self.learner = HeaderFooterLearner()
        self.healer = HeadingHealer()
        self.stitcher = ChunkStitcher()

    def repair_pipeline(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run full repair suite on a list of chunks."""
        if not chunks: return []

        # Step 1: Learn recurring patterns
        self.learner.learn(chunks)

        # Step 2: Clean content and heal headings
        cleaned_chunks = []
        for chunk in chunks:
            repaired = self._clean_and_heal(chunk)
            cleaned_chunks.append(repaired)

        # Step 3: Stitch chunks across boundaries
        repaired_chunks = self._stitch_chunks(cleaned_chunks)
        
        # Step 4: Final ID reassignment
        for idx, chunk in enumerate(repaired_chunks):
            chunk["chunk_id"] = idx + 1
            
        return repaired_chunks

    def _clean_and_heal(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Remove artifacts from content and attempt to repair heading."""
        c = chunk.copy()
        content = c.get("content", "")
        lines = content.split("\n")
        
        # Remove learned artifacts
        clean_lines = [l for l in lines if not self.learner.is_artifact(l)]
        
        # Safety: If we stripped everything, keep the longest line to avoid empty content
        if not clean_lines and lines:
            longest_line = max(lines, key=len)
            clean_lines = [longest_line]
            
        clean_content = "\n".join(clean_lines).strip()
        
        # Heal heading if it looks broken
        heading = c.get("heading", "")
        if self.healer.is_broken_heading(heading, clean_content):
            new_h, new_c = self.healer.reconstruct(heading, clean_content)
            if new_h != heading:
                c["heading"] = new_h
                clean_content = new_c
                c.setdefault("_repair_actions", []).append(f"Healed heading: '{heading}' -> '{new_h}'")

        c["content"] = clean_content
        return c

    def _stitch_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Iteratively merge adjacent chunks that belong together."""
        if not chunks: return []
        
        result = []
        current = chunks[0]
        
        for next_chunk in chunks[1:]:
            if self.stitcher.should_merge(current, next_chunk):
                current = self.stitcher.merge(current, next_chunk)
            else:
                result.append(current)
                current = next_chunk
        
        result.append(current)
        return result
