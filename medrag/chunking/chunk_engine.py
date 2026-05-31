"""Chunk engine – splits hierarchy nodes into RAG-ready chunks.

Implements semantic chunking with:
  - Sentence-aware splitting
  - Configurable target/min/max word counts (200–600)
  - Overlap for retrieval continuity
  - Bullet-list preservation
  - Tiny-chunk merging
  - Large-chunk semantic splitting
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medrag import config as cfg
from medrag.chunking.hierarchy_builder import HierarchyNode

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single RAG-ready chunk."""

    chunk_id: int
    source_file: str
    level1: str | None
    level2: str | None
    level3: str | None
    heading: str
    content: str
    page_start: int | None
    page_end: int | None
    token_count: int  # word count

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "level1": self.level1,
            "level2": self.level2,
            "level3": self.level3,
            "heading": self.heading,
            "content": self.content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "token_count": self.token_count,
        }


@dataclass
class ChunkingStats:
    """Track chunking metrics."""

    sections_processed: int = 0
    chunks_created: int = 0
    chunks_merged: int = 0
    chunks_split: int = 0
    min_words: int = 0
    max_words: int = 0
    avg_words: float = 0.0


# Sentence splitting with medical abbreviation awareness
_MEDICAL_ABBREVS = {
    "dr", "mr", "mrs", "ms", "vs", "fig", "prof", "inc", "jr", "sr",
    "st", "no", "appt", "mg", "ml", "oz", "dept", "etc", "approx",
    "e.g", "i.e", "vol", "min", "max", "avg", "pt", "dx", "tx", "rx",
    "hx", "bx", "fx", "sx",
}

# Build regex for sentence splitting
_SENTENCE_END = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'(])'
    r'|(?<=[.!?])\s*\n'
)

# Bullet line detection
_BULLET_RE = re.compile(r"^\s*[-*•●◦▪▸►‣⁃]\s|^\s*\d+[.)]\s|^\s*[a-zA-Z][.)]\s")


def _count_words(text: str) -> int:
    return len(text.split())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting medical abbreviations."""
    if not text.strip():
        return []

    # Handle bullet lists: each bullet is its own "sentence"
    lines = text.split("\n")
    sentences: list[str] = []
    buffer: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if _BULLET_RE.match(line):
            # Flush buffer
            if buffer:
                joined = " ".join(buffer)
                sentences.extend(_split_plain_sentences(joined))
                buffer = []
            sentences.append(line)
        else:
            buffer.append(line)

    if buffer:
        joined = " ".join(buffer)
        sentences.extend(_split_plain_sentences(joined))

    return [s.strip() for s in sentences if s.strip()]


def _split_plain_sentences(text: str) -> list[str]:
    """Split plain text (no bullets) into sentences."""
    # Use regex-based splitting
    parts = _SENTENCE_END.split(text)
    if len(parts) <= 1:
        return [text] if text.strip() else []

    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Check if the period before split was an abbreviation
        if result and _is_abbreviation_end(result[-1]):
            result[-1] = result[-1] + " " + part
        else:
            result.append(part)

    return result


def _is_abbreviation_end(text: str) -> bool:
    """Check if text ends with a known abbreviation."""
    words = text.lower().split()
    if not words:
        return False
    last_word = words[-1].rstrip(".")
    return last_word in _MEDICAL_ABBREVS


class ChunkEngine:
    """Semantic chunk engine for medical documents."""

    def __init__(self) -> None:
        self._cfg = cfg.get("chunking", default={})
        self.min_words = self._cfg.get("min_words", 200)
        self.target_words = self._cfg.get("target_words", 400)
        self.max_words = self._cfg.get("max_words", 600)
        self.overlap_words = self._cfg.get("overlap_words", 50)
        self.overlap_sentences = self._cfg.get("overlap_sentences", 2)
        self.min_chunk_words = self._cfg.get("min_chunk_words", 40)
        self.merge_tiny_threshold = self._cfg.get("merge_tiny_threshold", 30)
        self.stats = ChunkingStats()
        self._chunk_counter = 0

    def reset(self) -> None:
        self.stats = ChunkingStats()
        self._chunk_counter = 0

    def process_sections(self, sections: list[HierarchyNode]) -> list[Chunk]:
        """Process a list of hierarchy sections into chunks."""
        all_chunks: list[Chunk] = []

        for section in sections:
            self.stats.sections_processed += 1
            chunks = self._chunk_section(section)
            all_chunks.extend(chunks)

        # Post-process: merge tiny chunks
        all_chunks = self._merge_tiny_chunks(all_chunks)

        # Update stats
        word_counts = [c.token_count for c in all_chunks]
        if word_counts:
            self.stats.min_words = min(word_counts)
            self.stats.max_words = max(word_counts)
            self.stats.avg_words = sum(word_counts) / len(word_counts)
        self.stats.chunks_created = len(all_chunks)

        # Re-number chunk IDs
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_id = i + 1

        logger.info(
            "Chunking complete: %d sections -> %d chunks (avg %d words)",
            self.stats.sections_processed,
            self.stats.chunks_created,
            int(self.stats.avg_words),
        )
        return all_chunks

    def _chunk_section(self, section: HierarchyNode) -> list[Chunk]:
        """Split a single section into appropriately sized chunks."""
        content = section.full_content
        if not content.strip():
            return []

        word_count = _count_words(content)

        # If content fits in one chunk, emit it directly
        if word_count <= self.max_words:
            if word_count < self.merge_tiny_threshold:
                # Too tiny – will be merged later
                pass
            chunk = self._make_chunk(section, content)
            return [chunk]

        # Split into sentences
        sentences = _split_sentences(content)
        if not sentences:
            return []

        # Build chunks from sentences
        return self._build_chunks_from_sentences(section, sentences)

    def _build_chunks_from_sentences(
        self,
        section: HierarchyNode,
        sentences: list[str],
    ) -> list[Chunk]:
        """Build chunks from sentences using semantic boundaries."""
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_words = 0

        for i, sentence in enumerate(sentences):
            sent_words = _count_words(sentence)

            # Check if adding this sentence exceeds max
            if current_words + sent_words > self.max_words and current_sentences:
                # Emit current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append(self._make_chunk(section, chunk_text))
                self.stats.chunks_split += 1

                # Overlap: keep last N sentences
                overlap = current_sentences[-self.overlap_sentences:]
                current_sentences = overlap[:]
                current_words = sum(_count_words(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_words += sent_words

            # Check if we've hit target and next sentence is a good break point
            if current_words >= self.target_words and i + 1 < len(sentences):
                next_sent = sentences[i + 1]
                if self._is_good_break_point(sentence, next_sent):
                    chunk_text = " ".join(current_sentences)
                    chunks.append(self._make_chunk(section, chunk_text))

                    overlap = current_sentences[-self.overlap_sentences:]
                    current_sentences = overlap[:]
                    current_words = sum(_count_words(s) for s in current_sentences)

        # Emit remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            wc = _count_words(chunk_text)

            # If tiny remainder, append to previous chunk
            if wc < self.min_chunk_words and chunks:
                prev = chunks[-1]
                merged = prev.content + "\n\n" + chunk_text
                chunks[-1] = self._make_chunk(
                    section, merged,
                    page_start=prev.page_start,
                )
            else:
                chunks.append(self._make_chunk(section, chunk_text))

        return chunks

    def _is_good_break_point(self, current: str, next_sent: str) -> bool:
        """Determine if the boundary between two sentences is a good chunk break."""
        # Strong break: paragraph boundary (already in separate sentences)
        # Strong break: new topic (capitalized, not continuation)

        # Weak break: next sentence starts with transition word
        transition_words = {
            "however", "moreover", "furthermore", "additionally",
            "also", "therefore", "thus", "hence", "consequently",
            "meanwhile", "nevertheless", "nonetheless",
        }
        first_word = next_sent.split()[0].lower().rstrip(".,;:") if next_sent.split() else ""

        # Good break if current ends sentence and next starts fresh
        if current.rstrip().endswith((".", "!", "?")):
            # Avoid breaking before continuation words
            if first_word in {"this", "that", "these", "those", "it", "they", "its"}:
                return False
            # Avoid breaking before transition words
            if first_word in transition_words:
                return False
            return True

        return False

    def _make_chunk(
        self,
        section: HierarchyNode,
        content: str,
        page_start: int | None = None,
    ) -> Chunk:
        """Create a Chunk from a section and content string."""
        self._chunk_counter += 1
        return Chunk(
            chunk_id=self._chunk_counter,
            source_file=section.source_file,
            level1=section.level1,
            level2=section.level2,
            level3=section.level3,
            heading=section.heading,
            content=content.strip(),
            page_start=page_start or section.page_start,
            page_end=section.page_end,
            token_count=_count_words(content),
        )

    def _merge_tiny_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge chunks below the minimum threshold with neighbors."""
        if not chunks:
            return []

        merged: list[Chunk] = [chunks[0]]

        for chunk in chunks[1:]:
            if chunk.token_count < self.min_chunk_words:
                # Try to merge with previous
                prev = merged[-1]
                if prev.heading == chunk.heading or prev.level1 == chunk.level1:
                    combined = prev.content + "\n\n" + chunk.content
                    merged[-1] = Chunk(
                        chunk_id=prev.chunk_id,
                        source_file=prev.source_file,
                        level1=prev.level1,
                        level2=prev.level2,
                        level3=prev.level3,
                        heading=prev.heading,
                        content=combined,
                        page_start=prev.page_start,
                        page_end=chunk.page_end or prev.page_end,
                        token_count=_count_words(combined),
                    )
                    self.stats.chunks_merged += 1
                    continue

            merged.append(chunk)

        return merged

    @staticmethod
    def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
        """Write chunks as JSONL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        logger.info("Saved %d chunks to %s", len(chunks), output_path)

    @staticmethod
    def load_chunks(input_path: Path) -> list[Chunk]:
        """Load chunks from JSONL."""
        chunks: list[Chunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                chunks.append(Chunk(**data))
        return chunks
