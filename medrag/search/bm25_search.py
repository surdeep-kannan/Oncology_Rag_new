"""BM25 sparse retrieval index for chunk search."""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any

from medrag import config as cfg

logger = logging.getLogger(__name__)

# Simple tokenizer with stopword removal
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "if", "in", "into", "is",
    "it", "its", "may", "more", "not", "of", "on", "or", "than",
    "that", "the", "their", "them", "there", "these", "they", "this",
    "those", "to", "was", "were", "what", "when", "which", "with",
    "you", "your", "can", "do", "does", "will", "would", "could",
    "should", "about", "after", "all", "also", "any", "been", "being",
    "between", "both", "each", "few", "get", "got", "her", "here",
    "him", "his", "how", "just", "like", "most", "must", "no", "nor",
    "only", "other", "our", "out", "own", "same", "she", "so", "some",
    "such", "very", "we",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize with stopword removal."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """BM25 index for sparse retrieval over chunks."""

    def __init__(self) -> None:
        self._cfg = cfg.get("search", default={})
        self.k1 = self._cfg.get("bm25_k1", 1.5)
        self.b = self._cfg.get("bm25_b", 0.75)
        self._index = None
        self._chunks: list[dict[str, Any]] = []
        self._corpus: list[list[str]] = []

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build BM25 index from chunk dicts."""
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        self._corpus = [
            _tokenize(c.get("content", "") + " " + c.get("heading", ""))
            for c in chunks
        ]

        self._index = BM25Okapi(
            self._corpus,
            k1=self.k1,
            b=self.b,
        )
        logger.info("BM25 index built with %d documents", len(chunks))

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the BM25 index."""
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Apply metadata filters
        results: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            if filters and not self._matches_filters(self._chunks[idx], filters):
                continue
            results.append((score, idx))

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:top_k]

        return [
            {**self._chunks[idx], "_bm25_score": float(score)}
            for score, idx in results
        ]

    def save(self, path: Path) -> None:
        """Persist the index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self._chunks,
            "corpus": self._corpus,
            "k1": self.k1,
            "b": self.b,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a persisted index."""
        from rank_bm25 import BM25Okapi

        with open(path, "rb") as f:
            data = pickle.load(f)

        self._chunks = data["chunks"]
        self._corpus = data["corpus"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)

        self._index = BM25Okapi(self._corpus, k1=self.k1, b=self.b)
        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._chunks))

    @staticmethod
    def _matches_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if a chunk matches all provided metadata filters."""
        for key, value in filters.items():
            chunk_val = chunk.get(key)
            if isinstance(value, list):
                if chunk_val not in value:
                    return False
            elif chunk_val != value:
                return False
        return True
