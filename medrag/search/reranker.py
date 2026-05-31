"""Cross-encoder reranker with parent-child retrieval support."""

from __future__ import annotations

import logging
from typing import Any

from medrag import config as cfg

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank search results using a cross-encoder model.

    Also supports parent-child retrieval: when a chunk matches,
    optionally include its parent section for richer context.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._cfg = cfg.get("search", default={})
        self.model_name = model_name or self._cfg.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.reranker_top_k = self._cfg.get("reranker_top_k", 10)
        self.parent_child_enabled = self._cfg.get("parent_child_enabled", True)
        self.parent_context_levels = self._cfg.get("parent_context_levels", 1)
        self._model = None

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return

        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: %s", self.model_name)
        self._model = CrossEncoder(self.model_name)
        logger.info("Cross-encoder loaded")

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank results using cross-encoder scoring."""
        if not results:
            return []

        self._load_model()
        k = top_k or self.reranker_top_k

        # Prepare query-document pairs
        pairs = [
            [query, r.get("content", "")]
            for r in results
        ]

        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored_results = []
        for result, score in zip(results, scores):
            r = result.copy()
            r["_reranker_score"] = float(score)
            scored_results.append(r)

        scored_results.sort(key=lambda x: x["_reranker_score"], reverse=True)

        logger.debug(
            "Reranked %d results -> top %d",
            len(scored_results), min(k, len(scored_results)),
        )
        return scored_results[:k]

    def rerank_with_parent_context(
        self,
        query: str,
        results: list[dict[str, Any]],
        all_chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank and enrich results with parent section context.

        For each top result, finds sibling chunks that share the same
        heading hierarchy and includes them as context.
        """
        reranked = self.rerank(query, results, top_k)

        if not self.parent_child_enabled:
            return reranked

        # Build a parent lookup
        parent_map = self._build_parent_map(all_chunks)

        enriched: list[dict[str, Any]] = []
        for result in reranked:
            r = result.copy()

            # Find parent context
            parent_key = self._get_parent_key(result)
            if parent_key and parent_key in parent_map:
                siblings = parent_map[parent_key]
                parent_content = "\n\n".join(
                    s.get("content", "") for s in siblings
                    if s.get("chunk_id") != result.get("chunk_id")
                )
                if parent_content:
                    r["_parent_context"] = parent_content[:800]  # cap length to avoid prompt truncation/overflow
                    r["_sibling_count"] = len(siblings)

            enriched.append(r)

        return enriched

    def _build_parent_map(
        self, chunks: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group chunks by their parent heading hierarchy."""
        parent_map: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            key = self._get_parent_key(chunk)
            if key:
                parent_map.setdefault(key, []).append(chunk)
        return parent_map

    def _get_parent_key(self, chunk: dict[str, Any]) -> str | None:
        """Create a grouping key based on hierarchy levels."""
        parts: list[str] = []
        for level_key in ["level1", "level2"]:
            val = chunk.get(level_key)
            if val:
                parts.append(str(val))

        if self.parent_context_levels == 1 and len(parts) >= 1:
            return " > ".join(parts[:1])
        return " > ".join(parts) if parts else None
