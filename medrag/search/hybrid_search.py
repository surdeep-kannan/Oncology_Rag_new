"""Hybrid search – fuses BM25 sparse and dense embedding results."""

from __future__ import annotations

import logging
from typing import Any

from medrag import config as cfg
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Reciprocal Rank Fusion (RRF) hybrid of BM25 + embedding search."""

    def __init__(
        self,
        bm25_index: BM25Index,
        embedding_index: EmbeddingIndex,
        alpha: float | None = None,
    ) -> None:
        self._cfg = cfg.get("search", default={})
        self.bm25 = bm25_index
        self.embedding = embedding_index
        self.alpha = alpha if alpha is not None else self._cfg.get("hybrid_alpha", 0.5)
        self.default_top_k = self._cfg.get("top_k", 20)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion.

        alpha controls the weight: 1.0 = pure embedding, 0.0 = pure BM25.
        """
        k = top_k or self.default_top_k
        rrf_k = 60  # RRF constant

        # Retrieve from both
        candidate_k = max(100, k * 3)
        bm25_results = self.bm25.search(query, top_k=candidate_k, filters=filters)
        emb_results = self.embedding.search(query, top_k=candidate_k, filters=filters)

        # Build RRF scores
        scores: dict[str, float] = {}
        chunk_map: dict[str, dict[str, Any]] = {}

        # BM25 contribution
        for rank, result in enumerate(bm25_results):
            ukey = f"{result.get('source_file', 'unknown')}_{result.get('chunk_id', rank)}"
            rrf_score = (1 - self.alpha) / (rrf_k + rank + 1)
            scores[ukey] = scores.get(ukey, 0.0) + rrf_score
            chunk_map[ukey] = result

        # Embedding contribution
        for rank, result in enumerate(emb_results):
            ukey = f"{result.get('source_file', 'unknown')}_{result.get('chunk_id', rank)}"
            rrf_score = self.alpha / (rrf_k + rank + 1)
            scores[ukey] = scores.get(ukey, 0.0) + rrf_score
            if ukey not in chunk_map:
                chunk_map[ukey] = result

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for ukey, fused_score in ranked:
            result = chunk_map[ukey].copy()
            result["_hybrid_score"] = fused_score
            # Carry forward individual scores
            bm25_score = next(
                (r.get("_bm25_score", 0) for r in bm25_results if f"{r.get('source_file', 'unknown')}_{r.get('chunk_id')}" == ukey),
                0.0,
            )
            emb_score = next(
                (r.get("_embedding_score", 0) for r in emb_results if f"{r.get('source_file', 'unknown')}_{r.get('chunk_id')}" == ukey),
                0.0,
            )
            result["_bm25_score"] = bm25_score
            result["_embedding_score"] = emb_score
            results.append(result)

        logger.debug(
            "Hybrid search: %d BM25 + %d embedding -> %d fused results",
            len(bm25_results), len(emb_results), len(results),
        )
        return results
