"""Dense embedding index for semantic search over chunks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from medrag import config as cfg

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """Dense vector index using sentence-transformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self._cfg = cfg.get("search", default={})
        self.model_name = model_name or self._cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.batch_size = self._cfg.get("embedding_batch_size", 64)
        self.device = self._cfg.get("embedding_device", "cpu")
        
        # Matryoshka (MRL) Settings
        self.matryoshka_dim = self._cfg.get("matryoshka_dim", 512)
        self.is_nomic = "nomic" in self.model_name.lower()

        self._model = None
        self._embeddings: np.ndarray | None = None
        self._chunks: list[dict[str, Any]] = []

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(
            self.model_name, 
            device=self.device,
            trust_remote_code=True
        )
        logger.info("Embedding model loaded on %s", self.device)

    def build(self, chunks: list[dict[str, Any]], show_progress: bool = True) -> None:
        """Build embedding index from chunk dicts."""
        self._load_model()
        self._chunks = chunks

        # Prepare texts: heading + content for richer embeddings
        # Add Nomic prefix if applicable
        prefix = "search_document: " if self.is_nomic else ""
        texts = [
            (prefix + c.get("heading", "") + ". " + c.get("content", "")).strip()
            for c in chunks
        ]

        logger.info("Encoding %d chunks (is_nomic=%s, dim=%d) ...", 
                    len(texts), self.is_nomic, self.matryoshka_dim)
        
        raw_embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Matryoshka Truncation & Re-Normalization
        if self.matryoshka_dim and self.matryoshka_dim < raw_embeddings.shape[1]:
            logger.info("Applying Matryoshka truncation to %d dims", self.matryoshka_dim)
            self._embeddings = raw_embeddings[:, :self.matryoshka_dim]
            # Re-normalize after truncation
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            self._embeddings = self._embeddings / (norms + 1e-10)
        else:
            self._embeddings = raw_embeddings

        logger.info(
            "Embeddings built: shape=%s", self._embeddings.shape
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using cosine similarity."""
        if self._embeddings is None or self._model is None:
            raise RuntimeError("Embedding index not built. Call build() first.")

        # Add Nomic prefix for query
        search_query = f"search_query: {query}" if self.is_nomic else query
        
        query_emb = self._model.encode(
            [search_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Matryoshka Truncation & Re-Normalization
        if self.matryoshka_dim and self.matryoshka_dim < query_emb.shape[1]:
            query_emb = query_emb[:, :self.matryoshka_dim]
            norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
            query_emb = query_emb / (norms + 1e-10)

        # Cosine similarity (embeddings are already L2-normalized)
        scores = np.dot(self._embeddings, query_emb.T).flatten()

        # Apply filters
        results: list[tuple[float, int]] = []
        for idx, score in enumerate(scores):
            if filters and not self._matches_filters(self._chunks[idx], filters):
                continue
            results.append((float(score), idx))

        results.sort(key=lambda x: x[0], reverse=True)
        results = results[:top_k]

        return [
            {**self._chunks[idx], "_embedding_score": score}
            for score, idx in results
        ]

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string."""
        self._load_model()
        return self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    def save(self, directory: Path) -> None:
        """Save embeddings and chunk metadata."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self._embeddings)
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False)
        with open(directory / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name}, f)
        logger.info("Embedding index saved to %s", directory)

    def load(self, directory: Path) -> None:
        """Load embeddings and chunk metadata."""
        self._embeddings = np.load(directory / "embeddings.npy")
        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            self._chunks = json.load(f)
        config_path = directory / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
                self.model_name = saved_cfg.get("model_name", self.model_name)
                self.is_nomic = "nomic" in self.model_name.lower()
        self._load_model()
        logger.info(
            "Embedding index loaded from %s (%d docs)",
            directory, len(self._chunks),
        )

    @staticmethod
    def _matches_filters(chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, value in filters.items():
            chunk_val = chunk.get(key)
            if isinstance(value, list):
                if chunk_val not in value:
                    return False
            elif chunk_val != value:
                return False
        return True
