import json
import argparse
import pickle
import re
import textwrap
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_store"
SPARSE_INDEX_PATH = BASE_DIR / "bm25_index.pkl"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "what",
    "when",
    "with",
}


def load_metadata(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token]


def normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def meaningful_query_tokens(query: str) -> list[str]:
    tokens = [normalize_token(token) for token in tokenize(query)]
    return [token for token in tokens if token not in STOPWORDS]


def build_search_text(item: dict) -> str:
    parts = [
        item.get("heading"),
        item.get("level1"),
        item.get("level2"),
        item.get("level3"),
        item.get("level4"),
        item.get("content"),
    ]
    return " ".join(part for part in parts if part)

def compute_phrase_boost(query: str, metadata: list[dict]) -> np.ndarray:
    lowered_query = query.lower().strip()
    query_tokens = meaningful_query_tokens(query)
    boosts = np.zeros(len(metadata), dtype=float)

    for index, item in enumerate(metadata):
        text = build_search_text(item).lower()
        boost = 0.0
        if lowered_query and lowered_query in text:
            boost += 2.0
        for token in query_tokens:
            if token in text:
                boost += 0.08
        for token in ("memory", "concentration", "cognition", "neurotoxicity", "neuropsychological", "chemotherapy"):
            if token in query_tokens and token in text:
                boost += 0.25
        boosts[index] = boost

    return boosts


def min_max_scale(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--dim", type=int, default=None, help="Use a Matryoshka sub-store, e.g. --dim 256")
    parser.add_argument("--query-prefix", default="", help='Optional prefix, e.g. "search_query: "')
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-width", type=int, default=100)
    parser.add_argument("--preview-max-chars", type=int, default=900)
    return parser.parse_args()


def build_preview(content: str, max_chars: int) -> str:
    paragraphs = [paragraph.strip() for paragraph in content.split("\n") if paragraph.strip()]
    preview_parts: list[str] = []
    current_length = 0

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        paragraph_preview: list[str] = []
        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            proposed = cleaned if not paragraph_preview else f"{paragraph_preview[-1]} {cleaned}"
            candidate_length = current_length + len(cleaned) + (2 if preview_parts else 0)
            if candidate_length > max_chars and preview_parts:
                break
            paragraph_preview.append(cleaned)
            current_length += len(cleaned) + (1 if paragraph_preview else 0)
            if current_length >= max_chars:
                break
        if paragraph_preview:
            preview_parts.append(" ".join(paragraph_preview))
        if current_length >= max_chars:
            break

    preview = "\n\n".join(preview_parts).strip()
    if not preview:
        preview = content[:max_chars].strip()
    if len(preview) < len(content.strip()):
        preview += "\n\n..."
    return preview


def format_preview(content: str, width: int, max_chars: int) -> str:
    preview = build_preview(content, max_chars)
    wrapped_paragraphs = [
        textwrap.fill(paragraph, width=width, break_long_words=False, break_on_hyphens=False)
        for paragraph in preview.split("\n\n")
        if paragraph.strip()
    ]
    return "\n\n".join(wrapped_paragraphs)


def main() -> None:
    args = parse_args()
    query = f"{args.query_prefix}{args.query}" if args.query_prefix else args.query

    base_dir = VECTOR_DIR if args.dim is None else VECTOR_DIR / f"dim_{args.dim}"
    config_path = base_dir / "config.json"
    embeddings_path = base_dir / "embeddings.npy"
    metadata_path = base_dir / "metadata.jsonl"

    if not config_path.exists() or not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Vector store not found. Run build_vectors.py first.")
    if not SPARSE_INDEX_PATH.exists():
        raise FileNotFoundError("Sparse index not found. Run build_sparse.py first.")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"Loading embedding model: {config['embedding_model']}")
    model = SentenceTransformer(
        config["embedding_model"],
        trust_remote_code=True,
        local_files_only=True,
    )

    print("Loading vectors...")
    embeddings = np.load(embeddings_path)
    metadata = load_metadata(metadata_path)
    with SPARSE_INDEX_PATH.open("rb") as file:
        sparse_index = pickle.load(file)
    sparse_metadata = sparse_index["metadata"]
    bm25 = sparse_index["bm25"]

    metadata_lookup = {}
    for index, item in enumerate(sparse_metadata):
        key = (
            item.get("heading"),
            item.get("chunk_id"),
            item.get("level1"),
            item.get("level2"),
            item.get("level3"),
            item.get("level4"),
        )
        metadata_lookup[key] = index

    print("Embedding query...")
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        truncate_dim=config.get("truncate_dim"),
    )[0]

    semantic_scores = embeddings @ query_vector
    query_tokens = meaningful_query_tokens(query)
    sparse_scores = np.array(bm25.get_scores(query_tokens), dtype=float)
    lexical_scores = np.zeros(len(metadata), dtype=float)
    for index, item in enumerate(metadata):
        key = (
            item.get("heading"),
            item.get("chunk_id"),
            item.get("level1"),
            item.get("level2"),
            item.get("level3"),
            item.get("level4"),
        )
        sparse_index_position = metadata_lookup.get(key)
        if sparse_index_position is not None:
            lexical_scores[index] = sparse_scores[sparse_index_position]
    phrase_boosts = compute_phrase_boost(query, metadata)
    scores = (
        0.65 * min_max_scale(semantic_scores)
        + 0.30 * min_max_scale(lexical_scores)
        + 0.05 * min_max_scale(phrase_boosts)
    )
    top_indices = np.argsort(scores)[::-1][: args.top_k]

    for rank, index in enumerate(top_indices, start=1):
        item = metadata[int(index)]
        print(f"\nResult {rank}")
        print(f"Score: {scores[index]:.4f}")
        print(f"Semantic: {semantic_scores[index]:.4f}")
        print(f"Lexical: {lexical_scores[index]:.4f}")
        print(f"Heading: {item.get('heading')}")
        print(f"Chunk ID: {item.get('chunk_id')}")
        print(f"Level1: {item.get('level1')}")
        print(f"Level2: {item.get('level2')}")
        print(f"Level3: {item.get('level3')}")
        print("Content:")
        print(format_preview(item.get("content", ""), args.preview_width, args.preview_max_chars))


if __name__ == "__main__":
    main()
