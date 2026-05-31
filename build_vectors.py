import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "chunks.jsonl"
OUTPUT_DIR = BASE_DIR / "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMS: list[int] = []


def load_chunks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_text_for_embedding(chunk: dict, document_prefix: str = "") -> str:
    parts = [
        chunk.get("level1"),
        chunk.get("level2"),
        chunk.get("level3"),
        chunk.get("level4"),
        chunk.get("heading"),
        chunk.get("content"),
    ]
    text = "\n".join(part for part in parts if part)
    return f"{document_prefix}{text}" if document_prefix else text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(CHUNKS_PATH), help="JSONL chunk file to embed")
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--document-prefix", default="", help='Optional prefix, e.g. "search_document: "')
    parser.add_argument(
        "--dims",
        nargs="*",
        type=int,
        default=DEFAULT_DIMS,
        help="Optional Matryoshka dimensions, e.g. --dims 768 512 256 128",
    )
    return parser.parse_args()


def save_store(
    output_dir: Path,
    chunks: list[dict],
    embeddings: np.ndarray,
    model_name: str,
    truncate_dim: Optional[int],
    document_prefix: str,
) -> None:
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir / "embeddings.npy", embeddings)

    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    config = {
        "embedding_model": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "truncate_dim": truncate_dim,
        "document_prefix": document_prefix,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    print("Loading chunks...")
    chunks = load_chunks(input_path)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
    )

    texts = [build_text_for_embedding(chunk, args.document_prefix) for chunk in chunks]

    dims = args.dims or [None]
    for truncate_dim in dims:
        label = "full" if truncate_dim is None else str(truncate_dim)
        print(f"Creating embeddings for dimension: {label}")
        embeddings = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            truncate_dim=truncate_dim,
        )

        target_dir = OUTPUT_DIR if truncate_dim is None else OUTPUT_DIR / f"dim_{truncate_dim}"
        save_store(target_dir, chunks, embeddings, args.model, truncate_dim, args.document_prefix)
        print(f"Saved vector store to: {target_dir}")


if __name__ == "__main__":
    main()
