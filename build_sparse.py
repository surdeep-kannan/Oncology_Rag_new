import argparse
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "semantic_chunks.jsonl"
OUTPUT_FILE = BASE_DIR / "bm25_index.pkl"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_token(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower()) if token]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(INPUT_FILE), help="JSONL chunk file to index")
    parser.add_argument("--output", default=str(OUTPUT_FILE), help="Pickle output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    documents = []
    metadata = []

    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            text = build_search_text(item)
            tokenized = tokenize(text)
            if not tokenized:
                continue
            documents.append(tokenized)
            metadata.append(item)

    bm25 = BM25Okapi(documents)

    with output_path.open("wb") as file:
        pickle.dump(
            {
                "bm25": bm25,
                "metadata": metadata,
                "documents": documents,
                "input_file": str(input_path),
            },
            file,
        )

    print(f"Built BM25 index for {len(documents)} semantic chunks")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
