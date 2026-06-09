#!/usr/bin/env python3
"""
repatch_asco.py — Re-chunk ASCO PDF with smaller chunks, re-embed with MRL-512, patch database.
"""
import json, re, pickle, time, sys, os, numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pathlib import Path
import subprocess

ASCO_PDFS = [
    "input/american_global_guidelines_ASCO.pdf",
]

OUT_DIR     = Path("output/repatch_asco")
CHUNKS_PATH = Path("output/indices/embeddings/chunks.json")
NPY_PATH    = Path("output/indices/embeddings/embeddings.npy")
BM25_PATH   = Path("output/indices/bm25_index.pkl")

TARGET_SOURCE_NAMES = {
    "american global guidelines asco.pdf",
    "american global guidelines asco (1).pdf",
    "american_global_guidelines_asco.pdf",
}

def strip_citations(text):
    text = re.sub(r'\[[\d\s,\-\u2013]+\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    TOKEN_RE = re.compile(r"[a-z0-9]+")
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]

def build_search_text(chunk):
    return " ".join(p for p in [
        chunk.get("heading"), chunk.get("level1"), chunk.get("level2"),
        chunk.get("level3"), chunk.get("content")
    ] if p)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 60)
    print("  REPATCHING ASCO PDF — Fine-grained chunking + MRL-512")
    print("=" * 60)

    new_chunks = []

    for pdf in ASCO_PDFS:
        if not Path(pdf).exists():
            print(f"Skipping {pdf} (not found)")
            continue
        base = Path(pdf).name
        parsed_out = OUT_DIR / f"parsed_{base}.jsonl"
        chunks_out = OUT_DIR / f"chunks_{base}.jsonl"

        print(f"\nParsing: {base}...")
        r = subprocess.run(
            ["python3", "scripts/parse_pdfs.py", "--parser", "pymupdf",
             "--input", pdf, "--output", str(parsed_out)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"  Parse error: {r.stderr[:200]}")
            continue
        print(f"  ✅ Parsed")

        print(f"  Chunking with new smaller sizes...")
        r = subprocess.run(
            ["python3", "scripts/build_chunks.py",
             "--input", str(parsed_out), "--output", str(chunks_out)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"  Chunk error: {r.stderr[:200]}")
            continue

        with open(chunks_out, "r") as f:
            for line in f:
                if line.strip():
                    c = json.loads(line)
                    c["content"] = strip_citations(c.get("content", ""))
                    new_chunks.append(c)
        print(f"  ✅ {len(new_chunks)} chunks so far")

    if not new_chunks:
        print("No new chunks generated. Exiting.")
        return

    print(f"\nTotal new ASCO chunks: {len(new_chunks)}")

    # Verify the key fact is present
    found = any("basic" in c["content"].lower() and "limited" in c["content"].lower()
                for c in new_chunks)
    print(f"  'Basic, Limited' present in chunks: {'✅ YES' if found else '❌ NO'}")
    if found:
        for c in new_chunks:
            if "basic" in c["content"].lower() and "limited" in c["content"].lower():
                print(f"  → Chunk: {c['content'][:200]}")
                break

    # Embed new chunks with Nomic MRL-512
    print("\nLoading Nomic embed model (int8 quantized)...")
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    model.max_seq_length = 256  # longer context for better factual grounding
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    texts = []
    for c in new_chunks:
        parts = [c.get("level1"), c.get("level2"), c.get("heading"), c.get("content")]
        ctx = " > ".join(p for p in parts[:-1] if p)
        full = f"{ctx}. {parts[-1]}" if ctx else parts[-1]
        texts.append(f"search_document: {full}")

    print(f"Encoding {len(texts)} chunks...")
    new_embs = model.encode(texts, batch_size=32, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)

    # MRL truncation to 512 dims
    if new_embs.shape[1] > 512:
        new_embs = new_embs[:, :512]
        norms = np.linalg.norm(new_embs, axis=1, keepdims=True)
        new_embs = new_embs / (norms + 1e-10)
    print(f"  MRL-512 embeddings shape: {new_embs.shape}")

    # Load active database, remove old ASCO chunks, append new
    print("\nPatching active database...")
    with open(CHUNKS_PATH) as f:
        active = json.load(f)
    active_embs = np.load(NPY_PATH)
    print(f"  Before: {len(active)} chunks")

    kept_chunks, kept_embs = [], []
    removed = 0
    for i, c in enumerate(active):
        if c.get("source_file", "").lower() in TARGET_SOURCE_NAMES:
            removed += 1
        else:
            kept_chunks.append(c)
            kept_embs.append(active_embs[i])

    print(f"  Removed {removed} old ASCO chunks")
    combined_chunks = kept_chunks + new_chunks
    combined_embs   = np.vstack([np.array(kept_embs), new_embs])

    np.save(NPY_PATH, combined_embs)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(combined_chunks, f, ensure_ascii=False)
    print(f"  After: {len(combined_chunks)} chunks | Embeddings: {combined_embs.shape}")

    # Rebuild BM25
    print("\nRebuilding BM25 index...")
    corpus = [tokenize(build_search_text(c)) for c in combined_chunks]
    bm25 = BM25Okapi(corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"chunks": combined_chunks, "corpus": corpus, "k1": 1.5, "b": 0.75}, f)
    print("  ✅ BM25 rebuilt")

    print(f"\n✅ Done in {time.time()-t0:.1f}s")
    print(f"   Total DB chunks: {len(combined_chunks)}")
    print(f"   Embeddings:      {combined_embs.shape}")

if __name__ == "__main__":
    main()
