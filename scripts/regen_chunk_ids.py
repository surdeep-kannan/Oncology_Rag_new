#!/usr/bin/env python3
"""
regen_chunk_ids.py — Regenerate relevant_chunk_ids in eval_dataset.json
to match the CURRENT active database after rebuilds/re-chunking.

For each question, we:
  1. Search the current chunk DB using BM25 on both the question AND ground truth text
  2. Take the top-5 chunk_ids as the new relevant_chunk_ids
  3. Save the updated eval_dataset.json

This ensures retrieval metrics are meaningful and not tied to stale IDs.
"""
import sys, os, json, re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medrag.search.bm25_search import BM25Index

CHUNKS_PATH  = "output/indices/embeddings/chunks.json"
BM25_PATH    = "output/indices/bm25_index.pkl"
EVAL_PATH    = "eval_dataset.json"
TOP_K        = 5

def main():
    print("Loading BM25 index...")
    bm25_idx = BM25Index()
    bm25_idx.load(BM25_PATH)

    print("Loading eval dataset...")
    with open(EVAL_PATH, "r") as f:
        dataset = json.load(f)

    print(f"Regenerating relevant_chunk_ids for {len(dataset)} questions...\n")

    for item in dataset:
        qid   = item["id"]
        q     = item["question"]
        gt    = item.get("ground_truth", "")

        # Search using ground truth (most precise signal)
        gt_results  = bm25_idx.search(gt,  top_k=TOP_K)
        q_results   = bm25_idx.search(q,   top_k=TOP_K)

        # Merge: gt hits first (they are the gold standard), then question hits
        seen = set()
        merged = []
        for r in (gt_results + q_results):
            cid = r.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                merged.append(cid)

        item["relevant_chunk_ids"] = merged[:TOP_K]

        # Report
        print(f"[{qid}] {q[:80]}...")
        print(f"  → New relevant_chunk_ids: {item['relevant_chunk_ids']}")
        if gt_results:
            top = gt_results[0]
            print(f"  → Top hit: ID={top.get('chunk_id')} | Source={top.get('source_file')} | Heading={top.get('heading','')[:60]}")
        print()

    # Save updated dataset
    with open(EVAL_PATH, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ Updated eval_dataset.json saved with fresh relevant_chunk_ids.")

if __name__ == "__main__":
    main()
