#!/usr/bin/env python3
"""search_chunks.py – Hybrid BM25 + embedding search with reranking.

Usage:
    python scripts/search_chunks.py "What is the treatment for breast cancer?"
    python scripts/search_chunks.py "chemotherapy side effects" --top-k 10
    python scripts/search_chunks.py "staging" --level1 "Treatment"
    python scripts/search_chunks.py "radiation" --build-index
    python scripts/search_chunks.py "diagnosis" --mode bm25
    python scripts/search_chunks.py "surgery" --rerank
    python scripts/search_chunks.py "fertility" --parent-context
"""
from __future__ import annotations
import argparse, json, logging, sys, time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag.search.hybrid_search import HybridSearcher
from medrag.search.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Search medical RAG chunks.")
    p.add_argument("query", type=str, help="Search query.")
    p.add_argument("--input","-i",type=Path,default=None,help="Chunks JSONL.")
    p.add_argument("--top-k","-k",type=int,default=10)
    p.add_argument("--mode",choices=["hybrid","bm25","embedding"],default="hybrid")
    p.add_argument("--alpha",type=float,default=None,help="Hybrid weight (0=BM25, 1=embedding).")
    p.add_argument("--rerank",action="store_true",help="Apply cross-encoder reranking.")
    p.add_argument("--parent-context",action="store_true",help="Include parent context.")
    p.add_argument("--level1",type=str,default=None,help="Filter by level1.")
    p.add_argument("--level2",type=str,default=None,help="Filter by level2.")
    p.add_argument("--source",type=str,default=None,help="Filter by source_file.")
    p.add_argument("--build-index",action="store_true",help="Force rebuild indices.")
    p.add_argument("--config","-c",type=Path,default=None)
    p.add_argument("--verbose","-v",action="store_true")
    p.add_argument("--json", action="store_true", help="Output results as JSON.")
    return p.parse_args()

def load_chunks(path):
    chunks = []
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): chunks.append(json.loads(line))
    return chunks

def build_filters(args):
    filters = {}
    if args.level1: filters["level1"] = args.level1
    if args.level2: filters["level2"] = args.level2
    if args.source: filters["source_file"] = args.source
    return filters or None

def display_result(rank, result, verbose=False):
    cid = result.get("chunk_id","?")
    heading = result.get("heading","")
    l1 = result.get("level1","—")
    content = result.get("content","")
    scores = []
    for key in ["_hybrid_score","_bm25_score","_embedding_score","_reranker_score"]:
        if key in result: scores.append(f"{key[1:]}={result[key]:.4f}")
    score_str = " | ".join(scores)

    print(f"\n  [{rank}] Chunk #{cid} — {heading}")
    print(f"      L1: {l1} | {score_str}")
    if verbose:
        for line in content.split("\n")[:10]: print(f"      {line}")
        if content.count("\n")>10: print(f"      ... ({len(content.split())} words total)")
    else:
        preview = content[:200].replace("\n"," ")
        if len(content)>200: preview+="..."
        print(f"      {preview}")
    if "_parent_context" in result:
        print(f"      [+parent: {result.get('_sibling_count',0)} siblings]")

def main():
    args = parse_args()
    if args.config: cfg.load_config(args.config)
    setup_logging(level="DEBUG" if args.verbose else "WARNING")

    # Use master_chunks.jsonl as default if it exists
    master_path = PROJECT_ROOT / "master_chunks.jsonl"
    input_path = args.input or (master_path if master_path.exists() else cfg.output_dir() / "chunks.jsonl")
    
    if not input_path.exists():
        if args.json:
            print(json.dumps({"error": f"Not found: {input_path}"}))
        else:
            print(f"❌ Not found: {input_path}")
        sys.exit(1)

    chunks = load_chunks(input_path)
    if not args.json:
        print(f"Loaded {len(chunks)} chunks")
    
    filters = build_filters(args)
    t0 = time.time()

    # Index paths
    idx_dir = cfg.index_dir()
    bm25_path = idx_dir / "bm25_index.pkl"
    emb_dir = idx_dir / "embeddings"

    # BM25 index
    bm25 = BM25Index()
    if bm25_path.exists() and not args.build_index:
        bm25.load(bm25_path)
    else:
        if not args.json: print("Building BM25 index...")
        bm25.build(chunks)
        bm25.save(bm25_path)

    results = []
    if args.mode == "bm25":
        results = bm25.search(args.query, top_k=args.top_k, filters=filters)
    elif args.mode in ("embedding", "hybrid"):
        emb = EmbeddingIndex()
        if emb_dir.exists() and not args.build_index:
            emb.load(emb_dir)
        else:
            if not args.json: print("Building embedding index...")
            emb.build(chunks)
            emb.save(emb_dir)

        if args.mode == "embedding":
            results = emb.search(args.query, top_k=args.top_k, filters=filters)
        else:
            hybrid = HybridSearcher(bm25, emb, alpha=args.alpha)
            results = hybrid.search(args.query, top_k=args.top_k, filters=filters)

    # Rerank
    if args.rerank and results:
        reranker = CrossEncoderReranker()
        if args.parent_context:
            results = reranker.rerank_with_parent_context(
                args.query, results, chunks, top_k=args.top_k
            )
        else:
            results = reranker.rerank(args.query, results, top_k=args.top_k)

    elapsed = time.time() - t0

    if args.json:
        print(json.dumps({
            "query": args.query,
            "results": results,
            "elapsed": elapsed,
            "total_chunks": len(chunks)
        }, indent=2))
        return

    # Display
    print(f"\n{'═'*60}")
    print(f"  Query: {args.query}")
    print(f"  Mode: {args.mode} | Top-K: {args.top_k} | Rerank: {args.rerank}")
    print(f"  Results: {len(results)} in {elapsed:.2f}s")
    print(f"{'═'*60}")

    for rank, result in enumerate(results, 1):
        display_result(rank, result, verbose=args.verbose)

    print()

if __name__ == "__main__": main()
