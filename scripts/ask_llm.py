#!/usr/bin/env python3
"""ask_llm.py – Full RAG pipeline: search for context and generate answer with Med42.

Usage:
    python scripts/ask_llm.py "What are the common side effects of pembrolizumab?"
"""
import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag.search.hybrid_search import HybridSearcher
from medrag.search.reranker import CrossEncoderReranker
from medrag.llm import MedLLM

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Ask a medical question to the RAG system.")
    p.add_argument("query", type=str, help="The medical question.")
    p.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve.")
    p.add_argument("--no-llm", action="store_true", help="Only search, don't generate (for testing).")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging(level="INFO")

    # 1. Load Chunks
    master_path = PROJECT_ROOT / "master_chunks.jsonl"
    if not master_path.exists():
        print(f"❌ Error: {master_path} not found.")
        return

    chunks = []
    with open(master_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # 2. Search for context
    idx_dir = cfg.index_dir()
    bm25 = BM25Index()
    bm25.load(idx_dir / "bm25_index.pkl")
    
    emb = EmbeddingIndex()
    emb.load(idx_dir / "embeddings")
    
    hybrid = HybridSearcher(bm25, emb)
    results = hybrid.search(args.query, top_k=args.top_k * 2) # Get more for reranking
    
    # Rerank
    reranker = CrossEncoderReranker()
    results = reranker.rerank(args.query, results, top_k=args.top_k)

    print(f"\n🔍 Found {len(results)} relevant context blocks.")

    if args.no_llm:
        for i, res in enumerate(results, 1):
            print(f"\n[{i}] {res.get('heading')} (Source: {res.get('source_file')})")
            print(f"    {res.get('content')[:200]}...")
        return

    # 3. Generate Answer
    print("🧠 Loading Med42 LLM and generating answer (this may take a moment)...")
    llm = MedLLM()
    prompt = llm.format_rag_prompt(args.query, results)
    
    answer = llm.generate(prompt)
    
    print("\n" + "═"*60)
    print("🤖 MED42 RESPONSE:")
    print("═"*60)
    print(answer)
    print("═"*60 + "\n")

if __name__ == "__main__":
    main()
