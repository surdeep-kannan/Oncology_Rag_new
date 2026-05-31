import sys
import json
from pathlib import Path
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag import config as cfg

def main():
    print("Loading indexes...")
    bm25 = BM25Index()
    bm25.load(cfg.index_dir() / "bm25_index.pkl")
    emb = EmbeddingIndex()
    emb.load(cfg.index_dir() / "embeddings")
    
    query = "What are the recommended first-line chemotherapy regimens for advanced non-small cell lung cancer?"
    print(f"\nAnalyzing ranks for original query: '{query}'")
    
    bm25_results = bm25.search(query, top_k=2000)
    emb_results = emb.search(query, top_k=2000)
    
    bm25_match_idx = -1
    for rank, r in enumerate(bm25_results):
        if r.get("level2") == "Frontline Chemotherapy for Advanced Non-Small Cell Lung Cancer":
            bm25_match_idx = rank
            print(f"  BM25 match rank: {rank} (score: {r.get('_bm25_score'):.4f})")
            break
            
    emb_match_idx = -1
    for rank, r in enumerate(emb_results):
        if r.get("level2") == "Frontline Chemotherapy for Advanced Non-Small Cell Lung Cancer":
            emb_match_idx = rank
            print(f"  Embedding match rank: {rank} (score: {r.get('_embedding_score'):.4f})")
            break
            
    if bm25_match_idx == -1:
        print("  Not found in top 2000 BM25 results!")
    if emb_match_idx == -1:
        print("  Not found in top 2000 Embedding results!")

if __name__ == "__main__":
    main()
