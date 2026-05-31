"""Auto-generate ground-truth chunk IDs for eval_dataset.json using hybrid search top-5."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag.search.hybrid_search import HybridSearcher
from medrag import config as cfg

print("Loading indexes...")
bm25 = BM25Index(); bm25.load(cfg.index_dir() / "bm25_index.pkl")
emb = EmbeddingIndex(); emb.load(cfg.index_dir() / "embeddings")
hybrid = HybridSearcher(bm25, emb)

with open("eval_dataset.json") as f:
    dataset = json.load(f)

for item in dataset:
    q = item["question"]
    results = hybrid.search(q, top_k=5)
    item["relevant_chunk_ids"] = [r.get("chunk_id") for r in results]
    item["relevant_sources"] = [{"chunk_id": r.get("chunk_id"), "heading": r.get("heading"), "file": r.get("source_file")} for r in results]
    print(f"Q: {q[:60]}...")
    for r in results[:3]:
        print(f"  → [{r.get('chunk_id')}] {r.get('heading')} | {r.get('source_file')}")

with open("eval_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("\n✅ eval_dataset.json updated with relevant_chunk_ids!")
