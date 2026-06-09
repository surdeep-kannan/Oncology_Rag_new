import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Import local modules
from medrag.search.embedding_search import EmbeddingIndex

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 standalone_rag.py 'Your medical question'")
        sys.exit(1)
        
    query = sys.argv[1]
    
    # 1. Load Embeddings
    emb_dir = Path("output/indices/embeddings")
    if not emb_dir.exists():
        print(f"Error: Embeddings not found at {emb_dir}")
        sys.exit(1)
        
    print(f"Loading embedding index from {emb_dir}...")
    emb_index = EmbeddingIndex()
    emb_index.load(emb_dir)
    
    # 2. Search
    print(f"\nSearching for: '{query}'...")
    results = emb_index.search(query, top_k=3)
    
    if not results:
        print("No results found in knowledge base.")
        sys.exit(1)
        
    print(f"Found {len(results)} chunks. Loading Llama3-Med42-8B GGUF model for CPU...")
    
    # 3. Load Llama 3 GGUF version with extreme memory optimizations
    gguf_path = "/home/surdeep/.cache/huggingface/hub/models--mradermacher--Llama3-Med42-8B-GGUF/snapshots/7e2883406aaaee888cefbba8a50420062b484fee/Llama3-Med42-8B.Q8_0.gguf"
    
    llm = Llama(
        model_path=gguf_path,
        n_ctx=2048,          # Context window
        n_threads=4,         # Number of CPU threads
        n_batch=128,
        verbose=False
    )
    
    # 4. Generate Answer
    print("\nPreparing prompt...")
    context_str = "\n\n".join([f"Source [{i+1}]: {c.get('content', '')}" for i, c in enumerate(results)])
    
    prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a professional medical assistant. Use the following retrieved context to answer the user's question. 
If the answer is not in the context, say you don't know based on the provided documents.

### Context:
{context_str}<|eot_id|><|start_header_id|>user<|end_header_id|>

{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""

    print("Generating answer...\n" + "="*50)
    
    response = llm(
        prompt,
        max_tokens=512,
        stop=["<|eot_id|>"],
        echo=False
    )
    answer = response["choices"][0]["text"].strip()

    print(answer)
    print("\n" + "="*50)
    
    print("\nSources used:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res.get('source_file')} - {res.get('heading', 'Unknown Heading')}")

if __name__ == "__main__":
    main()
