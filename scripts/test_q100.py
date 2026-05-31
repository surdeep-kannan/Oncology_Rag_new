import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.hm_rag import HMRAGPipeline

def main():
    print("Initializing RAG pipeline...")
    pipeline = HMRAGPipeline()
    
    query = "What is the primary treatment for Stage IVC laryngeal cancer?"
    print(f"\nRunning RAG pipeline for Query: '{query}'")
    
    # Run the pipeline
    result = pipeline.run(query)
    
    print("\n" + "="*60)
    # Output results
    print(f"Generated Answer:\n{result['answer']}")
    print("="*60)
    print("\nRetrieved Context Chunks:")
    for idx, c in enumerate(result['context'], 1):
        print(f"\n[{idx}] File: {c.get('source_file')}, Heading: {c.get('heading')}")
        print(f"Content: {c.get('content')[:300]}...")

if __name__ == "__main__":
    main()
