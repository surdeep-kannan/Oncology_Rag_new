import sys
import json
from pathlib import Path
from medrag.hm_rag import HMRAGPipeline

def main():
    print("Initializing HMRAGPipeline...")
    pipeline = HMRAGPipeline()
    
    query = "What is the recommended treatment approach for a patient presenting with early-stage, HER2-positive breast cancer?"
    print(f"\nRunning pipeline on query: '{query}'")
    
    res = pipeline.run(query)
    
    print("\nDecomposed Sub-Queries:")
    for sq in res.get("sub_queries", []):
        print(f"- {sq}")
        
    print("\nRetrieved Context Sources:")
    for i, r in enumerate(res.get("context", []), 1):
        print(f"\n{i}. Heading: {r.get('heading')} | File: {r.get('source_file')} | Level2: {r.get('level2')} | Has parent context? {'_parent_context' in r}")
        print(f"   Content Preview: {r.get('content')[:300]}...")
        
    print("\nGenerated Answer:")
    print(res.get("answer"))
    
    print("\nEvaluation Metrics:")
    for k, v in res.get("eval_metrics", {}).items():
        print(f"- {k}: {v}")

if __name__ == "__main__":
    main()
