import sys
import time
from pathlib import Path
from medrag.hm_rag import HMRAGPipeline
from medrag.llm import clean_source_name

def run_comparison():
    print("=============================================================")
    print("  LIVE SIDE-BY-SIDE RAG COMPARISON: BASELINE VS OPTIMIZED RL")
    print("=============================================================")

    # Initialize Pipeline
    pipeline = HMRAGPipeline()

    query = "When is surgery preferred over radiation in localized prostate cancer according to standard clinical guidelines?"
    print(f"\nTarget Query: '{query}'\n")

    # -------------------------------------------------------------------------
    # CONFIGURATION 1: BASELINE (WITHOUT RL)
    # -------------------------------------------------------------------------
    print("--- 🔴 RUNNING BASELINE (WITHOUT RL) ---")
    
    # Baseline defaults
    baseline_top_k = 8
    baseline_reranker_top_n = 2
    pipeline.hybrid.alpha = 0.5
    pipeline.hybrid.rrf_k = 60
    
    # Generic unoptimized prompt
    baseline_prompt = {
        "system": "You are a helpful assistant. Answer the question using the provided context.",
        "user_template": "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    }

    start_time = time.time()
    sub_queries = pipeline.decompose_query(query)
    search_queries = [query] + [sq for sq in sub_queries if sq != query]
    
    # Retrieval
    baseline_context = pipeline.retrieve_context(search_queries, top_k=baseline_top_k, reranker_top_n=baseline_reranker_top_n)
    
    # Synthesis
    baseline_answer = pipeline.synthesize_answer(query, baseline_context, prompt_override=baseline_prompt)
    baseline_latency = time.time() - start_time

    # Evaluate
    # Compute semantic SBERT similarity
    emb_query = pipeline.sbert_model.encode(query, convert_to_tensor=True)
    emb_ans = pipeline.sbert_model.encode(baseline_answer, convert_to_tensor=True)
    import torch
    sbert_sim = float(torch.nn.functional.cosine_similarity(emb_query, emb_ans, dim=0).cpu().item())

    print(f"\n[Baseline Prompt Used]: Helpful General Assistant")
    print(f"[Baseline Answer Generated]:\n{baseline_answer}")
    print(f"\n[Baseline Metrics]:")
    print(f"  - Latency: {baseline_latency:.1f}s")
    print(f"  - SBERT Semantic Relevance: {sbert_sim:.4f}")
    print(f"  - Number of Sources Retrieved: {len(baseline_context)}")

    # -------------------------------------------------------------------------
    # CONFIGURATION 2: CHAMPION (WITH RL OPTIMIZED)
    # -------------------------------------------------------------------------
    print("\n--- 🟢 RUNNING OPTIMIZED CHAMPION (WITH RL) ---")
    
    # Learned parameters loaded automatically by HMRAGPipeline constructor
    # We explicitly read them here to verify
    rl_top_k = pipeline._rl_params.get("top_k", 12) if pipeline._rl_params else 12
    rl_reranker_top_n = pipeline._rl_params.get("reranker_top_n", 2) if pipeline._rl_params else 2
    
    if pipeline._rl_params:
        pipeline.hybrid.alpha = pipeline._rl_params.get("alpha", 0.46)
        pipeline.hybrid.rrf_k = pipeline._rl_params.get("rrf_k", 40)
    
    rl_prompt = pipeline._rl_prompt

    start_time = time.time()
    sub_queries_rl = pipeline.decompose_query(query)
    search_queries_rl = [query] + [sq for sq in sub_queries_rl if sq != query]
    
    # Retrieval
    rl_context = pipeline.retrieve_context(search_queries_rl, top_k=rl_top_k, reranker_top_n=rl_reranker_top_n)
    
    # Synthesis
    rl_answer = pipeline.synthesize_answer(query, rl_context, prompt_override=rl_prompt)
    rl_latency = time.time() - start_time

    # Evaluate
    emb_ans_rl = pipeline.sbert_model.encode(rl_answer, convert_to_tensor=True)
    sbert_sim_rl = float(torch.nn.functional.cosine_similarity(emb_query, emb_ans_rl, dim=0).cpu().item())

    print(f"\n[RL Prompt Used]: {pipeline._rl_prompt.get('system')[:80]}...")
    print(f"[RL Answer Generated]:\n{rl_answer}")
    print(f"\n[RL Metrics]:")
    print(f"  - Latency: {rl_latency:.1f}s")
    print(f"  - SBERT Semantic Relevance: {sbert_sim_rl:.4f}")
    print(f"  - Number of Sources Retrieved: {len(rl_context)}")
    print("=============================================================")

if __name__ == "__main__":
    run_comparison()
