"""
compare_generators_and_ablation.py — Compares 6 medical generator models (llama3med42,
medgemma, medalpaca, meditron, pmcllama, biomistral) and performs a live empirical 
ablation study on the best model (Llama3-Med42-8B) using the evaluation dataset.
"""

import sys
import os
import time
import json
import csv
import torch
from pathlib import Path
from collections import defaultdict

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.hm_rag import HMRAGPipeline
from medrag import config as cfg
from sentence_transformers import util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def answer_f1(ref, hyp):
    ref_toks = set(ref.lower().split())
    hyp_toks = set(hyp.lower().split())
    common = ref_toks & hyp_toks
    if not common: return 0.0
    p = len(common)/len(hyp_toks)
    r = len(common)/len(ref_toks)
    return 2*p*r/(p+r)

def sbert_faithfulness(sbert_model, context_str, answer):
    sentences = [s.strip() for s in answer.replace(".\n",". ").split(". ") if len(s.strip())>20]
    if not sentences or not context_str.strip(): return 0.0
    ctx_emb = sbert_model.encode(context_str[:3000], convert_to_tensor=True)
    ans_embs = sbert_model.encode(sentences, convert_to_tensor=True)
    sims = util.cos_sim(ans_embs, ctx_emb.unsqueeze(0)).squeeze(1).tolist()
    if not sims: return 0.0
    return round(sum(1 for s in sims if s > 0.45)/len(sims), 4)

def run_ablation_study(pipeline, sbert, rouge, dataset, num_questions=3):
    print("\n" + "="*70)
    print(f"  🚀 RUNNING LIVE ABLATION STUDY ON LLAMA3-MED42-8B ({num_questions} Questions)")
    print("="*70)
    
    # Selected questions for efficient CPU ablation execution
    eval_questions = dataset[:num_questions]
    
    configs = {
        "1_full_rag": {
            "name": "Full RAG Pipeline",
            "decomp": True,
            "rerank": True,
            "parent": True,
            "context": True
        },
        "2_no_decomp": {
            "name": "Ablation: No Query Decomposition",
            "decomp": False,
            "rerank": True,
            "parent": True,
            "context": True
        },
        "3_no_rerank": {
            "name": "Ablation: No Cross-Encoder Reranking",
            "decomp": True,
            "rerank": False,
            "parent": True,
            "context": True
        },
        "4_no_parent": {
            "name": "Ablation: No Parent Context (Child only)",
            "decomp": True,
            "rerank": True,
            "parent": False,
            "context": True
        },
        "5_direct_llm": {
            "name": "Ablation: Direct LLM (No RAG/Context)",
            "decomp": False,
            "rerank": False,
            "parent": False,
            "context": False
        }
    }
    
    ablation_results = []
    
    for cfg_key, cfg_info in configs.items():
        print(f"\nEvaluating Configuration: {cfg_info['name']}")
        print("-" * 50)
        
        running_latency = 0.0
        running_bleu = 0.0
        running_rouge = 0.0
        running_sbert = 0.0
        running_faith = 0.0
        running_f1 = 0.0
        
        for q_idx, item in enumerate(eval_questions):
            query = item["question"]
            gt = item["ground_truth"]
            print(f"  Q{q_idx+1}: {query[:60]}...")
            
            start_time = time.time()
            
            # --- 1. Query Decomposition Stage ---
            if cfg_info["decomp"]:
                sub_queries = pipeline.decompose_query(query)
                search_queries = [query] + [sq for sq in sub_queries if sq != query]
            else:
                search_queries = [query]
                
            # --- 2. Retrieval & Reranking Stage ---
            if not cfg_info["context"]:
                context = []
            else:
                if cfg_info["rerank"]:
                    context = pipeline.retrieve_context(search_queries)
                else:
                    # Raw hybrid search without reranking
                    context = []
                    seen_chunks = set()
                    for sq in search_queries:
                        raw_results = pipeline.hybrid.search(sq, top_k=5)
                        for r in raw_results:
                            cid = f"{r.get('source_file', 'unknown')}_{r.get('chunk_id', '')}"
                            if cid not in seen_chunks:
                                seen_chunks.add(cid)
                                context.append(r)
                    context = context[:2] # Top 2 chunks to mirror the rerank top-k
            
            # --- 3. Parent Context Stage ---
            context_to_send = []
            for c in context:
                c_copy = c.copy()
                if not cfg_info["parent"] and "_parent_context" in c_copy:
                    del c_copy["_parent_context"]
                context_to_send.append(c_copy)
                
            # --- 4. Synthesis Stage ---
            if not cfg_info["context"]:
                # Direct LLM generation without context
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional medical assistant. Answer the user's question directly using your medical knowledge.
Output ONLY the direct medical answer. Do not include introductory remarks.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                answer = pipeline.llm.generate(prompt)
            else:
                answer = pipeline.synthesize_answer(query, context_to_send)
                
            latency = time.time() - start_time
            
            # --- 5. Metric Calculations ---
            # Lexical Metrics
            ref_toks = gt.lower().split()
            gen_toks = answer.lower().split()
            smoothie = SmoothingFunction().method4
            bleu = sentence_bleu([ref_toks], gen_toks, smoothing_function=smoothie)
            
            rouge_scores = rouge.score(gt, answer)
            rouge_l = rouge_scores['rougeL'].fmeasure
            ans_f1_score = answer_f1(gt, answer)
            
            # Semantic Similarity (SBERT to Ground Truth)
            emb_gt = sbert.encode(gt, convert_to_tensor=True)
            emb_gen = sbert.encode(answer, convert_to_tensor=True)
            sbert_sim = float(util.cos_sim(emb_gt, emb_gen).item())
            
            # Faithfulness
            context_str = " ".join([c.get("content", "") for c in context_to_send])
            faith = sbert_faithfulness(sbert, context_str, answer)
            
            # Print individual question stats
            print(f"    - Latency: {latency:.2f}s | SBERT Sim: {sbert_sim:.4f} | Faithfulness: {faith:.4f} | ROUGE-L: {rouge_l:.4f}")
            
            running_latency += latency
            running_bleu += bleu
            running_rouge += rouge_l
            running_sbert += sbert_sim
            running_faith += faith
            running_f1 += ans_f1_score
            
        # Averages
        avg_latency = running_latency / num_questions
        avg_bleu = running_bleu / num_questions
        avg_rouge = running_rouge / num_questions
        avg_sbert = running_sbert / num_questions
        avg_faith = running_faith / num_questions
        avg_f1 = running_f1 / num_questions
        
        print(f"📊 [{cfg_info['name']}] AVERAGES:")
        print(f"   Latency: {avg_latency:.2f}s | BLEU: {avg_bleu:.4f} | ROUGE-L: {avg_rouge:.4f} | SBERT Sim: {avg_sbert:.4f} | Faithfulness: {avg_faith:.4f} | Answer F1: {avg_f1:.4f}")
        
        ablation_results.append({
            "config_id": cfg_key,
            "config_name": cfg_info["name"],
            "avg_latency_s": round(avg_latency, 2),
            "avg_bleu": round(avg_bleu, 4),
            "avg_rouge_l": round(avg_rouge, 4),
            "avg_sbert_sim": round(avg_sbert, 4),
            "avg_faithfulness": round(avg_faith, 4),
            "avg_answer_f1": round(avg_f1, 4)
        })
        
    # Save Ablation Results to CSV
    csv_path = PROJECT_ROOT / "ablation_study_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ablation_results[0].keys())
        writer.writeheader()
        writer.writerows(ablation_results)
    print(f"\n✅ Ablation Study Results saved successfully to {csv_path}")
    
    return ablation_results

def main():
    print("="*70)
    print("  ONCOLOGY RAG GENERATOR BENCHMARK & ABLATION STUDY")
    print("="*70)
    
    # Load pipeline
    print("Loading pipeline models (Llama3-Med42-8B)...")
    pipeline = HMRAGPipeline()
    sbert = pipeline.sbert_model
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Load evaluation dataset
    eval_path = PROJECT_ROOT / "eval_dataset.json"
    with open(eval_path) as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} evaluation questions.")
    
    # Run the ablation study on first 3 questions
    ablation_results = run_ablation_study(pipeline, sbert, rouge, dataset, num_questions=3)
    
    # We will output a summary to stdout, which we can parse to write the artifact
    print("\n" + "="*70)
    print("  📊 SUMMARY OF EXPERIMENTAL ABLATION RESULTS")
    print("="*70)
    print(f"{'Configuration':<38} | {'Latency':<7} | {'SBERT Sim':<9} | {'Faithfulness':<12} | {'ROUGE-L':<7}")
    print("-" * 75)
    for res in ablation_results:
        print(f"{res['config_name']:<38} | {res['avg_latency_s']:<7.2f} | {res['avg_sbert_sim']:<9.4f} | {res['avg_faithfulness']:<12.4f} | {res['avg_rouge_l']:<7.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
