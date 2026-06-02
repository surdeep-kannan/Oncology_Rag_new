"""
evaluate_rag.py — Full Academic Evaluation Report
Matches the reference format:
  Retrieval: Precision@5, Recall@5, MRR, NDCG@5, Hit-Rate@5
  Lexical:   BLEU-1/2/4, GLEU, ROUGE-1/2/L, METEOR, Answer-F1
  Semantic:  BERTScore-F1, SBERT-Sim
  Faithfulness: Sentence-SBERT faithfulness, Context Relevancy, Answer Relevance
"""
import json, csv, sys, os, math
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── imports ────────────────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

from medrag.hm_rag import HMRAGPipeline
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag.search.hybrid_search import HybridSearcher
from medrag import config as cfg

# ── helpers ────────────────────────────────────────────────────────────────
def distinct_n(text, n):
    tokens = text.lower().split()
    if len(tokens) < n: return 0.0
    ngrams = set(" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(ngrams) / (len(tokens)-n+1)

def answer_f1(ref, hyp):
    ref_toks = set(ref.lower().split())
    hyp_toks = set(hyp.lower().split())
    common = ref_toks & hyp_toks
    if not common: return 0.0
    p = len(common)/len(hyp_toks)
    r = len(common)/len(ref_toks)
    return 2*p*r/(p+r)

def ndcg_at_k(retrieved_ids, relevant_ids, k):
    dcg = sum(
        1/math.log2(rank+2)
        for rank, cid in enumerate(retrieved_ids[:k])
        if cid in relevant_ids
    )
    ideal = sum(1/math.log2(rank+2) for rank in range(min(len(relevant_ids), k)))
    return dcg/ideal if ideal > 0 else 0.0

def mrr(retrieved_ids, relevant_ids):
    for rank, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1/(rank+1)
    return 0.0

def bertscore_f1(sbert_model, ref, hyp):
    """Approximate BERTScore F1 via SBERT token-level cosine (fast proxy)."""
    ref_toks = ref.split()[:64]
    hyp_toks = hyp.split()[:64]
    if not ref_toks or not hyp_toks: return 0.0
    ref_embs = sbert_model.encode(ref_toks, convert_to_tensor=True)
    hyp_embs = sbert_model.encode(hyp_toks, convert_to_tensor=True)
    sim_matrix = util.cos_sim(hyp_embs, ref_embs)
    precision = sim_matrix.max(dim=1).values.mean().item()
    recall    = sim_matrix.max(dim=0).values.mean().item()
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return round(f1, 4)

def sbert_faithfulness(sbert_model, context_str, answer):
    sentences = [s.strip() for s in answer.replace(".\n",". ").split(". ") if len(s.strip())>20]
    if not sentences or not context_str.strip(): return 0.0
    ctx_emb = sbert_model.encode(context_str[:3000], convert_to_tensor=True)
    ans_embs = sbert_model.encode(sentences, convert_to_tensor=True)
    sims = util.cos_sim(ans_embs, ctx_emb.unsqueeze(0)).squeeze(1).tolist()
    return round(sum(1 for s in sims if s > 0.45)/len(sims), 4)

# ── main ───────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate HM-RAG Pipeline.")
    parser.add_argument("--q", type=str, default=None, help="Evaluate a single question by ID (e.g. q22)")
    parser.add_argument("--index", type=int, default=None, help="Evaluate a single question by 0-based index")
    args = parser.parse_args()

    print("="*66)
    print("  ONCOLOGY RAG — COMPLETE EVALUATION REPORT")
    print("  HM-RAG (HybridSearch + CrossEncoder + SBERT Faithfulness)")
    print("="*66)

    print("\nLoading models...")
    pipeline = HMRAGPipeline()
    sbert = pipeline.sbert_model
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"], use_stemmer=True)

    # Load search stack for retrieval metrics
    bm25 = pipeline.bm25
    emb  = pipeline.emb
    hybrid = pipeline.hybrid

    with open("eval_dataset.json") as f:
        dataset = json.load(f)

    if args.q is not None:
        dataset = [item for item in dataset if item.get("id").strip().lower() == args.q.strip().lower()]
        print(f"Filtered evaluation to question ID: {args.q} ({len(dataset)} items)")
    elif args.index is not None:
        dataset = [dataset[args.index]]
        print(f"Filtered evaluation to index: {args.index} ({len(dataset)} items)")

    all_rows = []
    agg = defaultdict(float)

    for i, item in enumerate(dataset):
        q        = item["question"]
        gt_text  = item["ground_truth"]
        gt_ids   = set(item.get("relevant_chunk_ids", []))

        print(f"\n{'─'*66}")
        print(f"Q{i+1}/{len(dataset)}: {q}")

        # ── Run pipeline ──────────────────────────────────────────────
        res          = pipeline.run(q)
        answer       = res["answer"]
        context      = res["context"]
        ret_ids      = [r.get("chunk_id") for r in context]

        # ── Retrieval metrics ─────────────────────────────────────────
        k = 5
        hits     = [1 if cid in gt_ids else 0 for cid in ret_ids[:k]]
        prec_k   = sum(hits)/k
        recall_k = sum(hits)/len(gt_ids) if gt_ids else 0.0
        hit_rate = 1.0 if any(hits) else 0.0
        ndcg_k   = ndcg_at_k(ret_ids, gt_ids, k)
        mrr_val  = mrr(ret_ids, gt_ids)

        # ── Lexical metrics ───────────────────────────────────────────
        ref_toks = gt_text.lower().split()
        gen_toks = answer.lower().split()
        smoothie = SmoothingFunction().method4

        bleu1 = sentence_bleu([ref_toks], gen_toks, weights=(1,0,0,0), smoothing_function=smoothie)
        bleu2 = sentence_bleu([ref_toks], gen_toks, weights=(.5,.5,0,0), smoothing_function=smoothie)
        bleu4 = sentence_bleu([ref_toks], gen_toks, smoothing_function=smoothie)
        gleu  = sentence_gleu([ref_toks], gen_toks)

        r = scorer.score(gt_text, answer)
        rouge1   = r["rouge1"].fmeasure
        rouge2   = r["rouge2"].fmeasure
        rougeL   = r["rougeL"].fmeasure
        rougeLsum= r["rougeLsum"].fmeasure
        meteor   = meteor_score([ref_toks], gen_toks)
        ans_f1   = answer_f1(gt_text, answer)

        # ── Semantic metrics ──────────────────────────────────────────
        bert_f1  = bertscore_f1(sbert, gt_text, answer)
        emb_gt   = sbert.encode(gt_text, convert_to_tensor=True)
        emb_gen  = sbert.encode(answer, convert_to_tensor=True)
        sbert_sim= util.cos_sim(emb_gt, emb_gen).item()

        # ── Faithfulness & Relevance ──────────────────────────────────
        faith_chunks = [c.get("content","") for c in context if "REFERENCES" not in c.get("heading","").upper()]
        faith_ctx    = " ".join(faith_chunks) if faith_chunks else " ".join(c.get("content","") for c in context)
        faith_score  = sbert_faithfulness(sbert, faith_ctx, answer)

        ctx_str      = " ".join(c.get("content","") for c in context)
        emb_q        = sbert.encode(q, convert_to_tensor=True)
        emb_ctx      = sbert.encode(ctx_str[:2000], convert_to_tensor=True)
        emb_ans      = sbert.encode(answer, convert_to_tensor=True)
        ctx_relevance  = util.cos_sim(emb_q, emb_ctx).item()
        ans_relevance  = util.cos_sim(emb_q, emb_ans).item()

        latency = res["eval_metrics"]["latency_seconds"]

        row = {
            "id": item["id"], "question": q, "latency_s": latency,
            # retrieval
            "precision_5": round(prec_k,4), "recall_5": round(recall_k,4),
            "mrr": round(mrr_val,4), "ndcg_5": round(ndcg_k,4), "hit_rate_5": round(hit_rate,4),
            # lexical
            "bleu_1": round(bleu1,4), "bleu_2": round(bleu2,4), "bleu_4": round(bleu4,4),
            "gleu": round(gleu,4), "rouge_1": round(rouge1,4), "rouge_2": round(rouge2,4),
            "rouge_l": round(rougeL,4), "rouge_lsum": round(rougeLsum,4),
            "meteor": round(meteor,4), "answer_f1": round(ans_f1,4),
            # semantic
            "bertscore_f1": bert_f1, "sbert_sim": round(sbert_sim,4),
            # faithfulness
            "faithfulness": faith_score,
            "context_relevancy": round(ctx_relevance,4),
            "answer_relevance": round(ans_relevance,4),
        }
        all_rows.append(row)

        # accumulate averages
        for k_,v_ in row.items():
            if isinstance(v_, float): agg[k_] += v_

        print(f"  Retrieval   → Prec@5={prec_k:.3f} | Recall@5={recall_k:.3f} | MRR={mrr_val:.3f} | NDCG@5={ndcg_k:.3f} | Hit-Rate={hit_rate:.1f}")
        print(f"  Lexical     → BLEU-1={bleu1:.3f} | ROUGE-L={rougeL:.3f} | METEOR={meteor:.3f} | F1={ans_f1:.3f}")
        print(f"  Semantic    → BERTScore={bert_f1:.3f} | SBERT={sbert_sim:.3f}")
        print(f"  Faithfulness→ {faith_score:.3f} | CtxRel={ctx_relevance:.3f} | AnsRel={ans_relevance:.3f}")
        print(f"  Latency     → {latency:.1f}s")

    # ── Aggregate Report ──────────────────────────────────────────────────
    n = len(all_rows)
    sep = "="*66
    print(f"\n{sep}")
    print("  ONCOLOGY RAG — COMPLETE EVALUATION REPORT")
    print("  HM-RAG (HybridSearch + CrossEncoder + SBERT Faithfulness)")
    print(sep)
    print(f"  Questions evaluated : {n}")
    print(f"\n  -- Retrieval Quality (k=5) {'-'*38}")
    print(f"  Precision@5        : {agg['precision_5']/n:.4f}")
    print(f"  Recall@5           : {agg['recall_5']/n:.4f}")
    print(f"  MRR                : {agg['mrr']/n:.4f}")
    print(f"  NDCG@5             : {agg['ndcg_5']/n:.4f}")
    print(f"  Hit-Rate@5         : {agg['hit_rate_5']/n:.4f}")
    print(f"\n  -- Generation Lexical {'-'*43}")
    print(f"  BLEU-1             : {agg['bleu_1']/n:.4f}")
    print(f"  BLEU-2             : {agg['bleu_2']/n:.4f}")
    print(f"  BLEU-4             : {agg['bleu_4']/n:.4f}")
    print(f"  GLEU               : {agg['gleu']/n:.4f}")
    print(f"  ROUGE-1            : {agg['rouge_1']/n:.4f}")
    print(f"  ROUGE-2            : {agg['rouge_2']/n:.4f}")
    print(f"  ROUGE-L            : {agg['rouge_l']/n:.4f}")
    print(f"  ROUGE-Lsum         : {agg['rouge_lsum']/n:.4f}")
    print(f"  METEOR             : {agg['meteor']/n:.4f}")
    print(f"  Answer F1          : {agg['answer_f1']/n:.4f}")
    print(f"\n  -- Generation Semantic {'-'*42}")
    print(f"  BERTScore F1       : {agg['bertscore_f1']/n:.4f}")
    print(f"  SBERT Sim          : {agg['sbert_sim']/n:.4f}")
    print(f"\n  -- Faithfulness & Relevance {'-'*37}")
    print(f"  Faithfulness(SBERT): {agg['faithfulness']/n:.4f}")
    print(f"  Context Relevancy  : {agg['context_relevancy']/n:.4f}")
    print(f"  Answer Relevance   : {agg['answer_relevance']/n:.4f}")
    print(f"\n  Avg Latency        : {agg['latency_s']/n:.1f}s")
    print(sep)

    # save CSV
    csv_path = "academic_evaluation_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n✅ Full report saved → {csv_path}")

if __name__ == "__main__":
    main()
