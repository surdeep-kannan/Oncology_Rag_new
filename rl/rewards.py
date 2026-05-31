"""
rewards.py — Reward function for RL-based RAG optimization.

The reward is a weighted combination of generation quality metrics.
This serves as the "environment feedback" that tells the RL agent
whether its chosen action (prompt, parameters) was good or bad.

Reward Formula:
  R = w1*SBERT_sim + w2*faithfulness + w3*ROUGE_L + w4*speed_bonus + w5*ctx_relevancy

Each component is normalized to [0, 1]. The weights sum to 1.0.
"""

from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer


class RewardComputer:
    """Computes a scalar reward from RAG pipeline outputs."""

    def __init__(self, sbert_model=None):
        self.sbert = sbert_model or SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        # Reward weights — these define what "good" means
        self.weights = {
            "sbert_sim": 0.35,        # semantic similarity to ground truth
            "faithfulness": 0.25,      # answer grounded in retrieved context
            "rouge_l": 0.20,           # lexical overlap with ground truth
            "speed_bonus": 0.10,       # faster = better (normalized)
            "ctx_relevancy": 0.10,     # retrieved context matches query
        }
        self.max_latency = 300.0  # seconds; anything slower gets 0 speed bonus

    def compute_reward(self, answer: str, ground_truth: str,
                       context_chunks: list[dict], query: str,
                       latency: float) -> dict:
        """
        Compute the RL reward signal.

        Args:
            answer:          Generated answer from the RAG pipeline
            ground_truth:    Reference answer (from eval_dataset.json)
            context_chunks:  Retrieved chunks used to generate the answer
            query:           Original user query
            latency:         Time taken in seconds

        Returns:
            dict with individual scores and total reward
        """
        # 1. SBERT Similarity (answer vs ground truth)
        emb_gt = self.sbert.encode(ground_truth, convert_to_tensor=True)
        emb_ans = self.sbert.encode(answer, convert_to_tensor=True)
        sbert_sim = util.cos_sim(emb_gt, emb_ans).item()
        sbert_sim = max(0.0, sbert_sim)  # clamp negatives

        # 2. Faithfulness (sentence-level SBERT)
        faith_chunks = [c.get("content", "") for c in context_chunks
                        if "REFERENCES" not in c.get("heading", "").upper()]
        ctx_text = " ".join(faith_chunks) if faith_chunks else ""
        faithfulness = self._sentence_faithfulness(answer, ctx_text)

        # 3. ROUGE-L
        rouge_scores = self.rouge.score(ground_truth, answer)
        rouge_l = rouge_scores["rougeL"].fmeasure

        # 4. Speed bonus (linear decay from 1.0 at 0s to 0.0 at max_latency)
        speed_bonus = max(0.0, 1.0 - latency / self.max_latency)

        # 5. Context relevancy (query vs retrieved context)
        if ctx_text.strip():
            emb_q = self.sbert.encode(query, convert_to_tensor=True)
            emb_ctx = self.sbert.encode(ctx_text[:2000], convert_to_tensor=True)
            ctx_relevancy = max(0.0, util.cos_sim(emb_q, emb_ctx).item())
        else:
            ctx_relevancy = 0.0

        # Weighted total reward
        scores = {
            "sbert_sim": sbert_sim,
            "faithfulness": faithfulness,
            "rouge_l": rouge_l,
            "speed_bonus": speed_bonus,
            "ctx_relevancy": ctx_relevancy,
        }
        total = sum(self.weights[k] * scores[k] for k in self.weights)

        return {**scores, "total_reward": round(total, 4)}

    def _sentence_faithfulness(self, answer: str, context: str) -> float:
        """Check what fraction of answer sentences are supported by context."""
        sentences = [s.strip() for s in answer.replace(".\n", ". ").split(". ")
                     if len(s.strip()) > 20]
        if not sentences or not context.strip():
            return 0.0

        ctx_emb = self.sbert.encode(context[:3000], convert_to_tensor=True)
        ans_embs = self.sbert.encode(sentences, convert_to_tensor=True)
        sims = util.cos_sim(ans_embs, ctx_emb.unsqueeze(0)).squeeze(1).tolist()
        supported = sum(1 for s in sims if s > 0.45)
        return supported / len(sims)


def compute_reward(answer, ground_truth, context_chunks, query, latency,
                   sbert_model=None):
    """Convenience function for one-shot reward computation."""
    rc = RewardComputer(sbert_model)
    return rc.compute_reward(answer, ground_truth, context_chunks, query, latency)
