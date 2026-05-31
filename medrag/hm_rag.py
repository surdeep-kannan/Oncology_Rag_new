"""
hm_rag.py — RL-Enhanced Medical RAG Pipeline.

This version supports RL-controlled parameters:
  - prompt_override: Custom system/user prompt templates (from PromptBandit)
  - top_k: Number of chunks to retrieve (from RetrievalOptimizer)
  - reranker_top_n: Chunks per sub-query after reranking
  - Dynamic alpha/rrf_k on the HybridSearcher

When called without RL overrides, it uses sensible defaults.
"""

import logging
import json
import time
from pathlib import Path
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from medrag.llm import MedLLM
from medrag.search.hybrid_search import HybridSearcher
from medrag.search.bm25_search import BM25Index
from medrag.search.embedding_search import EmbeddingIndex
from medrag.search.reranker import CrossEncoderReranker
from medrag import config as cfg

logger = logging.getLogger(__name__)


def clean_source_name(filename: str) -> str:
    """Convert raw PDF filename into a clean, human-readable book title."""
    if not filename:
        return "Unknown Reference"
    name = filename.split('/')[-1]
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    name = name.replace('-', ' ').replace('_', ' ')
    
    # Capitalize properly
    words = name.split()
    clean_words = []
    for w in words:
        if w.lower() in ['of', 'and', 'the', 'for', 'in', 'to', 'with', 'on', 'at', 'by', 'from', 'an']:
            clean_words.append(w.lower())
        else:
            clean_words.append(w.capitalize())
    
    clean_name = " ".join(clean_words)
    if clean_name:
        clean_name = clean_name[0].upper() + clean_name[1:]
    return clean_name


class HMRAGPipeline:
    def __init__(self):
        self.llm = MedLLM()
        self.bm25 = BM25Index()
        self.bm25.load(cfg.index_dir() / "bm25_index.pkl")
        self.emb = EmbeddingIndex()
        self.emb.load(cfg.index_dir() / "embeddings")
        self.hybrid = HybridSearcher(self.bm25, self.emb)
        self.reranker = CrossEncoderReranker()
        print("Loading SBERT for live evaluation...")
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Try to load RL-optimized settings
        self._rl_prompt = None
        self._rl_params = None
        self._load_rl_state()

    def _load_rl_state(self):
        """Load learned RL parameters if available."""
        try:
            rl_state = Path("rl/rl_state.json")
            if rl_state.exists():
                from rl.prompt_bandit import PromptBandit
                bandit = PromptBandit(state_path=str(rl_state))
                _, best_template = bandit.get_best_arm()
                self._rl_prompt = {
                    "system": best_template["system"],
                    "user_template": best_template["user_template"],
                }
                logger.info(f"Loaded RL-optimized prompt: {best_template['name']}")

            ret_state = Path("rl/retrieval_state.json")
            if ret_state.exists():
                from rl.retrieval_optimizer import RetrievalOptimizer
                opt = RetrievalOptimizer(state_path=str(ret_state))
                self._rl_params = opt.get_best_params()
                logger.info(f"Loaded RL-optimized retrieval params: {self._rl_params}")
        except Exception as e:
            logger.warning(f"Could not load RL state: {e}")

    def decompose_query(self, query: str) -> list[str]:
        prompt = f"""<|start_header_id|>system<|end_header_id|>
You are an expert medical search decomposition agent. Break down the complex medical query into 2-3 distinct, concise sub-queries to maximize retrieval of relevant textbook data.
Output ONLY a JSON array of strings, for example: ["sub-query 1", "sub-query 2"]
Do not output anything else.<|eot_id|><|start_header_id|>user<|end_header_id|>
Query: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        try:
            response = self.llm.generate(prompt, max_new_tokens=100)
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != -1:
                sub_queries = json.loads(response[start:end])
                if isinstance(sub_queries, list) and len(sub_queries) > 0:
                    return sub_queries
        except Exception as e:
            logger.error(f"Decomposition failed: {e}")

        return [query]

    def retrieve_context(self, sub_queries: list[str],
                         top_k: int = 25, reranker_top_n: int = 2) -> list[dict]:
        """Retrieve and rerank context chunks."""
        # Strict context window safety budget:
        # Stop adding chunks once total character count exceeds 24,000 (~6,000 tokens)
        budget_char_limit = 24000
        current_chars = 0
        all_results = []
        seen_chunks = set()

        for sq in sub_queries:
            raw_results = self.hybrid.search(sq, top_k=top_k)
            results = self.reranker.rerank_with_parent_context(
                sq, raw_results, self.bm25._chunks, top_k=reranker_top_n
            )
            for r in results:
                cid = f"{r.get('source_file', 'unknown')}_{r.get('chunk_id', '')}"
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    
                    # Estimate token size by characters
                    chunk_len = len(r.get('content', '')) + len(r.get('_parent_context', ''))
                    if current_chars + chunk_len > budget_char_limit:
                        logger.warning(f"Exceeded context budget limit. Skipping remaining chunks to prevent OOM/crash.")
                        break
                    
                    all_results.append(r)
                    current_chars += chunk_len

        return all_results

    def synthesize_answer(self, query: str, context_chunks: list[dict],
                          prompt_override: dict = None) -> str:
        """
        Generate answer using either RL-optimized prompt or default.

        Args:
            prompt_override: {"system": ..., "user_template": ...}
                             user_template should have {context} and {query} placeholders
        """
        if prompt_override:
            # Build custom prompt from RL template
            context_text = ""
            for i, chunk in enumerate(context_chunks, 1):
                book_name = clean_source_name(chunk.get('source_file', ''))
                context_text += f"[Source {i}: {book_name}]: {chunk.get('content', '')}\n\n"

            user_msg = prompt_override["user_template"].format(
                context=context_text, query=query
            )

            prompt = f"""<|start_header_id|>system<|end_header_id|>
{prompt_override['system']}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

            return self.llm.generate(prompt)
        else:
            # Default prompt
            prompt = self.llm.format_rag_prompt(query, context_chunks)
            return self.llm.generate(prompt)

    def run(self, query: str, progress_callback=None,
            prompt_override: dict = None, top_k: int = None,
            reranker_top_n: int = None, use_rl: bool = True):
        """
        Run the RAG pipeline.

        If RL state exists and no explicit overrides given, uses learned params.
        During RL training, overrides are passed explicitly.
        """
        start_time = time.time()

        if not use_rl:
            # Force unoptimized baseline settings
            prompt_override = None
            top_k = 8
            reranker_top_n = 2
            self.hybrid.alpha = 0.5
            self.hybrid.rrf_k = 60
        else:
            # Use RL-learned params as defaults if available
            if prompt_override is None and self._rl_prompt is not None:
                prompt_override = self._rl_prompt
            if top_k is None:
                top_k = self._rl_params.get("top_k", 8) if self._rl_params else 8
            if reranker_top_n is None:
                reranker_top_n = self._rl_params.get("reranker_top_n", 2) if self._rl_params else 2

            # Apply RL retrieval params
            if self._rl_params:
                self.hybrid.alpha = self._rl_params.get("alpha", 0.5)
                self.hybrid.rrf_k = self._rl_params.get("rrf_k", 60)

        if progress_callback:
            progress_callback("Decomposing query...")
        sub_queries = self.decompose_query(query)
        search_queries = [query] + [sq for sq in sub_queries if sq != query]

        if progress_callback:
            progress_callback(f"Retrieving for queries: {', '.join(search_queries)}")
        context = self.retrieve_context(search_queries, top_k=top_k,
                                         reranker_top_n=reranker_top_n)

        if progress_callback:
            progress_callback("Synthesizing final answer...")
        answer = self.synthesize_answer(query, context,
                                         prompt_override=prompt_override)

        end_time = time.time()

        # ── Evaluation Metrics ────────────────────────────────────────
        avg_hybrid_score = 0.0
        if context:
            scores = [c.get('_hybrid_score', 0) for c in context]
            avg_hybrid_score = sum(scores) / len(scores) if scores else 0.0

        context_str = "\n".join([c.get("content", "") for c in context])
        ref_tokens = context_str.lower().split()
        gen_tokens = answer.lower().split()

        def get_dist(n):
            if len(gen_tokens) < n: return 0.0
            ngrams = set([" ".join(gen_tokens[i:i+n]) for i in range(len(gen_tokens)-n+1)])
            return len(ngrams) / (len(gen_tokens) - n + 1)

        dist1, dist2 = get_dist(1), get_dist(2)

        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu([ref_tokens], gen_tokens,
                             smoothing_function=smoothie) if ref_tokens else 0.0

        rouge_scores = self.rouge.score(context_str, answer)
        rouge_l = rouge_scores['rougeL'].fmeasure

        emb_gt = self.sbert_model.encode(context_str[:2000])
        emb_gen = self.sbert_model.encode(answer)
        sbert_sim = util.cos_sim(emb_gt, emb_gen).item()

        # Faithfulness
        answer_sentences = [s.strip() for s in answer.replace(".\n", ". ").split(". ")
                            if len(s.strip()) > 20]
        faith_chunks = [c.get("content", "") for c in context
                        if "REFERENCES" not in c.get("heading", "").upper()]
        faith_context_str = " ".join(faith_chunks) if faith_chunks else context_str

        if answer_sentences and faith_context_str.strip():
            ctx_emb = self.sbert_model.encode(faith_context_str[:3000], convert_to_tensor=True)
            ans_embs = self.sbert_model.encode(answer_sentences, convert_to_tensor=True)
            sims = util.cos_sim(ans_embs, ctx_emb.unsqueeze(0))
            sentence_sims = sims.squeeze(1).tolist()
            faithfulness = float(sum(1 for s in sentence_sims if s > 0.45) / len(sentence_sims))
        else:
            faithfulness = 0.0

        eval_metrics = {
            "latency_seconds": round(end_time - start_time, 2),
            "avg_context_relevance": round(avg_hybrid_score, 4),
            "bleu": round(bleu, 3),
            "rouge_l": round(rouge_l, 3),
            "distinct_1": round(dist1, 3),
            "distinct_2": round(dist2, 3),
            "sbert_sim": round(sbert_sim, 3),
            "ragas_faithfulness": faithfulness
        }

        return {
            "sub_queries": sub_queries,
            "context": context,
            "answer": answer,
            "eval_metrics": eval_metrics
        }
