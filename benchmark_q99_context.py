"""
Benchmark Q99 at different context window sizes.
Compares latency and answer quality for each n_ctx setting.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.table import Table

# Q99 details
Q99_QUESTION = "What is the significance of perineural invasion in oral cancer prognosis?"
Q99_EXPECTED = "Perineural invasion is an adverse prognostic factor and a high-risk feature indicating the need for adjuvant therapy."

CONTEXT_SIZES = [2048, 3072, 4096]
N_THREADS = 10

GGUF_PATH = "/home/surdeep/.cache/huggingface/hub/models--mradermacher--Llama3-Med42-8B-GGUF/snapshots/7e2883406aaaee888cefbba8a50420062b484fee/Llama3-Med42-8B.Q8_0.gguf"

console = Console()

def run_benchmark():
    # We need the search indices + reranker loaded once (they don't depend on n_ctx)
    console.print("[bold cyan]Loading search indices (one-time)...[/bold cyan]")
    
    from medrag.search.hybrid_search import HybridSearcher
    from medrag.search.bm25_search import BM25Index
    from medrag.search.embedding_search import EmbeddingIndex
    from medrag.search.reranker import CrossEncoderReranker
    from medrag.llm import MedLLM, clean_source_name
    from medrag import config as cfg
    from sentence_transformers import SentenceTransformer, util
    
    bm25 = BM25Index()
    bm25.load(cfg.index_dir() / "bm25_index.pkl")
    emb = EmbeddingIndex()
    emb.load(cfg.index_dir() / "embeddings")
    hybrid = HybridSearcher(bm25, emb)
    reranker = CrossEncoderReranker()
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    console.print("[bold green]Search indices loaded.[/bold green]")
    
    # Step 1: Retrieve context ONCE (retrieval doesn't depend on n_ctx)
    console.print(f"\n[bold yellow]Retrieving context for Q99...[/bold yellow]")
    
    # Simple retrieval (no decomposition needed for this direct question)
    raw_results = hybrid.search(Q99_QUESTION, top_k=40)
    context = reranker.rerank_with_parent_context(
        Q99_QUESTION, raw_results, bm25._chunks, top_k=3
    )
    
    console.print(f"[bold green]Retrieved {len(context)} chunks.[/bold green]")
    
    results = []
    
    for ctx_size in CONTEXT_SIZES:
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold magenta]Testing n_ctx = {ctx_size}[/bold magenta]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")
        
        # Load model with this context size
        console.print(f"[dim]Loading model with n_ctx={ctx_size}, n_threads={N_THREADS}...[/dim]")
        from llama_cpp import Llama
        
        model = Llama(
            model_path=GGUF_PATH,
            n_ctx=ctx_size,
            n_threads=N_THREADS,
            verbose=False
        )
        
        # Build the RAG prompt (same as MedLLM.format_rag_prompt)
        formatted_chunks = []
        for i, c in enumerate(context):
            content = c.get('content', '')
            if '_parent_context' in c:
                content = f"{c['_parent_context']}\n\n{content}"
            book_name = clean_source_name(c.get('source_file', ''))
            formatted_chunks.append(f"Source [{i+1}: {book_name}]: {content}")
        context_str = "\n\n".join(formatted_chunks)
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>

You are a professional medical assistant. Answer the user's question directly using ONLY the provided context. 
CRITICAL INSTRUCTIONS:
- Do NOT use conversational filler like "Based on the context" or "I can provide".
- Output ONLY the direct medical answer. Do not include introductory remarks.
- SPECIFICITY OVER FREQUENCY: If one source directly and specifically addresses the question's exact mechanism, disease, or clinical entity while other sources discuss related but different topics, you MUST prioritize the most specific source.
- CONCISENESS: Once you have answered the specific question asked, STOP immediately.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Context:
{context_str}

### Question:
{Q99_QUESTION}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        prompt_tokens = len(prompt) // 4  # rough estimate
        console.print(f"[dim]Estimated prompt tokens: ~{prompt_tokens}[/dim]")
        
        # Generate answer
        console.print(f"[yellow]Generating answer...[/yellow]")
        start_t = time.time()
        
        output = model(
            prompt,
            max_tokens=512,
            stop=["<|eot_id|>", "###", "</s>"],
            echo=False
        )
        
        latency = time.time() - start_t
        answer = output["choices"][0]["text"].strip()
        
        # Compute SBERT similarity
        emb_expected = sbert.encode(Q99_EXPECTED)
        emb_answer = sbert.encode(answer)
        similarity = util.cos_sim(emb_expected, emb_answer).item()
        
        console.print(f"\n[bold green]Answer (n_ctx={ctx_size}):[/bold green]")
        console.print(f"[white]{answer}[/white]")
        console.print(f"\n[bold]Latency: {latency:.1f}s | SBERT Sim: {similarity:.3f}[/bold]")
        
        results.append({
            "n_ctx": ctx_size,
            "latency": round(latency, 1),
            "answer": answer,
            "sbert_sim": round(similarity, 3),
            "answer_len": len(answer.split())
        })
        
        # Free the model before loading next
        del model
    
    # Final comparison table
    console.print(f"\n\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]  Q99 CONTEXT LENGTH BENCHMARK RESULTS[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]")
    
    console.print(f"\n[dim]Question: {Q99_QUESTION}[/dim]")
    console.print(f"[dim]Expected: {Q99_EXPECTED}[/dim]\n")
    
    table = Table(title="Context Length vs Performance", show_lines=True)
    table.add_column("n_ctx", justify="center", style="bold yellow")
    table.add_column("Latency (s)", justify="center", style="bold green")
    table.add_column("SBERT Sim", justify="center", style="bold blue")
    table.add_column("Words", justify="center", style="bold")
    table.add_column("Answer Preview", style="white", max_width=60)
    
    for r in results:
        preview = r["answer"][:120] + "..." if len(r["answer"]) > 120 else r["answer"]
        table.add_row(
            str(r["n_ctx"]),
            str(r["latency"]),
            str(r["sbert_sim"]),
            str(r["answer_len"]),
            preview
        )
    
    console.print(table)
    
    # Save results
    with open("benchmark_q99_results.json", "w") as f:
        json.dump(results, f, indent=2)
    console.print("\n[bold green]Results saved to benchmark_q99_results.json[/bold green]")

if __name__ == "__main__":
    run_benchmark()
