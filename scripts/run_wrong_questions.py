import sys
import json
import time
from pathlib import Path

# Add project root to Python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag.hm_rag import HMRAGPipeline

# Import Rich library components for the premium terminal dashboard
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

def make_dashboard(
    current_idx: int,
    total: int,
    current_item: dict,
    prev_item: dict,
    avg_latency: float,
    avg_sim: float,
    avg_faith: float,
    status_text: str
) -> Panel:
    """
    Builds a beautifully structured Rich Dashboard for live-monitoring the RAG runs.
    """
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)

    # 1. Header Section
    header_text = Text("\n🩺 CLINICAL RAG EVALUATION - TARGETED FAILING/INCOMPLETE RUNNER", style="bold red")
    sub_header = Text("Empirical Thompson-Sampling & Ligature Spacing Repair Verification\n", style="italic dim white")
    grid.add_row(header_text)
    grid.add_row(sub_header)

    # 2. Key Metrics Grid
    metrics_table = Table(title="📈 Target Cohort Performance Metrics (Cumulative)", show_header=True, header_style="bold magenta", expand=True)
    metrics_table.add_column("Progress", justify="center", style="bold yellow")
    metrics_table.add_column("Avg Latency", justify="center", style="bold green")
    metrics_table.add_column("Avg SBERT Sim", justify="center", style="bold blue")
    metrics_table.add_column("Avg Faithfulness", justify="center", style="bold dark_green")
    
    pct = (current_idx / total) * 100 if total > 0 else 0
    progress_str = f"{current_idx}/{total} ({pct:.1f}%)"
    latency_str = f"{avg_latency:.2f}s" if avg_latency > 0 else "N/A"
    sim_str = f"{avg_sim:.3f}" if avg_sim > 0 else "N/A"
    faith_str = f"{avg_faith:.3f}" if avg_faith > 0 else "N/A"
    
    metrics_table.add_row(progress_str, latency_str, sim_str, faith_str)
    grid.add_row(metrics_table)
    grid.add_row("")

    # 3. Active Process Monitoring
    if current_item:
        active_table = Table(title="⚡ Active Query Processing", show_header=False, expand=True, box=None)
        active_table.add_column("Label", style="bold yellow", width=15)
        active_table.add_column("Val", style="white")
        
        active_table.add_row("Question ID:", current_item.get("id", "N/A"))
        active_table.add_row("Category:", current_item.get("category", "N/A").upper())
        active_table.add_row("Difficulty:", current_item.get("difficulty", "N/A").upper())
        active_table.add_row("Question:", current_item.get("q", ""))
        active_table.add_row("Status:", f"[bold orange3]{status_text}[/bold orange3]")
        
        grid.add_row(Panel(active_table, border_style="yellow", title="[bold yellow]Current Task[/bold yellow]"))
    else:
        grid.add_row(Panel("[bold green]All Targeted Questions Processed Successfully![/bold green]", border_style="green"))
    grid.add_row("")

    # 4. Previous Question Results Panel
    if prev_item:
        prev_table = Table(title="⏮️ Previous Question Generation Snapshot", show_header=False, expand=True, box=None)
        prev_table.add_column("Label", style="bold cyan", width=15)
        prev_table.add_column("Val", style="white")
        
        prev_table.add_row("Question ID:", prev_item.get("id", "N/A"))
        prev_table.add_row("Question:", prev_item.get("q", "")[:100] + "...")
        prev_table.add_row("Generated:", prev_item.get("generated_answer", "")[:180] + "...")
        
        metrics = prev_item.get("eval_metrics", {})
        metric_line = (
            f"[bold green]BLEU:[/bold green] {metrics.get('bleu', 0.0):.3f} | "
            f"[bold green]ROUGE-L:[/bold green] {metrics.get('rouge_l', 0.0):.3f} | "
            f"[bold green]SBERT Sim:[/bold green] {metrics.get('sbert_sim', 0.0):.3f} | "
            f"[bold green]Faithfulness:[/bold green] {metrics.get('ragas_faithfulness', 0.0):.3f}"
        )
        prev_table.add_row("Metrics:", metric_line)
        
        grid.add_row(Panel(prev_table, border_style="cyan", title="[bold cyan]Last Answer Recorded[/bold cyan]"))
    
    return Panel(grid, title="[bold red]Med42 Targeted Cohort Runner[/bold red]", border_style="red")

def main():
    console = Console()
    
    # Target Lists
    wrong_cohort = ['Q002', 'Q015', 'Q024', 'Q035', 'Q038', 'Q040']
    partial_cohort = ['Q005', 'Q013', 'Q018', 'Q020', 'Q022', 'Q023', 'Q025', 'Q027', 'Q028', 'Q029', 'Q039', 'Q050']
    target_ids = wrong_cohort + partial_cohort
    
    # Paths setup
    json_path = Path("/home/surdeep/Downloads/oncology_questions_100 (3).json")
    output_json_path = PROJECT_ROOT / "targeted_answers_recorded.json"
    
    console.print("[bold red]Initializing targeted RAG runner for failing/incomplete cohort...[/bold red]")
    
    # Initialize pipeline
    try:
        pipeline = HMRAGPipeline()
    except Exception as e:
        console.print(f"[bold red]Failed to load pipeline: {e}[/bold red]")
        sys.exit(1)
        
    console.print(f"[bold green]Successfully initialized pipeline.[/bold green]")
    
    # Load questions dataset
    if not json_path.exists():
        console.print(f"[bold red]Dataset not found at {json_path}[/bold red]")
        sys.exit(1)
        
    with open(json_path) as f:
        all_questions = json.load(f)
        
    # Filter exactly the 18 targeted questions
    targeted_questions = [item for item in all_questions if item.get("id") in target_ids]
    
    # Sort them to follow original numeric sequence
    targeted_questions.sort(key=lambda x: int(x.get("id")[1:]))
    total_q = len(targeted_questions)
    
    console.print(f"[bold green]Loaded {total_q} targeted questions (6 Wrong, 12 Incomplete) for live audit.[/bold green]")
    
    # State tracking variables
    processed_questions = []
    latencies = []
    sbert_sims = []
    faithfulness_scores = []
    
    prev_item = None
    
    console.print("\n[bold cyan]Launching Target Monitor Board...[/bold cyan]")
    time.sleep(1)
    
    # Start live display dashboard
    with Live(Panel("Initializing Live Monitor dashboard...", title="Med42 Targeted Cohort"), refresh_per_second=4, console=console) as live:
        
        for idx, item in enumerate(targeted_questions):
            current_q_id = item.get("id", f"Q{idx+1}")
            query = item.get("q", "")
            
            # Setup initial dashboard view
            avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
            avg_sim = sum(sbert_sims) / len(sbert_sims) if sbert_sims else 0.0
            avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
            
            cohort_type = "🔴 WRONG" if current_q_id in wrong_cohort else "🟠 INCOMPLETE"
            status_msg = f"[{cohort_type}] Decomposing Clinical query..."
            
            live.update(make_dashboard(idx, total_q, item, prev_item, avg_lat, avg_sim, avg_faith, status_msg))
            
            # 1. Run pipeline
            start_t = time.time()
            
            def progress_callback(text):
                live.update(make_dashboard(idx, total_q, item, prev_item, avg_lat, avg_sim, avg_faith, f"[{cohort_type}] {text}"))
            
            # Execute RAG run
            result = pipeline.run(query, progress_callback=progress_callback, use_rl=True)
            latency = time.time() - start_t
            
            # 2. Extract results
            gen_answer = result.get("answer", "")
            metrics = result.get("eval_metrics", {})
            context = result.get("context", [])
            
            # Formulate the updated item
            answered_item = {
                "id": item.get("id"),
                "q": item.get("q"),
                "a": item.get("a"),
                "category": item.get("category"),
                "difficulty": item.get("difficulty"),
                
                # Generated properties
                "generated_answer": gen_answer,
                "latency_seconds": round(latency, 2),
                "eval_metrics": metrics,
                
                # Active retrieval context
                "rag_context": [
                    {
                        "source_file": c.get("source_file"),
                        "heading": c.get("heading"),
                        "content_preview": c.get("content", "")[:350]
                    }
                    for c in context
                ]
            }
            
            # 3. Add to tracking
            processed_questions.append(answered_item)
            latencies.append(latency)
            sbert_sims.append(metrics.get("sbert_sim", 0.0))
            faithfulness_scores.append(metrics.get("ragas_faithfulness", 0.0))
            
            # Set as previous item for the dashboard view
            prev_item = answered_item
            
            # 4. Save JSON file incrementally
            with open(output_json_path, "w") as out_f:
                json.dump(processed_questions, out_f, indent=2)
                
        # Final completed dashboard view
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        avg_sim = sum(sbert_sims) / len(sbert_sims) if sbert_sims else 0.0
        avg_faith = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        live.update(make_dashboard(total_q, total_q, None, prev_item, avg_lat, avg_sim, avg_faith, "Completed!"))

    console.print("\n[bold green]=============================================================[/bold green]")
    console.print(f"[bold green]🎉 DONE! Targeted Cohort Q&A evaluations completed successfully![/bold green]")
    console.print(f"[bold green]Saved output dataset: {output_json_path}[/bold green]")
    console.print("[bold green]=============================================================[/bold green]\n")

if __name__ == "__main__":
    main()
