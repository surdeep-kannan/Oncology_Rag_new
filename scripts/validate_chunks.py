#!/usr/bin/env python3
"""validate_chunks.py – Automated quality validation for medical RAG chunks.

Usage:
    python scripts/validate_chunks.py output/chunks.jsonl
    python scripts/validate_chunks.py output/chunks.jsonl --strict --export-csv
    python scripts/validate_chunks.py output/chunks.jsonl --min-score 70
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.validation import ValidationEngine

console = Console()

def parse_args():
    p = argparse.ArgumentParser(description="Validate chunk quality for RAG.")
    p.add_argument("input", type=Path, help="Input chunks.jsonl file.")
    p.add_argument("--output-dir", "-o", type=Path, default=None, help="Directory for reports.")
    p.add_argument("--strict", action="store_true", help="Higher bar for validation.")
    p.add_argument("--export-csv", action="store_true", help="Export full results to CSV.")
    p.add_argument("--min-score", type=float, default=0.0, help="Only show/flag chunks below this score.")
    p.add_argument("--sample-bad", type=int, default=10, help="Number of bad chunks to sample in terminal.")
    p.add_argument("--config", "-c", type=Path, default=None)
    return p.parse_args()

def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks

def main():
    args = parse_args()
    if args.config: cfg.load_config(args.config)
    setup_logging(level="WARNING")

    if not args.input.exists():
        console.print(f"[bold red]Error:[/] File not found: {args.input}")
        sys.exit(1)

    # 1. Load data
    chunks = load_chunks(args.input)
    console.print(f"Loaded [bold blue]{len(chunks)}[/] chunks from {args.input.name}")

    # 2. Run Validation
    engine = ValidationEngine()
    qualities = engine.validate_chunks(chunks)

    # 3. Aggregate Stats
    stats = engine.aggregate_stats(chunks, qualities)
    
    # 4. Generate Reports
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # validation_report.json
    report = {
        "summary": stats,
        "parameters": cfg.get("validation") or {},
        "results": [
            {
                "chunk_id": q.chunk_id,
                "score": round(q.overall_score, 1),
                "issues": [{"code": i.code, "msg": i.message} for i in q.issues]
            } for q in qualities
        ]
    }
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # flagged_chunks.jsonl
    flagged_ids = {q.chunk_id for q in qualities if q.overall_score < 70 or q.issues}
    flagged_data = [c for c in chunks if c['chunk_id'] in flagged_ids]
    with open(out_dir / "flagged_chunks.jsonl", "w") as f:
        for c in flagged_data:
            f.write(json.dumps(c) + "\n")

    # CSV Export
    if args.export_csv:
        df_chunks = pd.DataFrame(chunks)
        df_qual = pd.DataFrame([{
            "chunk_id": q.chunk_id,
            "overall_score": q.overall_score,
            "heading_score": q.heading_score,
            "content_score": q.content_score,
            "semantic_score": q.semantic_score,
            "issue_count": len(q.issues),
            "issues": "|".join(f"{i.code}:{i.message}" for i in q.issues)
        } for q in qualities])
        df_merged = pd.merge(df_chunks, df_qual, on="chunk_id")
        df_merged.to_csv(out_dir / "validation_results.csv", index=False)

    # 5. Terminal Summary Table
    console.print("\n[bold]Validation Summary[/]")
    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric")
    summary_table.add_column("Value")
    
    summary_table.add_row("Total Chunks", str(stats['total_chunks']))
    summary_table.add_row("Average Quality Score", f"{stats['avg_score']:.1f}/100")
    summary_table.add_row("Flagged Chunks", f"[bold yellow]{stats['flagged_count']}[/] ({stats['flagged_count']*100/stats['total_chunks']:.1f}%)")
    summary_table.add_row("Critical Failures", f"[bold red]{stats['critical_count']}[/]")
    summary_table.add_row("Avg Words/Chunk", f"{stats['avg_words']:.0f}")
    console.print(summary_table)

    # Issues Distribution
    if stats['issue_distribution']:
        console.print("\n[bold]Issues Detected[/]")
        issue_table = Table(show_header=True, header_style="bold magenta")
        issue_table.add_column("Code")
        issue_table.add_column("Count")
        for code, count in sorted(stats['issue_distribution'].items(), key=lambda x: x[1], reverse=True):
            issue_table.add_row(code, str(count))
        console.print(issue_table)

    # 6. Sample Bad Chunks
    if args.sample_bad > 0:
        bad_qualities = sorted([q for q in qualities if q.overall_score < 70], key=lambda x: x.overall_score)
        samples = bad_qualities[:args.sample_bad]
        
        if samples:
            console.print(f"\n[bold red]Samples of Low Quality Chunks (Score < 70)[/]")
            for q in samples:
                chunk = next(c for c in chunks if c['chunk_id'] == q.chunk_id)
                console.print(Panel(
                    f"[bold]Heading:[/] {chunk.get('heading')}\n"
                    f"[bold]Score:[/] {q.overall_score:.1f} (H:{q.heading_score:.0f} C:{q.content_score:.0f} S:{q.semantic_score:.0f})\n"
                    f"[bold]Issues:[/] {', '.join(i.message for i in q.issues)}\n"
                    f"[italic]Content:[/] {chunk.get('content')[:150]}...",
                    title=f"Chunk #{q.chunk_id}",
                    border_style="red" if q.overall_score < 50 else "yellow"
                ))

    console.print(f"\n[bold green]Reports generated in:[/] {out_dir}")
    console.print(f"- validation_report.json")
    console.print(f"- flagged_chunks.jsonl")
    if args.export_csv: console.print(f"- validation_results.csv")

if __name__ == "__main__":
    main()
