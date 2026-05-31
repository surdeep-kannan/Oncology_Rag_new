#!/usr/bin/env python3
"""repair_chunks.py – Production-grade chunk repair and reconstruction.

Repairs page-boundary splits, broken headings, and removes learned headers/footers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.repair import RepairEngine
from medrag.validation import ValidationEngine

def parse_args():
    p = argparse.ArgumentParser(description="Repair malformed RAG chunks.")
    p.add_argument("input", type=Path, help="Input chunks.jsonl file.")
    p.add_argument("--output-dir", "-o", type=Path, default=None, help="Directory for reports.")
    p.add_argument("--config", "-c", type=Path, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
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
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    if not args.input.exists():
        print(f"Error: File not found {args.input}")
        sys.exit(1)

    t0 = time.time()
    
    # 1. Load
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks for repair.")

    # 2. Repair
    engine = RepairEngine()
    repaired = engine.repair_pipeline(chunks)

    # 3. Validate Repaired Chunks
    print("Running quality validation on repaired chunks...")
    val_engine = ValidationEngine()
    qualities = val_engine.validate_chunks(repaired)
    stats = val_engine.aggregate_stats(repaired, qualities)

    # 4. Save Outputs
    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # repaired_chunks.jsonl
    repaired_path = out_dir / "repaired_chunks.jsonl"
    with open(repaired_path, "w", encoding="utf-8") as f:
        for c, q in zip(repaired, qualities):
            c["quality_score"] = round(q.overall_score, 1)
            c["validation_issues"] = [i.code for i in q.issues]
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # summary_report.json
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_count": len(chunks),
        "output_count": len(repaired),
        "merged_count": len(chunks) - len(repaired),
        "quality_stats": stats,
        "repair_actions_total": sum(len(c.get("_repair_actions", [])) for c in repaired)
    }
    with open(out_dir / "repair_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # suspicious_chunks.jsonl (Low score after repair)
    suspicious_path = out_dir / "suspicious_chunks.jsonl"
    with open(suspicious_path, "w") as f:
        for c in repaired:
            if c.get("quality_score", 0) < 60:
                f.write(json.dumps(c) + "\n")

    # 5. Report
    elapsed = time.time() - t0
    print("\n" + "="*50)
    print("REPAIR & RECONSTRUCTION COMPLETE")
    print("="*50)
    print(f"Input chunks:    {len(chunks)}")
    print(f"Repaired chunks: {len(repaired)}")
    print(f"Merged:          {len(chunks) - len(repaired)}")
    print(f"Total actions:   {summary['repair_actions_total']}")
    print(f"Avg Quality:     {stats['avg_score']:.1f}/100")
    print(f"Time:            {elapsed:.1f}s")
    print(f"\nOutputs:")
    print(f"- {repaired_path}")
    print(f"- {out_dir}/repair_summary.json")
    print(f"- {suspicious_path}")
    print("="*50)

if __name__ == "__main__":
    main()
