#!/usr/bin/env python3
"""build_chunks.py – Build hierarchical RAG chunks from parsed blocks.

Usage:
    python scripts/build_chunks.py                                # default paths
    python scripts/build_chunks.py --input output/parsed_blocks.jsonl
    python scripts/build_chunks.py --min-words 200 --max-words 600
    python scripts/build_chunks.py --visualize                    # show hierarchy tree
    python scripts/build_chunks.py --debug                        # debug mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.parsers.base import ParsedBlock
from medrag.cleaning import HeadingDetector, NoiseFilter, TextCleaner
from medrag.chunking import HierarchyBuilder, ChunkEngine
from medrag.utils.visualization import (
    print_hierarchy_tree,
    export_hierarchy_tree,
    print_chunk_stats,
    render_tree_text,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchical RAG chunks from parsed PDF blocks."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Input JSONL from parse_pdfs.py. Default: output/parsed_blocks.jsonl.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Directly process a PDF (parse + chunk in one step).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSONL path. Default: output/chunks.jsonl.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=None,
        help="Override minimum chunk words.",
    )
    parser.add_argument(
        "--target-words",
        type=int,
        default=None,
        help="Override target chunk words.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=None,
        help="Override maximum chunk words.",
    )
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Print document hierarchy tree.",
    )
    parser.add_argument(
        "--export-tree",
        type=Path,
        default=None,
        help="Export hierarchy tree to JSON file.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and evaluation output.",
    )
    return parser.parse_args()


def load_blocks_from_jsonl(path: Path) -> dict[str, list[ParsedBlock]]:
    """Load parsed blocks grouped by source file."""
    groups: dict[str, list[ParsedBlock]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            source = data.get("source_file", "unknown")
            bbox = data.get("bbox", [0, 0, 0, 0])

            block = ParsedBlock(
                text=data.get("text", ""),
                page_index=data.get("page_index", 0),
                block_index=data.get("block_index", 0),
                min_x=bbox[0] if len(bbox) > 0 else 0,
                min_y=bbox[1] if len(bbox) > 1 else 0,
                max_x=bbox[2] if len(bbox) > 2 else 0,
                max_y=bbox[3] if len(bbox) > 3 else 0,
                max_font_size=data.get("max_font_size", 0),
                font_names=tuple(data.get("font_names", ())),
                is_bold=data.get("is_bold", False),
                doc_item_type=data.get("doc_item_type", "text"),
                heading_level=data.get("heading_level", 0),
            )
            groups.setdefault(source, []).append(block)

    return groups


def main() -> None:
    args = parse_args()

    if args.config:
        cfg.load_config(args.config)

    setup_logging(level="DEBUG" if args.debug else None)

    # Override config with CLI args
    chunk_cfg = cfg.get("chunking", default={})
    if args.min_words:
        chunk_cfg["min_words"] = args.min_words
    if args.target_words:
        chunk_cfg["target_words"] = args.target_words
    if args.max_words:
        chunk_cfg["max_words"] = args.max_words

    output_path = args.output or cfg.output_dir() / "chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Option 1: Direct PDF processing
    if args.pdf:
        from medrag.parsers import parse_pdf

        logger.info("Direct PDF mode: %s", args.pdf)
        doc = parse_pdf(args.pdf)
        noise_filter = NoiseFilter(page_heights=doc.page_heights)
        clean_blocks = noise_filter.filter_blocks(doc.blocks)
        groups = {doc.source_file: clean_blocks}

    # Option 2: Load from parsed blocks JSONL
    else:
        input_path = args.input or cfg.output_dir() / "parsed_blocks.jsonl"
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            logger.error("Run parse_pdfs.py first, or use --pdf for direct processing.")
            sys.exit(1)

        logger.info("Loading blocks from %s", input_path)
        groups = load_blocks_from_jsonl(input_path)

    logger.info("Processing %d source file(s)", len(groups))

    all_chunks = []

    try:
        from tqdm import tqdm
        iterator = tqdm(groups.items(), desc="Building chunks")
    except ImportError:
        iterator = groups.items()

    for source_file, blocks in iterator:
        logger.info("Building hierarchy for: %s (%d blocks)", source_file, len(blocks))

        # Determine body font size
        body_fonts = [b.max_font_size for b in blocks if b.word_count > 8 and b.max_font_size > 0]
        body_font = median(body_fonts) if body_fonts else 11.0

        # Build hierarchy
        builder = HierarchyBuilder(source_file, body_font_size=body_font)
        hierarchy = builder.build(blocks)

        # Visualize if requested
        if args.visualize:
            print(f"\n{'═' * 60}")
            print(f"  HIERARCHY: {source_file}")
            print(f"{'═' * 60}")
            print_hierarchy_tree(hierarchy)

        # Export tree if requested
        if args.export_tree:
            export_hierarchy_tree(hierarchy, args.export_tree)
            logger.info("Hierarchy tree exported to %s", args.export_tree)

        # Flatten to sections
        sections = builder.flatten_to_sections(hierarchy)
        logger.info("Flattened to %d sections", len(sections))

        # Chunk
        engine = ChunkEngine()
        chunks = engine.process_sections(sections)
        all_chunks.extend(chunks)

        # Debug: print per-file stats
        if args.debug:
            print(f"\n  {source_file}:")
            print(f"    Hierarchy nodes: {len(hierarchy)}")
            print(f"    Sections: {len(sections)}")
            print(f"    Chunks: {len(chunks)}")
            if chunks:
                wc = [c.token_count for c in chunks]
                print(f"    Word range: {min(wc)} – {max(wc)} (avg {sum(wc)//len(wc)})")

    # Save all chunks
    ChunkEngine.save_chunks(all_chunks, output_path)

    elapsed = time.time() - t0

    # Print stats
    chunk_dicts = [c.to_dict() for c in all_chunks]
    print_chunk_stats(chunk_dicts)

    print(f"\n✅ Chunking complete in {elapsed:.1f}s")
    print(f"   Total chunks: {len(all_chunks)}")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    main()
