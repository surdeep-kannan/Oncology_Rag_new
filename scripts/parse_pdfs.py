#!/usr/bin/env python3
"""parse_pdfs.py – Parse one or more medical PDFs into structured blocks.

Usage:
    python scripts/parse_pdfs.py                              # all PDFs in input/
    python scripts/parse_pdfs.py --input my_file.pdf          # single PDF
    python scripts/parse_pdfs.py --input-dir /path/to/pdfs    # custom directory
    python scripts/parse_pdfs.py --parser pymupdf             # force parser
    python scripts/parse_pdfs.py --config custom_config.yaml  # custom config
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import median

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from medrag import config as cfg
from medrag.logging_setup import setup_logging
from medrag.parsers import parse_pdf
from medrag.cleaning import NoiseFilter, TextCleaner

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse medical PDFs into structured blocks (JSONL)."
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to a single PDF file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing PDFs. Defaults to input/.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to output/parsed_blocks.jsonl.",
    )
    parser.add_argument(
        "--parser",
        choices=["docling", "pymupdf", "auto"],
        default="auto",
        help="Force a specific parser. Default: auto (uses config).",
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
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip noise filtering (output raw blocks).",
    )
    return parser.parse_args()


def find_pdfs(args: argparse.Namespace) -> list[Path]:
    """Resolve input PDF paths."""
    if args.input:
        path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
        if not path.exists():
            logger.error("Input file not found: %s", path)
            sys.exit(1)
        return [path]

    input_dir = args.input_dir or cfg.input_dir()
    if not input_dir.exists():
        # Also check project root for PDFs
        input_dir = PROJECT_ROOT
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", input_dir)
        sys.exit(1)
    return pdfs


def main() -> None:
    args = parse_args()

    # Load config
    if args.config:
        cfg.load_config(args.config)

    # Force CLI parser override in global config
    if args.parser and args.parser != "auto":
        config_dict = cfg.load_config()
        if "parsing" not in config_dict:
            config_dict["parsing"] = {}
        config_dict["parsing"]["primary_parser"] = args.parser
        config_dict["parsing"]["fallback_parser"] = args.parser

    setup_logging(level="DEBUG" if args.debug else None)

    pdfs = find_pdfs(args)
    output_path = args.output or cfg.output_dir() / "parsed_blocks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Found %d PDF(s) to process", len(pdfs))

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    cleaner = TextCleaner()
    all_blocks: list[dict] = []
    iterator = tqdm(pdfs, desc="Parsing PDFs") if tqdm else pdfs

    for pdf_path in iterator:
        t0 = time.time()
        logger.info("─" * 50)
        logger.info("Processing: %s", pdf_path.name)

        try:
            doc = parse_pdf(pdf_path)
            logger.info(
                "Parsed %d blocks from %d pages (parser: %s)",
                len(doc.blocks), doc.page_count, doc.parser_used,
            )

            # Filter noise
            if not args.no_clean:
                noise_filter = NoiseFilter(page_heights=doc.page_heights)
                filtered_blocks = noise_filter.filter_blocks(doc.blocks)
            else:
                filtered_blocks = doc.blocks

            # Convert to dicts
            for block in filtered_blocks:
                cleaned_text = cleaner.clean_text(block.text) if not args.no_clean else block.text
                if not cleaned_text:
                    continue

                all_blocks.append({
                    "source_file": doc.source_file,
                    "parser": doc.parser_used,
                    "page_index": block.page_index,
                    "block_index": block.block_index,
                    "text": cleaned_text,
                    "doc_item_type": block.doc_item_type,
                    "heading_level": block.heading_level,
                    "max_font_size": block.max_font_size,
                    "is_bold": block.is_bold,
                    "bbox": [block.min_x, block.min_y, block.max_x, block.max_y],
                })

            elapsed = time.time() - t0
            logger.info("Done in %.1fs: %d blocks retained", elapsed, len(filtered_blocks))

        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_path.name, exc, exc_info=True)
            continue

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for block in all_blocks:
            f.write(json.dumps(block, ensure_ascii=False) + "\n")

    logger.info("═" * 50)
    logger.info("PARSING COMPLETE")
    logger.info("  Total blocks: %d", len(all_blocks))
    logger.info("  Output: %s", output_path)
    logger.info("═" * 50)

    # Print cleaning stats
    print(f"\n✅ Parsing complete: {len(all_blocks)} blocks from {len(pdfs)} PDF(s)")
    print(f"   Output: {output_path}")
    print(f"   OCR fixes: {cleaner.stats.ocr_merges_fixed}")
    print(f"   Unicode normalized: {cleaner.stats.unicode_normalized}")
    print(f"   Lines merged: {cleaner.stats.lines_merged}")


if __name__ == "__main__":
    main()
