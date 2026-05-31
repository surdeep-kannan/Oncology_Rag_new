"""Hierarchy tree visualization for debugging and inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medrag.chunking.hierarchy_builder import HierarchyNode


def print_hierarchy_tree(
    nodes: list[HierarchyNode],
    max_content_preview: int = 80,
    indent: int = 0,
) -> None:
    """Print a visual tree of the document hierarchy to stdout."""
    prefix = "  " * indent

    for node in nodes:
        # Node header
        level_marker = "█" * node.level
        words = node.word_count
        pages = ""
        if node.page_start:
            pages = f" (p{node.page_start}"
            if node.page_end and node.page_end != node.page_start:
                pages += f"-{node.page_end}"
            pages += ")"

        print(f"{prefix}{level_marker} L{node.level}: {node.heading}{pages} [{words}w]")

        # Content preview
        if node.content_blocks and max_content_preview > 0:
            preview = node.full_content[:max_content_preview].replace("\n", " ")
            if len(node.full_content) > max_content_preview:
                preview += "..."
            print(f"{prefix}  └─ {preview}")

        # Recurse into children
        if node.children:
            print_hierarchy_tree(node.children, max_content_preview, indent + 1)


def export_hierarchy_tree(
    nodes: list[HierarchyNode],
    output_path: Path,
) -> None:
    """Export hierarchy tree to a JSON file for external visualization."""
    tree = _nodes_to_dicts(nodes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree, f, indent=2, ensure_ascii=False)


def _nodes_to_dicts(nodes: list[HierarchyNode]) -> list[dict[str, Any]]:
    result = []
    for node in nodes:
        d = node.to_dict()
        d["children"] = _nodes_to_dicts(node.children) if node.children else []
        result.append(d)
    return result


def render_tree_text(nodes: list[HierarchyNode], max_depth: int = 4) -> str:
    """Render hierarchy as a formatted text string."""
    lines: list[str] = []
    _render_nodes(nodes, lines, depth=0, max_depth=max_depth)
    return "\n".join(lines)


def _render_nodes(
    nodes: list[HierarchyNode],
    lines: list[str],
    depth: int,
    max_depth: int,
) -> None:
    if depth >= max_depth:
        return

    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└── " if is_last else "├── "
        indent = "│   " * depth

        words = node.word_count
        pages = ""
        if node.page_start:
            pages = f" [p{node.page_start}"
            if node.page_end and node.page_end != node.page_start:
                pages += f"-{node.page_end}"
            pages += "]"

        lines.append(f"{indent}{connector}{node.heading}{pages} ({words}w)")

        if node.children:
            _render_nodes(node.children, lines, depth + 1, max_depth)


def print_chunk_stats(chunks: list[dict[str, Any]]) -> None:
    """Print summary statistics about chunks."""
    if not chunks:
        print("No chunks to analyze.")
        return

    word_counts = [c.get("token_count", len(c.get("content", "").split())) for c in chunks]
    total = len(chunks)
    total_words = sum(word_counts)
    avg_words = total_words / total if total else 0

    # Distribution
    ranges = [(0, 100), (100, 200), (200, 400), (400, 600), (600, 1000), (1000, float("inf"))]
    range_counts = {f"{lo}-{hi}": 0 for lo, hi in ranges}
    for wc in word_counts:
        for lo, hi in ranges:
            if lo <= wc < hi:
                key = f"{lo}-{hi}"
                range_counts[key] = range_counts.get(key, 0) + 1
                break

    print(f"\n{'═' * 50}")
    print(f"  CHUNK STATISTICS")
    print(f"{'═' * 50}")
    print(f"  Total chunks:     {total}")
    print(f"  Total words:      {total_words:,}")
    print(f"  Average words:    {avg_words:.0f}")
    print(f"  Min words:        {min(word_counts)}")
    print(f"  Max words:        {max(word_counts)}")
    print(f"\n  Word count distribution:")
    for label, count in range_counts.items():
        bar = "█" * (count * 40 // max(total, 1))
        print(f"    {label:>10}: {count:>4} {bar}")

    # Unique headings
    headings = set(c.get("heading", "") for c in chunks)
    l1s = set(c.get("level1", "") for c in chunks if c.get("level1"))
    l2s = set(c.get("level2", "") for c in chunks if c.get("level2"))
    print(f"\n  Unique level1:    {len(l1s)}")
    print(f"  Unique level2:    {len(l2s)}")
    print(f"  Unique headings:  {len(headings)}")
    print(f"{'═' * 50}\n")
