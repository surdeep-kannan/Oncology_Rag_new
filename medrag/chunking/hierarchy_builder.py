"""Hierarchy builder – constructs a document tree from parsed blocks.

Takes cleaned blocks with heading annotations and builds a hierarchical
tree of sections, tracking level1 → level2 → level3 relationships.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from medrag.parsers.base import ParsedBlock
from medrag.cleaning.heading_detector import HeadingDetector
from medrag.cleaning.text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


@dataclass
class HierarchyNode:
    """A node in the document hierarchy tree."""

    heading: str
    level: int                     # 1 = chapter, 2 = section, 3 = subsection
    level1: str | None = None
    level2: str | None = None
    level3: str | None = None
    content_blocks: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    children: list[HierarchyNode] = field(default_factory=list)
    source_file: str = ""

    @property
    def full_content(self) -> str:
        """Join all content blocks into a single string."""
        return "\n\n".join(b for b in self.content_blocks if b.strip())

    @property
    def word_count(self) -> int:
        return len(self.full_content.split())

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "level": self.level,
            "level1": self.level1,
            "level2": self.level2,
            "level3": self.level3,
            "content": self.full_content,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "word_count": self.word_count,
            "children_count": len(self.children),
        }


class HierarchyBuilder:
    """Build a hierarchy tree from a flat list of parsed blocks."""

    def __init__(
        self,
        source_file: str,
        body_font_size: float = 11.0,
    ) -> None:
        self.source_file = source_file
        self.heading_detector = HeadingDetector(body_font_size)
        self.text_cleaner = TextCleaner()

    def build(self, blocks: list[ParsedBlock]) -> list[HierarchyNode]:
        """Build hierarchy from cleaned blocks.

        Returns a list of top-level nodes (level 1), each containing
        nested children (level 2, level 3).
        """
        if not blocks:
            return []

        # Track hierarchy state
        current_l1: HierarchyNode | None = None
        current_l2: HierarchyNode | None = None
        current_l3: HierarchyNode | None = None
        current_breadcrumb: list[str] = []
        root_nodes: list[HierarchyNode] = []

        # Accumulator for blocks not yet assigned to a heading
        pending_blocks: list[str] = []
        pending_page_start: int | None = None

        for block in blocks:
            # Handle breadcrumbs
            if self.heading_detector.is_breadcrumb(block):
                current_breadcrumb = self.heading_detector.parse_breadcrumb(block.text)
                continue

            # Evaluate as heading
            candidate = self.heading_detector.evaluate(block)

            if candidate.is_heading:
                # Flush pending blocks to current node
                self._flush_pending(
                    pending_blocks, pending_page_start, block.page_index,
                    current_l1, current_l2, current_l3,
                )
                pending_blocks = []
                pending_page_start = None

                # Determine hierarchy level
                level = candidate.level
                heading_text = self.text_cleaner.clean_text(block.text)

                # Resolve hierarchy from breadcrumb
                l1, l2, l3 = self._resolve_levels(
                    heading_text, level, current_breadcrumb,
                    current_l1, current_l2,
                )

                node = HierarchyNode(
                    heading=heading_text,
                    level=level,
                    level1=l1,
                    level2=l2,
                    level3=l3,
                    page_start=block.page_index + 1,
                    source_file=self.source_file,
                )

                if level == 1:
                    current_l1 = node
                    current_l2 = None
                    current_l3 = None
                    root_nodes.append(node)
                elif level == 2:
                    current_l2 = node
                    current_l3 = None
                    if current_l1:
                        current_l1.children.append(node)
                    else:
                        root_nodes.append(node)
                else:  # level 3+
                    current_l3 = node
                    if current_l2:
                        current_l2.children.append(node)
                    elif current_l1:
                        current_l1.children.append(node)
                    else:
                        root_nodes.append(node)
            else:
                # Content block
                cleaned = self.text_cleaner.clean_text(block.text)
                if cleaned:
                    pending_blocks.append(cleaned)
                    if pending_page_start is None:
                        pending_page_start = block.page_index + 1

        # Flush remaining
        last_page = blocks[-1].page_index + 1 if blocks else None
        self._flush_pending(
            pending_blocks, pending_page_start, last_page,
            current_l1, current_l2, current_l3,
        )

        # If no headings were detected, create a single root node
        if not root_nodes and pending_blocks:
            root_nodes.append(
                HierarchyNode(
                    heading=self.source_file,
                    level=1,
                    level1=self.source_file,
                    content_blocks=pending_blocks,
                    page_start=1,
                    page_end=last_page,
                    source_file=self.source_file,
                )
            )

        logger.info(
            "Built hierarchy: %d top-level nodes from %s",
            len(root_nodes), self.source_file,
        )
        return root_nodes

    def _flush_pending(
        self,
        blocks: list[str],
        page_start: int | None,
        page_end: int | None,
        l1: HierarchyNode | None,
        l2: HierarchyNode | None,
        l3: HierarchyNode | None,
    ) -> None:
        """Attach pending content blocks to the most specific current node."""
        if not blocks:
            return

        target = l3 or l2 or l1
        if target:
            target.content_blocks.extend(blocks)
            if target.page_end is None or (page_end and page_end > (target.page_end or 0)):
                target.page_end = page_end

    def _resolve_levels(
        self,
        heading: str,
        level: int,
        breadcrumb: list[str],
        current_l1: HierarchyNode | None,
        current_l2: HierarchyNode | None,
    ) -> tuple[str | None, str | None, str | None]:
        """Determine level1/level2/level3 values for a heading."""

        # If breadcrumb is available, use it
        if breadcrumb:
            parts = breadcrumb[:]
            if not parts or parts[-1] != heading:
                parts.append(heading)
            parts = parts[:3]
            while len(parts) < 3:
                parts.append(None)
            return parts[0], parts[1], parts[2]

        # Otherwise infer from level and current context
        if level == 1:
            return heading, None, None
        elif level == 2:
            l1 = current_l1.heading if current_l1 else None
            return l1, heading, None
        else:
            l1 = current_l1.heading if current_l1 else None
            l2 = current_l2.heading if current_l2 else None
            return l1, l2, heading

    def flatten_to_sections(self, nodes: list[HierarchyNode]) -> list[HierarchyNode]:
        """Flatten the tree into a list of leaf-level sections for chunking."""
        sections: list[HierarchyNode] = []
        self._collect_leaves(nodes, sections)
        return sections

    def _collect_leaves(
        self,
        nodes: list[HierarchyNode],
        acc: list[HierarchyNode],
    ) -> None:
        for node in nodes:
            if node.children:
                # If node has content AND children, emit the content first
                if node.content_blocks:
                    acc.append(node)
                self._collect_leaves(node.children, acc)
            else:
                if node.content_blocks or node.heading:
                    acc.append(node)
