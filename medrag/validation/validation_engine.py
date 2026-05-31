"""Validation engine to orchestrate chunk quality analysis."""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

from medrag import config as cfg
from medrag.validation.validators import (
    HeadingValidator,
    ContentValidator,
    SemanticValidator,
    HierarchyValidator,
    ChunkQuality,
    ValidationIssue
)

logger = logging.getLogger(__name__)

class ValidationEngine:
    """Orchestrates multiple validators to score and flag chunks."""
    
    def __init__(self):
        self._cfg = cfg.get("validation") or {}
        self.weights = self._cfg.get("weights", {"heading": 0.3, "content": 0.4, "semantic": 0.3})
        
        self.heading_v = HeadingValidator()
        self.content_v = ContentValidator()
        self.semantic_v = SemanticValidator()
        self.hierarchy_v = HierarchyValidator()

    def validate_chunks(self, chunks: list[dict[str, Any]], show_progress: bool = True) -> list[ChunkQuality]:
        """Run full validation suite on a list of chunks."""
        results: dict[int, ChunkQuality] = {c['chunk_id']: ChunkQuality(chunk_id=c['chunk_id']) for c in chunks}
        
        # 1. Heading and Content validation (independent per chunk)
        for chunk in chunks:
            cid = chunk['chunk_id']
            h_score, h_issues = self.heading_v.validate(chunk.get('heading', ''))
            c_score, c_issues = self.content_v.validate(chunk.get('content', ''))
            
            results[cid].heading_score = h_score
            results[cid].content_score = c_score
            results[cid].issues.extend(h_issues)
            results[cid].issues.extend(c_issues)

        # 2. Semantic validation (batch processed for efficiency)
        headings = [c.get('heading', '') for c in chunks]
        contents = [c.get('content', '') for c in chunks]
        
        logger.info("Computing semantic alignment for %d chunks...", len(chunks))
        semantic_results = self.semantic_v.validate_batch(headings, contents)
        
        for idx, (s_score, s_issues) in enumerate(semantic_results):
            cid = chunks[idx]['chunk_id']
            results[cid].semantic_score = s_score
            results[cid].issues.extend(s_issues)

        # 3. Hierarchy validation (document-wide context)
        # Group by source file
        by_source = defaultdict(list)
        for chunk in chunks:
            by_source[chunk.get('source_file', 'unknown')].append(chunk)
            
        for source, source_chunks in by_source.items():
            h_results = self.hierarchy_v.validate_document(source_chunks)
            for cid, h_issues in h_results:
                results[cid].issues.extend(h_issues)

        # 4. Final scoring
        final_list = []
        for cid in sorted(results.keys()):
            q = results[cid]
            q.overall_score = (
                q.heading_score * self.weights.get("heading", 0.3) +
                q.content_score * self.weights.get("content", 0.4) +
                q.semantic_score * self.weights.get("semantic", 0.3)
            )
            final_list.append(q)
            
        return final_list

    @staticmethod
    def aggregate_stats(chunks: list[dict[str, Any]], qualities: list[ChunkQuality]) -> dict[str, Any]:
        """Generate statistical summary of validation results."""
        total = len(chunks)
        if total == 0: return {}

        scores = [q.overall_score for q in qualities]
        all_issues = [issue for q in qualities for issue in q.issues]
        issue_counts = Counter(i.code for i in all_issues)
        
        # Word counts
        word_counts = [len(c.get('content', '').split()) for c in chunks]
        
        return {
            "total_chunks": total,
            "avg_score": sum(scores) / total,
            "min_score": min(scores),
            "max_score": max(scores),
            "flagged_count": sum(1 for q in qualities if q.overall_score < 70 or q.issues),
            "critical_count": sum(1 for q in qualities if any(i.severity == "error" for i in q.issues)),
            "issue_distribution": dict(issue_counts),
            "avg_words": sum(word_counts) / total,
            "word_range": (min(word_counts), max(word_counts))
        }
