"""Core validators for MedRAG chunk quality."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from medrag import config as cfg

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    code: str
    severity: str  # "warning" | "error"
    message: str

@dataclass
class ChunkQuality:
    chunk_id: int
    overall_score: float = 0.0
    heading_score: float = 0.0
    content_score: float = 0.0
    semantic_score: float = 0.0
    issues: list[ValidationIssue] = field(default_factory=list)

class HeadingValidator:
    """Validate heading quality using heuristics and patterns."""
    
    def __init__(self):
        self.v_cfg = cfg.get("validation")
        if not self.v_cfg: self.v_cfg = {}
        self.min_words = self.v_cfg.get("heading_min_words", 2)
        self.max_words = self.v_cfg.get("heading_max_words", 15)
        self.max_symbol_ratio = self.v_cfg.get("heading_max_symbol_ratio", 0.2)
        self.bad_patterns = [re.compile(p) for p in self.v_cfg.get("bad_heading_patterns", [])]

    def validate(self, heading: str) -> tuple[float, list[ValidationIssue]]:
        issues = []
        score = 100.0
        
        if not heading or not heading.strip():
            return 0.0, [ValidationIssue("H001", "error", "Heading is empty")]

        text = heading.strip()
        words = text.split()
        word_count = len(words)

        # 1. Length checks
        if word_count < self.min_words:
            score -= 30
            issues.append(ValidationIssue("H002", "warning", f"Heading too short ({word_count} words)"))
        if word_count > self.max_words:
            score -= 20
            issues.append(ValidationIssue("H003", "warning", f"Heading too long ({word_count} words)"))

        # 2. Capitalization patterns
        if text.islower() and word_count > 1:
            score -= 20
            issues.append(ValidationIssue("H004", "warning", "Heading is unexpectedly all lowercase"))

        # 3. OCR and Boilerplate patterns
        for pattern in self.bad_patterns:
            if pattern.search(text):
                score -= 50
                issues.append(ValidationIssue("H005", "error", f"Heading matches bad pattern: {pattern.pattern}"))

        # 4. Symbol ratio
        symbols = len(re.findall(r"[^\w\s]", text))
        ratio = symbols / len(text) if len(text) > 0 else 0
        if ratio > self.max_symbol_ratio:
            score -= 20
            issues.append(ValidationIssue("H006", "warning", f"High symbol ratio in heading ({ratio:.2f})"))

        # 5. Semantic completion
        if text.rstrip().endswith(("and", "or", "the", "of", "with")):
            score -= 30
            issues.append(ValidationIssue("H007", "warning", "Heading ends with incomplete phrase"))

        return max(0.0, score), issues

class ContentValidator:
    """Validate content quality including OCR artifacts and structure."""
    
    def __init__(self):
        self.v_cfg = cfg.get("validation")
        if not self.v_cfg: self.v_cfg = {}
        self.min_words = self.v_cfg.get("content_min_words", 40)
        self.max_words = self.v_cfg.get("content_max_words", 800)
        self.min_alpha = self.v_cfg.get("content_min_alpha_ratio", 0.7)
        self.min_punc = self.v_cfg.get("content_min_punctuation_ratio", 0.01)

    def validate(self, content: str) -> tuple[float, list[ValidationIssue]]:
        issues = []
        score = 100.0
        
        if not content or not content.strip():
            return 0.0, [ValidationIssue("C001", "error", "Content is empty")]

        text = content.strip()
        words = text.split()
        word_count = len(words)

        # 1. Length checks
        if word_count < self.min_words:
            score -= 40
            issues.append(ValidationIssue("C002", "warning", f"Content too short ({word_count} words)"))
        if word_count > self.max_words:
            score -= 10
            issues.append(ValidationIssue("C003", "warning", f"Content exceeds target length ({word_count} words)"))

        # 2. OCR Merge detection
        merged_words = re.findall(r"[a-z][A-Z][a-z]", text)
        if merged_words:
            score -= min(len(merged_words) * 10, 50)
            issues.append(ValidationIssue("C004", "warning", f"Possible OCR word merges detected: {', '.join(merged_words[:3])}"))

        # 3. Alphabetic ratio
        alpha_chars = sum(1 for c in text if c.isalpha())
        ratio = alpha_chars / len(text) if len(text) > 0 else 0
        if ratio < self.min_alpha:
            score -= 30
            issues.append(ValidationIssue("C005", "warning", f"Low alphabetic character ratio ({ratio:.2f})"))

        # 4. Punctuation density
        punc_chars = sum(1 for c in text if c in ".!?,;:")
        punc_ratio = punc_chars / len(text) if len(text) > 0 else 0
        if punc_ratio < self.min_punc:
            score -= 20
            issues.append(ValidationIssue("C006", "warning", f"Low punctuation density ({punc_ratio:.4f})"))

        # 5. Excessive line breaks
        newlines = text.count("\n")
        if newlines > word_count * 0.2:
            score -= 20
            issues.append(ValidationIssue("C007", "warning", "Excessive line breaks relative to word count"))

        return max(0.0, score), issues

class SemanticValidator:
    """Compute semantic alignment between heading and content."""
    
    def __init__(self):
        self.v_cfg = cfg.get("validation")
        if not self.v_cfg: self.v_cfg = {}
        self.model_name = self.v_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.threshold = self.v_cfg.get("semantic_min_threshold", 0.25)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def validate_batch(self, headings: list[str], contents: list[str]) -> list[tuple[float, list[ValidationIssue]]]:
        model = self._load_model()
        
        # Batch encode
        h_embs = model.encode(headings, convert_to_numpy=True, normalize_embeddings=True)
        c_embs = model.encode(contents, convert_to_numpy=True, normalize_embeddings=True)
        
        # Compute cosine similarity
        similarities = np.sum(h_embs * c_embs, axis=1)
        
        results = []
        for sim in similarities:
            issues = []
            score = float(sim) * 100.0
            
            if sim < self.threshold:
                issues.append(ValidationIssue("S001", "warning", f"Low semantic alignment between heading and content ({sim:.2f})"))
                # Penalize score more heavily if very low
                if sim < self.threshold / 2:
                    score = max(0, score - 30)
            
            results.append((max(0.0, score), issues))
            
        return results

class HierarchyValidator:
    """Validate hierarchy consistency across chunks."""
    
    def validate_document(self, chunks: list[dict[str, Any]]) -> list[tuple[int, list[ValidationIssue]]]:
        issues_by_chunk = {c['chunk_id']: [] for c in chunks}
        
        # 1. Orphaned sections (e.g. level3 without level2)
        for chunk in chunks:
            cid = chunk['chunk_id']
            if chunk.get('level3') and not chunk.get('level2'):
                issues_by_chunk[cid].append(ValidationIssue("H008", "warning", "Orphaned level3 section (missing level2)"))
            if chunk.get('level2') and not chunk.get('level1'):
                issues_by_chunk[cid].append(ValidationIssue("H009", "warning", "Orphaned level2 section (missing level1)"))

        # 2. Duplicate headings under same parent
        hierarchy_counts = {}
        for chunk in chunks:
            parent_key = f"{chunk.get('level1')} > {chunk.get('level2')}"
            heading = chunk.get('heading')
            full_key = f"{parent_key} > {heading}"
            
            # Note: Multiple chunks can have same heading if it was split, 
            # but hierarchy chains should be consistent.
            pass

        return [(cid, issues) for cid, issues in issues_by_chunk.items()]
