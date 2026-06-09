#!/usr/bin/env python3
"""
fast_rebuild.py — Instant Database Rebuild Pipeline.
Author: Antigravity AI Pair Programmer
Step-by-step Execution:
  1. Load raw chunks (26,501 entries) from output/indices/embeddings/chunks.json
  2. Load existing Nomic dense embedding matrix (26,501 rows, 512 dimensions)
  3. Deduplicate chunks using unique (source_file, chunk_id) and keep track of unique indices
  4. Slice the dense embedding matrix at those unique indices (16,487 unique rows)
  5. Re-index and clean hierarchical headers/content (strip brackets/citations)
  6. Rebuild and serialize the BM25 sparse index
  7. Save all cleaned files back to output/indices
"""

import json
import re
import numpy as np
import pickle
import time
from pathlib import Path
from rank_bm25 import BM25Okapi

def strip_inline_citations(text: str) -> str:
    if not text:
        return ""
    # 1. Remove brackets like [21], [5, 6, 7], [15-18]
    text = re.sub(r'\[[\d\s,\-\–]+\]', '', text)
    
    # 2. Remove parentheses like (1), (1,2), (6-9) while preserving 4-digit study years like (1990)
    def clean_paren(match):
        val = match.group(1).strip()
        if re.match(r'^(19|20)\d{2}$', val):
            return match.group(0)
        return ''
    text = re.sub(r'\(\s*(\d+(?:\s*(?:,|\-|–)\s*\d+)*)\s*\)', clean_paren, text)
    
    # 3. Clean up spacing and punctuation offsets
    text = re.sub(r'\s+([\.,;:\?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_hierarchical_field(field_val: str) -> str:
    if not field_val:
        return ""
    cleaned = strip_inline_citations(field_val)
    cleaned = cleaned.replace("»", "").replace(">", "").strip()
    return cleaned

def main():
    print("=================================================================")
    print("   ⚡ RUNNING INSTANT LIGHTNING-FAST MEDICAL DATABASE REBUILD   ")
    print("=================================================================")
    t_start = time.time()
    
    emb_dir = Path("output/indices/embeddings")
    chunks_path = emb_dir / "chunks.json"
    npy_path = emb_dir / "embeddings.npy"
    bm25_path = Path("output/indices/bm25_index.pkl")
    
    if not chunks_path.exists():
        print(f"❌ ERROR: Active chunks file not found at {chunks_path}")
        return
    if not npy_path.exists():
        print(f"❌ ERROR: Dense embeddings file not found at {npy_path}")
        return
        
    # 1. Load active chunks
    print("[STEP 1] Loading active chunks metadata...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    print(f"Loaded {len(raw_chunks)} raw chunks.")
    
    # 2. Load dense embeddings matrix
    print("[STEP 2] Loading existing dense embedding matrix...")
    embeddings = np.load(npy_path)
    print(f"Loaded embedding matrix with shape: {embeddings.shape}")
    
    if len(raw_chunks) != embeddings.shape[0]:
        print(f"❌ ERROR: Count mismatch! Chunks ({len(raw_chunks)}) != Embeddings ({embeddings.shape[0]})")
        return
        
    # 3. Deduplicate chunks and track corresponding indices
    print("[STEP 3] Performing structural deduplication...")
    unique_keys = set()
    deduped_chunks = []
    unique_indices = []
    duplicate_count = 0
    
    for idx, chunk in enumerate(raw_chunks):
        key = (chunk.get("source_file"), chunk.get("chunk_id"))
        if key not in unique_keys:
            unique_keys.add(key)
            deduped_chunks.append(chunk)
            unique_indices.append(idx)
        else:
            duplicate_count += 1
            
    print(f"Deduplication Complete:")
    print(f"  - Removed Duplicates : {duplicate_count} chunks")
    print(f"  - Unique Active Chunks : {len(deduped_chunks)} chunks")
    
    # 4. Slice the dense embedding matrix instantly
    print("[STEP 4] Slicing dense embedding matrix...")
    sliced_embeddings = embeddings[unique_indices]
    print(f"New Sliced Embeddings Shape: {sliced_embeddings.shape}")
    
    # 5. Clean hierarchical headings & strip citations
    print("[STEP 5] Re-indexing and cleaning headings & content...")
    clean_count = 0
    for chunk in deduped_chunks:
        # Clean Content
        orig_content = chunk.get("content", "")
        cleaned_content = strip_inline_citations(orig_content)
        chunk["content"] = cleaned_content
        
        # Clean Headings
        chunk["heading"] = clean_hierarchical_field(chunk.get("heading") or "")
        chunk["level1"] = clean_hierarchical_field(chunk.get("level1") or "")
        chunk["level2"] = clean_hierarchical_field(chunk.get("level2") or "")
        chunk["level3"] = clean_hierarchical_field(chunk.get("level3") or "")
        
        if cleaned_content != orig_content:
            clean_count += 1
            
    print(f"Aligned hierarchy and cleared citations for {clean_count} chunks.")
    
    # 6. Save new dense embeddings and updated chunks.json
    print("[STEP 6] Saving updated clean chunks and dense embedding matrix...")
    np.save(npy_path, sliced_embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(deduped_chunks, f, ensure_ascii=False)
    print(f"Saved dense matrix to {npy_path}")
    print(f"Saved deduplicated chunks metadata to {chunks_path}")
    
    # 7. Rebuild BM25 Lexical Index
    print("[STEP 7] Rebuilding BM25 lexical sparse index...")
    TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
    def normalize_token(token: str) -> str:
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 4 and not token.endswith("ss"):
            return token[:-1]
        return token
        
    def tokenize(text: str) -> list[str]:
        return [normalize_token(token) for token in TOKEN_PATTERN.findall(text.lower()) if token]
        
    def build_search_text(item: dict) -> str:
        parts = [
            item.get("heading"),
            item.get("level1"),
            item.get("level2"),
            item.get("level3"),
            item.get("level4"),
            item.get("content"),
        ]
        return " ".join(part for part in parts if part)
        
    documents = []
    metadata = []
    for chunk in deduped_chunks:
        text = build_search_text(chunk)
        tokenized = tokenize(text)
        if not tokenized:
            continue
        documents.append(tokenized)
        metadata.append(chunk)
        
    bm25 = BM25Okapi(documents)
    with open(bm25_path, "wb") as f:
        pickle.dump({
            "chunks": metadata,
            "corpus": documents,
            "k1": 1.5,
            "b": 0.75,
        }, f)
    print(f"Sparse BM25 index built and serialized to {bm25_path}")
    
    print("\n=================================================================")
    print("        ⚡ INSTANT LIGHTNING-FAST REBUILD COMPLETED SUCCESSFULLY")
    print("=================================================================")
    print(f"Total Raw Chunks       : {len(raw_chunks)}")
    print(f"Deduplicated Chunks    : {len(deduped_chunks)}")
    print(f"Citation Cleaned Chunks: {clean_count}")
    print(f"Vector Space Matrix    : {sliced_embeddings.shape}")
    print(f"Total Execution Time   : {time.time() - t_start:.2f} seconds")
    print("=================================================================")

if __name__ == "__main__":
    main()
