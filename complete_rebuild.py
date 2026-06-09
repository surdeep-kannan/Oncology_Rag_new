#!/usr/bin/env python3
"""
complete_rebuild.py — Global Database Rebuild Pipeline.
Author: Antigravity AI Pair Programmer
Step-by-step Execution:
  1. Load active chunks from output/indices/embeddings/chunks.json
  2. Deduplicate chunks by unique (source_file, chunk_id) (Removes ~10,000 duplicate entries!)
  3. Re-index and clean headings, level1, level2, level3 fields to guarantee rich hierarchical structures.
  4. Strip all bracketed/parenthesized numeric citations from headings and content.
  5. Generate dense vector embeddings using instruction-tuned Nomic Embed Text v1.5.
  6. Rebuild and serialize the BM25 sparse lexical search index.
"""

import json
import re
import numpy as np
import pickle
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
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
    # Clean up standard layout artifact remnants
    cleaned = cleaned.replace("»", "").replace(">", "").strip()
    return cleaned

def main():
    print("=================================================================")
    print("   🚀 STARTING COMPREHENSIVE MEDICAL DATABASE REBUILD PIPELINE   ")
    print("=================================================================")
    t_start = time.time()
    
    # Define directories
    emb_dir = Path("output/indices/embeddings")
    chunks_path = emb_dir / "chunks.json"
    npy_path = emb_dir / "embeddings.npy"
    bm25_path = Path("output/indices/bm25_index.pkl")
    
    if not chunks_path.exists():
        print(f"❌ ERROR: Active chunks file not found at {chunks_path}")
        return
        
    print("\n[STEP 1] Loading active chunks from database...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)
    print(f"Loaded {len(raw_chunks)} raw chunks.")
    
    print("\n[STEP 2] Removing exact duplicate chunks...")
    unique_keys = set()
    deduped_chunks = []
    duplicate_count = 0
    for chunk in raw_chunks:
        # Use (source_file, chunk_id) as the unique signature
        key = (chunk.get("source_file"), chunk.get("chunk_id"))
        if key not in unique_keys:
            unique_keys.add(key)
            deduped_chunks.append(chunk)
        else:
            duplicate_count += 1
            
    print(f"Deduplication Complete:")
    print(f"  - Removed Duplicates : {duplicate_count} chunks")
    print(f"  - Unique Active Chunks : {len(deduped_chunks)} chunks")
    
    print("\n[STEP 3] Re-indexing and cleaning hierarchical headings & stripping citations...")
    clean_count = 0
    for chunk in deduped_chunks:
        # Clean Content
        orig_content = chunk.get("content", "")
        cleaned_content = strip_inline_citations(orig_content)
        chunk["content"] = cleaned_content
        
        # Clean Headings and Hierarchical levels
        chunk["heading"] = clean_hierarchical_field(chunk.get("heading") or "")
        chunk["level1"] = clean_hierarchical_field(chunk.get("level1") or "")
        chunk["level2"] = clean_hierarchical_field(chunk.get("level2") or "")
        chunk["level3"] = clean_hierarchical_field(chunk.get("level3") or "")
        
        if cleaned_content != orig_content:
            clean_count += 1
            
    print(f"Hierarchy alignment & citation stripping completed for {clean_count} chunks.")
    
    print("\n[STEP 4] Loading Nomic Embed Text v1.5 Model...")
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    print("Nomic model loaded successfully.")
    
    print("\n[STEP 5] Encoding chunks using instruction-tuned prefixes...")
    # Nomic embed-text-v1.5 is trained with instruction tuning: 
    # Must use 'search_document: ' prefix for documents and 'search_query: ' for queries.
    texts = []
    for c in deduped_chunks:
        parts = [
            c.get("level1"),
            c.get("level2"),
            c.get("level3"),
            c.get("heading"),
            c.get("content")
        ]
        text_content = " > ".join(p for p in parts[:-1] if p)
        if text_content:
            full_text = f"{text_content}. {parts[-1]}"
        else:
            full_text = parts[-1]
        texts.append(f"search_document: {full_text}")
        
    print(f"Encoding {len(texts)} document chunks in batches of 32...")
    t_enc_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    t_enc_end = time.time()
    print(f"Dense vector embeddings generated successfully in {t_enc_end - t_enc_start:.2f} seconds.")
    print(f"Embeddings Matrix Shape: {embeddings.shape}")
    
    print("\n[STEP 6] Saving updated clean chunks and dense embedding matrix...")
    np.save(npy_path, embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(deduped_chunks, f, ensure_ascii=False)
    print(f"Saved dense matrix to {npy_path}")
    print(f"Saved deduplicated chunks metadata to {chunks_path}")
    
    print("\n[STEP 7] Rebuilding BM25 lexical sparse index...")
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
        
    print(f"Tokenizing {len(metadata)} documents for sparse BM25 retrieval...")
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
    print("        🎉 PIPELINE EXECUTION SUCCESSFUL & COMPLETE Summary")
    print("=================================================================")
    print(f"Total Raw Chunks       : {len(raw_chunks)}")
    print(f"Deduplicated Chunks    : {len(deduped_chunks)}")
    print(f"Citation Cleaned Chunks: {clean_count}")
    print(f"Vector Space Matrix    : {embeddings.shape}")
    print(f"Total Execution Time   : {time.time() - t_start:.2f} seconds")
    print("=================================================================")

if __name__ == "__main__":
    main()
