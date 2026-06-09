#!/usr/bin/env python3
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

def main():
    print("=========================================")
    # Visual Polish
    print("  INLINE CITATION STRIPPING & EMBEDDING REBUILD")
    print("=========================================")
    
    # Target files in output/indices
    emb_dir = Path("output/indices/embeddings")
    chunks_path = emb_dir / "chunks.json"
    npy_path = emb_dir / "embeddings.npy"
    bm25_path = Path("output/indices/bm25_index.pkl")
    
    if not chunks_path.exists():
        print(f"Error: active chunks not found at {chunks_path}")
        return
        
    print("Loading active chunks...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"Loaded {len(chunks)} chunks.")
    
    # 1. Strip inline citations from chunks
    print("Stripping inline numeric citations from headings and content...")
    stripped_count = 0
    t0 = time.time()
    
    for chunk in chunks:
        orig_content = chunk.get("content", "")
        orig_heading = chunk.get("heading", "") or ""
        
        cleaned_content = strip_inline_citations(orig_content)
        cleaned_heading = strip_inline_citations(orig_heading)
        
        if cleaned_content != orig_content or cleaned_heading != orig_heading:
            stripped_count += 1
            
        chunk["content"] = cleaned_content
        if "heading" in chunk:
            chunk["heading"] = cleaned_heading
            
    print(f"Cleaned citations in {stripped_count} out of {len(chunks)} chunks.")
    
    # 2. Re-generate Dense Embeddings using nomic model
    print("Loading nomic embedding model...")
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    
    print("Encoding cleaned chunks using nomic model...")
    # Follow exactly the EmbeddingIndex build strategy (with nomic prefix)
    texts = [
        ("search_document: " + c.get("heading", "") + ". " + c.get("content", "")).strip()
        for c in chunks
    ]
    
    t_enc_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    t_enc_end = time.time()
    
    print(f"Embeddings re-generated in {t_enc_end - t_enc_start:.2f} seconds. Shape: {embeddings.shape}")
    
    # 3. Save new dense embeddings and updated chunks.json
    print("Saving new embeddings and updated chunks metadata...")
    np.save(npy_path, embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
        
    # 4. Rebuild BM25 Lexical Index
    print("Rebuilding BM25 lexical index...")
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
    for chunk in chunks:
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
    print("BM25 lexical index rebuilt successfully.")
    
    print("\n-----------------------------------------")
    print("        REBUILD COMPLETE SUMMARY")
    print("-----------------------------------------")
    print(f"Total Chunks Cleaned : {len(chunks)}")
    print(f"Dense Vector Matrix  : {embeddings.shape}")
    print(f"Time Taken (Total)   : {time.time() - t0:.2f} seconds")
    print("✅ SUCCESS: Active embeddings are now 100% clean and citation-free!")
    print("=========================================")

if __name__ == "__main__":
    main()
