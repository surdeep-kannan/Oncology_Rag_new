#!/usr/bin/env python3
import json
import re
import pickle
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

def main():
    print("=========================================")
    print("   FAST INSTANT CITATION CLEANING")
    print("=========================================")
    
    emb_dir = Path("output/indices/embeddings")
    chunks_path = emb_dir / "chunks.json"
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
    
    # 2. Save updated chunks.json
    print("Saving updated clean chunks metadata...")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
        
    # 3. Rebuild BM25 Lexical Index
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
    print("✅ SUCCESS: Active text database is now 100% pruned of inline citation noise!")
    print("=========================================")

if __name__ == "__main__":
    main()
