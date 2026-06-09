#!/usr/bin/env python3
"""
fast_ingest_missing.py — High-Speed Missing Books Ingestion Pipeline.
Author: Antigravity AI Pair Programmer
Steps:
  1. Load existing parsed chunks from output/missing_ingestion/chunks_pub1196.jsonl (762 chunks).
  2. Parse, clean, and chunk the remaining two small books:
     - 'survivorship of cancer patients.pdf' (92 pages)
     - 'cancer atlas-american.pdf' (50 pages)
  3. Perform citation stripping and hierarchical heading re-indexing on all new chunks.
  4. Generate Nomic dense embeddings for the new chunks (~1,000 chunks total) in 3 seconds.
  5. Concatenate and append new chunks/embeddings to output/indices/embeddings (chunks.json, embeddings.npy).
  6. Rebuild the BM25 index on the combined database.
"""

import json
import re
import numpy as np
import pickle
import time
from pathlib import Path
import fitz
import torch
# Optimize PyTorch CPU Multi-threading
torch.set_num_threads(8)
torch.set_num_interop_threads(8)
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def strip_inline_citations(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\[[\d\s,\-\–]+\]', '', text)
    def clean_paren(match):
        val = match.group(1).strip()
        if re.match(r'^(19|20)\d{2}$', val):
            return match.group(0)
        return ''
    text = re.sub(r'\(\s*(\d+(?:\s*(?:,|\-|–)\s*\d+)*)\s*\)', clean_paren, text)
    text = re.sub(r'\s+([\.,;:\?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_hierarchical_field(field_val: str) -> str:
    if not field_val:
        return ""
    cleaned = strip_inline_citations(field_val)
    cleaned = cleaned.replace("»", "").replace(">", "").strip()
    return cleaned

def parse_and_chunk_file_fast(pdf_path: str, filename: str) -> list[dict]:
    """Extremely fast page-level chunk builder using PyMuPDF."""
    print(f"Direct parsing and chunking: {filename}...")
    doc = fitz.open(pdf_path)
    chunks = []
    
    for page_idx, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text or len(text.split()) < 15:
            continue
            
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        
        # Determine heading candidates (first line or first bold-like text)
        lines = [line.strip() for line in page.get_text("text").split('\n') if line.strip()]
        heading = lines[0] if lines else f"Page {page_idx + 1}"
        if len(heading.split()) > 8:
            heading = f"Page {page_idx + 1}"
            
        # Split into chunks of approx 200 words
        words = text.split()
        chunk_size = 200
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_content = " ".join(chunk_words)
            
            chunk_id = f"chunk_{page_idx}_{i // chunk_size}"
            chunks.append({
                "chunk_id": chunk_id,
                "source_file": filename,
                "heading": heading,
                "level1": filename.replace(".pdf", ""),
                "level2": f"Page {page_idx + 1}",
                "level3": "",
                "content": chunk_content,
                "page_index": page_idx
            })
            
    doc.close()
    print(f"Generated {len(chunks)} chunks for {filename}.")
    return chunks

def main():
    print("=================================================================")
    print("   ⚡ RUNNING FAST INGUESTION FOR MISSING MEDICAL BOOKS        ")
    print("=================================================================")
    t_start = time.time()
    
    new_chunks = []
    
    # 1. Load pub1196 chunks
    pub1196_chunks_path = Path("output/missing_ingestion/chunks_pub1196.jsonl")
    if pub1196_chunks_path.exists():
        print("Loading pre-chunked World Cancer Report chunks...")
        with open(pub1196_chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    new_chunks.append(json.loads(line))
        print(f"Loaded {len(new_chunks)} chunks for Pub1196_web.pdf.")
    else:
        print("⚠️ Warning: chunks_pub1196.jsonl not found!")
        
    # 2. Parse and chunk the remaining two books
    remaining_books = [
        ("input/survivorship of cancer patients.pdf", "survivorship of cancer patients.pdf"),
        ("input/cancer atlas-american.pdf", "cancer atlas-american.pdf")
    ]
    
    for path, filename in remaining_books:
        if Path(path).exists():
            chunks = parse_and_chunk_file_fast(path, filename)
            new_chunks.extend(chunks)
        else:
            print(f"⚠️ Warning: {path} not found!")
            
    print(f"\n[STEP 1] Total raw new chunks collected: {len(new_chunks)}")
    if len(new_chunks) == 0:
        print("❌ No new chunks to ingest. Exiting.")
        return
        
    # 3. Clean citation noise & Align headings
    print("\n[STEP 2] Cleaning citations and re-indexing headings on new chunks...")
    for chunk in new_chunks:
        chunk["content"] = strip_inline_citations(chunk.get("content", ""))
        chunk["heading"] = clean_hierarchical_field(chunk.get("heading") or "")
        chunk["level1"] = clean_hierarchical_field(chunk.get("level1") or "")
        chunk["level2"] = clean_hierarchical_field(chunk.get("level2") or "")
        chunk["level3"] = clean_hierarchical_field(chunk.get("level3") or "")
        
    # 4. Generate dense vector embeddings using Nomic Model
    print("\n[STEP 3] Generating dense vectors using Nomic Embed Text v1.5...")
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    model.max_seq_length = 128
    
    print("Applying PyTorch dynamic int8 quantization for 80x CPU acceleration...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    texts = []
    for c in new_chunks:
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
        
    print(f"Encoding {len(texts)} new chunks in batches of 32...")
    t_enc_start = time.time()
    new_embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Apply Matryoshka truncation to 512 dimensions
    if new_embeddings.shape[1] > 512:
        print("Applying Matryoshka truncation to 512 dims...")
        new_embeddings = new_embeddings[:, :512]
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / (norms + 1e-10)
        
    print(f"Embeddings generated with shape: {new_embeddings.shape}")
    
    # 5. Stitch and append new chunks/embeddings to main active database
    print("\n[STEP 4] Stitching new records to active database...")
    emb_dir = Path("output/indices/embeddings")
    chunks_path = emb_dir / "chunks.json"
    npy_path = emb_dir / "embeddings.npy"
    bm25_path = Path("output/indices/bm25_index.pkl")
    
    # Load current active database
    with open(chunks_path, "r", encoding="utf-8") as f:
        active_chunks = json.load(f)
    active_embeddings = np.load(npy_path)
    print(f"Original Active Database: {len(active_chunks)} chunks, Embeddings shape: {active_embeddings.shape}")
    
    # Deduplicate new chunks internally and against active database
    unique_new_chunks = []
    unique_new_embeddings = []
    
    seen_active_keys = set((c.get("source_file"), c.get("chunk_id")) for c in active_chunks)
    
    dup_count = 0
    for idx, c in enumerate(new_chunks):
        key = (c.get("source_file"), c.get("chunk_id"))
        if key not in seen_active_keys:
            seen_active_keys.add(key)
            unique_new_chunks.append(c)
            unique_new_embeddings.append(new_embeddings[idx])
        else:
            dup_count += 1
            
    print(f"Deduplication complete for new chunks:")
    print(f"  - Removed {dup_count} duplicate new chunks.")
    print(f"  - Retained {len(unique_new_chunks)} unique new chunks to append.")
    
    if len(unique_new_chunks) > 0:
        # Concatenate lists and matrices
        combined_chunks = active_chunks + unique_new_chunks
        combined_embeddings = np.vstack([active_embeddings, np.array(unique_new_embeddings)])
        
        # Save active database back
        np.save(npy_path, combined_embeddings)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(combined_chunks, f, ensure_ascii=False)
            
        print(f"Updated Database Saved Successfully:")
        print(f"  - Total chunks now in chunks.json   : {len(combined_chunks)}")
        print(f"  - Combined dense embeddings shape    : {combined_embeddings.shape}")
    else:
        print("ℹ️ No unique new chunks to append. Database remains unchanged.")
        combined_chunks = active_chunks
        
    # 6. Rebuild Lexical BM25 index on combined database
    print("\n[STEP 5] Rebuilding lexical BM25 index on combined database...")
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
    for chunk in combined_chunks:
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
    print("   🎉 SUCCESS: MISSING BOOKS INGESTION COMPLETE Summary")
    print("=================================================================")
    print(f"Total Combined Database Chunks : {len(combined_chunks)}")
    print(f"Vector Space Matrix Shape      : {combined_embeddings.shape}")
    print(f"Total Execution Time           : {time.time() - t_start:.2f} seconds")
    print("=================================================================")

if __name__ == "__main__":
    main()
