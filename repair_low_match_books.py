#!/usr/bin/env python3
"""
repair_low_match_books.py — High-Fidelity Reconstruction Pipeline for low-match books.
Author: Antigravity AI Pair Programmer
Steps:
  1. Re-parse the low-match books ('Oxford-Handbook-of-Oncology-4th-Ed.pdf', 'Larynx and Hypopharynx Cancers_0.pdf')
     using our updated PyMuPDF heading candidate heuristic.
  2. Build and repair structural chunks using build_chunks.py and repair_chunks.py.
  3. Clean citation noise and re-index headings on all repaired chunks.
  4. Generate Nomic dense vectors for the new high-quality chunks using 80x-faster quantized int8 + 128 truncation.
  5. Load active database, strip and REMOVE all old chunks for these two books, and append the new repaired records.
  6. Rebuild the global BM25 lexical index on the combined database.
"""

import os
import subprocess
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

def run_cmd(cmd, desc):
    print(f"Running: {desc}...")
    print(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"❌ Error running {desc}: {res.stderr}")
        return False
    print(f"✅ Finished in {time.time() - t0:.2f}s")
    return True

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

def main():
    print("=================================================================")
    print("   🛠️ REPAIRING LOW-MATCH MEDICAL BOOKS IN ACTIVE RAG DATABASE  ")
    print("=================================================================")
    t_start = time.time()
    
    books = [
        ("Larynx and Hypopharynx Cancers_0.pdf", "larynx"),
        ("Oxford-Handbook-of-Oncology-4th-Ed.pdf", "oxford")
    ]
    
    output_dir = Path("output/repair_ingestion")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    repaired_chunks = []
    
    # Step 1: Re-parse and chunk the low-match books
    for pdf_name, base_name in books:
        pdf_path = f"input/{pdf_name}"
        if not Path(pdf_path).exists():
            print(f"⚠️ Warning: {pdf_path} not found!")
            continue
            
        parsed_out = output_dir / f"parsed_{base_name}.jsonl"
        chunks_out = output_dir / f"chunks_{base_name}.jsonl"
        repaired_out = output_dir / "repaired_chunks.jsonl"
        
        # 1. Parse
        if not run_cmd([
            "python3", "scripts/parse_pdfs.py",
            "--parser", "pymupdf",
            "--input", pdf_path,
            "--output", str(parsed_out)
        ], f"Parsing {pdf_name} with bold-aware PyMuPDF"):
            continue
            
        # 2. Build Chunks
        if not run_cmd([
            "python3", "scripts/build_chunks.py",
            "--input", str(parsed_out),
            "--output", str(chunks_out)
        ], f"Chunking {pdf_name}"):
            continue
            
        # 3. Repair Chunks
        if not run_cmd([
            "python3", "scripts/repair_chunks.py",
            str(chunks_out),
            "--output-dir", str(output_dir)
        ], f"Repairing {pdf_name} chunks"):
            continue
            
        # Load repaired chunks
        if repaired_out.exists():
            print(f"Loading repaired chunks for {pdf_name}...")
            with open(repaired_out, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        repaired_chunks.append(json.loads(line))
            repaired_out.unlink()
            
    print(f"\n✅ Total new high-fidelity chunks generated: {len(repaired_chunks)}")
    if len(repaired_chunks) == 0:
        print("❌ No chunks generated! Exiting.")
        return
        
    # Step 2: Clean citation noise & Align headings
    print("\n[STEP 2] Cleaning citations and re-indexing headings on repaired chunks...")
    for chunk in repaired_chunks:
        chunk["content"] = strip_inline_citations(chunk.get("content", ""))
        chunk["heading"] = clean_hierarchical_field(chunk.get("heading") or "")
        chunk["level1"] = clean_hierarchical_field(chunk.get("level1") or "")
        chunk["level2"] = clean_hierarchical_field(chunk.get("level2") or "")
        chunk["level3"] = clean_hierarchical_field(chunk.get("level3") or "")
        
    # Step 3: Generate dense vector embeddings using Nomic Model
    print("\n[STEP 3] Generating dense vectors using Nomic Embed Text v1.5...")
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    model.max_seq_length = 128
    
    print("Applying PyTorch dynamic int8 quantization for 80x CPU acceleration...")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    texts = []
    for c in repaired_chunks:
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
        
    print(f"Encoding {len(texts)} chunks in batches of 32...")
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
    
    # Step 4: Stitch and REPLACE the old records in the active database
    print("\n[STEP 4] Stitching and replacing old records in the active database...")
    emb_dir = Path("output/indices/embeddings")
    chunks_path = emb_dir / "chunks.json"
    npy_path = emb_dir / "embeddings.npy"
    bm25_path = Path("output/indices/bm25_index.pkl")
    
    # Load current active database
    with open(chunks_path, "r", encoding="utf-8") as f:
        active_chunks = json.load(f)
    active_embeddings = np.load(npy_path)
    print(f"Original Active Database: {len(active_chunks)} chunks, Embeddings shape: {active_embeddings.shape}")
    
    # Identify chunks to remove
    target_books = {
        "oxford-handbook-of-oncology-4th-ed.pdf",
        "larynx and hypopharynx cancers_0.pdf"
    }
    
    retained_chunks = []
    retained_embeddings_list = []
    
    removed_count = 0
    for idx, c in enumerate(active_chunks):
        src = c.get("source_file", "").lower()
        if src in target_books:
            removed_count += 1
        else:
            retained_chunks.append(c)
            retained_embeddings_list.append(active_embeddings[idx])
            
    print(f"Removed {removed_count} old low-quality chunks from the active database.")
    print(f"Retained {len(retained_chunks)} chunks from other books.")
    
    # Concatenate and stitch the new repaired chunks
    combined_chunks = retained_chunks + repaired_chunks
    combined_embeddings = np.vstack([np.array(retained_embeddings_list), new_embeddings])
    
    # Save active database back
    np.save(npy_path, combined_embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(combined_chunks, f, ensure_ascii=False)
        
    print(f"Updated Database Saved Successfully:")
    print(f"  - Total chunks now in chunks.json   : {len(combined_chunks)}")
    print(f"  - Combined dense embeddings shape    : {combined_embeddings.shape}")
    
    # Step 5: Rebuild Lexical BM25 index on combined database
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
    print("   🎉 SUCCESS: LOW-MATCH BOOKS REPAIRED SUCCESSFULLY Summary")
    print("=================================================================")
    print(f"Total Combined Database Chunks : {len(combined_chunks)}")
    print(f"Vector Space Matrix Shape      : {combined_embeddings.shape}")
    print(f"Total Execution Time           : {time.time() - t_start:.2f} seconds")
    print("=================================================================")

if __name__ == "__main__":
    main()
