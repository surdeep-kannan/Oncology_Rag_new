# MedRAG – Hierarchical PDF Chunking Pipeline for Medical RAG

A production-quality Python pipeline for extracting structured, hierarchical chunks from medical PDFs (NCCN guidelines, clinical documents) for Retrieval-Augmented Generation.

## Architecture

```
medrag/
├── parsers/           # Docling (primary) + PyMuPDF (fallback)
├── cleaning/          # Noise filter, OCR repair, heading detection
├── chunking/          # Hierarchy builder, semantic chunk engine
├── search/            # BM25, embeddings, hybrid, reranker
└── utils/             # Visualization, tree export

scripts/
├── parse_pdfs.py      # Step 1: Parse PDFs → structured blocks
├── build_chunks.py    # Step 2: Build hierarchical chunks
├── inspect_chunks.py  # Step 3: Inspect & evaluate quality
└── search_chunks.py   # Step 4: Search with hybrid retrieval
```

```

## 🚀 Super-Fast AI Agent Setup (For Antigravity Users)

If you are using the **Antigravity AI coding assistant**, you don't need to do any manual setup! 

### 1. The 1-Step Setup Prompt
Once you have cloned this repository and downloaded the `indices.zip` database folder from your friend, simply **copy and paste this exact prompt** to Antigravity:

> "Hey Antigravity! I just cloned my friend's Medical RAG repository and downloaded their `indices.zip` database. Please:
> 1. Unzip indices.zip into the `output/` folder so it creates `output/indices/`
> 2. Enable system site-packages in my virtual environment (open `venv/pyvenv.cfg` if it exists, and set `include-system-site-packages = true`)
> 3. Run check_gpu.py to detect my GPU device and configure config.yaml
> 4. Start the Streamlit application in the background on port 8502 so I can test it!"

### 2. Manual Quick Start (If setting up without an AI Agent)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place PDFs in input/ directory
cp my_guidelines.pdf input/

# 3. Parse PDFs
python scripts/parse_pdfs.py

# 4. Build chunks
python scripts/build_chunks.py

# 5. Inspect quality
python scripts/inspect_chunks.py --stats --eval

# 6. Search
python scripts/search_chunks.py "treatment for stage III breast cancer"
```

## Pipeline Steps

### Step 1: Parse PDFs

```bash
# Parse all PDFs in input/
python scripts/parse_pdfs.py

# Parse a single PDF
python scripts/parse_pdfs.py --input "adult cancer guidelinespdf.pdf"

# Force PyMuPDF parser
python scripts/parse_pdfs.py --parser pymupdf

# Debug mode
python scripts/parse_pdfs.py --debug
```

### Step 2: Build Chunks

```bash
# Build from parsed blocks
python scripts/build_chunks.py

# Direct PDF → chunks (skip step 1)
python scripts/build_chunks.py --pdf "adult cancer guidelinespdf.pdf"

# Custom chunk sizes
python scripts/build_chunks.py --min-words 200 --max-words 600

# Visualize hierarchy
python scripts/build_chunks.py --visualize

# Export hierarchy tree
python scripts/build_chunks.py --export-tree output/tree.json
```

### Step 3: Inspect & Evaluate

```bash
# Summary statistics
python scripts/inspect_chunks.py --stats

# Quality evaluation
python scripts/inspect_chunks.py --eval

# Random samples
python scripts/inspect_chunks.py --sample 5

# Filter by heading
python scripts/inspect_chunks.py --heading "Treatment"

# Search content
python scripts/inspect_chunks.py --search "chemotherapy"

# Hierarchy tree
python scripts/inspect_chunks.py --tree

# Specific chunk
python scripts/inspect_chunks.py --chunk-id 42
```

### Step 4: Search

```bash
# Hybrid search (BM25 + embeddings)
python scripts/search_chunks.py "chemotherapy side effects"

# BM25 only
python scripts/search_chunks.py "staging" --mode bm25

# With reranking
python scripts/search_chunks.py "radiation therapy" --rerank

# With parent context
python scripts/search_chunks.py "fertility" --rerank --parent-context

# Metadata filter
python scripts/search_chunks.py "surgery" --level1 "Treatment"

# Rebuild indices
python scripts/search_chunks.py "diagnosis" --build-index
```

## Output Format

Each chunk in the JSONL output:

```json
{
  "chunk_id": 1,
  "source_file": "guidelines.pdf",
  "level1": "Treatment",
  "level2": "Chemotherapy",
  "level3": "Side Effects",
  "heading": "Side Effects",
  "content": "Chemotherapy can cause several side effects...",
  "page_start": 42,
  "page_end": 43,
  "token_count": 312
}
```

## Configuration

Edit `config.yaml` to customize:

- **Parsing**: Parser selection, OCR settings
- **Cleaning**: Header/footer thresholds, noise patterns
- **Heading Detection**: Font deltas, title-case rules
- **Chunking**: Word count targets (200–600), overlap
- **Search**: Embedding model, BM25 params, reranker

## Key Features

| Feature | Implementation |
|---|---|
| Primary parser | Docling (structural parsing) |
| Fallback parser | PyMuPDF (font-based heuristics) |
| Heading detection | Hybrid: parser hints + font + regex + title-case |
| OCR repair | camelCase splitting, unicode normalization |
| Noise removal | Repeated headers/footers, page numbers, TOC |
| Chunking | Sentence-aware semantic splitting |
| Overlap | Configurable sentence/word overlap |
| Search | BM25 + sentence-transformers + RRF fusion |
| Reranking | Cross-encoder (ms-marco-MiniLM) |
| Parent-child | Sibling chunk context for enriched retrieval |

## Design Principles

1. **Hierarchy quality** – breadcrumb parsing, multi-level heading detection
2. **Chunk boundary quality** – sentence-aware splits, no mid-sentence breaks
3. **Retrieval quality** – hybrid search, reranking, parent context
4. **Scalability** – batch processing, configurable, 2000+ pages tested
5. **Maintainability** – modular codebase, logging, config-driven
