# Medical RAG Explorer — Techniques & How It Works
=====================================================

## What is RAG?

RAG = Retrieval Augmented Generation

Instead of asking an LLM (AI) to answer from memory (which can hallucinate),
we first SEARCH our own documents for relevant text, then feed that text to
the LLM so it answers based on real evidence.

Flow:
  User Question → Search Knowledge Base → Retrieve Chunks → LLM Generates Answer


## Architecture Overview

```
User Question
     ↓
[1] Query Decomposition (LLM)
     ↓
[2] Hybrid Search (BM25 + Nomic Embeddings)
     ↓
[3] Reciprocal Rank Fusion (RRF)
     ↓
[4] CrossEncoder Reranker
     ↓
[5] LLM Answer Generation (Llama 3.2 3B)
     ↓
Answer
```

---

## [1] Query Decomposition

The original question is broken into 2-3 focused sub-queries using the LLM.

Example:
  Input:  "What are the recommended first-line chemotherapy regimens for advanced NSCLC?"
  Output: ["recommended first-line chemotherapy regimens",
           "advanced non-small cell lung cancer",
           "platinum-based doublet therapy"]

WHY: A single broad query may miss specific chunks. Multiple focused queries
     cast a wider net across the knowledge base.

The original full query is also always included to preserve semantic precision.


---

## [2a] BM25 — Sparse Retrieval

BM25 = Best Match 25 (a keyword-based ranking algorithm)

Formula:
  Score(D, Q) = Σ IDF(qᵢ) × [ f(qᵢ,D) × (k₁+1) ] / [ f(qᵢ,D) + k₁×(1 - b + b×|D|/avgdl) ]

Where:
  qᵢ       = each word in the query
  f(qᵢ, D) = how many times word qᵢ appears in document D
  |D|       = length of document D
  avgdl    = average document length in the corpus
  k₁       = 1.5 (term frequency saturation)
  b        = 0.75 (length normalization)
  IDF(qᵢ)  = log((N - n(qᵢ) + 0.5) / (n(qᵢ) + 0.5))
             N = total docs, n(qᵢ) = docs containing qᵢ

Simple English: BM25 counts keyword matches but penalizes very long documents
                and rewards rare words (IDF).

GOOD FOR: Exact keyword matching (drug names, gene names, specific terms)
BAD FOR:  Synonyms, paraphrasing ("heart attack" vs "myocardial infarction")


---

## [2b] Nomic Embeddings — Dense Retrieval

Model: nomic-ai/nomic-embed-text-v1.5

Each chunk of text is converted into a vector (a list of numbers) representing its meaning.
The query is also converted to a vector. Then similarity is computed between them.

Similarity Formula — Cosine Similarity:

  CosSim(A, B) = (A · B) / (||A|| × ||B||)

Where:
  A · B    = dot product of vectors A and B (multiply each pair, then sum)
  ||A||    = magnitude (length) of vector A
  Result   = number between -1 and 1  (1 = identical, 0 = unrelated)

Simple English: Sentences with similar meanings have vectors pointing in similar
                directions. This catches synonyms that BM25 misses entirely.

GOOD FOR: Semantic understanding, synonyms, related concepts
BAD FOR:  Exact rare terms (a brand new drug name may not be well represented)

### Matryoshka Representation Learning (MRL)

Nomic v1.5 uses MRL — named after Russian nesting dolls.

The model trains so the FIRST N dimensions are always the most important.
You can slice the vector at any size and it still works well:

  Full:      [d1, d2, d3, ..., d512, d513, ..., d768]   ← 768 dims
  Truncated: [d1, d2, d3, ..., d512]                     ← 512 dims ✅

  Sizes:    768   512   256   128   64
  Quality:  100%  99%   98%   96%   94%

WHY: Smaller vectors = faster search + less memory.
     32K chunks × 512 floats = 66MB vs 99MB at full 768.

OUR SETTING: 512 dimensions (matryoshka_dim: 512 in config.yaml)


Nomic uses a special prefix for queries:
  "search_query: What are the symptoms of cancer?"   ← query
  "search_document: Symptoms include..."              ← document


---

## [3] Reciprocal Rank Fusion (RRF)

Combines BM25 rankings and Embedding rankings into one fused ranking.

Formula:
  RRF(d) = Σ 1 / (k + rank(d))

Where:
  rank(d) = position of document d in each ranking list
  k       = 60 (constant to reduce impact of top ranks)

Example:
  Chunk A: BM25 rank=2, Embedding rank=5
  Chunk B: BM25 rank=10, Embedding rank=1

  RRF(A) = 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315
  RRF(B) = 1/(60+10) + 1/(60+1) = 0.0143 + 0.0164 = 0.0307

  → Chunk A wins because it ranks high in BOTH methods

WHY: Neither BM25 nor embeddings is perfect alone. RRF leverages both strengths.

Alpha parameter controls weighting:
  alpha = 0.5 → equal weight to BM25 and embeddings
  alpha = 1.0 → pure embedding
  alpha = 0.0 → pure BM25

Key fix we made: Used unique key = source_file + chunk_id to avoid
collision when multiple PDFs share the same chunk_id number.


---

## [4] CrossEncoder Reranker

Model: cross-encoder/ms-marco-MiniLM-L-6-v2

After hybrid search returns top 25 candidates, the reranker scores each one.

Unlike embeddings (which encode query and doc separately), a CrossEncoder
reads the query AND the document together:

  Input:  [Query] [SEP] [Document Chunk]
  Output: Relevance score (single number)

This is much more accurate than embedding similarity because it can
understand the relationship between the specific question and the text.

It's slower (processes pairs, not pre-computed), so we only apply it
to the top 25 candidates from hybrid search → outputs top 2 per sub-query.


---

## [5] LLM Answer Generation

Model: Llama 3.2 3B Instruct (GGUF quantized, runs on CPU)

The retrieved chunks are combined into a context prompt:

  System: You are a medical assistant. Answer using ONLY the provided context.
  Context: [Source 1]: ... [Source 2]: ...
  Question: [User's question]

The LLM generates a grounded answer based only on what was retrieved.

Context window: 4096 tokens (~3000 words)


---

## Evaluation Metrics We Compute

### Retrieval Metrics
  Precision@K  = (relevant chunks in top K) / K
  Recall@K     = (relevant chunks in top K) / (total relevant chunks)
  MRR          = 1 / (rank of first relevant chunk)
  NDCG@K       = normalized discounted cumulative gain (rewards top-ranked hits more)
  Hit-Rate@K   = 1 if ANY relevant chunk in top K, else 0

### Lexical Generation Metrics
  BLEU-N       = n-gram overlap between generated answer and reference
  ROUGE-L      = longest common subsequence between generated and reference
  METEOR       = BLEU + synonym matching + stemming
  Answer F1    = token-level precision/recall F1 between answer and reference

### Semantic Metrics
  SBERT Sim    = cosine similarity of sentence embeddings (all-MiniLM-L6-v2)
  BERTScore F1 = token-level BERT embedding similarity

### Faithfulness Metrics
  Faithfulness = fraction of answer sentences supported by retrieved context
                 (measured via SBERT cosine similarity > 0.45 threshold)
  Context Relevancy = cosine similarity between query and retrieved context
  Answer Relevance  = cosine similarity between query and generated answer


---

## Knowledge Base Stats

  Source files: Medical textbooks (MD Anderson, Oxford Handbook, etc.)
  Total chunks: ~32,000
  Chunk format: JSONL (one JSON object per line)
  Indexes built: BM25 (pickle) + Nomic Embeddings (numpy arrays)


---

## Key Files

  app.py                          ← Streamlit UI (chat interface)
  medrag/hm_rag.py               ← Full RAG pipeline
  medrag/search/hybrid_search.py ← RRF fusion
  medrag/search/bm25_search.py  ← BM25 index
  medrag/search/embedding_search.py ← Nomic dense index
  medrag/search/reranker.py     ← CrossEncoder reranker
  medrag/llm.py                 ← Llama LLM wrapper
  scripts/evaluate_rag.py       ← Full academic evaluation
  master_chunks.jsonl           ← All 32k chunks
  output/indices/bm25_index.pkl ← Pre-built BM25 index
  output/indices/embeddings/    ← Pre-built vector embeddings

=====================================================
