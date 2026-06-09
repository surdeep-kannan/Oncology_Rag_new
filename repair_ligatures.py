import pickle
import re
from pathlib import Path

# Load BM25 Index
pkl_path = Path("/home/surdeep/Downloads/bm25_index.pkl")
if not pkl_path.exists():
    raise FileNotFoundError(f"BM25 index not found at: {pkl_path}")

print("Loading BM25 index...")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
print(f"Loaded {len(chunks)} chunks.")

# Define ligature correction map
LIGATURE_MAP = {
    "/uniFB00": "ff",
    "/uniFB01": "fi",
    "/uniFB02": "fl",
    "/uniFB03": "ffi",
    "/uniFB04": "ffl",
    "\uFB00": "ff",
    "\uFB01": "fi",
    "\uFB02": "fl",
    "\uFB03": "ffi",
    "\uFB04": "ffl",
    "\u2010": "-",
    "\u2013": "-",
    "\u2014": "-",
}

# Stopwords & Tokenization logic to match bm25_search.py _tokenize
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "if", "in", "into", "is",
    "it", "its", "may", "more", "not", "of", "on", "or", "than",
    "that", "the", "their", "them", "there", "these", "they", "this",
    "those", "to", "was", "were", "what", "when", "which", "with",
    "you", "your", "can", "do", "does", "will", "would", "could",
    "should", "about", "after", "all", "also", "any", "been", "being",
    "between", "both", "each", "few", "get", "got", "her", "here",
    "him", "his", "how", "just", "like", "most", "must", "no", "nor",
    "only", "other", "our", "out", "own", "same", "she", "so", "some",
    "such", "very", "we",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

def clean_text(text: str) -> str:
    if not text:
        return ""
    
    # 1. Replace raw ligature strings/unicode
    for k, v in LIGATURE_MAP.items():
        text = text.replace(k, v)
        
    # 2. Merge split ligature space sequences
    # Rule A: Middle-split words (e.g. strati fi cation -> stratification)
    text = re.sub(r'\b([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)\b', r'\1\2\3', text)
    # Rule B: Start-split words (e.g. fi nal -> final, fl uid -> fluid)
    text = re.sub(r'\b(fi|fl|ff|ffi|ffl)\s+([a-zA-Z]+)\b', r'\1\2', text)
    # Rule C: End-split words (e.g. life -> life)
    text = re.sub(r'\b([a-zA-Z]+)\s+(fi|fl|ff|ffi|ffl)\b', r'\1\2', text)
    
    return text

print("Repairing ligatures and space splits in chunks...")
repaired_count = 0
new_corpus = []

for idx, c in enumerate(chunks):
    orig_content = c.get("content", "")
    orig_heading = c.get("heading", "")
    
    cleaned_content = clean_text(orig_content)
    cleaned_heading = clean_text(orig_heading)
    
    if cleaned_content != orig_content or cleaned_heading != orig_heading:
        repaired_count += 1
        c["content"] = cleaned_content
        c["heading"] = cleaned_heading
    
    # Re-tokenize
    combined_text = cleaned_content + " " + cleaned_heading
    tokenized = _tokenize(combined_text)
    new_corpus.append(tokenized)

print(f"Repaired and merged split words in {repaired_count} chunks.")

# Update the data structure
data["chunks"] = chunks
data["corpus"] = new_corpus

# Save the index back to disk
print("Saving repaired index back to disk...")
with open(pkl_path, "wb") as f:
    pickle.dump(data, f)

print("Repair complete! BM25 index is now fully corrected.")
