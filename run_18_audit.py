"""
Quick sanity check: run the 18 problem questions (6 WRONG + 12 PARTIAL)
through the fixed pipeline and print answers for manual scoring.
"""
import sys, json, time
sys.path.append(".")
from medrag.hm_rag import HMRAGPipeline

QUESTIONS = [
    # 6 WRONG
    ("Q002", "What is the leading cause of cancer death worldwide?"),
    ("Q015", "What is the standard treatment for Stage IVa esophageal cancer?"),
    ("Q024", "Name the three resource stratification levels used by ASCO Global Guidelines."),
    ("Q035", "What is the 5-year overall survival rate for Stage I Nasopharyngeal Carcinoma?"),
    ("Q038", "What is the recommended screening strategy for colorectal cancer in adults aged 50–74?"),
    ("Q040", "What defines a T4b laryngeal tumor?"),
    # 12 PARTIAL
    ("Q005", "What is the most common early symptom of laryngeal cancer?"),
    ("Q013", "How does the Human Development Index (HDI) influence cancer trends?"),
    ("Q018", "What are the side effects of cisplatin in pediatric hepatoblastoma treatment?"),
    ("Q020", "What is the PRETEXT staging system for hepatoblastoma?"),
    ("Q022", "How does early diagnosis differ from cancer screening?"),
    ("Q023", "What is the recommended mammography screening interval for women aged 50–69?"),
    ("Q025", "How is a T2 glottic tumor staged?"),
    ("Q027", "What is the significance of EBV in nasopharyngeal carcinoma?"),
    ("Q028", "What is the treatment for Stage IVA laryngeal cancer with cartilage involvement?"),
    ("Q029", "What are the five Epstein grade groups for prostate cancer?"),
    ("Q039", "What is the NOURISHING framework?"),
    ("Q050", "What is the palliative radiotherapy dose for bone metastases?"),
]

pipeline = HMRAGPipeline()
results = []

for qid, question in QUESTIONS:
    print(f"\n{'='*60}")
    print(f"  {qid}: {question}")
    print(f"{'='*60}")
    t0 = time.time()
    res = pipeline.run(question)
    elapsed = time.time() - t0
    
    answer = res["answer"]
    sources = [c.get("source_file", "?") for c in res["context"]]
    
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Sources: {', '.join(set(sources))}")
    print(f"  ANSWER: {answer[:500]}")
    
    results.append({
        "id": qid,
        "question": question,
        "answer": answer,
        "sources": list(set(sources)),
        "latency": round(elapsed, 1),
    })

# Save results
with open("audit_18_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*60}")
print(f"  DONE — {len(results)} questions processed")
print(f"  Results saved to audit_18_results.json")
print(f"{'='*60}")
