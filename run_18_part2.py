"""Continue audit from Q023 onwards (7 remaining questions)."""
import sys, json, time
sys.path.append(".")
from medrag.hm_rag import HMRAGPipeline

QUESTIONS = [
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

with open("audit_18_part2.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\n{'='*60}")
print(f"  DONE — {len(results)} questions processed")
print(f"{'='*60}")
