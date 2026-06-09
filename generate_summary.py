import json

with open("oncology_questions_51_to_200_answered.json") as f:
    data = json.load(f)

print(f"# Clinical RAG Generation Review\n")
print(f"**Total Questions Completed:** {len(data)}\n")

for item in data:
    print(f"### {item['id']}: {item['category']}")
    print(f"**Q:** {item['q']}")
    print(f"**Expected:** {item.get('a', 'N/A')[:200]}...")
    print(f"**Generated:** {item.get('generated_answer', 'N/A')}")
    metrics = item.get('eval_metrics', {})
    print(f"**Metrics:** SBERT Sim: {metrics.get('sbert_sim', 0.0):.3f} | Faithfulness: {metrics.get('ragas_faithfulness', 0.0):.3f}")
    print("---\n")
