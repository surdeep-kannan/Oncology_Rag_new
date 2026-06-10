import json
from medrag.hm_rag import HMRAGPipeline

print('Loading pipeline with Groq...')
pipeline = HMRAGPipeline()

with open('oncology_questions_100 (3) (2).json') as f:
    questions = json.load(f)[:3]

print('\nTesting 3 questions using Groq API for speed...')

sbert_total = 0
faith_total = 0

for idx, q in enumerate(questions):
    print(f'\n--- Q{idx+1}: {q["q"][:60]}... ---')
    result = pipeline.run(q['q'])
    
    metrics = result['eval_metrics']
    sbert = metrics.get('sbert_sim', 0)
    faith = metrics.get('ragas_faithfulness', 0)
    rouge = metrics.get('rouge_l', 0)
    
    sbert_total += sbert
    faith_total += faith
    
    print(f'SBERT: {sbert:.3f} | Faithfulness: {faith:.3f} | ROUGE-L: {rouge:.3f}')
    print(f'Generated Answer: {result["answer"][:150]}...')

print('\n--- NEW AVERAGES (GROQ + EXTRACTIVE) ---')
print(f'SBERT: {sbert_total/3:.3f} (Old: 0.473)')
print(f'Faithfulness: {faith_total/3:.3f} (Old: 0.438)')
