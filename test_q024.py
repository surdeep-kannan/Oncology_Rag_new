import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add medicine_rl path
sys.path.append('/home/surdeep/Downloads/medicine_rl')

from medrag.hm_rag import HMRAGPipeline

print("Initializing pipeline...")
pipeline = HMRAGPipeline()

q024 = "What are the resource stratification levels in ASCO Global Guidelines?"
print("="*80)
print(f"RUNNING Q024: {q024}")
res = pipeline.run(q024, use_rl=True)
print("\n[GENERATED ANSWER]:")
print(res['answer'])
print("="*80)
print("\n--- RETRIEVED SOURCES ---")
for i, ctx in enumerate(res.get('context_chunks', [])):
    book_name = ctx.get('source_file', '')
    print(f"Source {i+1}: {book_name} | Heading: {ctx.get('heading')}")
    print(ctx.get('content')[:300].strip())
    print('-'*50)
print("="*80)
