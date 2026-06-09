import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add medicine_rl path
sys.path.append('/home/surdeep/Downloads/medicine_rl')

from medrag.hm_rag import HMRAGPipeline

print("Initializing pipeline...")
pipeline = HMRAGPipeline()

q002 = "What is the leading cause of cancer death worldwide?"
print("="*80)
print(f"RUNNING Q002: {q002}")
res = pipeline.run(q002, use_rl=True)
print("\n[GENERATED ANSWER]:")
print(res['answer'])
print("="*80)
