import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add local directory to path
sys.path.append('/home/surdeep/Downloads/medicine_rl')

from medrag.hm_rag import HMRAGPipeline

print("Initializing pipeline...")
pipeline = HMRAGPipeline()

# Run Q002
q002 = "Which cancer type is currently the leading cause of cancer death worldwide?"
print("="*80)
print(f"RUNNING Q002: {q002}")
res_q002 = pipeline.run(q002, use_rl=True)
print("\n[GENERATED ANSWER]:")
print(res_q002['answer'])
print("="*80)

# Run Q038
q038 = "What is the recommended screening for colorectal cancer in individuals aged 50 to 74?"
print("="*80)
print(f"RUNNING Q038: {q038}")
res_q038 = pipeline.run(q038, use_rl=True)
print("\n[GENERATED ANSWER]:")
print(res_q038['answer'])
print("="*80)
