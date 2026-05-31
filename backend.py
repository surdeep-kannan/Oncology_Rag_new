import os
import json
import subprocess
import time
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from medrag.llm import MedLLM

app = FastAPI(title="MedRAG Backend")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing MedLLM...")
try:
    llm = MedLLM()
    print("MedLLM initialized successfully.")
except Exception as e:
    print(f"Error initializing MedLLM: {e}")
    llm = None

@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
        
    originalPrompt = data.get("originalPrompt", "")
    optimizedPrompt = data.get("optimizedPrompt", originalPrompt)
    sessionId = data.get("sessionId", "")
    
    print(f"Received query: {optimizedPrompt}")

    async def generate_candidates(opt_prompt):
        cmd = ["python", "scripts/search_chunks.py", opt_prompt, "--top-k", "5", "--json", "--rerank"]
        result = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True)
        results = []
        if result.returncode == 0:
            try:
                parsed = json.loads(result.stdout)
                results = parsed.get("results", [])
            except:
                pass
        
        cands = []
        if llm:
            rag_prompt = llm.format_rag_prompt(opt_prompt, results)
            reasoning_prompt = f"Provide ONLY a direct clinical reasoning chain for: {opt_prompt}. Do NOT use conversational filler."
            base_prompt = f"Answer as a medical expert: {opt_prompt}. Provide ONLY the direct medical answer. Do NOT use conversational filler."

            
            def gen(p):
                try:
                    # Limit token generation to stop rambling and context drift
                    return llm.generate(p, max_new_tokens=200)
                except Exception as e:
                    return f"Error: {str(e)}"
            
            cands = []
            cands.append(gen(base_prompt))
            cands.append(gen(rag_prompt))
            cands.append(gen(reasoning_prompt))
            cands.append(gen(rag_prompt + "\nBe extremely concise and stop after answering."))

        else:
            cands = [
                "LLM is not initialized correctly.",
                "Fallback response due to missing MedLLM.",
                "Error: No model available.",
                "Clinical reasoning disabled."
            ]
            
        return cands

    def lightweight_evaluate(cands, original_prompt, optimized_prompt):
        scores = []
        prompt_words = set(optimized_prompt.lower().split())
        
        for cand in cands:
            if not cand or "Error:" in cand or "LLM is not initialized" in cand:
                scores.append({"accuracy": 0, "f1": 0, "rouge_l": 0, "distinct": 0, "total": 0})
                continue
                
            cand_words = cand.lower().split()
            length = len(cand_words)
            
            # DISTINCT (Vocabulary richness)
            unique_words = len(set(cand_words))
            distinct_score = (unique_words / length) if length > 0 else 0
            
            # Semantic Overlap
            overlap = len(prompt_words.intersection(set(cand_words)))
            precision = overlap / length if length > 0 else 0
            recall = overlap / len(prompt_words) if len(prompt_words) > 0 else 0
            
            # F1 Score & ROUGE
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            rouge_l = recall * 0.85
            
            # Extreme penalty for context drift (Long rambling answers get destroyed)
            accuracy = precision * 2.0
            tangent_penalty = 1.0
            if length > 120:
                tangent_penalty = 0.3  # Drifting into other diseases
            elif length > 80:
                tangent_penalty = 0.6  # Getting too wordy
                
            total_score = (f1_score * 0.4 + rouge_l * 0.2 + distinct_score * 0.1 + accuracy * 0.3) * tangent_penalty
            
            scores.append({
                "accuracy": accuracy,
                "f1": f1_score,
                "rouge_l": rouge_l,
                "distinct": distinct_score,
                "total": total_score
            })
            
        return scores

    def apply_rrf(cands, scores_dicts):
        # Reciprocal Rank Fusion (RRF) over multiple advanced metrics
        k = 60
        rrf_scores = [0] * len(cands)
        
        metrics = ["accuracy", "f1", "rouge_l", "distinct", "total"]
        for metric in metrics:
            ranked = sorted(list(enumerate(scores_dicts)), key=lambda x: x[1][metric], reverse=True)
            for rank, (idx, _) in enumerate(ranked):
                rrf_scores[idx] += 1.0 / (k + rank + 1)
                
        best_idx = max(range(len(rrf_scores)), key=rrf_scores.__getitem__)
        return cands[best_idx]

    async def generate_response():
        cands = await generate_candidates(optimizedPrompt)
        
        # Pass both original and optimized prompt for advanced relevance scoring
        scores_dicts = lightweight_evaluate(cands, originalPrompt, optimizedPrompt)

        
        # Step 3 & 4: Apply RRF and select the highest ranked answer
        best_answer = apply_rrf(cands, scores_dicts)
        
        if not best_answer:
            yield "Sorry, no valid response could be generated."
            return
            
        # Step 5: Return ONLY best final answer to frontend (streamed)
        words = best_answer.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01) # Simulate stream token generation

    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
