import logging
import platform
from pathlib import Path
from medrag import config as cfg

logger = logging.getLogger(__name__)


def clean_source_name(filename: str) -> str:
    """Convert raw PDF filename into a clean, human-readable book title."""
    if not filename:
        return "Unknown Reference"
    name = filename.split('/')[-1]
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    name = name.replace('-', ' ').replace('_', ' ')
    
    # Capitalize properly
    words = name.split()
    clean_words = []
    for w in words:
        if w.lower() in ['of', 'and', 'the', 'for', 'in', 'to', 'with', 'on', 'at', 'by', 'from', 'an']:
            clean_words.append(w.lower())
        else:
            clean_words.append(w.capitalize())
            
    clean_name = " ".join(clean_words)
    if clean_name:
        clean_name = clean_name[0].upper() + clean_name[1:]
    return clean_name


class MedLLM:
    def __init__(self, model_id: str = None):
        self.model_id = model_id or cfg.get("llm", "model_id", "m42-health/Llama3-Med42-8B")
        self.use_mlx = platform.system() == "Darwin" and platform.processor() == "arm"
        
        # Hardcoded fast GGUF path for Linux
        self.gguf_path = "/home/surdeep/.cache/huggingface/hub/models--mradermacher--Llama3-Med42-8B-GGUF/snapshots/7e2883406aaaee888cefbba8a50420062b484fee/Llama3-Med42-8B.Q4_K_M.gguf"
        
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.use_mlx:
            logger.info("Using MLX for Apple Silicon")
            import mlx_lm
            self.model, self.tokenizer = mlx_lm.load(self.model_id)
        else:
            logger.info(f"Using llama_cpp_python with {self.gguf_path}")
            from llama_cpp import Llama
            self.model = Llama(
                model_path=self.gguf_path,
                n_ctx=8192, # 8K native context window
                n_threads=8, # Use 8 CPU cores
                verbose=False
            )

    def generate(self, prompt: str, max_new_tokens: int = None, **kwargs) -> str:
        if max_new_tokens is None:
            max_new_tokens = cfg.get("llm", "max_new_tokens", 512)
        
        if self.use_mlx:
            import mlx_lm
            return mlx_lm.generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                max_tokens=max_new_tokens,
                verbose=False
            )
        else:
            output = self.model(
                prompt,
                max_tokens=max_new_tokens,
                stop=["<|eot_id|>", "###", "</s>"],
                echo=False
            )
            return output["choices"][0]["text"].strip()

    def format_rag_prompt(self, query: str, context_chunks: list[dict]) -> str:
        formatted_chunks = []
        for i, c in enumerate(context_chunks):
            content = c.get('content', '')
            if '_parent_context' in c:
                content = f"{c['_parent_context']}\n\n{content}"
            book_name = clean_source_name(c.get('source_file', ''))
            formatted_chunks.append(f"Source [{i+1}: {book_name}]: {content}")
        context_str = "\n\n".join(formatted_chunks)
        
        # Use Llama 3 Prompt format
        return f"""<|start_header_id|>system<|end_header_id|>

You are a professional medical assistant. Answer the user's question directly using ONLY the provided context. 
CRITICAL INSTRUCTIONS:
- Do NOT use conversational filler like "Based on the context" or "I can provide".
- Output ONLY the direct medical answer. Do not include introductory remarks.
- STRICT RELEVANCE: Ignore chunks discussing diseases not asked about.
- CONCISENESS: Once you have answered the specific question asked, STOP immediately.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Context:
{context_str}

### Question:
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
