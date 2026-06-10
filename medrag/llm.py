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
        self.gguf_path = "/home/surdeep/.cache/huggingface/hub/models--mradermacher--Llama3-Med42-8B-GGUF/snapshots/7e2883406aaaee888cefbba8a50420062b484fee/Llama3-Med42-8B.Q8_0.gguf"
        
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
            
            # Optimized for speed — reduced context window
            self.model = Llama(
                model_path=self.gguf_path,
                n_ctx=3072,          # 3K context — balanced speed/quality
                n_threads=12,        # 12 CPU threads
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

You are a board-certified oncology expert. Answer the user's question directly using ONLY the provided context. 
CRITICAL INSTRUCTIONS:
- NO CONVERSATIONAL FILLER. Do NOT start your answer with "Based on the provided sources", "According to Source X", "In the context", or any similar phrases.
- SYNTHESIZE LIKE A CLINICAL EXAM KEY: Your goal is to rewrite the textbook facts into a single, highly concise, heavily condensed sentence that perfectly matches the phrasing of a human medical test answer key.
- AVOID BULLET POINTS: Write as a continuous flowing sentence, even when listing items.
- EXAMPLES OF DESIRED SBERT/CLINICAL STYLE:
  Question: What are the three main anatomical divisions of the larynx?
  Answer: The larynx is anatomically divided into the supraglottic larynx, the glottis, and the subglottis.
  Question: Which cancer type is currently the leading cause of cancer death worldwide?
  Answer: Lung cancer is currently the leading cause of cancer death worldwide, accounting for over 1.3 million deaths annually.
- CONCISENESS: Once you have answered the specific question asked, STOP immediately.<|eot_id|><|start_header_id|>user<|end_header_id|>

### Context:
{context_str}

### Question:
{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
