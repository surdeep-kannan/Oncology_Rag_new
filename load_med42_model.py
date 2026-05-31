"""
Download and run the Hugging Face model `med42/llama3-med42-8b`.

Prerequisites:
1. Install dependencies:
   python -m pip install transformers accelerate torch
2. Make sure you are already authenticated:
   hf auth login

Note:
- This is an 8B model. On many Macs, `device_map="auto"` will offload part of the
  model to disk, which can make generation extremely slow.
"""

from pathlib import Path

import torch
from transformers import modeling_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "m42-health/Llama3-Med42-8B"
MAX_NEW_TOKENS = 32
OFFLOAD_DIR = Path("offload")


def get_input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "hf_device_map"):
        for mapped_device in model.hf_device_map.values():
            if mapped_device not in {"cpu", "disk"}:
                return torch.device(mapped_device)
    return next(model.parameters()).device


def disable_mps_allocator_warmup() -> None:
    original_warmup = modeling_utils.caching_allocator_warmup

    def patched_warmup(model, expanded_device_map, hf_quantizer):
        device_values = {str(device) for device in expanded_device_map.values()}
        if any(device.startswith("mps") for device in device_values):
            return
        return original_warmup(model, expanded_device_map, hf_quantizer)

    modeling_utils.caching_allocator_warmup = patched_warmup


def main() -> None:
    disable_mps_allocator_warmup()

    print(f"Loading tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model for {MODEL_ID}...")
    OFFLOAD_DIR.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder=str(OFFLOAD_DIR),
    )

    if hasattr(model, "hf_device_map") and "disk" in set(model.hf_device_map.values()):
        print("Warning: part of the model was offloaded to disk. Generation may be very slow on this Mac.\n")

    input_device = get_input_device(model)

    print("Interactive mode is ready.")
    print("Type a medical question and press Enter.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        prompt = input("Question: ").strip()
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(input_device)

        print(f"Generating response with max_new_tokens={MAX_NEW_TOKENS}...\n")
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = generated_text[len(prompt) :].strip() if generated_text.startswith(prompt) else generated_text.strip()
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
