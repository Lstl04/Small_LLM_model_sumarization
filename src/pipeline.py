# src/pipeline.py
import torch
from .prompts import load_prompt

def pick_batch_size(model_name: str, chunk_tokens: int) -> int:
    """
    Return a conservative, hard-coded batch size based on model and chunk size.
    Adjust these if your GPU has more VRAM.
    """
    name = (model_name or "").lower()
    # You can alias model families to avoid having to list every checkpoint string
    if "qwen" in name:
        table = {2048: 4, 4096: 2, 8192: 1}
    elif "llama" in name:
        table = {2048: 3, 4096: 2, 8192: 1}
    elif "gemma" in name or "google" in name:
        table = {2048: 2, 4096: 1, 8192: 1}
    else:
        table = {2048: 4, 4096: 2, 8192: 1}  # safe default

    # Fall back to 1 if chunk_tokens isn't one of your three options
    return table.get(int(chunk_tokens), 1)

def summarize_chunks(model, tokenizer, device, chunks, prompt_index, model_name, chunk_tokens):
    bs = pick_batch_size(model_name, chunk_tokens)
    # Build prompts
    prompts = [load_prompt(prompt_index, c) for c in chunks]
    out = [None]*len(chunks)
    for start in range(0, len(prompts), bs):
        batch = prompts[start:start+bs]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k,v in enc.items()}
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=350, temperature=0.7, top_p=0.9, do_sample=True)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, txt in enumerate(decoded):
            out[start+j] = txt.split("Summary:")[-1].strip()
    return out
