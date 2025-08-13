import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ALIASES = {
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-4b-it",
    "llama": "unsloth/Llama-3.1-8B-Instruct",
}

def load_model_and_tokenizer(choice: str):
    name = MODEL_ALIASES[choice]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_HUB_TOKEN", None)

    if device == "cuda":
        quant = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=quant, device_map="auto",
            trust_remote_code=True, token=token or None
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True, token=token or None
        ).to(device)

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, token=token or None)
    model.eval()
    return model, tok, device, name
