## TxtLab — Small-Model Book Summarization (Streamlit App)

A local Streamlit web app that demonstrates end‑to‑end “small‑model” book summarization on your own machine. It supports three open models, token‑accurate chunking with overlap and sentence‑aware splitting, five prompt variants, and a hierarchical “summaries‑of‑summaries” aggregator. The app is GPU‑first (8‑bit quantization with bitsandbytes) with CPU fallback.

### Key Features
- **Models**: Qwen2.5‑7B‑Instruct, Gemma‑3‑4B‑IT, Llama‑3.1‑8B‑Instruct
- **Chunking**: tokenizer‑based chunking by token length, overlap, and optional sentence‑aware boundaries
- **Prompts**: five single‑paragraph summary prompts with different focuses
- **Batching**: conservative, model‑aware batch size heuristics
- **Aggregation**: hierarchical intermediate summaries → final summary
- **UI**: device and VRAM info, chunk preview table, live per‑chunk summaries table, final summary, CSV/TXT downloads
- **Control**: context‑length warning, pad_token_id fallback, and a Stop button to abort at any time

## Project Structure
```text
TxtLab-LLM_sumarization/
  app/
    streamlit_app.py         # Streamlit UI
  src/
    models/
      loader.py              # Model/tokenizer loader (8-bit on CUDA; CPU fallback)
    chunking.py              # Tokenizer-based chunking + overlap + sentence-aware splitter
    prompts.py               # Five prompt variants
    pipeline.py              # Batch-size heuristics and per-chunk summarization helpers
    aggregate.py             # Hierarchical aggregator (no globals; takes model/tokenizer/device)
    utils.py                 # Reserved for shared helpers (currently empty)
  requirements.txt           # Transformers stack + Streamlit
  requirements-cuda121.txt   # Torch (CUDA 12.1 index)
  README.md                  # This file
```

## Requirements
- Python 3.10+ (tested with 3.11)
- For GPU: recent NVIDIA GPU with CUDA drivers; VRAM recommendations:
  - Qwen2.5‑7B‑Instruct (8‑bit): ~10–12 GB
  - Llama‑3.1‑8B‑Instruct (8‑bit): ~10–12 GB
  - Gemma‑3‑4B‑IT (8‑bit): ~6–8 GB
- For CPU: works but is slow for long texts

### About Hugging Face access tokens
Some checkpoints (e.g., Gemma) require license acceptance and a token. Create a token in your HF account and set it in the environment before running the app.

PowerShell (Windows):
```powershell
$env:HF_HUB_TOKEN = "YOUR_HF_TOKEN"
```

Bash (macOS/Linux/WSL):
```bash
export HF_HUB_TOKEN="YOUR_HF_TOKEN"
```

## Installation

### 1) Create and activate a virtual environment
PowerShell (Windows):
```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
```

Bash (macOS/Linux/WSL):
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install PyTorch
- GPU (CUDA 12.1):
```bash
pip install -r requirements-cuda121.txt
```

- CPU only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3) Install the remaining dependencies
```bash
pip install -r requirements.txt
```

Notes for Windows users:
- bitsandbytes has limited native support on Windows. The app only uses 8‑bit quantization when CUDA is available. If you are running CPU‑only on Windows and bitsandbytes installation fails, you can remove `bitsandbytes` from `requirements.txt` and run CPU‑only inference (slower). Alternatively, use WSL2 + CUDA for best GPU support.

## Running the App
From the repo root:
```bash
streamlit run app/streamlit_app.py
```

Then open the provided local URL in your browser.

## Using the App
1. Pick a model: `gemma`, `llama`, or `qwen`.
2. Adjust controls in the sidebar:
   - **Chunk length (tokens)**, **Overlap (tokens)**, **Sentence‑aware splitting**
   - **Prompt index (0–4)** to choose a summary style
   - **Final summary length** (max_new_tokens for aggregation)
3. Upload a `.txt` file (book or long document).
4. Click “Start summarization”. Use “Stop summarization” at any time to abort safely.
5. Watch:
   - Device/VRAM info, text preview
   - Chunk table (token counts and excerpts)
   - Live per‑chunk summaries table
   - Final aggregated summary
6. Download results as CSV (per‑chunk) or TXT (final summary).

### What happens under the hood
- The app loads the selected model in 8‑bit on CUDA (if available) or on CPU otherwise. It ensures `pad_token_id` is set.
- Your text is tokenized into chunks with overlap. If sentence‑aware splitting is enabled, chunk ends are trimmed to sentence boundaries so the next chunk starts at a clean sentence start.
- One of five prompts is applied to each chunk. Generation runs in batches, guided by a model‑specific batch size heuristic to limit VRAM usage.
- A two‑stage aggregator combines chunk summaries into intermediate summaries, then into a final summary.
- A basic context‑length warning appears if a prompt may exceed the model’s context window.
- The “Stop summarization” button interrupts between batches (and before final aggregation), preserving already generated partial results.

## Configuration and Customization
- Prompts: edit `src/prompts.py` (the `load_prompt` function) to change prompt styles.
- Chunking: edit `src/chunking.py` to customize the regex or overlap policy.
- Batch size heuristics: adjust `pick_batch_size` in `src/pipeline.py` or the UI logic in `app/streamlit_app.py`.
- Models: add aliases or modify defaults in `src/models/loader.py`.
- Aggregation: tweak group size and token budgets in `src/aggregate.py` or expose more controls in the Streamlit sidebar.

## Troubleshooting
- “Model requires access” or 401/403: accept the model’s license on Hugging Face and set `HF_HUB_TOKEN`.
- CUDA/driver issues: ensure your NVIDIA drivers and CUDA runtime are compatible with the selected PyTorch wheel.
- Out‑of‑memory (GPU): reduce chunk length, final summary length, or switch to a smaller model; the app already uses 8‑bit quantization on CUDA.
- Very slow on CPU: expected for 7–8B models; try `gemma` first or switch to a GPU/WSL2 environment.
- bitsandbytes build errors on Windows CPU: remove `bitsandbytes` from `requirements.txt` or use WSL2; the app will still run on CPU.

## Acknowledgments
- Models: `Qwen/Qwen2.5-7B-Instruct`, `google/gemma-3-4b-it`, `unsloth/Llama-3.1-8B-Instruct`.
- Built with `transformers`, `accelerate`, `safetensors`, `sentencepiece`, and `streamlit`.

## License
Add your project license here (e.g., MIT). Ensure that any model licenses you accept on Hugging Face are also respected when distributing this project.


