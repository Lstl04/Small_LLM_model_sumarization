# app/streamlit_app.py
import os, sys
from pathlib import Path

import streamlit as st
import pandas as pd
import torch

# Let us import your modules from the repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.loader import load_model_and_tokenizer
from src.chunking import process_chunks
from src.prompts import load_prompt
from src.aggregate import aggregate_summaries  # requires the fixed (no-globals) version

st.set_page_config(page_title="LLM Small Summarizer", layout="wide")
st.title("📚 LLM Small Summarizer")
st.caption("Local, GPU-first (CPU fallback). Upload a .txt book, watch chunking & per-chunk summaries, then get the final summary.")

# -------------------------------
# Cache — keep model in memory across reruns
# -------------------------------
@st.cache_resource(show_spinner=False)
def _load_model(choice: str):
    model, tok, device, name = load_model_and_tokenizer(choice)
    # Ensure pad_token_id is set to avoid generate() warnings
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        tok.pad_token_id = tok.eos_token_id
    model.eval()
    return model, tok, device, name

# -------------------------------
# Helpers
# -------------------------------
def device_banner(device: str, model_name: str):
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(props.total_memory / (1024**3), 1)
            st.success(f"Using **GPU**: {torch.cuda.get_device_name(0)} — {vram_gb} GB VRAM")
        except Exception:
            st.success("Using **GPU (CUDA)**")
    else:
        st.warning("Using **CPU** — this will be slower.")
    st.write(f"Selected model: `{model_name}`")

def count_tokens(tok, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def pick_batch_size_heuristic(model_name: str, chunk_tokens: int) -> int:
    name = (model_name or "").lower()
    if "qwen" in name:
        table = {2048: 4, 4096: 2, 8192: 1}
    elif "llama" in name:
        table = {2048: 3, 4096: 2, 8192: 1}
    elif "gemma" in name or "google" in name:
        table = {2048: 2, 4096: 1, 8192: 1}
    else:
        table = {2048: 4, 4096: 2, 8192: 1}
    return table.get(int(chunk_tokens), 1)

def summarize_chunks_streaming(model, tok, device, chunks, prompt_index, chunk_tokens, model_name, status_container):
    bs = pick_batch_size_heuristic(model_name, chunk_tokens)

    # Build prompts and sanity-check context
    prompts = [load_prompt(prompt_index, c) for c in chunks]
    ctx = int(getattr(model.config, "max_position_embeddings", 4096))
    reserve = 128
    over = sum(1 for p in prompts if count_tokens(tok, p) > ctx - reserve)
    if over:
        st.warning(f"{over} prompt(s) may exceed the model context ({ctx}). They will be truncated. Consider reducing chunk length.")

    out = [None] * len(chunks)
    progress = st.progress(0)
    total = len(prompts)

    for start in range(0, len(prompts), bs):
        batch = prompts[start:start+bs]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=350,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        for j, txt in enumerate(decoded):
            # Your prompts end with "Summary:" — grab everything after
            out[start + j] = txt.split("Summary:")[-1].strip()

        progress.progress(min(1.0, (start + bs) / total))

        # Live table preview
        with status_container:
            df = pd.DataFrame({
                "chunk_id": list(range(len(out))),
                "summary_ready": [o is not None for o in out],
                "summary_excerpt": [("" if o is None else (o[:200] + ("..." if len(o) > 200 else ""))) for o in out]
            })
            st.dataframe(df, use_container_width=True)

    progress.progress(1.0)
    return out

# -------------------------------
# Sidebar controls
# -------------------------------
with st.sidebar:
    st.header("Controls")
    model_choice   = st.selectbox("Model", ["gemma", "llama", "qwen"], index=0)
    chunk_tokens   = st.slider("Chunk length (tokens)", 512, 8192, 2048, step=256)
    overlap_tokens = st.slider("Overlap (tokens)", 0, 512, 150, step=10)
    sentence_aware = st.checkbox("Sentence-aware splitting", value=True)
    prompt_index   = st.number_input("Prompt index (0-4)", min_value=0, max_value=4, value=0, step=1)
    final_len      = st.slider("Final summary length (max_new_tokens)", 200, 1200, 600, step=50)

    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    start = st.button("Start summarization")

# -------------------------------
# Main flow
# -------------------------------
if not start:
    st.info("Upload a .txt book and click **Start summarization**.")
    st.stop()

if uploaded is None:
    st.error("Please upload a .txt file.")
    st.stop()

# Load model once per selection
with st.spinner("Loading model and tokenizer…"):
    model, tok, device, full_name = _load_model(model_choice)
device_banner(device, full_name)

# Read text
text = uploaded.read().decode("utf-8", errors="ignore")
st.subheader("Step 1 — File loaded")
st.write(f"Characters: {len(text):,}")
st.text_area("Preview (first 1200 chars)", value=text[:1200], height=200)

# Chunking
st.subheader("Step 2 — Chunking")
with st.spinner("Splitting into chunks…"):
    chunks = process_chunks(
        text,
        tok,
        chunk_length=chunk_tokens,
        chunk_overlap=overlap_tokens,
        sentence_aware=sentence_aware,
    )
st.success(f"Created {len(chunks)} chunks.")

# Show chunk table
chunk_token_counts = [count_tokens(tok, c) for c in chunks]
df_chunks = pd.DataFrame({
    "chunk_id": list(range(len(chunks))),
    "tokens": chunk_token_counts,
    "start_excerpt": [c[:120] + ("…" if len(c) > 120 else "") for c in chunks],
    "end_excerpt":   [c[-120:] if len(c) > 120 else c for c in chunks],
})
st.dataframe(df_chunks, use_container_width=True, height=300)

# Per-chunk summaries
st.subheader("Step 3 — Per-chunk summarization")
status_container = st.container()
with st.spinner("Generating per-chunk summaries…"):
    chunk_summaries = summarize_chunks_streaming(
        model, tok, device,
        chunks, prompt_index, chunk_tokens, full_name,
        status_container
    )
st.success("Per-chunk summaries ready.")

# Aggregate
st.subheader("Step 4 — Aggregate to final summary")
with st.spinner("Aggregating…"):
    final_summary = aggregate_summaries(
        model, tok, device,
        summaries=chunk_summaries,
        group_size=10,
        max_mid=500,
        max_final=final_len,
    )
st.text_area("Final Summary", value=final_summary, height=250)

# Export
st.subheader("Download results")
df_out = pd.DataFrame({
    "chunk_id": list(range(len(chunk_summaries))),
    "summary_text": chunk_summaries,
})
st.download_button(
    "Download per-chunk summaries CSV",
    df_out.to_csv(index=False).encode("utf-8"),
    file_name="chunk_summaries.csv",
    mime="text/csv",
)
st.download_button(
    "Download final summary (txt)",
    final_summary.encode("utf-8"),
    file_name="final_summary.txt",
    mime="text/plain",
)
