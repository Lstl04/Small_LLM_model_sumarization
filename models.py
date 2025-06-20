import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from typing import List
import os
from Data_preparation import get_data

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
try:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # keep overflow layers in fp32 on CPU
    )
    
    # Load model with explicit device mapping
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        # For CPU, load without device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model = model.to(device)

    print(next(model.parameters()).device)
    print(f"Model loaded successfully on {device}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def process_chunks(text: str, chunk_length: int = 50, chunk_overlap: int = 10) -> List[str]:
    """
    Tokenizes text into chunks with specified length and overlap.
    
    Args:
        text (str): Input text to process
        chunk_length (int): Maximum number of tokens per chunk
        chunk_overlap (int): Number of overlapping tokens between chunks
    
    Returns:
        List[str]: List of text chunks
    """
    # Tokenize the input text
    tokens = tokenizer.encode(text)
    chunks = []
    
    # Process tokens into chunks
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_length, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Move start index forward, accounting for overlap
        start_idx = end_idx - chunk_overlap if end_idx < len(tokens) else end_idx
    
    return chunks

def summarize_chunks(chunks: List[str]) -> List[str]:
    """
    Summarizes each chunk using the Qwen model.
    
    Args:
        chunks (List[str]): List of text chunks to summarize
    
    Returns:
        List[str]: List of chunk summaries
    """
    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}…")

        prompt = f"Please summarize the following text:\n\n{chunk}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt")

        # **key change**: find the device of the model’s parameters
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(summary)
        summaries.append(summary.split("Summary:")[-1].strip())
    return summaries

def aggregate_summaries(summaries: List[str]) -> str:
    """
    Combines chunk summaries into a final cohesive summary.
    
    Args:
        summaries (List[str]): List of chunk summaries
    
    Returns:
        str: Final aggregated summary
    """
    combined_text = " ".join(summaries)
    print("Generating final summary…")
    prompt = f"Please provide a coherent summary of the following text:\n\n{combined_text}\n\nFinal Summary:"
    inputs = tokenizer(prompt, return_tensors="pt")

    # send inputs to the same device as the model
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=750,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Final Summary:")[-1].strip()
        
  

if __name__ == "__main__":
    # Example usage
    try:
        # Get sample books from data preparation
        df = get_data()
        
        # Use first book for testing
        text = df.iloc[0]['full_text']
        original_summary = df.iloc[0]['summary']
        
        print(f"\nTesting summary generation for book: {df.iloc[0]['title']}")
        print(f"Original summary length: {len(original_summary.split())}")
        
        # Process text into chunks
        chunks = process_chunks(text)
        print(f"Text split into {len(chunks)} chunks")
        
        # Summarize chunks
        chunk_summaries = summarize_chunks(chunks)
        print(f"Generated {len(chunk_summaries)} chunk summaries")
        
        # Generate final summary
        final_summary = aggregate_summaries(chunk_summaries)
        print("\nFinal Summary:")
        print(final_summary)
        
        print("\nOriginal Summary (for comparison):")
        print(original_summary)
        
    except Exception as e:
        print(f"An error occurred: {e}")
