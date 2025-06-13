import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import os

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
try:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def process_chunks(text: str, chunk_length: int = 1000, chunk_overlap: int = 100) -> List[str]:
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
    
    for chunk in chunks:
        try:
            # Prepare prompt for summarization
            prompt = f"Please summarize the following text:\n\n{chunk}\n\nSummary:"
            
            # Generate summary
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the summary part (after "Summary:")
            summary = summary.split("Summary:")[-1].strip()
            summaries.append(summary)
            
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            summaries.append("")  # Add empty string for failed chunks
    
    return summaries

def aggregate_summaries(summaries: List[str]) -> str:
    """
    Combines chunk summaries into a final cohesive summary.
    
    Args:
        summaries (List[str]): List of chunk summaries
    
    Returns:
        str: Final aggregated summary
    """
    # Combine all summaries
    combined_text = " ".join(summaries)
    
    try:
        # Prepare prompt for final summary
        prompt = f"Please provide a coherent summary of the following text:\n\n{combined_text}\n\nFinal Summary:"
        
        # Generate final summary
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        final_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the summary part (after "Final Summary:")
        final_summary = final_summary.split("Final Summary:")[-1].strip()
        return final_summary
        
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return "Error generating final summary"

if __name__ == "__main__":
    # Example usage
    try:
        # Read sample text file
        with open("sample.txt", "r", encoding="utf-8") as f:
            text = f.read()
        
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
        
    except FileNotFoundError:
        print("Error: sample.txt file not found")
    except Exception as e:
        print(f"An error occurred: {e}")
