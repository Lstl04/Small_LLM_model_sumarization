import torch
from typing import List, Optional, Callable


def _summarize_list(model, tokenizer, device, text_list: List[str], header: str, trailer: str, max_new_tokens: int) -> str:
    """
    Summarize a list of short texts in a single generate() call.
    """
    prompt = header + "\n\n" + "\n\n".join(text_list) + "\n\n" + trailer
    enc = tokenizer(prompt, return_tensors="pt", truncation=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def aggregate_summaries(
    model,
    tokenizer,
    device,
    summaries: List[str],
    group_size: int = 10,
    max_mid: int = 500,
    max_final: int = 1000,
    should_stop: Optional[Callable[[], bool]] = None,
) -> str:
    """
    Hierarchical aggregation to avoid OOM:
      - First summarize chunk summaries in groups (size=group_size)
      - Then summarize those intermediate summaries into the final
    """
    if not summaries:
        return ""

    # 1) Group stage: chunk summaries -> intermediate summaries
    mids: List[str] = []
    header = "Combine the following chunk summaries into a single, coherent intermediate summary. Keep all major plot points, characters, and themes. Provide your answer as a single paragraph and nothing else:"
    trailer = "Intermediate Summary:"
    for i in range(0, len(summaries), group_size):
        if should_stop is not None and should_stop():
            return "[Stopped by user during aggregation]"
        group = summaries[i:i + group_size]
        mid_text = _summarize_list(model, tokenizer, device, group, header, trailer, max_new_tokens=max_mid)
        mids.append(mid_text.split("Intermediate Summary:")[-1].strip())

    # 2) Final stage: intermediate summaries -> final summary
    if should_stop is not None and should_stop():
        return "[Stopped by user before final aggregation]"
    final_header = "Combine these intermediate summaries into one cohesive book summary (500-750 words). Maintain chronology and avoid repetition. Provide your answer as a single paragraph and nothing else:"
    final_trailer = "Final Summary:"
    final_text = _summarize_list(model, tokenizer, device, mids, final_header, final_trailer, max_new_tokens=max_final)
    return final_text.split("Final Summary:")[-1].strip()