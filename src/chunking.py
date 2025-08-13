import re

SENT_END_RE = re.compile(r'([.!?][\'")\]]*\s+)', flags=re.S)

def trim_end_to_sentence_start_backward(tokenizer, input_ids, start_tok, end_tok, max_shift_tokens=128):
    """
    Move end_tok backward so the NEXT chunk starts at a sentence start.
    Never exceeds chunk_length; only backs up (<= max_shift_tokens).
    Returns new_end_tok (<= end_tok).
    """
    if end_tok <= start_tok:
        return end_tok

    # Decode a small tail near the boundary to find the last sentence end.
    tail_start = max(start_tok, end_tok - max_shift_tokens)
    tail_text = tokenizer.decode(
        input_ids[tail_start:end_tok],
        skip_special_tokens=True
    )

    # Find the LAST sentence terminator in the tail.
    last = None
    for m in SENT_END_RE.finditer(tail_text):
        last = m
    if last is None:
        return end_tok  # no good boundary nearby → keep original end

    # Cut *after* the sentence end + whitespace so next chunk begins clean.
    cut_char_idx = last.end()
    trimmed_tail = tail_text[:cut_char_idx]

    # Re-encode trimmed tail to measure token length of the kept portion
    kept_tail_ids = tokenizer.encode(
        trimmed_tail,
        add_special_tokens=False
    )
    new_end_tok = tail_start + len(kept_tail_ids)

    # Safety: never move forward beyond original end
    return min(new_end_tok, end_tok)

def process_chunks(text, tokenizer, chunk_length=4096, chunk_overlap=150, sentence_aware=True):
    ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(ids); chunks = []; start = 0
    while start < n:
        raw_end = min(start + chunk_length, n)
        end = raw_end
        if sentence_aware:
            end = trim_end_to_sentence_start_backward(tokenizer, ids, start, raw_end, max_shift_tokens=128)
        if end <= start: end = raw_end  # safety fallback
        chunk_text = tokenizer.decode(ids[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        span = end - start
        if span <= 0: break
        ov = min(chunk_overlap, max(0, span - 1))
        start = end - ov if ov > 0 else end
    return chunks