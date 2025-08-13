def load_prompt(i: int, chunk: str) -> str:
    p1 = f"Please summarize the following text. Provide your answer as a single paragraph and nothing else:\n\n{chunk}\n\nSummary:"

    p2 = f"Summarize the following passage from the novel. Your summary should be concise and focus on the main events, key actions taken by characters, and any significant information revealed. Capture the essential plot points and character interactions without interpretation or analysis. Provide your answer as a single paragraph and nothing else. The passage is: \n\n{chunk} \n\nSummary:"

    p3 = f"Read the following excerpt. Provide a summary that focuses on the characters' development and emotional landscape. What do we learn about the characters' personalities, motivations, and internal thoughts? Describe the emotional tone of the passage and how the characters' feelings evolve. Provide your answer as a single paragraph and nothing else. The passage is: \n\n{chunk} \n\nSummary:"

    p4 = f"After reading the text provided, summarize the key events and character interactions. Following the summary, identify and list the primary themes that are present in this specific section (e.g., betrayal, ambition, loss, justice). Provide your answer as a single paragraph and nothing else. The passage is: \n\n{chunk} \n\nSummary:"

    p5 = f"""Analyze the following passage and provide a summary that explains its role in the overall narrative. Your summary should address the following:
    
    Plot Advancement: What key events occur that move the story forward?
    Worldbuilding/Setting: What new details are revealed about the world, its rules, or the immediate setting?
    Foreshadowing/Setup: Are there any hints or setups for future events or conflicts?

    Provide your answer as a single paragraph and nothing else.

    The passage is: \n\n{chunk} \n\nSummary:"""

    prompts = [p1, p2, p3, p4, p5]
    
    return prompts[i % len(prompts)]