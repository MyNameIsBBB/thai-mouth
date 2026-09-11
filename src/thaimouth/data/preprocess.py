"""
Data formatting and preprocessing utilities for ThaiMouth.
Converts conversation and reasoning JSONL formats to causal LM training strings.
"""

import json
from pathlib import Path
from typing import Dict, Union


def format_sample_to_text(sample: Dict) -> str:
    """
    Formats a structured dictionary into a standardized text sequence with special tokens.
    """
    sample_type = sample.get("type", "text")
    
    if sample_type == "conversation":
        parts = ["<s>"]
        for msg in sample.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "").strip()
            if role == "user":
                parts.append(f"<user> {content} </user>")
            elif role == "assistant":
                parts.append(f"<assistant> {content} </assistant>")
        parts.append("</s>")
        return "".join(parts)
        
    elif sample_type == "reasoning":
        q = sample.get("question", "").strip()
        reasoning_list = sample.get("reasoning", [])
        r = " \n ".join(reasoning_list).strip()
        a = sample.get("answer", "").strip()
        
        parts = [
            "<s>",
            f"<question> {q} </question>",
            f"<reasoning> {r} </reasoning>",
            f"<answer> {a} </answer>",
            "</s>"
        ]
        return "".join(parts)
        
    elif "text" in sample:
        text = sample["text"].strip()
        return f"<s> {text} </s>"
        
    else:
        # Fallback to json string
        return f"<s> {json.dumps(sample, ensure_ascii=False)} </s>"


def load_formatted_texts(jsonl_path: Union[str, Path]) -> list[str]:
    """Loads all lines from a JSONL file and formats them into training texts."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
        
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                text = format_sample_to_text(data)
                texts.append(text)
            except json.JSONDecodeError:
                texts.append(f"<s> {line_str} </s>")
    return texts
