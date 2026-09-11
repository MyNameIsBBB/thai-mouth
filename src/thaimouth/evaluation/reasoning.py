"""
Reasoning accuracy benchmark across recurrent compute steps.
Evaluates model on synthetic and natural Thai reasoning benchmarks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch

from thaimouth.generation.generate import generate_text
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer


def evaluate_reasoning_step_accuracy(
    model: torch.nn.Module,
    test_jsonl_path: Union[str, Path],
    tokenizer: ThaiMouthTokenizer,
    recurrent_steps: int = 1,
    max_samples: int = 100,
    device: Optional[torch.device] = None
) -> Dict[str, Union[float, int]]:
    """
    Evaluates reasoning task accuracy at a fixed recurrent compute step setting.
    """
    if device is None:
        device = next(model.parameters()).device

    test_path = Path(test_jsonl_path)
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")

    samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            if item.get("type") == "reasoning":
                samples.append(item)

    if not samples:
        return {"accuracy": 0.0, "total_evaluated": 0, "recurrent_steps": recurrent_steps}

    samples = samples[:max_samples]
    correct = 0

    model.eval()
    with torch.no_grad():
        for sample in samples:
            q = sample["question"].strip()
            gold_answer = sample["answer"].strip()
            
            prompt = f"<s><question> {q} </question><reasoning>"
            
            # Generate continuation
            out = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=48,
                temperature=0.1,  # Near greedy for evaluation
                recurrent_steps=recurrent_steps,
                device=device
            )
            
            # Check if gold answer is in generated answer tag or text
            if f"<answer> {gold_answer}" in out or gold_answer in out:
                correct += 1

    accuracy = (correct / len(samples)) * 100.0
    return {
        "recurrent_steps": recurrent_steps,
        "correct": correct,
        "total_evaluated": len(samples),
        "accuracy_pct": round(accuracy, 2)
    }


def benchmark_all_depths(
    model: torch.nn.Module,
    test_jsonl_path: Union[str, Path],
    tokenizer: ThaiMouthTokenizer,
    step_list: List[int] = [1, 2, 4, 8, 16],
    max_samples: int = 100,
    device: Optional[torch.device] = None
) -> List[Dict]:
    """
    Benchmarks reasoning accuracy across a spectrum of recurrent steps.
    Answers: Does the SAME model perform better with more recurrent compute?
    """
    results = []
    print(f"\n[*] Benchmarking Reasoning Accuracy across Recurrent Steps {step_list}...")
    
    for steps in step_list:
        res = evaluate_reasoning_step_accuracy(
            model=model,
            test_jsonl_path=test_jsonl_path,
            tokenizer=tokenizer,
            recurrent_steps=steps,
            max_samples=max_samples,
            device=device
        )
        results.append(res)
        print(f"  - Steps: {steps:2d} | Accuracy: {res['accuracy_pct']:5.1f}% ({res['correct']}/{res['total_evaluated']})")

    return results
