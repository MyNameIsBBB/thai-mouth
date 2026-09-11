"""
Multi-Hop Reasoning Depth Benchmark & Latent Convergence Evaluation.
Evaluates Thai reasoning accuracy across explicit reasoning depth tiers (1-hop, 2-hop, 4-hop, 8-hop, 16-hop)
under varying test-time recurrent compute steps (1, 2, 4, 8, 16, 32).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from thaimouth.data.synthetic_reasoning import (
    THAI_NAMES_TEST,
    generate_arithmetic_sample,
    generate_comparison_sample,
    generate_logic_sample
)
from thaimouth.evaluation.latent_dynamics import compute_latent_step_dynamics
from thaimouth.generation.generate import sample_top_p_top_k
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer


def generate_depth_partitioned_benchmark(
    hop_depths: List[int] = [1, 2, 4, 8, 16],
    samples_per_depth: int = 40,
    seed: int = 42
) -> Dict[int, List[Dict]]:
    """
    Generates deterministic benchmark test suites partitioned strictly by reasoning hops.
    Uses test-set Thai names to guarantee zero entity overlap with training.
    """
    rng = random.Random(seed)
    benchmark_by_hop = {}
    
    for hops in hop_depths:
        samples = []
        for _ in range(samples_per_depth):
            # Mix comparison, logic, and arithmetic
            cat = rng.choice(["comp", "logic", "arith"])
            if cat == "comp":
                samples.append(generate_comparison_sample(THAI_NAMES_TEST, hops=hops, rng=rng))
            elif cat == "logic":
                samples.append(generate_logic_sample(hops=hops, rng=rng))
            else:
                samples.append(generate_arithmetic_sample(THAI_NAMES_TEST, steps=hops, rng=rng))
        benchmark_by_hop[hops] = samples
        
    return benchmark_by_hop


@torch.no_grad()
def evaluate_reasoning_batch(
    model: torch.nn.Module,
    tokenizer: ThaiMouthTokenizer,
    samples: List[Dict],
    recurrent_steps: int,
    device: torch.device,
    max_new_tokens: int = 36,
    return_predictions: bool = False,
) -> Union[
    Tuple[int, int, float, Optional[Dict]],
    Tuple[int, int, float, Optional[Dict], List[Dict]],
]:
    """
    Batched evaluation over a list of reasoning samples.
    Accurately passes recurrent_steps and dynamic attention masks in every step.
    Returns (correct_count, total_samples, accuracy_pct, latent_dynamics).
    """
    if not samples:
        empty = (0, 0, 0.0, None)
        return (*empty, []) if return_predictions else empty

    prompts = [f"<s><question> {s['question'].strip()} </question><reasoning>" for s in samples]
    gold_answers = [s["answer"].strip() for s in samples]
    
    encoded_prompts = [tokenizer.encode(p, add_special_tokens=False) or [tokenizer.bos_id] for p in prompts]
    max_len = max(len(ids) for ids in encoded_prompts)
    
    # Left pad with pad_id so generated tokens align cleanly
    pad_id = tokenizer.pad_id
    padded_inputs = []
    attention_masks = []
    
    for ids in encoded_prompts:
        pad_len = max_len - len(ids)
        padded_inputs.append([pad_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))
        
    input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=device)
    attn_mask = torch.tensor(attention_masks, dtype=torch.float32, device=device)
    
    # 1. Forward prompt pass with trajectory tracking
    forward_args = {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "recurrent_steps": recurrent_steps,
    }
    if getattr(model.config, "model_type", "") != "baseline":
        forward_args["return_trajectory"] = True
    out = model(**forward_args)
    
    dynamics = None
    if "trajectory_latents" in out and len(out["trajectory_latents"]) > 1:
        dynamics = compute_latent_step_dynamics(
            trajectory_latents=out["trajectory_latents"],
            trajectory_logits=out.get("trajectory_logits")
        )

    # 2. Batched greedy decoding with proper attention mask extension
    cur_input_ids = input_ids
    cur_attn_mask = attn_mask
    generated_ids = [list(ids) for ids in encoded_prompts]
    
    next_logits = out["logits"][:, -1, :]  # [B, V]
    next_tokens = torch.argmax(next_logits, dim=-1)  # [B]
    
    for b_idx in range(len(samples)):
        generated_ids[b_idx].append(next_tokens[b_idx].item())
        
    cur_input_ids = torch.cat([cur_input_ids, next_tokens.unsqueeze(-1)], dim=1)
    cur_attn_mask = torch.cat([cur_attn_mask, torch.ones((len(samples), 1), device=device)], dim=1)
    
    for _ in range(max_new_tokens - 1):
        step_out = model(
            input_ids=cur_input_ids[:, -model.config.max_seq_len:],
            attention_mask=cur_attn_mask[:, -model.config.max_seq_len:],
            recurrent_steps=recurrent_steps
        )
        step_logits = step_out["logits"][:, -1, :]
        next_tokens = torch.argmax(step_logits, dim=-1)
        
        for b_idx in range(len(samples)):
            generated_ids[b_idx].append(next_tokens[b_idx].item())
            
        cur_input_ids = torch.cat([cur_input_ids, next_tokens.unsqueeze(-1)], dim=1)
        cur_attn_mask = torch.cat([cur_attn_mask, torch.ones((len(samples), 1), device=device)], dim=1)

    # 3. Check correctness: parse generated answer tag
    correct_cnt = 0
    predictions = []
    for b_idx, gold in enumerate(gold_answers):
        gen_text = tokenizer.decode(generated_ids[b_idx], skip_special_tokens=False)
        completion_text = tokenizer.decode(
            generated_ids[b_idx][len(encoded_prompts[b_idx]):],
            skip_special_tokens=False,
        )
        # Check if gold answer is in the answer portion or exact tag
        is_correct = False
        if "<answer>" in gen_text:
            ans_part = gen_text.split("<answer>")[-1].replace("</answer>", "").replace("</s>", "").strip()
            if gold in ans_part:
                is_correct = True
        elif f"<answer> {gold}" in gen_text:
            is_correct = True
        correct_cnt += int(is_correct)
        if return_predictions:
            predictions.append({
                "question": samples[b_idx]["question"],
                "category": samples[b_idx].get("category"),
                "hops": samples[b_idx].get("hops"),
                "gold_answer": gold,
                "generated_text": gen_text,
                "generated_completion": completion_text,
                "correct": is_correct,
            })

    acc = (correct_cnt / len(samples)) * 100.0
    result = (correct_cnt, len(samples), acc, dynamics)
    return (*result, predictions) if return_predictions else result


def run_reasoning_depth_eval_for_model(
    model: torch.nn.Module,
    tokenizer: ThaiMouthTokenizer,
    benchmark_by_hop: Dict[int, List[Dict]],
    recurrent_step_list: List[int] = [1, 2, 4, 8, 16, 32],
    device: Optional[torch.device] = None
) -> Dict[str, Union[Dict, List]]:
    """
    Evaluates a single model over all hop depths across all recurrent steps using batched execution.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    results_by_depth_and_step = {}
    aggregated_dynamics_by_step = {}
    
    for hops, samples in benchmark_by_hop.items():
        results_by_depth_and_step[hops] = {}
        for r_step in recurrent_step_list:
            correct_cnt, total, acc_pct, dynamics = evaluate_reasoning_batch(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                recurrent_steps=r_step,
                device=device,
                max_new_tokens=36
            )
            
            results_by_depth_and_step[hops][r_step] = {
                "correct": correct_cnt,
                "total": total,
                "accuracy": round(acc_pct, 2)
            }
            
            if dynamics is not None and r_step not in aggregated_dynamics_by_step:
                aggregated_dynamics_by_step[r_step] = dynamics

    return {
        "accuracy_matrix": results_by_depth_and_step,
        "latent_dynamics": aggregated_dynamics_by_step
    }
