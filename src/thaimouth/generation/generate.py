"""
Autoregressive Text Generation for ThaiMouth language models.
Supports greedy decoding, temperature sampling, top-k, top-p, and recurrent depth control.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Union
import torch
import torch.nn.functional as F

from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.utils.device import get_device


def sample_top_p_top_k(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    generated_tokens: Optional[List[int]] = None
) -> int:
    """
    Applies temperature, repetition penalty, top-k, and top-p sampling to logits.
    """
    logits = logits.clone()
    
    # 1. Repetition penalty
    if generated_tokens and repetition_penalty != 1.0:
        for token_id in set(generated_tokens):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    # 2. Temperature
    if temperature <= 1e-4:
        return torch.argmax(logits).item()
        
    logits = logits / max(1e-4, temperature)
    
    # 3. Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    # 4. Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")

    # 5. Sample token
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()
    return next_token


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: ThaiMouthTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    recurrent_steps: Optional[int] = None,
    device: Optional[torch.device] = None
) -> str:
    """
    Generates continuation for a given prompt string.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.bos_id]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = list(token_ids)

    for _ in range(max_new_tokens):
        # Truncate if context grows beyond max_seq_len
        curr_input = input_ids[:, -model.config.max_seq_len:]
        
        outputs = model(curr_input, recurrent_steps=recurrent_steps)
        next_token_logits = outputs["logits"][0, -1, :]
        
        next_token_id = sample_top_p_top_k(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated
        )
        
        generated.append(next_token_id)
        if next_token_id == tokenizer.eos_id:
            break
            
        next_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_tensor], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=False)


def main():
    parser = argparse.ArgumentParser(description="Generate text using trained ThaiMouth model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--prompt", type=str, default="<s><user> วันนี้เหนื่อยมากเลย </user><assistant>", help="Prompt string")
    parser.add_argument("--max_tokens", type=int, default=64, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling")
    parser.add_argument("--recurrent_steps", type=int, default=None, help="Recurrent compute steps (e.g. 1, 2, 4, 8)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    args = parser.parse_args()
    device = get_device(args.device)
    
    model, config, _ = load_checkpoint(args.checkpoint, device=device)
    tokenizer = ThaiMouthTokenizer(args.tokenizer)
    
    print(f"[*] Prompt: {args.prompt}")
    steps = args.recurrent_steps if args.recurrent_steps is not None else config.model.recurrent_steps
    print(f"[*] Generating with Recurrent Steps: {steps} on {device.type.upper()}...")
    
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        recurrent_steps=args.recurrent_steps,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("GENERATED OUTPUT:")
    print("=" * 60)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
