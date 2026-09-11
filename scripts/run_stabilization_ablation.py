"""
ThaiMouth Milestone 0.3: Recurrent Latent Stabilization Ablation Runner.
Trains and evaluates PreNorm (Variant B), PostNorm (Variant C), and Gated PreNorm (Variant D)
against the un-normalized Baseline (Variant A).
Measures:
1. Latent Norm stability (||Z_t|| at R=1..32)
2. Relative delta Z dynamics
3. Qualitative text generation (preventing degenerate repetition)
4. Multi-Hop Reasoning Accuracy scaling
"""

import argparse
import json
import time
from pathlib import Path
import torch

from thaimouth.config import Config
from thaimouth.evaluation.latent_dynamics import compute_latent_step_dynamics
from thaimouth.evaluation.reasoning_depth import (
    generate_depth_partitioned_benchmark,
    run_reasoning_depth_eval_for_model
)
from thaimouth.generation.generate import generate_text
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.training.trainer import Trainer
from thaimouth.utils.device import get_device


def train_ablation_model(config_path: str, device: torch.device):
    """Trains a model variant if not already trained."""
    config = Config.from_yaml(config_path)
    output_dir = Path(config.training.output_dir)
    final_ckpt = output_dir / "model_final.pt"
    
    if final_ckpt.exists():
        print(f"[+] Found existing trained checkpoint at {final_ckpt}")
        return str(final_ckpt)
        
    print(f"\n[+] Starting training for {config_path} on {device.type.upper()}...")
    trainer = Trainer(config)
    trainer.train()
    return str(final_ckpt)


def inspect_latent_stability_and_generation(
    ckpt_path: str,
    tokenizer_path: str,
    device: torch.device
):
    """Inspects latent norm bounds and text generations at R=1,2,4,8,16,32."""
    model, config, _ = load_checkpoint(ckpt_path, device=device)
    tokenizer = ThaiMouthTokenizer(tokenizer_path)
    model.eval()
    
    sample_prompt = "<s><question> วีระ สูงกว่า ดวงใจ และ ดวงใจ สูงกว่า อรุณ ถาม: ใครสูงที่สุด? </question><reasoning>"
    input_ids = torch.tensor([tokenizer.encode(sample_prompt, add_special_tokens=False)], dtype=torch.long, device=device)
    
    print("\n" + "=" * 85)
    print(f" LATENT DYNAMICS & GENERATION INSPECTION: {ckpt_path}")
    print(f" Recurrent Norm Mode: {getattr(model.config, 'recurrent_norm', 'none')}")
    print("=" * 85)
    print(f"{'Step R':<8} | {'||Z_R||':<12} | {'Rel ΔZ':<10} | {'Generated Preview (first 40 chars)':<45}")
    print("-" * 85)
    
    # Run full trajectory up to R=32
    with torch.no_grad():
        out_32 = model(input_ids, recurrent_steps=32, return_trajectory=True)
        dynamics = compute_latent_step_dynamics(out_32["trajectory_latents"], out_32.get("trajectory_logits"))
        
    for r in [1, 2, 4, 8, 16, 32]:
        z_r = out_32["trajectory_latents"][r]
        norm_val = torch.norm(z_r, dim=-1).mean().item()
        rel_d = dynamics["relative_deltas"][r-1] if r-1 < len(dynamics["relative_deltas"]) else 0.0
        
        # Generate text at this R
        gen = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=sample_prompt,
            max_new_tokens=32,
            temperature=0.0,
            recurrent_steps=r,
            device=device
        )
        preview = gen.replace("\n", " ")[:42]
        print(f"R={r:<6} | {norm_val:<12.4f} | {rel_d:<10.4f} | {preview}")
        
    return dynamics


def main():
    parser = argparse.ArgumentParser(description="Run Milestone 0.3 Recurrent Stabilization Ablation.")
    parser.add_argument("--config", type=str, default="configs/ablation_prenorm.yaml", help="Path to ablation config")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, mps, cuda, cpu)")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and run evaluation only")
    args = parser.parse_args()

    device = get_device(args.device)
    
    # 1. Train or load checkpoint
    if not args.eval_only:
        ckpt_path = train_ablation_model(args.config, device)
    else:
        config = Config.from_yaml(args.config)
        ckpt_path = str(Path(config.training.output_dir) / "model_final.pt")

    # 2. Inspect latent stability & sample generations
    inspect_latent_stability_and_generation(ckpt_path, args.tokenizer, device)

    # 3. Run multi-hop reasoning benchmark
    print("\n[+] Evaluating Multi-Hop Reasoning Benchmark across R=1..32...")
    model, _, _ = load_checkpoint(ckpt_path, device=device)
    tokenizer = ThaiMouthTokenizer(args.tokenizer)
    
    benchmark_by_hop = generate_depth_partitioned_benchmark(
        hop_depths=[1, 2, 4, 8, 16],
        samples_per_depth=20,
        seed=42
    )
    
    res = run_reasoning_depth_eval_for_model(
        model=model,
        tokenizer=tokenizer,
        benchmark_by_hop=benchmark_by_hop,
        recurrent_step_list=[1, 2, 4, 8, 16, 32],
        device=device
    )
    
    print("\n" + "-" * 75)
    print(" REASONING ACCURACY BY HOP DEPTH & RECURRENT STEPS")
    print("-" * 75)
    header = f"{'Hop Depth':<15} | " + " | ".join([f"R={r:<2}" for r in [1, 2, 4, 8, 16, 32]])
    print(header)
    print("-" * len(header))
    
    for hop, steps_dict in res["accuracy_matrix"].items():
        row = [f"{steps_dict[r]['accuracy']:4.1f}%" for r in [1, 2, 4, 8, 16, 32]]
        print(f"{hop}-Hop Reasoning | " + " | ".join(row))


if __name__ == "__main__":
    main()
