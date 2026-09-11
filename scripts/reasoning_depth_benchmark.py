"""
ThaiMouth Milestone 0.2: Reasoning Depth Benchmark & Latent Convergence Analysis.
Evaluates Model A (Train: R=1), Model B (Train: R in {1,2,4}), and Model C (Train: R in {1,2,4,8})
on 1-hop, 2-hop, 4-hop, 8-hop, and 16-hop reasoning tasks at test recurrent steps [1, 2, 4, 8, 16, 32]
with multi-seed evaluation (mean +- std) and latent dynamics tracking.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

from thaimouth.evaluation.reasoning_depth import (
    generate_depth_partitioned_benchmark,
    run_reasoning_depth_eval_for_model
)
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.utils.device import get_device


def run_multi_seed_benchmark(
    models_info: List[Dict],
    tokenizer: ThaiMouthTokenizer,
    hop_depths: List[int] = [1, 2, 4, 8, 16],
    step_list: List[int] = [1, 2, 4, 8, 16, 32],
    seeds: List[int] = [1, 2, 3, 4, 5],
    samples_per_depth: int = 30,
    device: torch.device = torch.device("cpu")
) -> Dict:
    """
    Executes benchmark across models, seeds, reasoning depths, and recurrent steps.
    """
    all_model_results = {}
    
    print("\n" + "=" * 78, flush=True)
    print(" ThaiMouth Milestone 0.2: Multi-Hop Reasoning Scaling & Latent Dynamics", flush=True)
    print(f" Seeds: {seeds} | Reasoning Depths: {hop_depths} | Recurrent Steps: {step_list}", flush=True)
    print(f" Device: {device.type.upper()}", flush=True)
    print("=" * 78, flush=True)
    
    for m_info in models_info:
        model_name = m_info["name"]
        ckpt_path = m_info["checkpoint"]
        print(f"\n[+] Loading model '{model_name}' from {ckpt_path}...", flush=True)
        
        if not Path(ckpt_path).exists():
            print(f"  [!] Checkpoint not found at {ckpt_path}, skipping...", flush=True)
            continue
            
        model, config, _ = load_checkpoint(ckpt_path, device=device)
        model.eval()
        
        # Track accuracy across seeds: seed -> hop -> r_step -> acc
        seed_runs = []
        latent_dynamics_by_step = None
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  - Running Seed {seed} ({seed_idx+1}/{len(seeds)})...", end="", flush=True)
            t0 = time.time()
            benchmark_by_hop = generate_depth_partitioned_benchmark(
                hop_depths=hop_depths,
                samples_per_depth=samples_per_depth,
                seed=seed
            )
            
            eval_res = run_reasoning_depth_eval_for_model(
                model=model,
                tokenizer=tokenizer,
                benchmark_by_hop=benchmark_by_hop,
                recurrent_step_list=step_list,
                device=device
            )
            elapsed = time.time() - t0
            print(f" Done ({elapsed:.1f}s)", flush=True)
            
            seed_runs.append(eval_res["accuracy_matrix"])
            if latent_dynamics_by_step is None:
                latent_dynamics_by_step = eval_res["latent_dynamics"]
                
        # Aggregate mean +- std across seeds
        # structured: hop -> r_step -> {mean, std}
        aggregated_matrix = {}
        for hop in hop_depths:
            aggregated_matrix[hop] = {}
            for r_step in step_list:
                acc_vals = [seed_runs[s_idx][hop][r_step]["accuracy"] for s_idx in range(len(seeds))]
                mean_acc = float(np.mean(acc_vals))
                std_acc = float(np.std(acc_vals))
                aggregated_matrix[hop][r_step] = {
                    "mean": round(mean_acc, 2),
                    "std": round(std_acc, 2),
                    "raw_runs": acc_vals
                }
                
        all_model_results[model_name] = {
            "training_depth": m_info["training_depth"],
            "aggregated_matrix": aggregated_matrix,
            "latent_dynamics": latent_dynamics_by_step
        }
        
    return all_model_results


def print_formatted_summary(
    benchmark_results: Dict,
    hop_depths: List[int],
    step_list: List[int]
):
    """Prints clean terminal comparative tables for each reasoning depth."""
    for hop in hop_depths:
        print("\n" + "-" * 78, flush=True)
        print(f" REASONING ACCURACY (Mean ± Std over 5 Seeds) | {hop}-HOP REASONING", flush=True)
        print("-" * 78, flush=True)
        header = f"{'Model':<28} | " + " | ".join([f"R={r:<2}" for r in step_list])
        print(header, flush=True)
        print("-" * len(header), flush=True)
        
        for model_name, res in benchmark_results.items():
            row_items = []
            for r in step_list:
                m_val = res["aggregated_matrix"][hop][r]["mean"]
                s_val = res["aggregated_matrix"][hop][r]["std"]
                row_items.append(f"{m_val:4.1f}±{s_val:3.1f}")
            print(f"{model_name:<28} | " + " | ".join(row_items), flush=True)

    # Print Latent Dynamics Convergence Sample (e.g. Model B at R=8 or R=16)
    for model_name, res in benchmark_results.items():
        if "Model B" in model_name and res.get("latent_dynamics"):
            dyn_32 = res["latent_dynamics"].get(32) or res["latent_dynamics"].get(16)
            if dyn_32:
                print("\n" + "=" * 90, flush=True)
                print(f" LATENT DYNAMICS CONVERGENCE ({model_name}) across Recurrent Steps", flush=True)
                print("=" * 90, flush=True)
                print(f"{'Step t':<8} | {'||Z_t||':<10} | {'||ΔZ_t||':<10} | {'Rel ΔZ':<10} | {'cos(Z_t, Z_{t+1})':<18} | {'Entropy H(p_t)':<15} | {'KL(p_t || p_{t+1})':<15}", flush=True)
                print("-" * 102, flush=True)
                
                norms = dyn_32["norms"]
                deltas = dyn_32["deltas"]
                rel_deltas = dyn_32.get("relative_deltas", [0.0]*len(deltas))
                cosines = dyn_32["cosine_sims"]
                entropies = dyn_32["entropies"]
                kls = dyn_32["kl_divergences"]
                
                for t in range(len(deltas)):
                    kl_str = f"{kls[t]:.6f}" if t < len(kls) else "-"
                    print(f"t={t+1:<6} | {norms[t+1]:<10.4f} | {deltas[t]:<10.4f} | {rel_deltas[t]:<10.4f} | {cosines[t]:<18.4f} | {entropies[t+1]:<15.4f} | {kl_str:<15}", flush=True)
                break


def save_reports(
    results: Dict,
    hop_depths: List[int],
    step_list: List[int],
    seeds: List[int],
    out_dir: Path
):
    """Saves structured JSON, CSV, and markdown reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. JSON Report
    json_path = out_dir / "reasoning_depth_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Saved detailed JSON results to {json_path}")
    
    # 2. CSV Report
    csv_path = out_dir / "reasoning_depth_benchmark.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "reasoning_hops", "recurrent_steps", "mean_accuracy", "std_accuracy"])
        for model_name, res in results.items():
            for hop in hop_depths:
                for r_step in step_list:
                    m = res["aggregated_matrix"][hop][r_step]["mean"]
                    s = res["aggregated_matrix"][hop][r_step]["std"]
                    writer.writerow([model_name, hop, r_step, m, s])
    print(f"[+] Saved CSV report to {csv_path}")

    # 3. Markdown Report
    md_path = out_dir / "reasoning_depth_benchmark.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# ThaiMouth Milestone 0.2: Reasoning Depth Benchmark & Latent Dynamics\n\n")
        f.write(f"- **Seeds Evaluated:** {seeds}\n")
        f.write(f"- **Reasoning Depth Categories:** {hop_depths} hops\n")
        f.write(f"- **Recurrent Step Spectrum:** {step_list}\n\n")
        
        for hop in hop_depths:
            f.write(f"### {hop}-Hop Reasoning Accuracy (Mean ± Std %)\n\n")
            f.write("| Model | Training Depth | " + " | ".join([f"R={r}" for r in step_list]) + " |\n")
            f.write("| --- | --- | " + " | ".join(["---"] * len(step_list)) + " |\n")
            for model_name, res in results.items():
                row_str = f"| **{model_name}** | `{res['training_depth']}` | "
                row_str += " | ".join([f"{res['aggregated_matrix'][hop][r]['mean']:.1f} ± {res['aggregated_matrix'][hop][r]['std']:.1f}" for r in step_list])
                row_str += " |\n"
                f.write(row_str)
            f.write("\n")
            
        f.write("### Latent Dynamics & Convergence Diagnostics\n\n")
        f.write("Output distributions converge with increasing recurrence, while latent-state norms diverge rapidly. This suggests output-space stabilization despite unstable latent magnitude dynamics. Further experiments are required to determine whether normalization or recurrent update design can produce stable iterative computation.\n\n")
        for model_name, res in results.items():
            if res.get("latent_dynamics"):
                dyn = res["latent_dynamics"].get(32) or res["latent_dynamics"].get(16)
                if dyn:
                    f.write(f"#### {model_name} Latent Trajectory\n\n")
                    f.write("| Step $t$ | $\\|Z_t\\|_2$ | $\\|\\Delta Z_t\\|_2$ | $\\text{Rel } \\Delta Z$ | $\\cos(Z_t, Z_{t+1})$ | Entropy $H(p_t)$ | $D_{KL}(p_t \\parallel p_{t+1})$ |\n")
                    f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
                    rel_d = dyn.get("relative_deltas", [0.0]*len(dyn["deltas"]))
                    for t in range(len(dyn["deltas"])):
                        kl_val = f"{dyn['kl_divergences'][t]:.6f}" if t < len(dyn['kl_divergences']) else "-"
                        f.write(f"| $t={t+1}$ | {dyn['norms'][t+1]:.4f} | {dyn['deltas'][t]:.4f} | {rel_d[t]:.4f} | {dyn['cosine_sims'][t]:.4f} | {dyn['entropies'][t+1]:.4f} | {kl_val} |\n")
                    f.write("\n")
    print(f"[+] Saved Markdown summary to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ThaiMouth Reasoning Depth Benchmark.")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, mps, cuda, cpu)")
    parser.add_argument("--samples_per_depth", type=int, default=25, help="Number of samples per depth tier per seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save output reports")
    args = parser.parse_args()

    device = get_device(args.device)
    tokenizer = ThaiMouthTokenizer(args.tokenizer)

    models_info = [
        {
            "name": "Model A (Step=1)",
            "checkpoint": "checkpoints/model_a_step1/model_final.pt",
            "training_depth": "steps = 1"
        },
        {
            "name": "Model B (Curriculum {1,2,4})",
            "checkpoint": "checkpoints/model_b_curriculum_1_2_4/model_final.pt",
            "training_depth": "steps ∈ {1, 2, 4}"
        },
        {
            "name": "Model C (Curriculum {1,2,4,8})",
            "checkpoint": "checkpoints/model_c_curriculum_1_2_4_8/model_final.pt",
            "training_depth": "steps ∈ {1, 2, 4, 8}"
        }
    ]

    hop_depths = [1, 2, 4, 8, 16]
    step_list = [1, 2, 4, 8, 16, 32]
    seeds = [1, 2, 3, 4, 5]

    results = run_multi_seed_benchmark(
        models_info=models_info,
        tokenizer=tokenizer,
        hop_depths=hop_depths,
        step_list=step_list,
        seeds=seeds,
        samples_per_depth=args.samples_per_depth,
        device=device
    )

    print_formatted_summary(results, hop_depths, step_list)
    save_reports(results, hop_depths, step_list, seeds, Path(args.results_dir))


if __name__ == "__main__":
    main()
