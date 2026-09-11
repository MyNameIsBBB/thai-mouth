"""
Controlled Empirical Experiment:
Comparing Model A (Train: steps=1), Model B (Train: steps in {1,2,4}), and Model C (Train: steps in {1,2,4,8})
across Test-time Recurrent Depths: 1, 2, 4, 8, 16, 32.
"""

import json
from pathlib import Path
import pandas as pd
import torch

from thaimouth.config import Config
from thaimouth.evaluation.efficiency import benchmark_efficiency
from thaimouth.evaluation.language import evaluate_perplexity
from thaimouth.evaluation.reasoning import evaluate_reasoning_step_accuracy
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.training.trainer import Trainer
from thaimouth.utils.device import get_device


def run_experiment():
    print("=" * 75, flush=True)
    print("      🧪 CONTROLLED EXPERIMENT: RECURRENT DEPTH SCALING STUDY 🧪      ", flush=True)
    print("=" * 75, flush=True)

    device = get_device("auto")
    tokenizer_path = "checkpoints/tokenizer.json"
    test_data_path = "data/synthetic/test.jsonl"
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    models_meta = [
        ("Model A (Train steps=1)", "configs/experiment_model_a.yaml", "checkpoints/model_a_step1/model_final.pt"),
        ("Model B (Train steps={1,2,4})", "configs/experiment_model_b.yaml", "checkpoints/model_b_curriculum_1_2_4/model_final.pt"),
        ("Model C (Train steps={1,2,4,8})", "configs/experiment_model_c.yaml", "checkpoints/model_c_curriculum_1_2_4_8/model_final.pt"),
    ]

    # 1. Train models if not present
    for name, cfg_path, ckpt_path in models_meta:
        if not Path(ckpt_path).exists():
            print(f"\n[*] Training {name} from {cfg_path}...", flush=True)
            cfg = Config.from_yaml(cfg_path)
            trainer = Trainer(cfg)
            trainer.train()
        else:
            print(f"[+] Loaded trained checkpoint for {name} from {ckpt_path}", flush=True)

    # 2. Evaluate all models across test depths [1, 2, 4, 8, 16, 32]
    test_steps_list = [1, 2, 4, 8, 16, 32]
    tokenizer = ThaiMouthTokenizer(tokenizer_path)
    all_results = []

    print("\n" + "=" * 75, flush=True)
    print("        RUNNING INFERENCE DEPTH SCALING (Steps: 1, 2, 4, 8, 16, 32)       ", flush=True)
    print("=" * 75, flush=True)

    for name, _, ckpt_path in models_meta:
        print(f"\n>>> Benchmarking {name}...", flush=True)
        model, config, _ = load_checkpoint(ckpt_path, device=device)
        
        for steps in test_steps_list:
            # A. Perplexity & Loss
            lang_metrics = evaluate_perplexity(
                model=model,
                test_path=test_data_path,
                tokenizer=tokenizer,
                recurrent_steps=steps,
                device=device
            )

            # B. Reasoning Accuracy on held-out tasks (25 samples per depth for snappy evaluation)
            reasoning_metrics = evaluate_reasoning_step_accuracy(
                model=model,
                test_jsonl_path=test_data_path,
                tokenizer=tokenizer,
                recurrent_steps=steps,
                max_samples=25,
                device=device
            )

            # C. Inference Speed / Latency
            eff_metrics = benchmark_efficiency(
                model=model,
                recurrent_steps=steps,
                device=device
            )

            record = {
                "Model": name,
                "Test_Steps": steps,
                "Perplexity": round(lang_metrics["perplexity"], 2),
                "Loss": round(lang_metrics["loss"], 4),
                "Reasoning_Acc_Pct": reasoning_metrics["accuracy_pct"],
                "Latency_ms": eff_metrics["latency_ms"],
                "Tokens_per_sec": eff_metrics["tokens_per_sec"]
            }
            all_results.append(record)
            print(f"  Step {steps:2d} -> PPL: {record['Perplexity']:6.2f} | Loss: {record['Loss']:.4f} | Reasoning Acc: {record['Reasoning_Acc_Pct']:5.1f}% | Latency: {record['Latency_ms']:4.2f}ms ({record['Tokens_per_sec']:6.0f} tok/s)", flush=True)

    df = pd.DataFrame(all_results)

    # 3. Export Summary Reports
    csv_path = results_dir / "depth_scaling_experiment.csv"
    json_path = results_dir / "depth_scaling_experiment.json"
    md_path = results_dir / "depth_scaling_experiment.md"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    try:
        table_str = df.to_markdown(index=False)
    except Exception:
        table_str = str(df)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Recurrent Depth Scaling Experiment: Model A vs Model B vs Model C\n\n")
        f.write(f"Evaluated on test steps: {test_steps_list}\n\n")
        f.write(table_str + "\n")

    print("\n" + "=" * 75, flush=True)
    print("FINAL EXPERIMENT SUMMARY TABLE:", flush=True)
    print("=" * 75, flush=True)
    print(table_str, flush=True)
    print(f"\n[+] Saved results to {results_dir}/", flush=True)
    return df


if __name__ == "__main__":
    run_experiment()
