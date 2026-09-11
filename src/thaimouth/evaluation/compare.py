"""
Model comparison suite.
Evaluates baseline, recurrent, and latent models across recurrent depths (1, 2, 4, 8, 16).
Outputs results to Markdown, CSV, and JSON in results/ directory.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import torch

from thaimouth.evaluation.efficiency import benchmark_efficiency
from thaimouth.evaluation.language import evaluate_perplexity
from thaimouth.evaluation.reasoning import benchmark_all_depths
from thaimouth.models.common import format_param_count
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.utils.device import get_device


def compare_checkpoints(
    checkpoint_paths: List[str],
    tokenizer_path: str = "checkpoints/tokenizer.json",
    test_jsonl_path: str = "data/synthetic/test.jsonl",
    step_list: List[int] = [1, 2, 4, 8, 16],
    max_reasoning_samples: int = 100,
    output_dir: str = "results",
    device_pref: str = "auto"
) -> pd.DataFrame:
    """
    Compares multiple model checkpoints across recurrent depth spectrum.
    """
    device = get_device(device_pref)
    tokenizer = ThaiMouthTokenizer(tokenizer_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for ckpt_str in checkpoint_paths:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            print(f"[!] Checkpoint not found: {ckpt_path}, skipping.")
            continue

        model_name = ckpt_path.parent.name if ckpt_path.parent.name != "checkpoints" else ckpt_path.stem
        model, config, _ = load_checkpoint(ckpt_path, device=device)
        total_params = model.count_params()
        model_type = config.model.model_type

        print("\n" + "=" * 60)
        print(f"Evaluating Model: {model_name} ({model_type.upper()}) | Params: {format_param_count(total_params)}")
        print("=" * 60)

        # 1. Evaluate language modeling perplexity across steps
        for steps in step_list:
            # Baseline doesn't change with recurrent steps, test once or report same
            if model_type == "baseline" and steps > 1:
                ppl_res = {"perplexity": records[-1]["Perplexity"], "loss": records[-1]["Loss"]}
            else:
                ppl_res = evaluate_perplexity(
                    model=model,
                    test_path=test_jsonl_path,
                    tokenizer=tokenizer,
                    recurrent_steps=steps,
                    device=device
                )

            # 2. Efficiency
            eff_res = benchmark_efficiency(
                model=model,
                recurrent_steps=steps,
                device=device
            )

            record = {
                "Model": model_name,
                "Type": model_type,
                "Params": total_params,
                "Params_Str": format_param_count(total_params),
                "Steps": steps,
                "Perplexity": round(ppl_res["perplexity"], 2),
                "Loss": round(ppl_res["loss"], 4),
                "Latency_ms": eff_res["latency_ms"],
                "Tokens_per_sec": eff_res["tokens_per_sec"]
            }
            records.append(record)

    df = pd.DataFrame(records)
    
    # Save CSV and JSON
    csv_file = out_dir / "comparison_summary.csv"
    json_file = out_dir / "comparison_summary.json"
    md_file = out_dir / "comparison_summary.md"

    df.to_csv(csv_file, index=False)
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        
    try:
        md_table = df[["Model", "Type", "Params_Str", "Steps", "Perplexity", "Loss", "Latency_ms", "Tokens_per_sec"]].to_markdown(index=False)
    except Exception:
        # Fallback string representation
        cols = ["Model", "Type", "Params_Str", "Steps", "Perplexity", "Loss", "Latency_ms", "Tokens_per_sec"]
        lines = [" | ".join(cols), " | ".join(["---"] * len(cols))]
        for _, row in df[cols].iterrows():
            lines.append(" | ".join(str(row[c]) for c in cols))
        md_table = "\n".join(lines)

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Model Comparison Summary across Recurrent Compute Steps\n\n")
        f.write(md_table + "\n")

    print("\n" + "=" * 60)
    print("COMPARISON RESULTS TABLE:")
    print("=" * 60)
    print(md_table)
    print(f"\n[+] Results saved to {out_dir}/")
    return df


def main():
    parser = argparse.ArgumentParser(description="Compare multiple ThaiMouth models across recurrent depths.")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of model checkpoint paths")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--test_data", type=str, default="data/synthetic/test.jsonl", help="Path to test JSONL")
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 4, 8, 16], help="Recurrent steps to test")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output reports")
    parser.add_argument("--device", type=str, default="auto", help="Device preference")

    args = parser.parse_args()
    compare_checkpoints(
        checkpoint_paths=args.checkpoints,
        tokenizer_path=args.tokenizer,
        test_jsonl_path=args.test_data,
        step_list=args.steps,
        output_dir=args.output_dir,
        device_pref=args.device
    )


if __name__ == "__main__":
    main()
