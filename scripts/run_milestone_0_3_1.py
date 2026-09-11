"""Run the locked Milestone 0.3.1 evaluation protocol.

This script evaluates existing checkpoints. Pass --train-missing to train every
missing seed/control checkpoint from the generated configs before evaluation.
Raw generated completions are saved alongside aggregate metrics.
"""

import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import yaml

from thaimouth.evaluation.efficiency import benchmark_efficiency
from thaimouth.evaluation.language import evaluate_perplexity
from thaimouth.evaluation.milestone import assess_milestone
from thaimouth.evaluation.reasoning_depth import (
    evaluate_reasoning_batch,
    generate_depth_partitioned_benchmark,
)
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.training.trainer import Trainer
from thaimouth.config import Config
from thaimouth.utils.device import get_device


def _load_protocol(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {})["experiment"]


def _load_plan(protocol: dict) -> dict:
    path = Path(protocol["generated_config_dir"]) / "plan.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_milestone_0_3_1.py first."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@contextlib.contextmanager
def _temporary_alpha(model: torch.nn.Module, alpha: Optional[float]) -> Iterator[None]:
    parameter = getattr(model, "alpha", None)
    if alpha is None or parameter is None:
        yield
        return
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.fill_(alpha)
        yield
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def _difficulty_for_hop(hop: int, difficulty: Dict[str, List[int]]) -> str:
    matches = [name for name, hops in difficulty.items() if hop in hops]
    if len(matches) != 1:
        raise ValueError(f"Hop {hop} must belong to exactly one difficulty tier")
    return matches[0]


def _train_missing(plan: dict) -> None:
    for run in plan["runs"]:
        checkpoint = Path(run["checkpoint"])
        if checkpoint.exists():
            continue
        print(f"Training {run['role']} seed={run['seed']} from {run['config']}", flush=True)
        Trainer(Config.from_yaml(run["config"])).train()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Training completed without expected checkpoint: {checkpoint}")


def _evaluate_setting(
    model: torch.nn.Module,
    tokenizer: ThaiMouthTokenizer,
    benchmark: Dict[int, List[Dict]],
    recurrent_steps: int,
    alpha: Optional[float],
    role: str,
    seed: int,
    target_recurrent_steps: int,
    test_path: str,
    protocol: dict,
    device: torch.device,
) -> tuple[List[dict], List[dict], dict]:
    metric_rows = []
    prediction_rows = []
    dynamics_by_hop = {}
    with _temporary_alpha(model, alpha):
        for hop, samples in benchmark.items():
            result = evaluate_reasoning_batch(
                model=model,
                tokenizer=tokenizer,
                samples=samples,
                recurrent_steps=recurrent_steps,
                device=device,
                max_new_tokens=protocol["max_new_tokens"],
                return_predictions=True,
            )
            correct, total, accuracy, dynamics, predictions = result
            difficulty = _difficulty_for_hop(hop, protocol["difficulty"])
            metric_rows.append({
                "role": role,
                "seed": seed,
                "alpha": alpha,
                "target_recurrent_steps": target_recurrent_steps,
                "executed_recurrent_steps": recurrent_steps,
                "hops": hop,
                "difficulty": difficulty,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            })
            for prediction in predictions:
                prediction_rows.append({
                    "role": role,
                    "seed": seed,
                    "alpha": alpha,
                    "target_recurrent_steps": target_recurrent_steps,
                    **prediction,
                })
            if dynamics is not None:
                dynamics_by_hop[str(hop)] = dynamics

        language = evaluate_perplexity(
            model=model,
            test_path=test_path,
            tokenizer=tokenizer,
            recurrent_steps=recurrent_steps,
            max_seq_len=model.config.max_seq_len,
            device=device,
        )
        efficiency_cfg = protocol["efficiency"]
        efficiency = benchmark_efficiency(
            model=model,
            recurrent_steps=recurrent_steps,
            batch_size=efficiency_cfg["batch_size"],
            seq_len=efficiency_cfg["seq_len"],
            num_warmup=efficiency_cfg["num_warmup"],
            num_runs=efficiency_cfg["num_runs"],
            device=device,
        )

    shared = {
        "role": role,
        "seed": seed,
        "alpha": alpha,
        "target_recurrent_steps": target_recurrent_steps,
        "executed_recurrent_steps": recurrent_steps,
        "loss": language["loss"],
        "perplexity": language["perplexity"],
        **efficiency,
        "latent_dynamics_by_hop": dynamics_by_hop,
    }
    return metric_rows, prediction_rows, shared


def _aggregate(metric_rows: List[dict], setting_rows: List[dict]) -> dict:
    per_seed_totals: Dict[tuple, List[int]] = {}
    for row in metric_rows:
        for difficulty in (row["difficulty"], "overall"):
            key = (
                row["role"], row["alpha"], row["target_recurrent_steps"],
                difficulty, row["seed"],
            )
            totals = per_seed_totals.setdefault(key, [0, 0])
            totals[0] += row["correct"]
            totals[1] += row["total"]
    accuracy_groups: Dict[tuple, List[float]] = {}
    for key, (correct, total) in per_seed_totals.items():
        role, alpha, recurrent_steps, difficulty, _seed = key
        group_key = (role, alpha, recurrent_steps, difficulty)
        accuracy_groups.setdefault(group_key, []).append(100.0 * correct / total)
    accuracy = []
    for key, values in accuracy_groups.items():
        role, alpha, recurrent_steps, difficulty = key
        accuracy.append({
            "role": role,
            "alpha": alpha,
            "recurrent_steps": recurrent_steps,
            "difficulty": difficulty,
            "mean_accuracy": float(np.mean(values)),
            "std_accuracy": float(np.std(values)),
            "n": len(values),
        })

    setting_groups: Dict[tuple, List[dict]] = {}
    for row in setting_rows:
        key = (row["role"], row["alpha"], row["target_recurrent_steps"])
        setting_groups.setdefault(key, []).append(row)
    settings = []
    for key, rows in setting_groups.items():
        role, alpha, recurrent_steps = key
        final_norms = []
        final_relative_deltas = []
        for row in rows:
            for dynamics in row.get("latent_dynamics_by_hop", {}).values():
                if dynamics.get("norms"):
                    final_norms.append(dynamics["norms"][-1])
                if dynamics.get("relative_deltas"):
                    final_relative_deltas.append(dynamics["relative_deltas"][-1])
        settings.append({
            "role": role,
            "alpha": alpha,
            "recurrent_steps": recurrent_steps,
            "mean_perplexity": float(np.mean([r["perplexity"] for r in rows])),
            "std_perplexity": float(np.std([r["perplexity"] for r in rows])),
            "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
            "mean_estimated_gflops": float(np.mean([r["estimated_gflops"] for r in rows])),
            "mean_final_latent_norm": float(np.mean(final_norms)) if final_norms else None,
            "mean_final_relative_delta": float(np.mean(final_relative_deltas)) if final_relative_deltas else None,
        })
    return {"accuracy": accuracy, "settings": settings}


def _save_heatmaps(aggregate: dict, protocol: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    alphas = protocol["alphas"]
    recurrent_steps = protocol["recurrent_steps"]
    accuracy_lookup = {
        (row["alpha"], row["recurrent_steps"]): row["mean_accuracy"]
        for row in aggregate["accuracy"]
        if row["role"] == "variant_b" and row["difficulty"] == "overall"
    }
    setting_lookup = {
        (row["alpha"], row["recurrent_steps"]): row
        for row in aggregate["settings"]
        if row["role"] == "variant_b"
    }
    plots = {
        "accuracy": (accuracy_lookup, None, "Accuracy (%)"),
        "perplexity": (setting_lookup, "mean_perplexity", "Perplexity"),
        "latency": (setting_lookup, "mean_latency_ms", "Latency (ms)"),
        "latent_norm": (setting_lookup, "mean_final_latent_norm", "Final latent norm"),
        "relative_delta": (setting_lookup, "mean_final_relative_delta", "Final relative delta Z"),
    }
    for filename, (lookup, field, title) in plots.items():
        matrix = np.full((len(alphas), len(recurrent_steps)), np.nan)
        for row_index, alpha in enumerate(alphas):
            for column_index, recurrent_step in enumerate(recurrent_steps):
                value = lookup.get((alpha, recurrent_step))
                if field is not None and value is not None:
                    value = value[field]
                if value is not None:
                    matrix[row_index, column_index] = value
        figure, axis = plt.subplots(figsize=(8, 4.5))
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        axis.set_xticks(range(len(recurrent_steps)), recurrent_steps)
        axis.set_yticks(range(len(alphas)), alphas)
        axis.set_xlabel("Recurrent steps R")
        axis.set_ylabel("Inference alpha")
        axis.set_title(title)
        for row_index in range(len(alphas)):
            for column_index in range(len(recurrent_steps)):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(column_index, row_index, f"{value:.3g}", ha="center", va="center", color="white")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / f"alpha_r_{filename}.png", dpi=160)
        plt.close(figure)


def _save_tradeoff_curves(aggregate: dict, protocol: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    primary_alpha = protocol["primary_alpha"]
    recurrent_steps = protocol["recurrent_steps"]
    accuracy = {
        (row["role"], row["alpha"], row["recurrent_steps"], row["difficulty"]): row["mean_accuracy"]
        for row in aggregate["accuracy"]
    }
    settings = {
        (row["role"], row["alpha"], row["recurrent_steps"]): row
        for row in aggregate["settings"]
    }
    b_accuracy = [
        accuracy.get(("variant_b", primary_alpha, r, "overall"), np.nan)
        for r in recurrent_steps
    ]
    b_hard = [
        accuracy.get(("variant_b", primary_alpha, r, "hard"), np.nan)
        for r in recurrent_steps
    ]
    c_accuracy = [
        accuracy.get(("compute_control", None, r, "overall"), np.nan)
        for r in recurrent_steps
    ]
    b_settings = [settings.get(("variant_b", primary_alpha, r), {}) for r in recurrent_steps]

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].plot(recurrent_steps, b_accuracy, marker="o", label="Variant B")
    axes[0, 0].plot(recurrent_steps, c_accuracy, marker="s", label="Fixed-depth control")
    axes[0, 0].set(title="Accuracy vs R", xlabel="R", ylabel="Accuracy (%)")
    axes[0, 0].legend()

    axes[0, 1].plot(recurrent_steps, b_accuracy, marker="o", label="Overall")
    axes[0, 1].plot(recurrent_steps, b_hard, marker="s", label="Hard")
    axes[0, 1].set(title="Difficulty response", xlabel="R", ylabel="Accuracy (%)")
    axes[0, 1].legend()

    b_compute = [row.get("mean_estimated_gflops") or np.nan for row in b_settings]
    c_compute = [
        settings.get(("compute_control", None, r), {}).get("mean_estimated_gflops", np.nan)
        for r in recurrent_steps
    ]
    axes[0, 2].plot(b_compute, b_accuracy, marker="o", label="Variant B")
    axes[0, 2].plot(c_compute, c_accuracy, marker="s", label="Fixed-depth control")
    axes[0, 2].set(title="Accuracy vs compute", xlabel="Estimated GFLOPs", ylabel="Accuracy (%)")
    axes[0, 2].legend()

    axes[1, 0].plot(
        recurrent_steps,
        [row.get("mean_perplexity") or np.nan for row in b_settings],
        marker="o",
    )
    axes[1, 0].set(title="PPL vs R", xlabel="R", ylabel="Perplexity")
    axes[1, 1].plot(
        recurrent_steps,
        [row.get("mean_final_latent_norm") or np.nan for row in b_settings],
        marker="o",
    )
    axes[1, 1].set(title="Latent norm vs R", xlabel="R", ylabel="Mean final norm")
    axes[1, 2].plot(
        recurrent_steps,
        [row.get("mean_final_relative_delta") or np.nan for row in b_settings],
        marker="o",
    )
    axes[1, 2].set(title="Relative delta Z vs R", xlabel="R", ylabel="Mean final relative delta")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "milestone_tradeoff_curves.png", dpi=160)
    plt.close(figure)


def _write_csv(path: Path, rows: List[dict], columns: List[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(protocol_path: str, train_missing: bool, device_name: str) -> None:
    protocol = _load_protocol(protocol_path)
    plan = _load_plan(protocol)
    if train_missing:
        _train_missing(plan)

    missing = [run["checkpoint"] for run in plan["runs"] if not Path(run["checkpoint"]).exists()]
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} required checkpoints:\n{preview}{suffix}\n"
            "Train them with --train-missing or the generated YAML configs."
        )

    device = get_device(device_name)
    tokenizer = ThaiMouthTokenizer(Config.from_yaml(protocol["variant_b_config"]).data.tokenizer_path)
    output_dir = Path(protocol["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[dict] = []
    prediction_rows: List[dict] = []
    setting_rows: List[dict] = []

    for run_info in plan["runs"]:
        role = run_info["role"]
        seed = run_info["seed"]
        model, config, _ = load_checkpoint(run_info["checkpoint"], device=device)
        benchmark = generate_depth_partitioned_benchmark(
            hop_depths=protocol["hop_depths"],
            samples_per_depth=protocol["samples_per_depth"],
            seed=protocol["benchmark_seed"],
        )
        if role == "variant_b":
            settings = [
                (recurrent_steps, alpha, recurrent_steps)
                for alpha in protocol["alphas"]
                for recurrent_steps in protocol["recurrent_steps"]
            ]
        elif role == "parameter_control":
            settings = [(1, None, 1)]
        else:
            settings = [(1, None, run_info["target_recurrent_steps"])]

        for executed_steps, alpha, target_steps in settings:
            print(
                f"Evaluating {role} seed={seed} alpha={alpha} target_R={target_steps}",
                flush=True,
            )
            metrics, predictions, setting = _evaluate_setting(
                model=model,
                tokenizer=tokenizer,
                benchmark=benchmark,
                recurrent_steps=executed_steps,
                alpha=alpha,
                role=role,
                seed=seed,
                target_recurrent_steps=target_steps,
                test_path=Config.from_yaml(run_info["config"]).data.test_path,
                protocol=protocol,
                device=device,
            )
            metric_rows.extend(metrics)
            prediction_rows.extend(predictions)
            setting_rows.append(setting)

    aggregate = _aggregate(metric_rows, setting_rows)
    assessment = assess_milestone(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        setting_rows=setting_rows,
        protocol=protocol,
    )
    with open(output_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump({
            "protocol": protocol,
            "plan": plan,
            "metrics": metric_rows,
            "settings": setting_rows,
            "aggregate": aggregate,
            "assessment": assessment,
        }, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv(
        output_dir / "accuracy_by_difficulty.csv",
        aggregate["accuracy"],
        ["role", "alpha", "recurrent_steps", "difficulty", "mean_accuracy", "std_accuracy", "n"],
    )
    _write_csv(
        output_dir / "efficiency_and_ppl.csv",
        aggregate["settings"],
        ["role", "alpha", "recurrent_steps", "mean_perplexity", "std_perplexity", "mean_latency_ms", "mean_estimated_gflops", "mean_final_latent_norm", "mean_final_relative_delta"],
    )
    _save_heatmaps(aggregate, protocol, output_dir)
    _save_tradeoff_curves(aggregate, protocol, output_dir)
    print(f"Saved Milestone 0.3.1 measurements to {output_dir}")
    print(json.dumps(assessment, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Milestone 0.3.1")
    parser.add_argument("--protocol", default="configs/milestone_0_3_1.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-missing", action="store_true")
    args = parser.parse_args()
    try:
        run(args.protocol, args.train_missing, args.device)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
