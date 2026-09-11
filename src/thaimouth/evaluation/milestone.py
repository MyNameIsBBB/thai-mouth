"""Milestone 0.3.1 exit-gate calculations from measured benchmark records."""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


def _accuracy_by_key(
    rows: List[dict], role: str, alpha: Optional[float]
) -> Dict[Tuple[int, int, str], float]:
    totals = defaultdict(lambda: [0, 0])
    for row in rows:
        if row["role"] != role or row.get("alpha") != alpha:
            continue
        base = (row["seed"], row["target_recurrent_steps"])
        for difficulty in ("overall", row["difficulty"]):
            key = (*base, difficulty)
            totals[key][0] += row["correct"]
            totals[key][1] += row["total"]
    return {
        key: 100.0 * correct / total
        for key, (correct, total) in totals.items()
        if total
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def assess_milestone(
    metric_rows: List[dict],
    prediction_rows: List[dict],
    setting_rows: List[dict],
    protocol: dict,
) -> dict:
    """Evaluate all pre-registered Milestone 0.3.1 gates."""
    thresholds = protocol["exit_criteria"]
    primary_alpha = protocol["primary_alpha"]
    recurrent_steps = protocol["recurrent_steps"]
    min_r, max_r = min(recurrent_steps), max(recurrent_steps)
    seeds = protocol["seeds"]

    b_accuracy = _accuracy_by_key(metric_rows, "variant_b", primary_alpha)
    c_accuracy = _accuracy_by_key(metric_rows, "compute_control", None)

    norm_growth_ratios = []
    dynamics_decrease_checks = []
    initial_relative_deltas = []
    finite_dynamics = True
    for setting in setting_rows:
        if setting["role"] != "variant_b" or setting.get("alpha") != primary_alpha:
            continue
        for dynamics in setting.get("latent_dynamics_by_hop", {}).values():
            norms = dynamics.get("norms", [])
            relative = dynamics.get("relative_deltas", [])
            finite_dynamics &= all(math.isfinite(value) for value in norms + relative)
            if norms and norms[0] > 0:
                norm_growth_ratios.append(max(norms) / norms[0])
            if len(relative) >= 2:
                width = max(1, len(relative) // 4)
                initial = _mean(relative[:width])
                final = _mean(relative[-width:])
                initial_relative_deltas.append(initial)
                dynamics_decrease_checks.append(final < initial)

    completion_groups = defaultdict(list)
    for row in prediction_rows:
        if row["role"] == "variant_b" and row.get("alpha") == primary_alpha:
            key = (row["seed"], row["target_recurrent_steps"])
            completion_groups[key].append(row.get("generated_completion", "").strip())
    unique_fractions = [
        len(set(completions)) / len(completions)
        for completions in completion_groups.values()
        if completions
    ]
    nonempty_fractions = [
        sum(bool(completion) for completion in completions) / len(completions)
        for completions in completion_groups.values()
        if completions
    ]

    seed_gains = {
        seed: b_accuracy.get((seed, max_r, "overall"), float("nan"))
        - b_accuracy.get((seed, min_r, "overall"), float("nan"))
        for seed in seeds
    }
    valid_gains = [gain for gain in seed_gains.values() if math.isfinite(gain)]
    positive_seed_fraction = (
        sum(gain >= thresholds["min_accuracy_gain_points"] for gain in valid_gains)
        / len(valid_gains)
        if valid_gains else 0.0
    )
    mean_curve = {
        recurrent_step: _mean(
            b_accuracy[(seed, recurrent_step, "overall")]
            for seed in seeds
            if (seed, recurrent_step, "overall") in b_accuracy
        )
        for recurrent_step in recurrent_steps
    }
    transitions = [
        mean_curve[right] >= mean_curve[left]
        for left, right in zip(recurrent_steps, recurrent_steps[1:])
        if math.isfinite(mean_curve[left]) and math.isfinite(mean_curve[right])
    ]
    monotonic_fraction = sum(transitions) / len(transitions) if transitions else 0.0

    margins = {"overall": [], "hard": []}
    for difficulty in margins:
        for seed in seeds:
            for recurrent_step in recurrent_steps:
                if recurrent_step == min_r:
                    continue
                key = (seed, recurrent_step, difficulty)
                if key in b_accuracy and key in c_accuracy:
                    margins[difficulty].append(b_accuracy[key] - c_accuracy[key])
    mean_margins = {name: _mean(values) for name, values in margins.items()}
    win_fractions = {
        name: (
            sum(value > thresholds["min_control_margin_points"] for value in values)
            / len(values)
            if values else 0.0
        )
        for name, values in margins.items()
    }

    setting_index = {
        (row["role"], row.get("alpha"), row["seed"], row["target_recurrent_steps"]): row
        for row in setting_rows
    }
    b_flops = _mean(
        setting_index[("variant_b", primary_alpha, seed, max_r)]["estimated_gflops"]
        - setting_index[("variant_b", primary_alpha, seed, min_r)]["estimated_gflops"]
        for seed in seeds
        if ("variant_b", primary_alpha, seed, min_r) in setting_index
        and ("variant_b", primary_alpha, seed, max_r) in setting_index
    )
    c_flops = _mean(
        setting_index[("compute_control", None, seed, max_r)]["estimated_gflops"]
        - setting_index[("compute_control", None, seed, min_r)]["estimated_gflops"]
        for seed in seeds
        if ("compute_control", None, seed, min_r) in setting_index
        and ("compute_control", None, seed, max_r) in setting_index
    )
    b_gain = mean_curve[max_r] - mean_curve[min_r]
    c_min = _mean(
        c_accuracy[(seed, min_r, "overall")]
        for seed in seeds
        if (seed, min_r, "overall") in c_accuracy
    )
    c_max = _mean(
        c_accuracy[(seed, max_r, "overall")]
        for seed in seeds
        if (seed, max_r, "overall") in c_accuracy
    )
    c_gain = c_max - c_min
    b_gain_per_gflop = b_gain / b_flops if b_flops and math.isfinite(b_flops) else float("nan")
    c_gain_per_gflop = c_gain / c_flops if c_flops and math.isfinite(c_flops) else float("nan")

    gates = {
        "numerical_stability": (
            finite_dynamics
            and bool(norm_growth_ratios)
            and max(norm_growth_ratios) <= thresholds["max_norm_growth_ratio"]
            and bool(unique_fractions)
            and min(unique_fractions) >= thresholds["min_unique_output_fraction"]
            and bool(nonempty_fractions)
            and min(nonempty_fractions) >= thresholds["min_nonempty_output_fraction"]
        ),
        "adaptive_computation": (
            bool(valid_gains)
            and positive_seed_fraction >= thresholds["min_positive_seed_fraction"]
            and monotonic_fraction >= thresholds["min_monotonic_transition_fraction"]
        ),
        "useful_computation": (
            all(math.isfinite(value) for value in mean_margins.values())
            and mean_margins["overall"] > thresholds["min_control_margin_points"]
            and mean_margins["hard"] > thresholds["min_control_margin_points"]
            and win_fractions["overall"] >= thresholds["min_compute_control_win_fraction"]
            and win_fractions["hard"] >= thresholds["min_compute_control_win_fraction"]
        ),
        "efficiency": (
            math.isfinite(b_gain_per_gflop)
            and math.isfinite(c_gain_per_gflop)
            and b_gain_per_gflop > c_gain_per_gflop
        ),
        "dynamics": (
            bool(dynamics_decrease_checks)
            and sum(dynamics_decrease_checks) / len(dynamics_decrease_checks)
            >= thresholds["min_dynamics_decrease_fraction"]
            and bool(initial_relative_deltas)
            and min(initial_relative_deltas) >= thresholds["min_initial_relative_delta"]
        ),
    }
    if gates["numerical_stability"] and gates["adaptive_computation"] and gates["useful_computation"]:
        interpretation = "ready_for_thaimouth_5m"
    elif gates["numerical_stability"] and gates["adaptive_computation"]:
        interpretation = "adaptive_recurrent_compute_evidence"
    elif gates["numerical_stability"]:
        interpretation = "numerical_stability_only"
    else:
        interpretation = "milestone_failed"

    return {
        "passed": all(gates.values()),
        "interpretation": interpretation,
        "gates": gates,
        "evidence": {
            "primary_alpha": primary_alpha,
            "max_norm_growth_ratio": max(norm_growth_ratios, default=float("nan")),
            "min_unique_output_fraction": min(unique_fractions, default=float("nan")),
            "min_nonempty_output_fraction": min(nonempty_fractions, default=float("nan")),
            "seed_accuracy_gains": seed_gains,
            "positive_seed_fraction": positive_seed_fraction,
            "mean_accuracy_curve": mean_curve,
            "monotonic_transition_fraction": monotonic_fraction,
            "mean_compute_control_margin_points": mean_margins,
            "compute_control_win_fraction": win_fractions,
            "variant_b_accuracy_gain_per_gflop": b_gain_per_gflop,
            "control_accuracy_gain_per_gflop": c_gain_per_gflop,
            "dynamics_decrease_fraction": (
                sum(dynamics_decrease_checks) / len(dynamics_decrease_checks)
                if dynamics_decrease_checks else 0.0
            ),
            "min_initial_relative_delta": min(initial_relative_deltas, default=float("nan")),
        },
    }
