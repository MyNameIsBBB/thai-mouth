"""Generate the locked training configs for Milestone 0.3.1.

The fixed-depth controls keep the Variant B width and execute the same number
of Transformer blocks as Variant B at each R. This gives an exact match under
ThaiMouth's documented dominant-matmul FLOPs convention.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

from thaimouth.config import Config
from thaimouth.evaluation.compute import estimate_forward_flops, relative_compute_gap
from thaimouth.models import build_model


def _load_protocol(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw["experiment"]


def _save_config(config: Config, path: Path, output_dir: Path, seed: int) -> None:
    config.training.seed = seed
    config.training.output_dir = str(output_dir)
    config.save_yaml(path)


def prepare(protocol_path: str) -> dict:
    protocol = _load_protocol(protocol_path)
    base = Config.from_yaml(protocol["variant_b_config"])
    if base.model.model_type != "latent" or base.model.recurrent_norm != "prenorm":
        raise ValueError("variant_b_config must describe the PreNorm latent Variant B")
    if base.model.think_layers != 1:
        raise ValueError("Variant B currently requires think_layers=1")

    config_root = Path(protocol["generated_config_dir"])
    checkpoint_root = Path(protocol["checkpoint_dir"])
    config_root.mkdir(parents=True, exist_ok=True)
    seeds = protocol["seeds"]
    recurrent_steps = protocol["recurrent_steps"]
    seq_len = protocol["efficiency"]["seq_len"]
    param_tolerance = protocol["matching"]["max_parameter_gap_fraction"]
    compute_tolerance = protocol["matching"]["max_compute_gap_fraction"]

    b_params = build_model(base.model).count_params()
    plan = {
        "protocol": protocol_path,
        "flops_convention": "multiply_add_is_2; dominant_matmuls_only",
        "variant_b_parameters": b_params,
        "runs": [],
    }

    for seed in seeds:
        variant = copy.deepcopy(base)
        variant_path = config_root / f"variant_b_seed_{seed}.yaml"
        variant_out = checkpoint_root / "variant_b" / f"seed_{seed}"
        _save_config(variant, variant_path, variant_out, seed)
        plan["runs"].append({
            "role": "variant_b",
            "seed": seed,
            "config": str(variant_path),
            "checkpoint": str(variant_out / "model_final.pt"),
        })

        parameter_control = copy.deepcopy(base)
        parameter_control.model.model_type = "baseline"
        parameter_control.model.n_layers = base.model.input_layers + 1
        parameter_control.model.recurrent_norm = "none"
        parameter_path = config_root / f"parameter_control_seed_{seed}.yaml"
        parameter_out = checkpoint_root / "parameter_control" / f"seed_{seed}"
        _save_config(parameter_control, parameter_path, parameter_out, seed)
        control_params = build_model(parameter_control.model).count_params()
        param_gap = abs(control_params - b_params) / b_params
        if param_gap > param_tolerance:
            raise ValueError(
                f"Parameter control gap {param_gap:.2%} exceeds tolerance {param_tolerance:.2%}"
            )
        plan["runs"].append({
            "role": "parameter_control",
            "seed": seed,
            "config": str(parameter_path),
            "checkpoint": str(parameter_out / "model_final.pt"),
            "parameters": control_params,
            "parameter_gap_fraction": param_gap,
        })

        for recurrent_step in recurrent_steps:
            if recurrent_step == 1:
                # At R=1 the parameter control is also the exact compute control.
                compute_control = parameter_control
                control_path = parameter_path
                control_out = parameter_out
            else:
                compute_control = copy.deepcopy(base)
                compute_control.model.model_type = "baseline"
                compute_control.model.n_layers = base.model.input_layers + recurrent_step
                compute_control.model.recurrent_norm = "none"
                control_path = config_root / f"compute_r{recurrent_step}_seed_{seed}.yaml"
                control_out = checkpoint_root / f"compute_r{recurrent_step}" / f"seed_{seed}"
                _save_config(compute_control, control_path, control_out, seed)

            target_compute = estimate_forward_flops(
                base.model, recurrent_steps=recurrent_step, seq_len=seq_len
            )
            control_compute = estimate_forward_flops(
                compute_control.model, recurrent_steps=1, seq_len=seq_len
            )
            compute_gap = relative_compute_gap(
                target_compute.total_flops, control_compute.total_flops
            )
            if compute_gap > compute_tolerance:
                raise ValueError(
                    f"R={recurrent_step} compute gap {compute_gap:.2%} exceeds "
                    f"tolerance {compute_tolerance:.2%}"
                )
            plan["runs"].append({
                "role": "compute_control",
                "target_recurrent_steps": recurrent_step,
                "seed": seed,
                "config": str(control_path),
                "checkpoint": str(control_out / "model_final.pt"),
                "parameters": build_model(compute_control.model).count_params(),
                "target_gflops": target_compute.gflops,
                "control_gflops": control_compute.gflops,
                "compute_gap_fraction": compute_gap,
            })

    plan_path = config_root / "plan.json"
    with open(plan_path, "w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2)
    print(f"Prepared {len(plan['runs'])} runs in {config_root}")
    print(f"Machine-readable plan: {plan_path}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Milestone 0.3.1 configs")
    parser.add_argument("--protocol", default="configs/milestone_0_3_1.yaml")
    args = parser.parse_args()
    prepare(args.protocol)


if __name__ == "__main__":
    main()
