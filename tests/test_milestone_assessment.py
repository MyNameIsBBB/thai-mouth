from thaimouth.evaluation.milestone import assess_milestone


def test_assessment_requires_compute_matched_advantage():
    protocol = {
        "primary_alpha": 0.5,
        "recurrent_steps": [1, 2],
        "seeds": [1, 2],
        "exit_criteria": {
            "max_norm_growth_ratio": 8.0,
            "min_unique_output_fraction": 0.1,
            "min_nonempty_output_fraction": 0.9,
            "min_accuracy_gain_points": 1.0,
            "min_positive_seed_fraction": 1.0,
            "min_monotonic_transition_fraction": 1.0,
            "min_control_margin_points": 0.0,
            "min_compute_control_win_fraction": 0.6,
            "min_dynamics_decrease_fraction": 1.0,
            "min_initial_relative_delta": 0.0001,
        },
    }
    metrics = []
    settings = []
    predictions = []
    for seed in protocol["seeds"]:
        for recurrent_steps, b_accuracy, c_accuracy in [(1, 50, 55), (2, 70, 75)]:
            for difficulty in ["easy", "hard"]:
                metrics.extend([
                    {
                        "role": "variant_b", "seed": seed, "alpha": 0.5,
                        "target_recurrent_steps": recurrent_steps,
                        "difficulty": difficulty, "correct": b_accuracy, "total": 100,
                    },
                    {
                        "role": "compute_control", "seed": seed, "alpha": None,
                        "target_recurrent_steps": recurrent_steps,
                        "difficulty": difficulty, "correct": c_accuracy, "total": 100,
                    },
                ])
            settings.extend([
                {
                    "role": "variant_b", "seed": seed, "alpha": 0.5,
                    "target_recurrent_steps": recurrent_steps,
                    "estimated_gflops": float(recurrent_steps),
                    "latent_dynamics_by_hop": {
                        "1": {"norms": [1.0, 1.5], "relative_deltas": [0.5, 0.2]}
                    },
                },
                {
                    "role": "compute_control", "seed": seed, "alpha": None,
                    "target_recurrent_steps": recurrent_steps,
                    "estimated_gflops": float(recurrent_steps),
                },
            ])
            predictions.extend([
                {
                    "role": "variant_b", "seed": seed, "alpha": 0.5,
                    "target_recurrent_steps": recurrent_steps,
                    "generated_completion": f"answer-{seed}-{recurrent_steps}-a",
                },
                {
                    "role": "variant_b", "seed": seed, "alpha": 0.5,
                    "target_recurrent_steps": recurrent_steps,
                    "generated_completion": f"answer-{seed}-{recurrent_steps}-b",
                },
            ])

    result = assess_milestone(metrics, predictions, settings, protocol)
    assert result["gates"]["numerical_stability"]
    assert result["gates"]["adaptive_computation"]
    assert not result["gates"]["useful_computation"]
    assert result["interpretation"] == "adaptive_recurrent_compute_evidence"
