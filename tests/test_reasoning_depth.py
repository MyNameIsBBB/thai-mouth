"""
Tests for Reasoning Depth Benchmark & Latent Dynamics calculations.
"""

import pytest
import torch
from thaimouth.data.synthetic_reasoning import (
    THAI_NAMES_TEST,
    generate_comparison_sample,
    generate_logic_sample,
    generate_arithmetic_sample
)
from thaimouth.evaluation.latent_dynamics import compute_latent_step_dynamics
from thaimouth.evaluation.reasoning_depth import generate_depth_partitioned_benchmark


def test_procedural_hop_generation():
    # Test up to 16 hops
    for hops in [1, 2, 4, 8, 16]:
        comp = generate_comparison_sample(THAI_NAMES_TEST, hops=hops)
        assert comp["hops"] == hops
        assert "ถาม:" in comp["question"]
        assert len(comp["answer"]) > 0

        logic = generate_logic_sample(hops=hops)
        assert logic["hops"] == hops
        assert len(logic["answer"]) > 0

        arith = generate_arithmetic_sample(THAI_NAMES_TEST, steps=min(hops, 8))
        assert arith["hops"] == min(hops, 8)
        assert "บาท" in arith["answer"]


def test_partitioned_benchmark_structure():
    benchmark = generate_depth_partitioned_benchmark(
        hop_depths=[1, 2, 4, 8, 16],
        samples_per_depth=5,
        seed=123
    )
    for hop in [1, 2, 4, 8, 16]:
        assert hop in benchmark
        assert len(benchmark[hop]) == 5


def test_latent_dynamics_computation():
    # Mock trajectory latents for 4 steps: Z_0, Z_1, Z_2, Z_3, Z_4
    batch_size = 2
    seq_len = 8
    d_model = 16
    vocab_size = 32

    latents = [torch.randn(batch_size, seq_len, d_model) for _ in range(5)]
    logits = [torch.randn(batch_size, seq_len, vocab_size) for _ in range(5)]

    metrics = compute_latent_step_dynamics(latents, logits)
    assert metrics["num_steps"] == 4
    assert len(metrics["norms"]) == 5
    assert len(metrics["deltas"]) == 4
    assert len(metrics["cosine_sims"]) == 4
    assert len(metrics["entropies"]) == 5
    assert len(metrics["kl_divergences"]) == 4
    assert all(d >= 0 for d in metrics["deltas"])
