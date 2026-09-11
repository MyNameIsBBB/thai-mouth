"""
Tests for Milestone 0.3 Recurrent Latent Stabilization Variants:
A. Current Baseline: Z_{t+1} = Z_t + alpha * F(Z_t, X)
B. PreNorm: Z_{t+1} = Z_t + alpha * F(RMSNorm(Z_t), X)
C. PostNorm: Z_{t+1} = RMSNorm(Z_t + alpha * F(Z_t, X))
D. Gated PreNorm: Z_{t+1} = Z_t + sigmoid(W_g * RMSNorm(Z_t)) * F(RMSNorm(Z_t), X)
"""

import pytest
import torch
from thaimouth.config import ModelConfig
from thaimouth.models.latent import RecurrentLatentLM


@pytest.mark.parametrize("norm_variant", ["none", "prenorm", "postnorm", "gated_prenorm"])
def test_recurrent_norm_variants(norm_variant):
    cfg = ModelConfig(
        model_type="latent",
        vocab_size=100,
        d_model=32,
        n_heads=2,
        d_ff=64,
        input_layers=1,
        think_layers=1,
        recurrent_steps=4,
        max_seq_len=64,
        latent_alpha=0.5,
        recurrent_norm=norm_variant
    )
    model = RecurrentLatentLM(cfg)
    
    input_ids = torch.randint(0, 100, (2, 16))
    labels = torch.randint(0, 100, (2, 16))
    
    # Forward with trajectory
    out = model(input_ids, labels=labels, recurrent_steps=4, return_trajectory=True)
    
    assert "logits" in out
    assert out["logits"].shape == (2, 16, 100)
    assert out["loss"] is not None
    assert len(out["trajectory_latents"]) == 5  # Z_0, Z_1, Z_2, Z_3, Z_4
    
    # Backward pass check
    out["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
