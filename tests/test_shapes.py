"""
Unit tests for model forward output shapes across all 3 architectures.
"""

import pytest
import torch
from thaimouth.config import ModelConfig
from thaimouth.models.baseline import TinyTransformerBaseline
from thaimouth.models.latent import RecurrentLatentLM
from thaimouth.models.recurrent import RecurrentTransformer


@pytest.mark.parametrize("model_cls", [TinyTransformerBaseline, RecurrentTransformer, RecurrentLatentLM])
def test_model_forward_shapes(model_cls):
    B, T, V, D = 2, 12, 256, 64
    cfg = ModelConfig(
        vocab_size=V,
        d_model=D,
        n_heads=4,
        d_ff=128,
        n_layers=2,
        input_layers=1,
        think_layers=1,
        recurrent_steps=2,
        max_seq_len=64
    )
    model = model_cls(cfg)
    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    
    out = model(input_ids=input_ids, labels=labels)
    
    assert "logits" in out
    assert "loss" in out
    assert out["logits"].shape == (B, T, V), f"Logits shape mismatch: {out['logits'].shape}"
    assert out["loss"] is not None
    assert torch.isfinite(out["loss"]), "Loss is not finite (NaN/Inf)"
