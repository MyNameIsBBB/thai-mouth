"""
Critical unit test: Verifies that recurrent computation steps
strictly REUSE the same parameter tensors and do NOT increase parameter count.
"""

import pytest
import torch
from thaimouth.config import ModelConfig
from thaimouth.models.common import count_parameters
from thaimouth.models.latent import RecurrentLatentLM
from thaimouth.models.recurrent import RecurrentTransformer


def test_recurrent_transformer_weight_sharing():
    cfg = ModelConfig(
        model_type="recurrent",
        vocab_size=512,
        d_model=64,
        n_heads=4,
        d_ff=128,
        input_layers=1,
        think_layers=1,
        recurrent_steps=1,
        max_seq_len=64
    )
    model = RecurrentTransformer(cfg)
    params_initial = count_parameters(model)
    
    x = torch.randint(0, 512, (2, 8))
    
    # 1. Run with 1 recurrent step
    out1 = model(x, recurrent_steps=1)
    params_step1 = count_parameters(model)
    assert params_initial == params_step1, "Parameter count altered after 1 step"
    
    # 2. Run with 8 recurrent steps
    out8 = model(x, recurrent_steps=8)
    params_step8 = count_parameters(model)
    assert params_initial == params_step8, "Parameter count altered after 8 steps"
    
    # 3. Outputs should differ between step 1 and step 8 due to additional compute depth
    diff = torch.max(torch.abs(out1["logits"] - out8["logits"])).item()
    assert diff > 1e-4, f"Recurrent steps 1 and 8 produced identical output (diff={diff}). Compute was not applied!"


def test_recurrent_latent_weight_sharing():
    cfg = ModelConfig(
        model_type="latent",
        vocab_size=512,
        d_model=64,
        n_heads=4,
        d_ff=128,
        input_layers=1,
        think_layers=1,
        recurrent_steps=1,
        latent_alpha=1.0,
        max_seq_len=64
    )
    model = RecurrentLatentLM(cfg)
    params_initial = count_parameters(model)
    
    x = torch.randint(0, 512, (2, 8))
    
    # 1. Run with 1 recurrent step
    out1 = model(x, recurrent_steps=1)
    params_step1 = count_parameters(model)
    assert params_initial == params_step1
    
    # 2. Run with 8 recurrent steps
    out8 = model(x, recurrent_steps=8)
    params_step8 = count_parameters(model)
    assert params_initial == params_step8
    
    # 3. Verify outputs differ
    diff = torch.max(torch.abs(out1["logits"] - out8["logits"])).item()
    assert diff > 1e-4, "Recurrent latent steps 1 and 8 produced identical output!"
