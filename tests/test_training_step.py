"""
Unit test for forward-backward optimization step and checkpointing.
"""

from pathlib import Path
import pytest
import torch
from thaimouth.config import Config, ModelConfig, TrainConfig
from thaimouth.models.latent import RecurrentLatentLM
from thaimouth.training.checkpoint import load_checkpoint, save_checkpoint


def test_optimization_step():
    cfg = ModelConfig(
        model_type="latent",
        vocab_size=256,
        d_model=64,
        n_heads=4,
        d_ff=128,
        input_layers=1,
        think_layers=1,
        recurrent_steps=2,
        max_seq_len=32
    )
    model = RecurrentLatentLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    input_ids = torch.randint(0, 256, (2, 16))
    labels = input_ids.clone()
    
    # Initial loss
    out = model(input_ids=input_ids, labels=labels)
    initial_loss = out["loss"]
    assert torch.isfinite(initial_loss), "Initial loss is not finite"
    
    # Backward + step
    initial_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Next step loss should be finite
    out2 = model(input_ids=input_ids, labels=labels)
    assert torch.isfinite(out2["loss"])


def test_checkpoint_save_and_load(tmp_path):
    cfg = Config(
        model=ModelConfig(
            model_type="latent",
            vocab_size=256,
            d_model=64,
            n_heads=4,
            d_ff=128
        )
    )
    model = RecurrentLatentLM(cfg.model)
    ckpt_file = save_checkpoint(
        model=model,
        config=cfg,
        output_dir=tmp_path,
        epoch=1,
        step=50,
        val_loss=2.345
    )
    assert ckpt_file.exists()
    
    loaded_model, loaded_cfg, meta = load_checkpoint(ckpt_file)
    assert meta["step"] == 50
    assert meta["val_loss"] == 2.345
    assert loaded_cfg.model.model_type == "latent"
