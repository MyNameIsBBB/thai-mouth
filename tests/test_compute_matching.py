import pytest

from thaimouth.config import ModelConfig
from thaimouth.evaluation.compute import estimate_forward_flops, relative_compute_gap


def _latent_config() -> ModelConfig:
    return ModelConfig(
        model_type="latent",
        vocab_size=256,
        d_model=32,
        n_heads=4,
        d_ff=128,
        input_layers=1,
        think_layers=1,
        recurrent_norm="prenorm",
    )


def test_same_width_deep_control_matches_recurrent_flops():
    latent = _latent_config()
    for recurrent_steps in [1, 2, 4, 8, 16, 32]:
        control = ModelConfig(
            model_type="baseline",
            vocab_size=latent.vocab_size,
            d_model=latent.d_model,
            n_heads=latent.n_heads,
            d_ff=latent.d_ff,
            n_layers=latent.input_layers + recurrent_steps,
        )
        target = estimate_forward_flops(latent, recurrent_steps, seq_len=64)
        matched = estimate_forward_flops(control, 1, seq_len=64)
        assert target.total_flops == matched.total_flops
        assert relative_compute_gap(target.total_flops, matched.total_flops) == 0.0


def test_latent_estimator_rejects_unimplemented_think_layer_multiplicity():
    config = _latent_config()
    config.think_layers = 2
    with pytest.raises(ValueError, match="think_layers=1"):
        estimate_forward_flops(config, recurrent_steps=4)


def test_compute_estimate_scales_with_recurrent_depth():
    config = _latent_config()
    r1 = estimate_forward_flops(config, recurrent_steps=1, seq_len=32)
    r8 = estimate_forward_flops(config, recurrent_steps=8, seq_len=32)
    assert r8.total_flops > r1.total_flops
    assert r8.executed_blocks == config.input_layers + 8
