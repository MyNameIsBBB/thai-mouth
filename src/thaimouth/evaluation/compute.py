"""Analytical compute estimates for compute-matched experiments.

The estimator counts one multiply and one add as two FLOPs. Embedding lookups,
normalization, activations, masking, and softmax are omitted and reported as
such; the dominant matrix multiplications and attention score products are
included. The same convention is used for every compared architecture.
"""

from dataclasses import dataclass
from typing import Union

import torch

from thaimouth.config import ModelConfig


@dataclass(frozen=True)
class ComputeEstimate:
    total_flops: int
    transformer_block_flops: int
    executed_blocks: int
    lm_head_flops: int
    batch_size: int
    seq_len: int
    recurrent_steps: int

    @property
    def gflops(self) -> float:
        return self.total_flops / 1e9


def _model_config(model_or_config: Union[torch.nn.Module, ModelConfig]) -> ModelConfig:
    if isinstance(model_or_config, ModelConfig):
        return model_or_config
    config = getattr(model_or_config, "config", None)
    if not isinstance(config, ModelConfig):
        raise TypeError("Expected a ModelConfig or a model with a ModelConfig at .config")
    return config


def _executed_blocks(config: ModelConfig, recurrent_steps: int) -> int:
    model_type = config.model_type.lower()
    if model_type == "baseline":
        return config.n_layers
    if model_type == "recurrent":
        return config.input_layers + recurrent_steps * config.think_layers
    if model_type == "latent":
        # RecurrentLatentLM currently owns exactly one shared LatentThinkBlock.
        if config.think_layers != 1:
            raise ValueError(
                "RecurrentLatentLM currently requires think_layers=1 for an honest "
                "analytical FLOPs estimate"
            )
        return config.input_layers + recurrent_steps
    raise ValueError(f"Unsupported model_type: {config.model_type!r}")


def estimate_forward_flops(
    model_or_config: Union[torch.nn.Module, ModelConfig],
    recurrent_steps: int = 1,
    batch_size: int = 1,
    seq_len: int = 128,
) -> ComputeEstimate:
    """Estimate FLOPs for one full-sequence causal LM forward pass."""
    if recurrent_steps < 1 or batch_size < 1 or seq_len < 1:
        raise ValueError("recurrent_steps, batch_size, and seq_len must be positive")

    config = _model_config(model_or_config)
    batch = batch_size
    tokens = seq_len
    width = config.d_model
    hidden = config.d_ff

    # Q/K/V/output projections, QK^T and AV, then the two MLP projections.
    block_macs = (
        4 * batch * tokens * width * width
        + 2 * batch * tokens * tokens * width
        + 2 * batch * tokens * width * hidden
    )
    block_flops = 2 * block_macs
    blocks = _executed_blocks(config, recurrent_steps)
    lm_head_flops = 2 * batch * tokens * width * config.vocab_size
    total_flops = blocks * block_flops + lm_head_flops

    return ComputeEstimate(
        total_flops=total_flops,
        transformer_block_flops=block_flops,
        executed_blocks=blocks,
        lm_head_flops=lm_head_flops,
        batch_size=batch,
        seq_len=tokens,
        recurrent_steps=recurrent_steps,
    )


def relative_compute_gap(target_flops: int, control_flops: int) -> float:
    """Return absolute relative FLOPs difference from the target."""
    if target_flops <= 0 or control_flops <= 0:
        raise ValueError("FLOPs must be positive")
    return abs(control_flops - target_flops) / target_flops
