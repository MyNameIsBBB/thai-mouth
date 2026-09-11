"""
Loss functions and perplexity computation for ThaiMouth language models.
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Computes standard next-token cross-entropy loss.
    
    Args:
        logits: [B, T, V]
        labels: [B, T]
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing
    )
    return loss


def compute_perplexity(loss_value: float) -> float:
    """Computes perplexity from cross-entropy loss value."""
    try:
        return math.exp(min(loss_value, 100.0))
    except OverflowError:
        return float("inf")
