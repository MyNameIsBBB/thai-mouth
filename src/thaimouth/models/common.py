"""
Common layers, normalizations, activations, and parameter counting utilities.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Normalizes inputs by their root mean square without subtracting mean.
    
    Tensor shapes:
        x: [B, T, D]
        output: [B, T, D]
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm


class MLP(nn.Module):
    """
    Standard Feed-Forward Network with GELU or SwiGLU activation.
    
    Tensor shapes:
        x: [B, T, D]
        hidden: [B, T, D_ff]
        output: [B, T, D]
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = F.gelu(self.fc1(x))  # [B, T, D_ff]
        out = self.fc2(h)        # [B, T, D]
        return self.dropout(out)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Applies rotary transformation to queries and keys.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: [B, H, T, Dh]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(dtype=q.dtype, device=q.device)  # [T, Dh]
        sin = self.sin_cached[:seq_len].to(dtype=q.dtype, device=q.device)  # [T, Dh]

        # Reshape for broadcasting over [B, H, T, Dh]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, Dh]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, T, Dh]

        q_rot = apply_rotary_emb(q, cos, sin)
        k_rot = apply_rotary_emb(k, cos, sin)
        return q_rot, k_rot


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE rotation to tensor x: (x * cos) + (rotate_half(x) * sin)."""
    return (x * cos) + (rotate_half(x) * sin)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Returns the total number of unique learned parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_param_count(num_params: int) -> str:
    """Formats integer parameter count into a human-readable string (e.g. 1.25M)."""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)
