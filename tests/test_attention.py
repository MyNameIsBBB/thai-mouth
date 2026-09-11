"""
Unit tests for causal self-attention, RoPE, and masking.
"""

import pytest
import torch
from thaimouth.models.attention import CausalSelfAttention


def test_causal_self_attention_shape():
    B, T, D, H = 2, 16, 64, 4
    attn = CausalSelfAttention(d_model=D, n_heads=H, max_seq_len=128)
    x = torch.randn(B, T, D)
    
    out = attn(x)
    assert out.shape == (B, T, D), f"Expected shape ({B}, {T}, {D}), got {out.shape}"


def test_causal_masking_no_future_leakage():
    """
    Verifies causal masking: modifying token at index t > 0 should NOT affect output at index 0.
    """
    B, T, D, H = 1, 8, 64, 4
    attn = CausalSelfAttention(d_model=D, n_heads=H, max_seq_len=128)
    attn.eval()
    
    x1 = torch.randn(B, T, D)
    x2 = x1.clone()
    # Modify last token
    x2[:, -1, :] = x2[:, -1, :] + 10.0
    
    with torch.no_grad():
        out1 = attn(x1)
        out2 = attn(x2)
        
    # Output at token 0 must be identical because token 0 cannot attend to token T-1
    diff_first_token = torch.max(torch.abs(out1[:, 0, :] - out2[:, 0, :])).item()
    assert diff_first_token < 1e-5, f"Future token leakage detected! Diff at t=0 was {diff_first_token}"
