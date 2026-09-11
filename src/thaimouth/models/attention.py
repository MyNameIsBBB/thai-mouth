"""
Manual PyTorch Multi-Head Causal Self-Attention and Cross-Attention.
Includes explicit tensor shape annotations and Rotary Positional Embedding (RoPE).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from thaimouth.models.common import RotaryEmbedding


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.
    
    Tensor shapes:
        B: Batch size
        T: Sequence length
        D: Hidden size (d_model)
        H: Number of attention heads (n_heads)
        Dh: Dimension per head (d_model / n_heads)
        
        Input x:                [B, T, D]
        Query / Key / Value:    [B, H, T, Dh]
        Attention weights:      [B, H, T, T]
        Output:                 [B, T, D]
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Combined QKV projection for computational efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # B = Batch size, T = Sequence length, D = Hidden dimension
        B, T, D = x.shape

        # 1. Project to Q, K, V
        # qkv: [B, T, 3 * D]
        qkv = self.qkv_proj(x)
        # Split into [B, T, D] each
        q, k, v = qkv.chunk(3, dim=-1)

        # 2. Reshape to multi-head format: [B, H, T, Dh]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Apply Rotary Positional Embeddings (RoPE)
        q, k = self.rope(q, k, seq_len=T)

        # 4. Scaled Dot-Product Attention with causal mask
        # is_causal=True automatically builds lower-triangular causal mask
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T]
            # Convert 0 -> -inf, 1 -> 0
            pad_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=q.dtype)) * -1e9
            # Build lower-triangular causal mask: [1, 1, T, T]
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device, dtype=q.dtype), diagonal=1)[None, None, :, :]
            combined_mask = causal_mask + pad_mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=combined_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )

        # 5. Recombine heads: [B, H, T, Dh] -> [B, T, H * Dh] = [B, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        # 6. Final linear output projection
        return self.dropout(self.out_proj(out))


class CrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention for conditioning latent states Z on context tokens X.
    
    Tensor shapes:
        q_in (latent Z):  [B, T_q, D]
        kv_in (context X):[B, T_kv, D]
        Output:           [B, T_q, D]
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T_q, D = q_in.shape
        _, T_kv, _ = kv_in.shape

        q = self.q_proj(q_in).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_in)
        k, v = kv.chunk(2, dim=-1)
        k = k.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, T_kv] -> [B, 1, 1, T_kv]
            attn_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=q.dtype)) * -1e9

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, T_q, D)
        return self.dropout(self.out_proj(out))
