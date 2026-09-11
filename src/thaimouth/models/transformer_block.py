"""
Pre-Norm Transformer Block and Shared ThinkBlock modules.
"""

from typing import Optional
import torch
import torch.nn as nn
from thaimouth.models.attention import CausalSelfAttention, CrossAttention
from thaimouth.models.common import MLP, RMSNorm


class TransformerBlock(nn.Module):
    """
    Standard Pre-Norm Transformer Block.
    
    Update:
        x = x + SelfAttention(Norm(x))
        x = x + MLP(Norm(x))
        
    Tensor shapes:
        x: [B, T, D]
        output: [B, T, D]
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm"
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model) if norm_type == "rmsnorm" else nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.norm2 = RMSNorm(d_model) if norm_type == "rmsnorm" else nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-Norm + Residual Self-Attention
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        # Pre-Norm + Residual MLP
        x = x + self.mlp(self.norm2(x))
        return x


class LatentThinkBlock(nn.Module):
    """
    Latent Refinement Block F_theta(Z, Context=X).
    Allows recurrent updates of latent states with context conditioning.
    
    Update:
        Z = Z + SelfAttention(Norm(Z))
        (optional) Z = Z + CrossAttention(Norm(Z), X)
        Z = Z + MLP(Norm(Z))
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        use_cross_attn: bool = False,
        norm_type: str = "rmsnorm"
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.norm_self = RMSNorm(d_model) if norm_type == "rmsnorm" else nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        if use_cross_attn:
            self.norm_cross = RMSNorm(d_model) if norm_type == "rmsnorm" else nn.LayerNorm(d_model)
            self.cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
            
        self.norm_mlp = RMSNorm(d_model) if norm_type == "rmsnorm" else nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        z: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 1. Self-attention over latent sequence
        z = z + self.self_attn(self.norm_self(z), attention_mask=attention_mask)
        
        # 2. Cross-attention over context (if enabled)
        if self.use_cross_attn and context is not None:
            z = z + self.cross_attn(self.norm_cross(z), context, attention_mask=attention_mask)
            
        # 3. Latent MLP
        z = z + self.mlp(self.norm_mlp(z))
        return z


class ThinkBlock(nn.Module):
    """
    Shared ThinkBlock consisting of one or more sequential Transformer layers.
    This module is repeatedly invoked across recurrent steps without duplicating parameters.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int = 1,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm"
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                norm_type=norm_type
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x
