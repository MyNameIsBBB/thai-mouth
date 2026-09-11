"""
Standard Decoder-Only Causal Transformer Baseline (TinyTransformerBaseline).
Fixed depth, next-token prediction, pre-norm architecture.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from thaimouth.config import ModelConfig
from thaimouth.models.common import RMSNorm, count_parameters
from thaimouth.models.transformer_block import TransformerBlock


class TinyTransformerBaseline(nn.Module):
    """
    Standard decoder-only causal Transformer baseline.
    
    Architecture:
        Tokens -> Embedding -> N x TransformerBlock -> FinalNorm -> LM Head
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                norm_type=config.norm_type
            )
            for _ in range(config.n_layers)
        ])
        
        self.final_norm = RMSNorm(config.d_model) if config.norm_type == "rmsnorm" else nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
            
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        recurrent_steps: Optional[int] = None  # Ignored on standard baseline for API consistency
    ) -> Dict[str, torch.Tensor]:
        # B = batch size, T = sequence length
        B, T = input_ids.shape
        
        # 1. Token Embeddings: [B, T] -> [B, T, D]
        h = self.drop(self.tok_embeddings(input_ids))
        
        # 2. Sequential Transformer Blocks
        for block in self.blocks:
            h = block(h, attention_mask=attention_mask)
            
        # 3. Final Normalization: [B, T, D]
        h = self.final_norm(h)
        
        # 4. Logits: [B, T, D] -> [B, T, vocab_size]
        logits = self.lm_head(h)
        
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling: logits[:-1] predicts labels[1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
        return {
            "logits": logits,
            "loss": loss,
            "last_hidden_state": h
        }

    def count_params(self) -> int:
        return count_parameters(self)
