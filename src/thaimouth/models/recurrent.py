"""
Recurrent Transformer Model (RecurrentTransformer).
Reuses the SAME shared ThinkBlock weights across multiple recurrent steps.
Increases compute depth without increasing parameter count.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from thaimouth.config import ModelConfig
from thaimouth.models.common import RMSNorm, count_parameters
from thaimouth.models.transformer_block import ThinkBlock, TransformerBlock


class RecurrentTransformer(nn.Module):
    """
    Recurrent Transformer with parameter-tied ThinkBlock.
    
    Architecture:
        Tokens -> Embedding -> InputBlock(s) -> [Shared ThinkBlock] x R -> FinalNorm -> LM Head
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Optional non-recurrent input layers (e.g., 0 or 1 layer)
        if config.input_layers > 0:
            self.input_blocks = nn.ModuleList([
                TransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    norm_type=config.norm_type
                )
                for _ in range(config.input_layers)
            ])
        else:
            self.input_blocks = None
            
        # Shared ThinkBlock (Strictly 1 set of parameters reused R times)
        self.think_block = ThinkBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.think_layers,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            norm_type=config.norm_type
        )
        
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
        recurrent_steps: Optional[int] = None,
        return_trajectory: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Determine number of recurrent iterations (uses config default if not provided)
        steps = recurrent_steps if recurrent_steps is not None else self.config.recurrent_steps
        
        # 1. Embed tokens: [B, T, D]
        h = self.drop(self.tok_embeddings(input_ids))
        
        # 2. Input Block(s) if configured
        if self.input_blocks is not None:
            for block in self.input_blocks:
                h = block(h, attention_mask=attention_mask)
                
        trajectory_latents = [h] if return_trajectory else None
        trajectory_logits = [self.lm_head(self.final_norm(h))] if return_trajectory else None

        # 3. Recurrent ThinkBlock execution (Parameter Sharing)
        # The exact same self.think_block is invoked 'steps' times
        for step_idx in range(steps):
            h = self.think_block(h, attention_mask=attention_mask)
            if return_trajectory:
                trajectory_latents.append(h)
                trajectory_logits.append(self.lm_head(self.final_norm(h)))
            
        # 4. Final Normalization: [B, T, D]
        h_norm = self.final_norm(h)
        
        # 5. LM Head Logits: [B, T, vocab_size]
        logits = self.lm_head(h_norm)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
            
        out_dict = {
            "logits": logits,
            "loss": loss,
            "last_hidden_state": h,
            "recurrent_steps": steps
        }
        if return_trajectory:
            out_dict["trajectory_latents"] = trajectory_latents
            out_dict["trajectory_logits"] = trajectory_logits
            
        return out_dict

    def count_params(self) -> int:
        return count_parameters(self)
