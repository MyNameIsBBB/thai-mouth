"""
Recurrent Latent Language Model (RecurrentLatentLM).
Refines a sequence of latent representations Z_0 -> Z_1 -> ... -> Z_R
via repeated shared-weight function F_theta(Z_t, X) before causal LM decoding.
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
from thaimouth.config import ModelConfig
from thaimouth.models.common import RMSNorm, count_parameters
from thaimouth.models.transformer_block import LatentThinkBlock, TransformerBlock


class RecurrentLatentLM(nn.Module):
    """
    Recurrent Latent Language Model.
    
    Architecture:
        Tokens -> Embedding (X)
               -> Input Encoder -> Initial Latent Z_0
               -> Loop R times:
                    Z_{t+1} = Z_t + alpha * F_theta(Norm(Z_t), context=X)
               -> FinalNorm -> LM Head
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        
        # Input Encoder: transforms raw embeddings X into initial latent state Z_0
        self.input_encoder = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                max_seq_len=config.max_seq_len,
                dropout=config.dropout,
                norm_type=config.norm_type
            )
            for _ in range(config.input_layers)
        ]) if config.input_layers > 0 else None
        
        # Shared Latent Refinement Block F_theta
        # Strictly shared weights across all recurrent steps
        self.latent_think_block = LatentThinkBlock(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            use_cross_attn=False,  # Can be enabled for cross-attention context injection
            norm_type=config.norm_type
        )
        
        # Learnable or fixed step scaling factor alpha
        self.alpha = nn.Parameter(torch.tensor(float(config.latent_alpha)))
        
        # Recurrent normalization / gating modules for Milestone 0.3
        self.recurrent_norm_type = getattr(config, "recurrent_norm", "none")
        if self.recurrent_norm_type in ["prenorm", "postnorm", "gated_prenorm"]:
            self.recurrent_norm = RMSNorm(config.d_model) if config.norm_type == "rmsnorm" else nn.LayerNorm(config.d_model)
        else:
            self.recurrent_norm = None

        if self.recurrent_norm_type == "gated_prenorm":
            self.gate_proj = nn.Linear(config.d_model, config.d_model, bias=False)
            # Initialize gate bias/weights towards modest residual update
            nn.init.zeros_(self.gate_proj.weight)
        else:
            self.gate_proj = None

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
        steps = recurrent_steps if recurrent_steps is not None else self.config.recurrent_steps
        
        # 1. Input Token Embedding: [B, T, D]
        x = self.drop(self.tok_embeddings(input_ids))
        
        # 2. Input Encoder -> Z_0
        z = x
        if self.input_encoder is not None:
            for block in self.input_encoder:
                z = block(z, attention_mask=attention_mask)
                
        trajectory_latents = [z] if return_trajectory else None
        trajectory_logits = [self.lm_head(self.final_norm(z))] if return_trajectory else None
        
        # 3. Recurrent Latent Refinement across steps
        for step_idx in range(steps):
            if self.recurrent_norm_type == "prenorm":
                # Variant B: Z_{t+1} = Z_t + alpha * F(RMSNorm(Z_t), X)
                z_in = self.recurrent_norm(z)
                delta_z = self.latent_think_block(z_in, context=x, attention_mask=attention_mask)
                z = z + self.alpha * delta_z
            elif self.recurrent_norm_type == "postnorm":
                # Variant C: Z_{t+1} = RMSNorm(Z_t + alpha * F(Z_t, X))
                delta_z = self.latent_think_block(z, context=x, attention_mask=attention_mask)
                z = self.recurrent_norm(z + self.alpha * delta_z)
            elif self.recurrent_norm_type == "gated_prenorm":
                # Variant D: U = F(RMSNorm(Z_t), X), Z_{t+1} = Z_t + alpha * sigmoid(W_g * RMSNorm(Z_t)) * U
                z_norm = self.recurrent_norm(z)
                delta_z = self.latent_think_block(z_norm, context=x, attention_mask=attention_mask)
                gate = torch.sigmoid(self.gate_proj(z_norm))
                z = z + self.alpha * gate * delta_z
            else:
                # Variant A (Baseline): Z_{t+1} = Z_t + alpha * F(Z_t, X)
                delta_z = self.latent_think_block(z, context=x, attention_mask=attention_mask)
                z = z + self.alpha * delta_z

            if return_trajectory:
                trajectory_latents.append(z)
                trajectory_logits.append(self.lm_head(self.final_norm(z)))
            
        # 4. Final Normalization
        z_norm = self.final_norm(z)
        
        # 5. LM Head Logits: [B, T, vocab_size]
        logits = self.lm_head(z_norm)
        
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
            "latent_state": z,
            "recurrent_steps": steps
        }
        if return_trajectory:
            out_dict["trajectory_latents"] = trajectory_latents
            out_dict["trajectory_logits"] = trajectory_logits
            
        return out_dict

    def count_params(self) -> int:
        return count_parameters(self)
