"""
Latent Dynamics Analysis for ThaiMouth Recurrent Models.
Tracks latent convergence diagnostics across recurrent thinking steps:
- ||Z_t|| (Latent state magnitude)
- ||Z_{t+1} - Z_t|| (Latent step delta / update magnitude)
- cosine(Z_t, Z_{t+1}) (Latent step directional alignment)
- Entropy H(p_t) of predicted output distributions
- KL(p_t || p_{t+1}) divergence between successive step prediction distributions
"""

from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


def compute_latent_step_dynamics(
    trajectory_latents: List[torch.Tensor],
    trajectory_logits: Optional[List[torch.Tensor]] = None
) -> Dict[str, List[float]]:
    """
    Computes convergence statistics from a sequence of latent states [Z_0, Z_1, ..., Z_R]
    and corresponding LM logits [logits_0, logits_1, ..., logits_R].
    
    Args:
        trajectory_latents: List of tensors [B, T, D] for t = 0 ... R
        trajectory_logits: Optional list of tensors [B, T, V] for t = 0 ... R
        
    Returns:
        Dictionary with per-step metrics:
        - norms: [ ||Z_0||, ||Z_1||, ... ]
        - deltas: [ ||Z_1 - Z_0||, ||Z_2 - Z_1||, ... ]
        - cosine_sims: [ cos(Z_0, Z_1), cos(Z_1, Z_2), ... ]
        - entropies: [ H(p_0), H(p_1), ... ]
        - kl_divergences: [ KL(p_0 || p_1), KL(p_1 || p_2), ... ]
    """
    num_steps = len(trajectory_latents) - 1
    
    # 1. Latent Norms ||Z_t||
    norms = []
    for z in trajectory_latents:
        # z: [B, T, D] -> mean L2 norm across batch and sequence tokens
        l2_norm = torch.linalg.norm(z, dim=-1).mean().item()
        norms.append(round(l2_norm, 4))
        
    # 2. Latent Deltas ||Z_{t+1} - Z_t||, Relative Deltas, and Cosine Similarity
    deltas = []
    rel_deltas = []
    cosine_sims = []
    for t in range(num_steps):
        z_t = trajectory_latents[t]
        z_next = trajectory_latents[t + 1]
        
        diff = torch.linalg.norm(z_next - z_t, dim=-1).mean().item()
        z_t_norm = torch.linalg.norm(z_t, dim=-1).mean().item()
        rel_diff = diff / max(1e-6, z_t_norm)
        
        deltas.append(round(diff, 4))
        rel_deltas.append(round(rel_diff, 4))
        
        # Flatten B*T to compute average cosine similarity
        cos = F.cosine_similarity(z_t, z_next, dim=-1).mean().item()
        cosine_sims.append(round(cos, 4))
        
    # 3. Output Distribution Diagnostics (Entropy & KL Divergence)
    entropies = []
    kl_divs = []
    
    if trajectory_logits is not None:
        for t in range(len(trajectory_logits)):
            # Take last token position logits for next-token prediction analysis: [B, V]
            last_logits = trajectory_logits[t][:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            log_probs = F.log_softmax(last_logits, dim=-1)
            
            # Entropy: - sum(p * log(p))
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            entropies.append(round(entropy, 4))
            
        for t in range(len(trajectory_logits) - 1):
            log_p_t = F.log_softmax(trajectory_logits[t][:, -1, :], dim=-1)
            probs_next = F.softmax(trajectory_logits[t + 1][:, -1, :], dim=-1)
            log_probs_next = F.log_softmax(trajectory_logits[t + 1][:, -1, :], dim=-1)
            
            # KL(p_t || p_{t+1}) = sum(p_t * (log_p_t - log_p_{t+1}))
            probs_t = F.softmax(trajectory_logits[t][:, -1, :], dim=-1)
            kl = (probs_t * (log_p_t - log_probs_next)).sum(dim=-1).mean().item()
            kl_divs.append(round(max(0.0, kl), 6))

    return {
        "num_steps": num_steps,
        "norms": norms,
        "deltas": deltas,
        "relative_deltas": rel_deltas,
        "cosine_sims": cosine_sims,
        "entropies": entropies,
        "kl_divergences": kl_divs
    }
