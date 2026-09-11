"""
Efficiency and inference compute benchmarking.
Measures latency (ms/token), throughput (tokens/sec), and memory.
"""

import time
from typing import Dict, List, Optional
import torch
from thaimouth.evaluation.compute import estimate_forward_flops
from thaimouth.models.common import count_parameters


@torch.no_grad()
def benchmark_efficiency(
    model: torch.nn.Module,
    recurrent_steps: int = 1,
    batch_size: int = 1,
    seq_len: int = 128,
    num_warmup: int = 5,
    num_runs: int = 20,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Measures latency per forward pass and estimated tokens/sec.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    dummy_input = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    for _ in range(num_warmup):
        _ = model(dummy_input, recurrent_steps=recurrent_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
            
    # Benchmark runs
    start_time = time.perf_counter()
    for _ in range(num_runs):
        _ = model(dummy_input, recurrent_steps=recurrent_steps)
        if device.type == "cuda":
            torch.cuda.synchronize()
            
    elapsed = time.perf_counter() - start_time
    avg_latency_ms = (elapsed / num_runs) * 1000.0
    total_tokens = batch_size * seq_len * num_runs
    tokens_per_sec = total_tokens / elapsed
    ms_per_token = avg_latency_ms / (batch_size * seq_len)
    compute = estimate_forward_flops(
        model,
        recurrent_steps=recurrent_steps,
        batch_size=batch_size,
        seq_len=seq_len,
    )
    
    return {
        "recurrent_steps": recurrent_steps,
        "latency_ms": round(avg_latency_ms, 2),
        "ms_per_token": round(ms_per_token, 4),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "num_params": count_parameters(model),
        "estimated_flops": compute.total_flops,
        "estimated_gflops": round(compute.gflops, 6),
        "flops_convention": "multiply_add_is_2; dominant_matmuls_only"
    }
