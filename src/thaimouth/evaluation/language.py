"""
Language modeling evaluation: Cross-Entropy loss and Perplexity.
"""

from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader
from thaimouth.data.dataset import CausalLMDataset, DataCollatorForCausalLM
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.losses import compute_perplexity


@torch.no_grad()
def evaluate_perplexity(
    model: torch.nn.Module,
    test_path: str,
    tokenizer: ThaiMouthTokenizer,
    recurrent_steps: Optional[int] = None,
    batch_size: int = 16,
    max_seq_len: int = 256,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Evaluates loss and perplexity on test set at specified recurrent depth.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    dataset = CausalLMDataset(test_path, tokenizer=tokenizer, max_seq_len=max_seq_len)
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_id, max_seq_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    
    total_loss = 0.0
    total_tokens = 0
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            recurrent_steps=recurrent_steps
        )
        
        num_tokens = (labels != -100).sum().item()
        total_loss += outputs["loss"].item() * num_tokens
        total_tokens += max(1, num_tokens)
        
    avg_loss = total_loss / max(1, total_tokens)
    ppl = compute_perplexity(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": ppl,
        "recurrent_steps": recurrent_steps if recurrent_steps is not None else getattr(model.config, "recurrent_steps", 1)
    }
