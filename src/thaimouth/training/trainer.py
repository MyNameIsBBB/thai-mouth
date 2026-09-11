"""
Trainer engine for ThaiMouth language models.
Supports standard decoder-only, recurrent, and recurrent-latent architectures.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from thaimouth.config import Config
from thaimouth.data.dataset import CausalLMDataset, DataCollatorForCausalLM
from thaimouth.models import build_model
from thaimouth.models.common import count_parameters, format_param_count
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.training.checkpoint import save_checkpoint
from thaimouth.training.losses import compute_perplexity
from thaimouth.training.scheduler import get_cosine_schedule_with_warmup
from thaimouth.utils.device import get_autocast_context, get_device
from thaimouth.utils.seed import set_seed


class Trainer:
    """
    Unified training loop for ThaiMouth language models.
    """
    def __init__(self, config: Config):
        self.config = config
        set_seed(config.training.seed)
        
        self.device = get_device(config.training.device)
        print(f"[*] Training Device: {self.device.type.upper()}")
        
        # 1. Load Tokenizer
        self.tokenizer = ThaiMouthTokenizer(config.data.tokenizer_path)
        # Ensure vocab size alignment
        if config.model.vocab_size != self.tokenizer.vocab_size:
            print(f"[!] Updating model vocab_size ({config.model.vocab_size}) to match tokenizer ({self.tokenizer.vocab_size})")
            config.model.vocab_size = self.tokenizer.vocab_size

        # 2. Build Model
        self.model = build_model(config.model)
        self.model.to(self.device)
        
        total_params = count_parameters(self.model)
        print("=" * 60)
        print(f"  MODEL ARCHITECTURE:  {config.model.model_type.upper()}")
        print(f"  TOTAL PARAMETERS:    {total_params:,} ({format_param_count(total_params)})")
        print(f"  HIDDEN DIM (d_model):{config.model.d_model} | HEADS: {config.model.n_heads}")
        print(f"  RECURRENT STEPS:     {config.model.recurrent_steps}")
        print(f"  CONTEXT LENGTH:      {config.model.max_seq_len}")
        print("=" * 60)

        # 3. Load Datasets
        self.train_dataset = CausalLMDataset(
            config.data.train_path,
            tokenizer=self.tokenizer,
            max_seq_len=config.data.max_seq_len
        )
        self.val_dataset = CausalLMDataset(
            config.data.val_path,
            tokenizer=self.tokenizer,
            max_seq_len=config.data.max_seq_len
        )
        
        collator = DataCollatorForCausalLM(
            pad_token_id=self.tokenizer.pad_id,
            max_seq_len=config.data.max_seq_len
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collator,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collator
        )

        # 4. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        total_steps = (len(self.train_loader) // config.training.grad_accum_steps) * config.training.max_epochs
        if config.training.max_steps is not None:
            total_steps = min(total_steps, config.training.max_steps)
            
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.training.warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=config.training.min_lr / config.training.learning_rate
        )
        self.total_steps = total_steps
        self.scaler = torch.amp.GradScaler('cuda') if (config.training.mixed_precision and self.device.type == 'cuda') else None

    def _get_current_recurrent_steps(self, step: int, total_steps: int) -> int:
        """Determines recurrent depth if a curriculum schedule is active."""
        curriculum = self.config.training.curriculum
        if not curriculum:
            return self.config.model.recurrent_steps
            
        progress = step / max(1, total_steps)
        for stage in curriculum:
            if progress <= stage.get("progress", 1.0):
                allowed_steps = stage.get("steps", [self.config.model.recurrent_steps])
                return random.choice(allowed_steps)
        return self.config.model.recurrent_steps

    def evaluate(self, recurrent_steps: Optional[int] = None) -> Dict[str, float]:
        """Runs evaluation over validation dataset."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        steps = recurrent_steps if recurrent_steps is not None else self.config.model.recurrent_steps
        max_eval_batches = self.config.training.eval_batches
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= max_eval_batches:
                    break
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with get_autocast_context(self.device, self.config.training.mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        recurrent_steps=steps
                    )
                    loss = outputs["loss"]
                    
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += max(1, num_tokens)
                
        val_loss = total_loss / max(1, total_tokens)
        val_ppl = compute_perplexity(val_loss)
        return {"val_loss": val_loss, "val_perplexity": val_ppl, "recurrent_steps": steps}

    def train(self) -> Dict[str, Any]:
        """Executes full training loop."""
        print(f"\n[*] Starting training: {self.config.training.max_epochs} epochs (~{self.total_steps} optimizer steps)...")
        self.model.train()
        
        global_step = 0
        total_tokens_trained = 0
        best_val_loss = float("inf")
        start_time = time.time()
        
        for epoch in range(self.config.training.max_epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.max_epochs}")
            accum_loss = 0.0
            
            for step_in_epoch, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Determine current recurrent steps (curriculum or fixed)
                rec_steps = self._get_current_recurrent_steps(global_step, self.total_steps)
                
                with get_autocast_context(self.device, self.config.training.mixed_precision):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        recurrent_steps=rec_steps
                    )
                    loss = outputs["loss"] / self.config.training.grad_accum_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss.item() * self.config.training.grad_accum_steps
                tokens_in_batch = (labels != -100).sum().item()
                total_tokens_trained += tokens_in_batch

                # Optimizer step on gradient accumulation boundary
                if (step_in_epoch + 1) % self.config.training.grad_accum_steps == 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens_trained / max(1e-5, elapsed)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    
                    pbar.set_postfix({
                        "loss": f"{accum_loss:.4f}",
                        "lr": f"{current_lr:.2e}",
                        "steps": rec_steps,
                        "tok/s": f"{tokens_per_sec:.0f}"
                    })
                    accum_loss = 0.0

                    # Periodic Validation
                    if global_step % self.config.training.eval_interval == 0:
                        eval_metrics = self.evaluate()
                        self.model.train()
                        print(f"\n[Step {global_step}] Validation Loss: {eval_metrics['val_loss']:.4f} | PPL: {eval_metrics['val_perplexity']:.2f}")

                    # Periodic Save
                    if global_step % self.config.training.save_interval == 0:
                        save_checkpoint(
                            model=self.model,
                            config=self.config,
                            output_dir=self.config.training.output_dir,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            step=global_step,
                            filename=f"checkpoint_step_{global_step}.pt"
                        )

                if self.config.training.max_steps and global_step >= self.config.training.max_steps:
                    break

            if self.config.training.max_steps and global_step >= self.config.training.max_steps:
                break

        # Final Validation
        final_eval = self.evaluate()
        print(f"\n[OK] Training Finished! Final Validation Loss: {final_eval['val_loss']:.4f} | PPL: {final_eval['val_perplexity']:.2f}")
        
        # Save Final Model Checkpoint
        final_ckpt = save_checkpoint(
            model=self.model,
            config=self.config,
            output_dir=self.config.training.output_dir,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.config.training.max_epochs,
            step=global_step,
            val_loss=final_eval["val_loss"],
            filename="model_final.pt"
        )
        
        return {
            "final_checkpoint": str(final_ckpt),
            "final_loss": final_eval["val_loss"],
            "final_ppl": final_eval["val_perplexity"],
            "total_tokens_trained": total_tokens_trained,
            "total_steps": global_step
        }


def main():
    parser = argparse.ArgumentParser(description="Train ThaiMouth language model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
