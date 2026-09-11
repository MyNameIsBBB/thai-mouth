"""
Checkpoint management for ThaiMouth models.
Supports PyTorch checkpoints and optional safetensors.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from thaimouth.config import Config, ModelConfig
from thaimouth.models import build_model


def save_checkpoint(
    model: nn.Module,
    config: Config,
    output_dir: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    val_loss: float = 0.0,
    filename: str = "model.pt"
) -> Path:
    """
    Saves model weights, optimizer state, and configuration to disk.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    ckpt_file = out_path / filename
    
    # Save config yaml alongside checkpoint
    config.save_yaml(out_path / "config.yaml")
    
    state_dict = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
    }
    
    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state_dict["scheduler_state_dict"] = scheduler.state_dict()
        
    torch.save(state_dict, ckpt_file)
    print(f"[+] Checkpoint saved to {ckpt_file} (Step {step}, Val Loss {val_loss:.4f})")
    return ckpt_file


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, Config, Dict[str, Any]]:
    """
    Loads model and configuration from a saved checkpoint file.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")
        
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    raw_cfg = checkpoint.get("config", {})
    model_cfg = ModelConfig(**raw_cfg.get("model", {}))
    config = Config(model=model_cfg)
    
    model = build_model(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, config, checkpoint
