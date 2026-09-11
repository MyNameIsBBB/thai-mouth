"""
Configuration management for ThaiMouth models and training.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ModelConfig:
    """
    Model Architecture Configuration.
    
    Attributes:
        model_type: 'baseline' | 'recurrent' | 'latent'
        vocab_size: Size of the tokenizer vocabulary
        d_model: Hidden dimension size D
        n_heads: Number of attention heads H
        d_ff: Feed-forward intermediate size (default 4 * d_model)
        n_layers: Number of transformer blocks (for standard baseline)
        input_layers: Number of non-recurrent input blocks (for recurrent/latent)
        think_layers: Number of transformer blocks inside the shared ThinkBlock
        recurrent_steps: Number of times ThinkBlock is repeatedly executed
        max_seq_len: Maximum sequence length (context window)
        dropout: Dropout probability
        latent_alpha: Residual scaling factor for latent update Z_{t+1} = Z_t + alpha * F(Z_t, X)
        tie_word_embeddings: Whether to tie LM head weights with input embeddings
        norm_type: 'rmsnorm' or 'layernorm'
    """
    model_type: str = "baseline"
    vocab_size: int = 4096
    d_model: int = 128
    n_heads: int = 4
    d_ff: Optional[int] = None
    n_layers: int = 4
    input_layers: int = 1
    think_layers: int = 1
    recurrent_steps: int = 1
    max_seq_len: int = 256
    dropout: float = 0.0
    latent_alpha: float = 1.0
    tie_word_embeddings: bool = True
    norm_type: str = "rmsnorm"
    recurrent_norm: str = "none"  # 'none', 'prenorm', 'postnorm', 'gated_prenorm'

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")


@dataclass
class TrainConfig:
    """Training hyperparameter configuration."""
    batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 5e-4
    min_lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: Optional[int] = None
    max_epochs: int = 5
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 500
    grad_clip: float = 1.0
    mixed_precision: bool = True
    device: str = "auto"
    seed: int = 42
    output_dir: str = "checkpoints"
    tensorboard: bool = False
    curriculum: Optional[List[Dict[str, Any]]] = None


@dataclass
class DataConfig:
    """Dataset and tokenizer configuration."""
    train_path: str = "data/processed/train.jsonl"
    val_path: str = "data/processed/val.jsonl"
    test_path: Optional[str] = "data/processed/test.jsonl"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    max_seq_len: int = 256


@dataclass
class Config:
    """Unified configuration holding Model, Training, and Data configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        model_cfg = ModelConfig(**raw.get("model", {}))
        train_cfg = TrainConfig(**raw.get("training", {}))
        data_cfg = DataConfig(**raw.get("data", {}))

        return cls(model=model_cfg, training=train_cfg, data=data_cfg)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
