"""
ThaiMouth Model Architectures:
- TinyTransformerBaseline
- RecurrentTransformer
- RecurrentLatentLM
"""

from typing import Union
import torch.nn as nn
from thaimouth.config import ModelConfig
from thaimouth.models.baseline import TinyTransformerBaseline
from thaimouth.models.latent import RecurrentLatentLM
from thaimouth.models.recurrent import RecurrentTransformer


def build_model(config: ModelConfig) -> Union[TinyTransformerBaseline, RecurrentTransformer, RecurrentLatentLM]:
    """
    Instantiates model based on config.model_type ('baseline' | 'recurrent' | 'latent').
    """
    model_type = config.model_type.lower()
    if model_type == "baseline":
        return TinyTransformerBaseline(config)
    elif model_type == "recurrent":
        return RecurrentTransformer(config)
    elif model_type == "latent":
        return RecurrentLatentLM(config)
    else:
        raise ValueError(f"Unknown model_type: '{config.model_type}'. Expected 'baseline', 'recurrent', or 'latent'.")
