"""
Reproducibility utilities for ThaiMouth.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
