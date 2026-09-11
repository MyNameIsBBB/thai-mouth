from contextlib import nullcontext
import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Selects torch.device based on availability and preference ('auto', 'cuda', 'mps', 'cpu').
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    return torch.device("cpu")


def get_autocast_context(device: torch.device, enabled: bool = True):
    """
    Returns an appropriate autocast context manager based on device.
    """
    if not enabled or device.type == "cpu":
        return nullcontext()
    elif device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    elif device.type == "mps":
        try:
            return torch.amp.autocast(device_type="mps", dtype=torch.float16)
        except Exception:
            return nullcontext()
    return nullcontext()
