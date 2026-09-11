"""
PyTorch Dataset and DataCollator for Causal Language Modeling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from torch.utils.data import Dataset
from thaimouth.data.preprocess import load_formatted_texts
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer


class CausalLMDataset(Dataset):
    """
    Dataset for next-token prediction language modeling.
    Tokenizes raw input strings and prepares input_ids and labels.
    """
    def __init__(
        self,
        file_path: Union[str, Path],
        tokenizer: ThaiMouthTokenizer,
        max_seq_len: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = load_formatted_texts(file_path)
        self.samples = []
        self._tokenize_all()

    def _tokenize_all(self):
        for text in self.texts:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 0:
                continue
            # Truncate if exceeds max_seq_len
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class DataCollatorForCausalLM:
    """
    Pads dynamic sequences in a batch to the longest sequence,
    producing input_ids, attention_mask, and labels (with pad tokens masked to -100).
    """
    def __init__(self, pad_token_id: int, max_seq_len: Optional[int] = None):
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_lens = [len(s) for s in batch]
        max_len = max(batch_lens)
        if self.max_seq_len is not None:
            max_len = min(max_len, self.max_seq_len)

        b_size = len(batch)
        input_ids = torch.full((b_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((b_size, max_len), dtype=torch.long)
        labels = torch.full((b_size, max_len), -100, dtype=torch.long)

        for i, s in enumerate(batch):
            seq_len = min(len(s), max_len)
            input_ids[i, :seq_len] = s[:seq_len]
            attention_mask[i, :seq_len] = 1
            labels[i, :seq_len] = s[:seq_len]
            # Ignore padding in loss computation
            labels[i, labels[i] == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
