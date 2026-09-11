"""
Tokenizer wrapper for ThaiMouth language models.
Handles tokenization, detokenization, and special conversational tokens.
"""

from pathlib import Path
from typing import List, Optional, Union
from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<s>",
    "</s>",
    "<user>",
    "<assistant>",
    "<think>",
    "</think>",
    "<question>",
    "<reasoning>",
    "<answer>",
]


class ThaiMouthTokenizer:
    """
    Tokenizer wrapper around HuggingFace Byte-level BPE Tokenizer for Thai text.
    """
    def __init__(self, tokenizer_path: Optional[Union[str, Path]] = None):
        self.tokenizer: Optional[HFTokenizer] = None
        if tokenizer_path is not None:
            self.load(tokenizer_path)

    def load(self, path: Union[str, Path]) -> "ThaiMouthTokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        self.tokenizer = HFTokenizer.from_file(str(path))
        return self

    def save(self, path: Union[str, Path]) -> None:
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @property
    def vocab_size(self) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        return self.tokenizer.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self.token_to_id("<pad>")

    @property
    def unk_id(self) -> int:
        return self.token_to_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self.token_to_id("<s>")

    @property
    def eos_id(self) -> int:
        return self.token_to_id("</s>")

    def token_to_id(self, token: str) -> int:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Special token '{token}' not found in vocabulary.")
        return token_id

    def id_to_token(self, token_id: int) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        return self.tokenizer.id_to_token(token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], add_special_tokens: bool = False) -> List[List[int]]:
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
