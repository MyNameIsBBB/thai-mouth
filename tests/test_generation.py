"""
Unit tests for autoregressive text generation.
"""

from pathlib import Path
import pytest
import torch
from thaimouth.config import ModelConfig
from thaimouth.generation.generate import generate_text
from thaimouth.models.latent import RecurrentLatentLM
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.tokenizer.train_tokenizer import train_thai_tokenizer


@pytest.fixture(scope="module")
def mock_tokenizer(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("tok_test")
    dummy_file = tmp_dir / "corpus.txt"
    dummy_file.write_text("สวัสดีครับ วันนี้อากาศดีมาก ถ้า A มากกว่า B และ B มากกว่า C", encoding="utf-8")
    
    out_file = tmp_dir / "tok.json"
    tok = train_thai_tokenizer([dummy_file], out_file, vocab_size=256, min_frequency=1)
    return tok


def test_generation_output(mock_tokenizer):
    cfg = ModelConfig(
        model_type="latent",
        vocab_size=mock_tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        d_ff=128,
        input_layers=1,
        think_layers=1,
        recurrent_steps=2,
        max_seq_len=64
    )
    model = RecurrentLatentLM(cfg)
    
    out = generate_text(
        model=model,
        tokenizer=mock_tokenizer,
        prompt="สวัสดี",
        max_new_tokens=10,
        temperature=0.8,
        recurrent_steps=2
    )
    
    assert isinstance(out, str)
    assert len(out) > len("สวัสดี"), "Generation did not produce continuation text"
