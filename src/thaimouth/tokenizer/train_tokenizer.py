"""
Trainer for Thai Byte-Level BPE Tokenizer.
"""

import argparse
import json
from pathlib import Path
from typing import Iterator, List
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from thaimouth.tokenizer.tokenizer import SPECIAL_TOKENS, ThaiMouthTokenizer


def batch_iterator(files: List[Path], batch_size: int = 1000) -> Iterator[List[str]]:
    """Yields batches of raw strings from text and jsonl files."""
    batch = []
    for file_path in files:
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                if file_path.suffix == ".jsonl":
                    try:
                        data = json.loads(line_str)
                        if data.get("type") == "conversation":
                            text = " ".join([m.get("content", "") for m in data.get("messages", [])])
                        elif data.get("type") == "reasoning":
                            q = data.get("question", "")
                            r = " ".join(data.get("reasoning", []))
                            a = data.get("answer", "")
                            text = f"{q} {r} {a}"
                        elif "text" in data:
                            text = data["text"]
                        else:
                            text = line_str
                    except Exception:
                        text = line_str
                else:
                    text = line_str

                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def train_thai_tokenizer(
    input_files: List[Path],
    output_path: Path,
    vocab_size: int = 4096,
    min_frequency: int = 2,
) -> ThaiMouthTokenizer:
    """
    Trains a Byte-level BPE tokenizer optimized for Thai language script and conversation.
    """
    print(f"[*] Training Byte-level BPE tokenizer with vocab_size={vocab_size}...")
    
    # 1. Initialize BPE model
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    
    # 2. Normalization: NFKC unicode normalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])
    
    # 3. Pre-tokenization: ByteLevel allows zero-OOV fallback for all Unicode characters
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 4. Decoder: ByteLevel decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # 5. Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 6. Train from iterator
    tokenizer.train_from_iterator(batch_iterator(input_files), trainer=trainer)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"[+] Tokenizer saved successfully to {output_path} (vocab size: {tokenizer.get_vocab_size()})")
    
    return ThaiMouthTokenizer(output_path)


def main():
    parser = argparse.ArgumentParser(description="Train ThaiMouth Byte-Level BPE Tokenizer.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input text or jsonl files.")
    parser.add_argument("--output", type=str, default="checkpoints/tokenizer.json", help="Path to save tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=4096, help="Target vocabulary size")
    parser.add_argument("--min_freq", type=int, default=1, help="Minimum token frequency")

    args = parser.parse_args()
    input_paths = [Path(p) for p in args.inputs]
    out_path = Path(args.output)

    train_thai_tokenizer(
        input_files=input_paths,
        output_path=out_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )


if __name__ == "__main__":
    main()
