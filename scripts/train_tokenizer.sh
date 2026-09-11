#!/usr/bin/env bash
set -e

VOCAB_SIZE=${1:-4096}
OUTPUT=${2:-"checkpoints/tokenizer.json"}

echo "=== Training Thai Byte-Level BPE Tokenizer (Vocab Size: ${VOCAB_SIZE}) ==="
PYTHONPATH=src python3 -m thaimouth.tokenizer.train_tokenizer \
  --inputs data/synthetic/train.jsonl data/synthetic/val.jsonl \
  --output "${OUTPUT}" \
  --vocab_size "${VOCAB_SIZE}" \
  --min_freq 1

echo "[✓] Tokenizer training completed -> ${OUTPUT}"
