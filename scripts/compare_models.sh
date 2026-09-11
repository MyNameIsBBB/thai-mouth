#!/usr/bin/env bash
set -e

echo "=== Comparing ThaiMouth Model Variants Across Recurrent Depths ==="
PYTHONPATH=src python3 -m thaimouth.evaluation.compare \
  --checkpoints "$@" \
  --tokenizer checkpoints/tokenizer.json \
  --test_data data/synthetic/test.jsonl \
  --steps 1 2 4 8 16 \
  --output_dir results
