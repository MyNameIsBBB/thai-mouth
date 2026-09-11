#!/usr/bin/env bash
set -e

CHECKPOINT=${1:-"checkpoints/smoke/model_final.pt"}
TOKENIZER=${2:-"checkpoints/tokenizer_smoke.json"}
TEST_DATA=${3:-"data/synthetic/test.jsonl"}

echo "=== Evaluating ThaiMouth Model (${CHECKPOINT}) ==="
PYTHONPATH=src python3 -c "
import sys
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.evaluation.reasoning import benchmark_all_depths
from thaimouth.evaluation.language import evaluate_perplexity

model, config, _ = load_checkpoint('${CHECKPOINT}')
tokenizer = ThaiMouthTokenizer('${TOKENIZER}')

print('\n--- Language Modeling Perplexity ---')
for s in [1, 2, 4, 8]:
    res = evaluate_perplexity(model, '${TEST_DATA}', tokenizer, recurrent_steps=s)
    print(f'Steps {s:2d} | Perplexity: {res[\"perplexity\"]:.2f} | Loss: {res[\"loss\"]:.4f}')

print('\n--- Reasoning Accuracy Benchmark ---')
benchmark_all_depths(model, '${TEST_DATA}', tokenizer, step_list=[1, 2, 4, 8], max_samples=100)
"
