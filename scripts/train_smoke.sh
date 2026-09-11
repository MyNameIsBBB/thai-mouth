#!/usr/bin/env bash
set -e

echo "============================================================"
echo "          ThaiMouth End-to-End Smoke Pipeline               "
echo "============================================================"

# 1. Generate synthetic dataset if not exists
if [ ! -f "data/synthetic/train.jsonl" ]; then
    echo ""
    echo "[*] Step 1: Generating synthetic Thai reasoning & conversation data..."
    PYTHONPATH=src python3 -m thaimouth.data.synthetic_reasoning \
      --out_dir data/synthetic \
      --num_train 600 \
      --num_val 100 \
      --num_test 150 \
      --seed 42
else
    echo ""
    echo "[*] Step 1: Synthetic data already exists in data/synthetic/"
fi

# 2. Train Smoke Tokenizer if not exists
if [ ! -f "checkpoints/tokenizer_smoke.json" ]; then
    echo ""
    echo "[*] Step 2: Training smoke tokenizer..."
    PYTHONPATH=src python3 -m thaimouth.tokenizer.train_tokenizer \
      --inputs data/synthetic/train.jsonl data/synthetic/val.jsonl \
      --output checkpoints/tokenizer_smoke.json \
      --vocab_size 1024 \
      --min_freq 1
else
    echo ""
    echo "[*] Step 2: Smoke tokenizer already exists at checkpoints/tokenizer_smoke.json"
fi

# 3. Train Smoke Model
echo ""
echo "[*] Step 3: Training Tiny Recurrent-Latent Model (Smoke)..."
PYTHONPATH=src python3 -m thaimouth.training.trainer --config configs/smoke.yaml

# 4. Generate Sample Text
echo ""
echo "[*] Step 4: Generating sample Thai text..."
PYTHONPATH=src python3 -m thaimouth.generation.generate \
  --checkpoint checkpoints/smoke/model_final.pt \
  --tokenizer checkpoints/tokenizer_smoke.json \
  --prompt "<s><user> วันนี้เหนื่อยมากเลย </user><assistant>" \
  --max_tokens 32 \
  --recurrent_steps 2

# 5. Run Multi-Depth Reasoning Benchmark
echo ""
echo "[*] Step 5: Benchmarking reasoning accuracy across recurrent depths (1, 2, 4, 8)..."
PYTHONPATH=src python3 -c "
import torch
from thaimouth.training.checkpoint import load_checkpoint
from thaimouth.tokenizer.tokenizer import ThaiMouthTokenizer
from thaimouth.evaluation.reasoning import benchmark_all_depths

model, config, _ = load_checkpoint('checkpoints/smoke/model_final.pt')
tokenizer = ThaiMouthTokenizer('checkpoints/tokenizer_smoke.json')
results = benchmark_all_depths(
    model=model,
    test_jsonl_path='data/synthetic/test.jsonl',
    tokenizer=tokenizer,
    step_list=[1, 2, 4, 8],
    max_samples=50
)
"

echo "\n============================================================"
echo "  [✓] ThaiMouth Smoke Pipeline Completed Successfully!      "
echo "============================================================"
