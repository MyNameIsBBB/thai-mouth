#!/usr/bin/env bash
set -e

CONFIG=${1:-"configs/baseline_1m.yaml"}
echo "=== Training Baseline Transformer Model (${CONFIG}) ==="
PYTHONPATH=src python3 -m thaimouth.training.trainer --config "${CONFIG}"
