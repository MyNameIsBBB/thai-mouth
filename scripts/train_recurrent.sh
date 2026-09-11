#!/usr/bin/env bash
set -e

CONFIG=${1:-"configs/recurrent_1m.yaml"}
echo "=== Training Recurrent Transformer Model (${CONFIG}) ==="
PYTHONPATH=src python3 -m thaimouth.training.trainer --config "${CONFIG}"
