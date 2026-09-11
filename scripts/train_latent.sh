#!/usr/bin/env bash
set -e

CONFIG=${1:-"configs/latent_1m.yaml"}
echo "=== Training Recurrent Latent Model (${CONFIG}) ==="
PYTHONPATH=src python3 -m thaimouth.training.trainer --config "${CONFIG}"
