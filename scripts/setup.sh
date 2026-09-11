#!/usr/bin/env bash
set -e

echo "=== ThaiMouth Environment Setup ==="
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

echo "[✓] Environment setup completed."
