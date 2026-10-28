#!/bin/bash
# NS-2026-05: Run training pipeline
set -e

cd /data2/css2025/NS-2026-05

echo "=== Installing dependencies ==="
pip install -q torch torchaudio transformers librosa scikit-learn numpy pandas soundfile modelscope

echo ""
echo "=== Starting AST training ==="
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES=1 python3 train_ast.py

echo ""
echo "=== Done! ==="
echo "Submission file: /data2/css2025/NS-2026-05/NS-2026-05-answer.zip"
