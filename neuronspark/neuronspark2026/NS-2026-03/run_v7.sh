#!/bin/bash
# v7: Qwen3-8B zero-shot 分类 (不微调，纯推理)
set -e
cd "$(dirname "$0")"

echo "=== v7: Qwen3-8B zero-shot 分类 ==="
CUDA_VISIBLE_DEVICES=0 python classify_zeroshot.py \
    --model_path /data2/css2025/models/Qwen/Qwen3-8B \
    --test_file test.jsonl \
    --output results_v7.json \
    --gpu 0 \
    2>&1 | tee v7_classify.log

cp results_v7.json results.json
rm -f NS-2026-03-answer-v7.zip
zip NS-2026-03-answer-v7.zip results.json
ls -la NS-2026-03-answer-v7.zip
echo "[v7] 完成"
