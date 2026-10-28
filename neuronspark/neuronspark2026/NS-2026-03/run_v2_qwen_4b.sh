#!/bin/bash
# Qwen3-4B LoRA 5-fold (3090，比 8B 快 2-3 倍且不会过热)
set -e
cd "$(dirname "$0")"

QWEN_PATH="/data2/css2025/models/Qwen/Qwen3-4B"

if [ ! -d "${QWEN_PATH}" ]; then
    echo "ERROR: ${QWEN_PATH} 不存在"
    ls /data2/css2025/models/Qwen/
    exit 1
fi

echo "=== Qwen3-4B LoRA 5-fold CV (3090) ==="
# 用 GPU1 (3090, 24G)
CUDA_VISIBLE_DEVICES=1 python train_qwen.py \
    --train_file train.jsonl \
    --output_dir qwen3_4b_out \
    --model_name $QWEN_PATH \
    --n_folds 5 \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 1e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --gpu 0 \
    2>&1 | tee qwen3_4b_train.log

echo ""
echo "=== 推理: Qwen3-4B + BERT v2 ensemble ==="
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_PATH \
    --qwen_lora_dir qwen3_4b_out \
    --bert_dir bert_v2 \
    --output results_qwen4b_ensemble.json \
    --gpu 0 \
    --use_qwen \
    --use_bert \
    --qwen_weight 0.6 \
    --bert_weight 0.4 \
    2>&1 | tee infer_qwen4b_ensemble.log

# 也单独 Qwen-only
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_PATH \
    --qwen_lora_dir qwen3_4b_out \
    --bert_dir bert_v2 \
    --output results_qwen4b_only.json \
    --gpu 0 \
    --use_qwen \
    --qwen_weight 1.0 \
    2>&1 | tee infer_qwen4b_only.log

# 验证 + 打包两个版本
for tag in qwen4b_ensemble qwen4b_only; do
    cp results_${tag}.json results.json
    rm -f NS-2026-03-answer-${tag}.zip
    zip NS-2026-03-answer-${tag}.zip results.json
    python -c "
import json
from collections import Counter
with open('results_${tag}.json') as f:
    data = json.load(f)
labels = Counter(d['label'] for d in data)
risks = [d['risk_score'] for d in data]
print(f'[$tag] 标签: {dict(labels)}')
print(f'[$tag] risk: min={min(risks):.3f} max={max(risks):.3f}')
"
done

ls -la NS-2026-03-answer-*.zip
echo "[Qwen3-4B] 完成"
