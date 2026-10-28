#!/bin/bash
# v2: Qwen3-8B LoRA SFT + ensemble with BERT v2
set -e
cd "$(dirname "$0")"

QWEN_PATH="/data2/css2025/models/Qwen/Qwen3-8B"

if [ ! -d "${QWEN_PATH}" ]; then
    echo "ERROR: ${QWEN_PATH} 不存在"
    ls /data2/css2025/models/Qwen/
    exit 1
fi

echo "=== v2: Qwen3-8B LoRA 5-fold CV ==="
CUDA_VISIBLE_DEVICES=0 python train_qwen.py \
    --train_file train.jsonl \
    --output_dir qwen3_8b_out \
    --model_name $QWEN_PATH \
    --n_folds 5 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --gpu 0 \
    2>&1 | tee qwen3_8b_train.log

echo ""
echo "=== Ensemble: Qwen3-8B + BERT v2, 无规则 ==="
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_PATH \
    --qwen_lora_dir qwen3_8b_out \
    --bert_dir bert_v2 \
    --output results_ensemble_v2.json \
    --gpu 0 \
    --use_qwen \
    --use_bert \
    --qwen_weight 0.6 \
    --bert_weight 0.4 \
    2>&1 | tee infer_ensemble_v2.log

# 也跑一份只用 Qwen 的对比
echo ""
echo "=== 只用 Qwen3-8B (对比) ==="
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_PATH \
    --qwen_lora_dir qwen3_8b_out \
    --bert_dir bert_v2 \
    --output results_qwen_only.json \
    --gpu 0 \
    --use_qwen \
    --qwen_weight 1.0 \
    2>&1 | tee infer_qwen_only.log

# 验证
for f in results_ensemble_v2.json results_qwen_only.json; do
    echo "=== $f ==="
    python -c "
import json
from collections import Counter
with open('$f') as f:
    data = json.load(f)
labels = Counter(d['label'] for d in data)
print(f'标签分布: {dict(labels)}')
risks = [d['risk_score'] for d in data]
print(f'risk_score: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,2) for r in risks))}')
"
done

cp results_ensemble_v2.json results.json
rm -f NS-2026-03-answer-ensemble-v2.zip
zip NS-2026-03-answer-ensemble-v2.zip results.json
ls -la NS-2026-03-answer-ensemble-v2.zip

cp results_qwen_only.json results.json
rm -f NS-2026-03-answer-qwen-only.zip
zip NS-2026-03-answer-qwen-only.zip results.json

ls -la NS-2026-03-answer-*.zip
echo "[v2 ensemble] 完成"
