#!/bin/bash
# v4: 数据增强 + 重训 BERT + 推理
set -e
cd "$(dirname "$0")"

BERT_LOCAL="models/chinese-roberta-wwm-ext"

# 1. 生成增强数据 (用 Qwen3-8B 在 A40 上)
if [ ! -f "train_aug.jsonl" ]; then
    echo "=== 1. 生成增强训练数据 (Qwen3-8B, ~10分钟) ==="
    CUDA_VISIBLE_DEVICES=0 python generate_aug.py \
        --model_path /data2/css2025/models/Qwen/Qwen3-8B \
        --output train_aug.jsonl \
        --num_rounds 4 \
        --gpu 0 \
        2>&1 | tee gen_aug.log
fi

wc -l train_aug.jsonl

# 2. 合并 train + train_aug
echo "=== 2. 合并训练集 ==="
cat train.jsonl train_aug.jsonl > train_full.jsonl
wc -l train_full.jsonl

# 3. 用合并数据重训 BERT
echo "=== 3. 训练 BERT v4 (5-fold CV) ==="
CUDA_VISIBLE_DEVICES=1 python train_bert.py \
    --train_file train_full.jsonl \
    --test_file test.jsonl \
    --output_dir bert_v4 \
    --model_name $BERT_LOCAL \
    --n_folds 5 \
    --epochs 4 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 384 \
    --gpu 0 \
    2>&1 | tee bert_v4_train.log

# 4. 推理（用 BERT v4 单独 + 与 Qwen3-8B ensemble）
echo "=== 4. 推理 BERT v4 only ==="
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --test_file test.jsonl \
    --bert_dir bert_v4 \
    --output results_bert_v4.json \
    --gpu 0 \
    --use_bert \
    --bert_weight 1.0 \
    --debug_dir infer_debug_v4 \
    2>&1 | tee infer_bert_v4.log

# Ensemble with Qwen3-8B
echo "=== 5. Ensemble v4: BERT v4 + Qwen3-8B ==="
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_file test.jsonl \
    --qwen_base /data2/css2025/models/Qwen/Qwen3-8B \
    --qwen_lora_dir qwen3_8b_out \
    --bert_dir bert_v4 \
    --output results_ensemble_v4.json \
    --gpu 0 \
    --use_qwen \
    --use_bert \
    --qwen_weight 0.4 \
    --bert_weight 0.6 \
    --debug_dir infer_debug_v4 \
    2>&1 | tee infer_ensemble_v4.log

# 验证 + 打包
for tag in bert_v4 ensemble_v4; do
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
print(f'[$tag] risk: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,3) for r in risks))}')
"
done

ls -la NS-2026-03-answer-*v4.zip
echo "[v4] 完成"
