#!/bin/bash
# v2 改进版:
# 1. BERT 减少 epoch (4) + 提交多个版本对比
# 2. Qwen 切到 Qwen3-8B
# 3. risk_score 用连续概率 (高风险概率之和)
# 4. 关闭规则后处理（规则覆盖太激进）
set -e
cd "$(dirname "$0")"

BERT_LOCAL="models/chinese-roberta-wwm-ext"

echo "=== v2: 重新训练 BERT (减少过拟合) ==="
CUDA_VISIBLE_DEVICES=1 python train_bert.py \
    --train_file train.jsonl \
    --test_file test.jsonl \
    --output_dir bert_v2 \
    --model_name $BERT_LOCAL \
    --n_folds 5 \
    --epochs 4 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 384 \
    --gpu 0 \
    2>&1 | tee bert_v2_train.log

echo ""
echo "=== 推理 (BERT only, 无规则, 连续 risk_score) ==="
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --test_file test.jsonl \
    --bert_dir bert_v2 \
    --output results_bert_v2.json \
    --gpu 0 \
    --use_bert \
    --bert_weight 1.0 \
    2>&1 | tee infer_bert_v2.log

# 验证
python -c "
import json
from collections import Counter
with open('results_bert_v2.json') as f:
    data = json.load(f)
labels = Counter(d['label'] for d in data)
print(f'标签分布: {dict(labels)}')
risks = [d['risk_score'] for d in data]
print(f'risk_score: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,2) for r in risks))}')
"

cp results_bert_v2.json results.json
rm -f NS-2026-03-answer-bert-v2.zip
zip NS-2026-03-answer-bert-v2.zip results.json
ls -la NS-2026-03-answer-bert-v2.zip
echo "[v2 BERT] 完成"
