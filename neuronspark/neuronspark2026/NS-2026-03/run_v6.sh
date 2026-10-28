#!/bin/bash
# v6: benign 大幅增强 + 重训 BERT + 用 v4 的 AUC 公式
set -e
cd "$(dirname "$0")"

BERT_LOCAL="models/chinese-roberta-wwm-ext"

# 1. 生成额外 benign
if [ ! -f "benign_extra.jsonl" ]; then
    echo "=== 1. 生成 benign 增强 (Qwen3-8B) ==="
    CUDA_VISIBLE_DEVICES=0 python gen_benign_boost.py \
        --model_path /data2/css2025/models/Qwen/Qwen3-8B \
        --output benign_extra.jsonl \
        --gpu 0 \
        2>&1 | tee gen_benign.log
fi
echo "benign_extra:"
wc -l benign_extra.jsonl

# 2. 合并: 原始 train + v4 增强 + benign extra
echo "=== 2. 合并数据集 ==="
cat train.jsonl train_aug.jsonl benign_extra.jsonl > train_v6.jsonl
echo "train_v6 总量:"
wc -l train_v6.jsonl
python -c "
import json
from collections import Counter
labels = Counter()
with open('train_v6.jsonl') as f:
    for line in f:
        labels[json.loads(line)['label']] += 1
print(f'分布: {dict(labels)}')
"

# 3. BERT v6
echo "=== 3. 训练 BERT v6 ==="
CUDA_VISIBLE_DEVICES=1 python train_bert.py \
    --train_file train_v6.jsonl \
    --test_file test.jsonl \
    --output_dir bert_v6 \
    --model_name $BERT_LOCAL \
    --n_folds 5 \
    --epochs 4 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 384 \
    --gpu 0 \
    2>&1 | tee bert_v6_train.log

# 4. 推理: BERT v6 only (用 v4 的好 risk_score 公式)
echo "=== 4. 推理 BERT v6 ==="
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --test_file test.jsonl \
    --bert_dir bert_v6 \
    --output results_v6.json \
    --gpu 0 \
    --use_bert \
    --bert_weight 1.0 \
    --debug_dir infer_debug_v6 \
    2>&1 | tee infer_v6.log

# 5. 验证
python -c "
import json
from collections import Counter
with open('results_v6.json') as f:
    data = json.load(f)
labels = Counter(d['label'] for d in data)
risks = [d['risk_score'] for d in data]
print(f'标签: {dict(labels)}')
print(f'risk: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,3) for r in risks))}')
"

cp results_v6.json results.json
rm -f NS-2026-03-answer-v6.zip
zip NS-2026-03-answer-v6.zip results.json
ls -la NS-2026-03-answer-v6.zip
echo "[v6] 完成"
