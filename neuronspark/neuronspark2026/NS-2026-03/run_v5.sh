#!/bin/bash
# v5: 三模型融合 + benign 偏置修正
set -e
cd "$(dirname "$0")"

# 推理用 BERT v2 + BERT v4 + Qwen3-8B 三路融合
echo "=== v5: BERT v2 + BERT v4 + Qwen3-8B 三模型 ==="

CUDA_VISIBLE_DEVICES=0 python infer_v5.py \
    --test_file test.jsonl \
    --bert_v2_dir bert_v2 \
    --bert_v4_dir bert_v4 \
    --qwen_base /data2/css2025/models/Qwen/Qwen3-8B \
    --qwen_lora_dir qwen3_8b_out \
    --output results_v5.json \
    --gpu 0 \
    --bert_v2_weight 0.3 \
    --bert_v4_weight 0.4 \
    --qwen_weight 0.3 \
    --benign_boost 0.05 \
    2>&1 | tee infer_v5.log

cp results_v5.json results.json
rm -f NS-2026-03-answer-v5.zip
zip NS-2026-03-answer-v5.zip results.json

python -c "
import json
from collections import Counter
with open('results_v5.json') as f:
    data = json.load(f)
labels = Counter(d['label'] for d in data)
risks = [d['risk_score'] for d in data]
print(f'[v5] 标签: {dict(labels)}')
print(f'[v5] risk: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,3) for r in risks))}')
"
ls -la NS-2026-03-answer-v5.zip
echo "[v5] 完成"
