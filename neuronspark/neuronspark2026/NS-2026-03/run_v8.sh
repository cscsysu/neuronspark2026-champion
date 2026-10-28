#!/bin/bash
# v8: 全量 LoRA SFT (不做 CV) + 直接推理测试集
set -e
cd "$(dirname "$0")"

QWEN_PATH="/data2/css2025/models/Qwen/Qwen3-8B"

# 合并所有训练数据
echo "=== 合并训练数据 ==="
cat train.jsonl train_aug.jsonl benign_extra.jsonl > train_v8.jsonl
wc -l train_v8.jsonl
python -c "
import json
from collections import Counter
labels = Counter()
with open('train_v8.jsonl') as f:
    for line in f:
        labels[json.loads(line)['label']] += 1
print(f'分布: {dict(labels)}')
"

# 全量训练 (不做 fold split，用全部数据)
echo ""
echo "=== Qwen3-8B 全量 LoRA SFT ==="
CUDA_VISIBLE_DEVICES=0 python train_qwen.py \
    --train_file train_v8.jsonl \
    --output_dir qwen_v8_full \
    --model_name $QWEN_PATH \
    --full_train \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --gpu 0 \
    2>&1 | tee qwen_v8_train.log

# 推理 (单模型，不 ensemble)
echo ""
echo "=== 推理测试集 ==="
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_PATH \
    --qwen_lora_dir qwen_v8_full \
    --output results_v8.json \
    --gpu 0 \
    --use_qwen \
    --qwen_weight 1.0 \
    --debug_dir infer_debug_v8 \
    2>&1 | tee infer_v8.log

# 验证 + 打包
python -c "
import json
from collections import Counter
with open('results_v8.json') as f:
    data = json.load(f)
labels = Counter(d['label'] for d in data)
risks = [d['risk_score'] for d in data]
print(f'标签: {dict(labels)}')
print(f'risk: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,3) for r in risks))}')
"
cp results_v8.json results.json
rm -f NS-2026-03-answer-v8.zip
zip NS-2026-03-answer-v8.zip results.json
ls -la NS-2026-03-answer-v8.zip
echo "[v8] 完成"
