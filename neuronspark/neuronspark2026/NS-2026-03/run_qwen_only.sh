#!/bin/bash
# 仅训练 Qwen + 复用已有 BERT，做最终 ensemble
set -e
cd "$(dirname "$0")"

QWEN_LOCAL="/data2/css2025/models/Qwen/Qwen2.5-7B-Instruct"

if [ ! -d "${QWEN_LOCAL}" ]; then
    echo "ERROR: Qwen 模型不存在 ${QWEN_LOCAL}"
    exit 1
fi

if [ ! -d "bert_out/fold_1" ]; then
    echo "ERROR: 请先跑 run_quick.sh 得到 bert_out"
    exit 1
fi

echo "=== 训练 Qwen2.5-7B LoRA (5-fold CV) ==="
CUDA_VISIBLE_DEVICES=0 python train_qwen.py \
    --train_file train.jsonl \
    --output_dir qwen_out \
    --model_name $QWEN_LOCAL \
    --n_folds 5 \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 8 \
    --lr 1e-4 \
    --max_length 1024 \
    --lora_r 16 \
    --lora_alpha 32 \
    --gpu 0 \
    2>&1 | tee qwen_train.log

echo ""
echo "=== Ensemble 推理 ==="
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_LOCAL \
    --qwen_lora_dir qwen_out \
    --bert_dir bert_out \
    --output results_ensemble.json \
    --gpu 0 \
    --use_qwen \
    --use_bert \
    --qwen_weight 0.55 \
    --bert_weight 0.45 \
    2>&1 | tee infer_ensemble.log

# 验证 + 打包
python -c "
import json
with open('results_ensemble.json') as f:
    data = json.load(f)
test_ids = set()
with open('test.jsonl') as f:
    for line in f:
        test_ids.add(json.loads(line)['id'])
result_ids = set(d['id'] for d in data)
print(f'总数: {len(data)}, 缺失: {len(test_ids-result_ids)}, 多余: {len(result_ids-test_ids)}')
"

cp results_ensemble.json results.json
rm -f NS-2026-03-answer-ensemble.zip
zip NS-2026-03-answer-ensemble.zip results.json
ls -la NS-2026-03-answer-ensemble.zip
echo "完成"
