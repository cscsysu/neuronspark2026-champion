#!/bin/bash
# 重新校准已有概率，不重训
set -e
cd "$(dirname "$0")"

# 用现有的 qwen + bert 概率，重新决策
echo "=== v3 重校准（不重训）==="
python recalibrate.py \
    --test_file test.jsonl \
    --qwen_probs infer_debug/qwen_probs.npy \
    --bert_probs infer_debug/bert_probs.npy \
    --qwen_weight 0.5 \
    --bert_weight 0.5 \
    --output results_v3.json

cp results_v3.json results.json
rm -f NS-2026-03-answer-v3.zip
zip NS-2026-03-answer-v3.zip results.json
ls -la NS-2026-03-answer-v3.zip
echo "[v3] 完成"
