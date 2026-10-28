#!/bin/bash
# NS-2026-03 一键训练 + 推理 + 打包
# GPU 映射:
#   CUDA_VISIBLE_DEVICES=0 -> A40 (46G, 跑 Qwen 7B)
#   CUDA_VISIBLE_DEVICES=1/2/3 -> 3090 (24G, 跑 BERT)
set -e

cd "$(dirname "$0")"

echo "========================================"
echo "  NS-2026-03 训练与推理 Pipeline"
echo "========================================"

BERT_LOCAL="models/chinese-roberta-wwm-ext"
QWEN_LOCAL="/data2/css2025/models/Qwen/Qwen2.5-7B-Instruct"

# 0. BERT 模型
if [ ! -f "${BERT_LOCAL}/config.json" ]; then
    echo ""
    echo "[0a] 从 ModelScope 下载 BERT..."
    mkdir -p models
    python << 'PYEOF'
import os, sys, shutil
try:
    from modelscope import snapshot_download
except ImportError:
    os.system("pip install -q modelscope")
    from modelscope import snapshot_download
candidates = ["dienstag/chinese-roberta-wwm-ext", "AI-ModelScope/chinese-roberta-wwm-ext"]
target = "models/chinese-roberta-wwm-ext"
for repo in candidates:
    try:
        print(f"尝试: {repo}")
        path = snapshot_download(repo, cache_dir="./ms_cache")
        if os.path.exists(target): shutil.rmtree(target)
        shutil.copytree(path, target)
        print(f"成功: {target}")
        break
    except Exception as e:
        print(f"  失败: {e}")
else:
    sys.exit(1)
PYEOF
fi
echo "BERT: ${BERT_LOCAL}"

# Qwen 用服务器现有的
if [ ! -d "${QWEN_LOCAL}" ]; then
    echo "ERROR: Qwen 模型不存在 ${QWEN_LOCAL}"
    exit 1
fi
echo "Qwen: ${QWEN_LOCAL}"

# 环境
echo ""
echo "[1/5] 环境检查"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
wc -l train.jsonl test.jsonl

# Step 1: BERT 5-fold
echo ""
echo "[2/5] 训练 BERT (5-fold CV) - 3090"
CUDA_VISIBLE_DEVICES=1 python train_bert.py \
    --train_file train.jsonl \
    --test_file test.jsonl \
    --output_dir bert_out \
    --model_name $BERT_LOCAL \
    --n_folds 5 \
    --epochs 6 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 384 \
    --gpu 0 \
    2>&1 | tee bert_train.log

# Step 2: Qwen LoRA 5-fold
echo ""
echo "[3/5] 训练 Qwen2.5-7B LoRA (5-fold CV) - A40"
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

# Step 3: 推理 ensemble
echo ""
echo "[4/5] 推理 + Ensemble + 规则后处理 - A40"
CUDA_VISIBLE_DEVICES=0 python infer.py \
    --test_file test.jsonl \
    --qwen_base $QWEN_LOCAL \
    --qwen_lora_dir qwen_out \
    --bert_dir bert_out \
    --output results.json \
    --gpu 0 \
    --use_qwen \
    --use_bert \
    --qwen_weight 0.55 \
    --bert_weight 0.45 \
    2>&1 | tee infer.log

# Step 4: 验证格式
echo ""
echo "[5/5] 验证 + 打包"
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
test_ids = set()
with open('test.jsonl') as f:
    for line in f:
        test_ids.add(json.loads(line)['id'])
result_ids = set(d['id'] for d in data)
print(f'总数: {len(data)}, 缺失: {len(test_ids-result_ids)}, 多余: {len(result_ids-test_ids)}')
assert len(test_ids - result_ids) == 0 and len(result_ids - test_ids) == 0
"

rm -f NS-2026-03-answer.zip
zip NS-2026-03-answer.zip results.json
ls -la NS-2026-03-answer.zip
echo "完成"
