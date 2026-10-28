#!/bin/bash
# 仅用 BERT 快速跑出第一版结果
# 用 3090 (CUDA_VISIBLE_DEVICES=1)
set -e
cd "$(dirname "$0")"

BERT_LOCAL="models/chinese-roberta-wwm-ext"

# 0. 下载 BERT 模型（优先用 ModelScope）
if [ ! -f "${BERT_LOCAL}/config.json" ]; then
    echo "[0] 从 ModelScope 下载 BERT 模型..."
    mkdir -p models
    # ModelScope 上 chinese-roberta-wwm-ext 镜像
    python << 'PYEOF'
import os, sys
try:
    from modelscope import snapshot_download
except ImportError:
    print("[install] modelscope")
    os.system("pip install -q modelscope")
    from modelscope import snapshot_download

# 已知可用镜像
candidates = [
    "dienstag/chinese-roberta-wwm-ext",
    "AI-ModelScope/chinese-roberta-wwm-ext",
    "thomas/chinese-roberta-wwm-ext",
]
target = "models/chinese-roberta-wwm-ext"
ok = False
for repo in candidates:
    try:
        print(f"尝试: {repo}")
        path = snapshot_download(repo, cache_dir="./ms_cache")
        # 把下载内容拷到 target
        import shutil
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.copytree(path, target)
        print(f"成功: {repo} -> {target}")
        ok = True
        break
    except Exception as e:
        print(f"  失败: {e}")
if not ok:
    sys.exit(1)
PYEOF
fi

echo "BERT 模型: ${BERT_LOCAL}"
ls ${BERT_LOCAL}

echo ""
echo "[1/3] 训练 BERT (5-fold CV) - 3090"
CUDA_VISIBLE_DEVICES=1 python train_bert.py \
    --train_file train.jsonl \
    --test_file test.jsonl \
    --output_dir bert_out \
    --model_name $BERT_LOCAL \
    --n_folds 5 \
    --epochs 8 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 384 \
    --gpu 0 \
    2>&1 | tee bert_train.log

echo ""
echo "[2/3] BERT 推理"
CUDA_VISIBLE_DEVICES=1 python infer.py \
    --test_file test.jsonl \
    --bert_dir bert_out \
    --output results.json \
    --gpu 0 \
    --use_bert \
    --bert_weight 1.0 \
    2>&1 | tee infer_bert.log

echo ""
echo "[3/3] 验证 + 打包"
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
"

rm -f NS-2026-03-answer.zip
zip NS-2026-03-answer.zip results.json
ls -la NS-2026-03-answer.zip
echo "[Quick] 完成"
