#!/bin/bash
# ============================================
# NS-2026-02 一键运行脚本
# 在服务器上执行: bash setup_and_run.sh
# ============================================
set -e

echo "============================================"
echo "NS-2026-02 RAG 知识库问答 - 一键部署"
echo "============================================"

# 检测 conda
if command -v conda &> /dev/null; then
    echo "[OK] conda detected"
else
    echo "[WARN] conda not found, using system pip"
fi

# 安装依赖
echo ""
echo "[1/4] Installing dependencies..."
pip install -q jieba rank_bm25 transformers torch accelerate modelscope 2>/dev/null || \
pip install jieba rank_bm25 transformers torch accelerate modelscope

# 检测 GPU
echo ""
echo "[2/4] Checking GPU..."
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 运行
echo ""
echo "[3/4] Running inference (Qwen2.5-3B-Instruct on GPU)..."
echo "  This takes ~3-5 min on RTX 3090..."
echo ""

CUDA_VISIBLE_DEVICES=0 python solve_best.py

# 检查结果
echo ""
echo "[4/4] Checking output..."
python -c "
import json
with open('results.json') as f:
    r = json.load(f)
u = sum(1 for x in r if x['answer']=='无法根据给定知识库回答')
print(f'  Total: {len(r)}')
print(f'  Answered: {len(r)-u}')
print(f'  Refused: {u}')
# Sanity check
assert len(r) == 332, f'Expected 332, got {len(r)}'
ids = set(x['id'] for x in r)
assert len(ids) == 332, 'Duplicate IDs!'
for x in r:
    assert isinstance(x['answer'], str) and x['answer'], 'Empty answer!'
    assert isinstance(x['citations'], list), 'Citations not list!'
print('  Format check: PASSED')
"

echo ""
echo "============================================"
echo "DONE! Submit file: NS-2026-02-answer.zip"
echo "============================================"
ls -lh NS-2026-02-answer.zip
