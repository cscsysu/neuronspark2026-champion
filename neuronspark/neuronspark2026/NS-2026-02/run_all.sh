#!/bin/bash
# ============================================
# NS-2026-02 完整流程：LoRA微调 + 推理
# 一键运行：bash run_all.sh
# ============================================
set -e

echo "============================================"
echo "NS-2026-02 Qwen3.5-4B + LoRA + Self-Consistency"
echo "============================================"

# 1. 安装依赖
echo ""
echo "[1/5] Installing dependencies..."
pip install -q jieba rank_bm25 transformers torch accelerate modelscope peft trl datasets bitsandbytes 2>/dev/null || \
pip install jieba rank_bm25 transformers torch accelerate modelscope peft trl datasets bitsandbytes

# 2. 检测GPU
echo ""
echo "[2/5] Checking GPU..."
python -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 3. 准备数据
echo ""
echo "[3/5] Preparing LoRA training data..."
python prepare_lora_data.py

# 4. LoRA微调（删除旧adapter重新训练）
echo ""
echo "[4/5] LoRA fine-tuning Qwen3.5-4B (2 epochs)..."
rm -rf lora_output
CUDA_VISIBLE_DEVICES=0 python train_lora.py

# 5. 推理
echo ""
echo "[5/5] Running inference with Self-Consistency..."
CUDA_VISIBLE_DEVICES=0 python solve_final.py

# 检查
echo ""
echo "=== Checking output ==="
python -c "
import json
with open('results.json') as f:
    r = json.load(f)
u = sum(1 for x in r if x['answer']=='无法根据给定知识库回答')
print(f'  Total: {len(r)}, Answered: {len(r)-u}, Refused: {u}')
assert len(r) == 332
ids = set(x['id'] for x in r)
assert len(ids) == 332
for x in r:
    assert isinstance(x['answer'], str) and x['answer']
    assert isinstance(x['citations'], list)
print('  Format: PASSED')
"

echo ""
echo "============================================"
echo "DONE! Submit: NS-2026-02-answer.zip"
echo "============================================"
ls -lh NS-2026-02-answer.zip
