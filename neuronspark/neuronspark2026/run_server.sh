#!/bin/bash
# NS-2026-04 服务器端一键运行脚本
# Usage: bash run_server.sh

set -e

echo "=========================================="
echo "NS-2026-04 分子炼金术 - 服务器端运行"
echo "=========================================="

# 1. 安装依赖
echo ""
echo ">>> Installing dependencies..."
pip install numpy scipy scikit-learn lightgbm xgboost pandas rdkit-pypi torch torchvision --quiet 2>/dev/null || true

# Install torch_geometric (需要匹配 PyTorch 和 CUDA 版本)
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
echo "PyTorch: $TORCH_VERSION, CUDA: $CUDA_VERSION"

pip install torch-geometric --quiet 2>/dev/null || true
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION//./}.html --quiet 2>/dev/null || true

echo ""
echo ">>> Dependencies installed!"
echo ""

# 2. Check GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

# 3. Run the solution
echo ""
echo ">>> Running solution..."
python3 solve_gpu.py

echo ""
echo ">>> Done! Submit NS-2026-04-answer.zip"
