#!/bin/bash
# 环境安装（如果 cscllm 环境没装齐）
set -e

# 关键包
pip install -U "transformers>=4.45.0" "peft>=0.12.0" "accelerate>=0.34.0" \
    "datasets>=2.20.0" "scikit-learn>=1.3.0" "numpy>=1.24"

# bitsandbytes 可选（如显存不够想 4bit 时用）
# pip install bitsandbytes

# 验证
python -c "
import torch, transformers, peft, sklearn
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('peft:', peft.__version__)
print('sklearn:', sklearn.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
"
