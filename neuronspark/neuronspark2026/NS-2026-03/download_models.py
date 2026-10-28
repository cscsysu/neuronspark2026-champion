#!/usr/bin/env python
"""测试服务器能下载哪些镜像 / 预下载模型"""
import os
import sys
import subprocess

# 测试三种方式: hf-mirror, modelscope, raw huggingface
endpoints = {
    "hf-mirror": "https://hf-mirror.com",
    "modelscope-via-hf": "https://www.modelscope.cn",
    "huggingface": "https://huggingface.co",
}

print("=== 测试端点连通性 ===")
for name, url in endpoints.items():
    try:
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "5", url],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  {name:25s} -> HTTP {r.stdout}")
    except Exception as e:
        print(f"  {name:25s} -> ERROR {e}")

# 选可用端点
print("\n=== 尝试用 hf-mirror 下载模型 ===")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

models_to_pull = {
    "hfl/chinese-roberta-wwm-ext": "models/chinese-roberta-wwm-ext",
    "Qwen/Qwen2.5-7B-Instruct": "models/Qwen2.5-7B-Instruct",
}

for repo, local in models_to_pull.items():
    if os.path.isdir(local) and os.listdir(local):
        print(f"  [SKIP] {repo} 已存在 -> {local}")
        continue
    print(f"\n下载: {repo} -> {local}")
    try:
        snapshot_download(
            repo_id=repo,
            local_dir=local,
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "tokenizer*", "vocab*", "merges*"],
        )
        print(f"  ✓ 下载完成")
    except Exception as e:
        print(f"  ✗ 失败: {e}")

print("\n=== 完成 ===")
print("可用模型路径:")
for repo, local in models_to_pull.items():
    if os.path.isdir(local):
        size = subprocess.run(["du", "-sh", local], capture_output=True, text=True).stdout.strip()
        print(f"  {local}: {size}")
