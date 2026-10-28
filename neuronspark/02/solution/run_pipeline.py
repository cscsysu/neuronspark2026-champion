"""跑测试集 → outputs/results.json
用法:
    cd /Users/tristonchen/Desktop/02/solution
    python run_pipeline.py                # 纯规则
    python run_pipeline.py --llm          # 启用 mlx Qwen2.5-1.5B 增强
"""
import json
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_corpus, load_train, load_test
from src.pipeline import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="enable mlx Qwen2.5-1.5B answer extraction")
    args = ap.parse_args()

    corpus = load_corpus()
    train = load_train()
    test = load_test()
    print(f"corpus={len(corpus)}, train={len(train)}, test={len(test)} | use_llm={args.llm}", flush=True)

    results = run(test, corpus, train, use_llm=args.llm)
    out_path = ROOT / "outputs" / ("results.json" if not args.llm else "results_llm.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}  ({len(results)} items)", flush=True)


if __name__ == "__main__":
    main()
