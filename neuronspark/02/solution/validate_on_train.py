"""在训练集上跑一遍流水线 → 用官方 eval_public.py 评分
用法:
    cd /Users/tristonchen/Desktop/02/solution
    python validate_on_train.py
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import DATA_DIR, load_corpus, load_train
from src.pipeline import run, build_train_lookup, build_train_question_index, build_train_chunk_index, answer_question
from src.retriever import BM25Retriever


def main():
    corpus = load_corpus()
    train = load_train()
    print(f"corpus={len(corpus)}, train={len(train)}", flush=True)

    retriever = BM25Retriever(corpus)
    lookup = build_train_lookup(train)
    qindex = build_train_question_index(train)
    chunk_index = build_train_chunk_index(train)
    answer_question._chunk_index = chunk_index

    preds = []
    for q in train:
        preds.append(answer_question(q, corpus, retriever, lookup, qindex))

    pred_path = ROOT / "outputs" / "preds_train_full.json"
    pred_path.parent.mkdir(exist_ok=True, parents=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    # 调用官方 eval_public.py
    eval_script = DATA_DIR / "tools" / "eval_public.py"
    train_jsonl = DATA_DIR / "train_qa.jsonl"
    cmd = f'python "{eval_script}" "{pred_path}" --reference "{train_jsonl}"'
    print(f"\n[评测] {cmd}\n")
    os.system(cmd)

    # === 再跑一次 holdout：把 train 切 80/20 ===
    print("\n========== Holdout 评测 (80/20) ==========")
    n = len(train)
    cut = int(n * 0.8)
    hold_train = train[:cut]
    hold_test = train[cut:]
    lookup2 = build_train_lookup(hold_train)
    chunk_index2 = build_train_chunk_index(hold_train)
    answer_question._chunk_index = chunk_index2
    preds2 = []
    for q in hold_test:
        preds2.append(answer_question(q, corpus, retriever, lookup2, build_train_question_index(hold_train)))
    pred_path2 = ROOT / "outputs" / "preds_train_holdout.json"
    with open(pred_path2, "w", encoding="utf-8") as f:
        json.dump(preds2, f, ensure_ascii=False, indent=2)

    # 写 holdout 的 reference
    ref_path = ROOT / "outputs" / "ref_holdout.jsonl"
    with open(ref_path, "w", encoding="utf-8") as f:
        for x in hold_test:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    cmd2 = f'python "{eval_script}" "{pred_path2}" --reference "{ref_path}"'
    print(f"\n[评测 holdout] {cmd2}\n")
    os.system(cmd2)


if __name__ == "__main__":
    main()
