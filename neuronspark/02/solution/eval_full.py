"""完整 train evaluate (全集 + holdout) — 比较 规则 vs +LLM"""
import json, sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import DATA_DIR, load_corpus, load_train
from src.pipeline import build_train_lookup, build_train_question_index, build_train_chunk_index, answer_question
from src.retriever import BM25Retriever

corpus = load_corpus()
train = load_train()
retriever = BM25Retriever(corpus)

# Holdout 80/20
n = len(train)
cut = int(n * 0.8)
hold_train = train[:cut]
hold_test = train[cut:]
lookup = build_train_lookup(hold_train)
qindex = build_train_question_index(hold_train)
chunk_index = build_train_chunk_index(hold_train)
answer_question._chunk_index = chunk_index

ref_path = ROOT / "outputs" / "ref_holdout.jsonl"
with open(ref_path, "w", encoding="utf-8") as f:
    for x in hold_test: f.write(json.dumps(x, ensure_ascii=False) + "\n")

for tag in ["rule", "llm"]:
    answer_question._use_llm = (tag == "llm")
    t0 = time.time()
    preds = [answer_question(q, corpus, retriever, lookup, qindex) for q in hold_test]
    print(f"\n[{tag}] {time.time()-t0:.1f}s")
    p = ROOT / "outputs" / f"preds_holdout_full_{tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    os.system(f'/Users/tristonchen/Desktop/02/solution/.venv/bin/python "{DATA_DIR}/tools/eval_public.py" "{p}" --reference "{ref_path}"')
