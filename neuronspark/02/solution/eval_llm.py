"""在 train holdout 上对比 规则 vs +LLM，看 LLM 是否真的提分"""
import json, sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import DATA_DIR, load_corpus, load_train
from src.pipeline import build_train_lookup, build_train_question_index, build_train_chunk_index, answer_question
from src.retriever import BM25Retriever
from src.templates import parse_template

corpus = load_corpus()
train = load_train()
retriever = BM25Retriever(corpus)

n = len(train)
cut = int(n * 0.8)
hold_train = train[:cut]
hold_test = train[cut:]

# 只评测 holdout 中的非模板可答题（这才是 LLM 能改善的部分）
non_tmpl_holdtest = [t for t in hold_test if t['answerable'] and parse_template(t['question'])[0] is None]
print(f"Holdout 非模板可答题: {len(non_tmpl_holdtest)}")

lookup = build_train_lookup(hold_train)
qindex = build_train_question_index(hold_train)
chunk_index = build_train_chunk_index(hold_train)
answer_question._chunk_index = chunk_index

# 1. 规则版
answer_question._use_llm = False
t0 = time.time()
preds_rule = [answer_question(q, corpus, retriever, lookup, qindex) for q in non_tmpl_holdtest]
print(f"[规则] {time.time()-t0:.1f}s")

# 2. LLM 版
answer_question._use_llm = True
t0 = time.time()
preds_llm = [answer_question(q, corpus, retriever, lookup, qindex) for q in non_tmpl_holdtest]
print(f"[+LLM] {time.time()-t0:.1f}s")

# 写两份预测，再调用官方 eval 比较
ref_path = ROOT / "outputs" / "ref_holdout_nt.jsonl"
with open(ref_path, "w", encoding="utf-8") as f:
    for x in non_tmpl_holdtest:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

for tag, preds in [("rule", preds_rule), ("llm", preds_llm)]:
    p = ROOT / "outputs" / f"preds_holdout_nt_{tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print(f"\n=== {tag.upper()} 评测（仅非模板可答 {len(non_tmpl_holdtest)} 题）===")
    os.system(f'/Users/tristonchen/Desktop/02/solution/.venv/bin/python "{DATA_DIR}/tools/eval_public.py" "{p}" --reference "{ref_path}"')

# diff 对比
print("\n=== Rule vs LLM 答案差异样例 ===")
diff_cnt = 0
for i, q in enumerate(non_tmpl_holdtest):
    r, l = preds_rule[i]['answer'], preds_llm[i]['answer']
    if r != l:
        diff_cnt += 1
        if diff_cnt <= 15:
            print(f"Q: {q['question'][:55]}")
            print(f"  GOLD: {q['answer']}")
            print(f"  rule: {r}")
            print(f"  llm:  {l}")
print(f"...共 {diff_cnt}/{len(non_tmpl_holdtest)} 题答案不同")
