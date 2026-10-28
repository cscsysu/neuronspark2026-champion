"""额外跑2轮 temperature=0.5 的 DS Pro, 然后做5次投票"""
import json, os, sys
sys.path.insert(0, '.')
from plan_770 import *

test_data = load_test_data()
system_prompt = build_system_prompt_v1()

for run_idx in [4, 5]:
    progress_file = f"sc_progress_run{run_idx}.jsonl"
    if os.path.exists(progress_file) and sum(1 for _ in open(progress_file)) >= 430:
        print(f"Run {run_idx} already done, skip")
        continue
    print(f"\n=== High-temp Run {run_idx} (temp=0.5) ===")
    results = run_one_pass(test_data, MODEL_DS_PRO, 0.5, progress_file, system_prompt)
    print(f"  Run {run_idx} done: {len(results)} items")

# 5次投票
print("\n=== 5-run Majority Vote ===")
all_runs = []
for i in range(1, 6):
    pf = f"sc_progress_run{i}.jsonl"
    if not os.path.exists(pf):
        print(f"Missing run {i}, skip")
        continue
    run_data = {}
    with open(pf) as f:
        for line in f:
            d = json.loads(line)
            run_data[d["id"]] = d
    if len(run_data) >= 430:
        all_runs.append(run_data)
        print(f"  Run {i}: {len(run_data)} items")
    else:
        print(f"  Run {i}: INCOMPLETE ({len(run_data)}), skip")

if len(all_runs) < 3:
    print("Not enough runs!")
    sys.exit(1)

from collections import Counter
test_ids = [d["id"] for d in test_data]
final_results = []
changes = 0
for tid in test_ids:
    labels = [run.get(tid, {}).get("label", "benign") for run in all_runs]
    scores = [run.get(tid, {}).get("risk_score", 0.5) for run in all_runs]
    
    # Weighted majority: give run1 (temp=0, 752 version) extra weight
    weighted_labels = [labels[0], labels[0]] + labels[1:]  # run1 counts double
    label_votes = Counter(weighted_labels)
    best_label = label_votes.most_common(1)[0][0]
    
    # Average risk_score from agreeing runs
    agreeing_scores = [s for l, s in zip(labels, scores) if l == best_label]
    avg_score = sum(agreeing_scores) / len(agreeing_scores) if agreeing_scores else sum(scores) / len(scores)
    
    if best_label != labels[0]:
        changes += 1
    
    final_results.append({"id": tid, "label": best_label, "risk_score": round(avg_score, 4)})

labels_dist = Counter(r["label"] for r in final_results)
risks = [r["risk_score"] for r in final_results]
print(f"\n标签分布: {dict(labels_dist)}")
print(f"risk_score unique={len(set(risks))}, range=[{min(risks):.4f}, {max(risks):.4f}]")
print(f"相比 Run1 变化: {changes} 条")

with open("results_5vote.json", "w") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
import shutil
shutil.copy("results_5vote.json", "results.json")
os.system("rm -f NS-2026-03-answer-5vote.zip && zip NS-2026-03-answer-5vote.zip results.json")
print(f"\n提交文件: NS-2026-03-answer-5vote.zip")
