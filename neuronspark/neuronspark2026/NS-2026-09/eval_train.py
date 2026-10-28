"""Eval current settings on train."""
import json, sys
sys.path.insert(0, '.')
import predict_v3

results = []
with open('train.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = predict_v3.best_profile_for(d)
        priors = predict_v3.context_priors(d)
        for q in d['queries']:
            results.append(predict_v3.predict_query(q, profile, priors, d['context_episodes']))
with open('results_train.json','w') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Score on train
import subprocess
out = subprocess.run(['python','score.py','results_train.json','train.jsonl'], capture_output=True, text=True).stdout
print(out)
