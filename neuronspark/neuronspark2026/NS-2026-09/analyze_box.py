"""Track B prediction accuracy: sim vs probe modal vs initial."""
import json, sys
from collections import Counter
sys.path.insert(0, '.')
from sim2 import simulate, EVENT_KEYS, DEFAULT_PROFILE
from predict_v3 import best_profile_for, context_priors, find_pos, find_all

with open('results_v22.json') as f:
    preds = {p['id']: p for p in json.load(f)}

n = 0
sim_correct = 0
modal_correct = 0
init_correct = 0  # B at initial position
combined = 0  # current pred (which uses sim)
disagree_modal = 0  # sim != modal
sim_wins_disagree = 0
modal_wins_disagree = 0
no_b_count = 0

with open('valid.jsonl') as f:
    for line in f:
        d = json.loads(line)
        profile, _ = best_profile_for(d)
        # Probe modal
        all_B_probe = [tuple(find_all(ep['observed_final_full_grid'], 'B')) for ep in d['context_episodes']]
        modal_B = Counter(all_B_probe).most_common(1)[0][0]
        # Initial B positions
        init_B = tuple(find_all(d['initial_full_grid'], 'B'))
        for q in d['queries']:
            r = simulate(q['initial_full_grid'], q['future_actions'], profile)
            sim_B = tuple(sorted([(x, y) for y, row in enumerate(r['final_grid']) for x, c in enumerate(row) if c == 'B']))
            label_B = tuple(sorted(find_all(q['label']['final_grid'], 'B')))
            if not label_B and not init_B:
                no_b_count += 1
                continue
            n += 1
            if sim_B == label_B: sim_correct += 1
            modal_B_sorted = tuple(sorted(modal_B))
            if modal_B_sorted == label_B: modal_correct += 1
            if init_B == label_B: init_correct += 1
            if sim_B != modal_B_sorted:
                disagree_modal += 1
                if sim_B == label_B: sim_wins_disagree += 1
                if modal_B_sorted == label_B: modal_wins_disagree += 1

print(f'B prediction accuracy on {n} queries with B:')
print(f'  Sim B correct: {sim_correct}/{n} = {sim_correct/n:.4f}')
print(f'  Probe modal B correct: {modal_correct}/{n} = {modal_correct/n:.4f}')
print(f'  Initial B correct (no movement): {init_correct}/{n} = {init_correct/n:.4f}')
print(f'  Disagree (sim vs modal): {disagree_modal}, sim_wins={sim_wins_disagree}, modal_wins={modal_wins_disagree}')
