# NS-2026-08 Data Notice

- version_id: `NS-2026-08-r20260525-07-score-calibrated`
- data_type: synthetic AIOps telemetry-world-model trajectories and proxy tasks.
- source_policy: The public data is generated locally. No external raw logs, metrics, labels, benchmark samples, production traces, map assets, or legacy materials are redistributed.
- train_trace_policy: `train_traces.jsonl` contains observations/actions/loss buckets only; observations may include missing fields, stale telemetry, delayed events and delayed diagnostic probe results with coarse `action_hint` sensor output. It does not contain formal root causes, private worlds, hidden latent coupling graphs, hidden seeds, scoring code, or official answers.
- proxy_boundary: `public_tasks.jsonl` and `valid_tasks.jsonl` contain proxy worlds for local debugging only; they are not formal hidden worlds and should not be treated as exact private simulator parameters.
- formal_boundary: `test_tasks.jsonl` contains public task information only and has no `proxy_world`, `private_world`, `latent_coupling_graph`, `event_channel_map`, hidden hard-slice membership, counterfactual variants, `scoring_state`, root-cause labels, rewards, or reference actions.
- submission_boundary: Final submissions must contain `agent.py`; model weights, checkpoints, cached rollouts and large intermediate files are not accepted.
- forbidden_material: Public packages must not contain internal seeds, private worlds, scoring program, official answers, old static-plan data, or review logs.
