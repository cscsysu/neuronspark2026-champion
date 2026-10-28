# NS-2026-08 Public Proxy Checker

This checker runs submitted `agent.py` on public proxy tasks. Public proxy scores are for local sanity only and are not formal hidden scores.

```powershell
python tools/check_format.py example_submission
python tools/run_public_eval.py example_submission --tasks public_tasks.jsonl
python checker/checker.py --tasks valid_tasks.jsonl --submission example_submission
python checker/visualize_trace.py train_traces.jsonl --limit 2
```
