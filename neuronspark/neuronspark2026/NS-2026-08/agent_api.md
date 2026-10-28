# Agent API

Submit a zip whose root contains `agent.py`.

`agent.py` must define:

```python
class Agent:
    def reset(self, task_info):
        ...
    def act(self, observation):
        return {"type": "NOOP"}
```

`reset(task_info)` is called once per task with public topology, service aliases, traffic forecast and action schema. `act(observation)` is called once per minute and must return exactly one action object. Observation fields may be missing or stale. `DIAGNOSTIC_PROBE` may return delayed noisy `probe_results` in later observations, including `source_hint`, bucketed confidence and a coarse `action_hint` such as `storage_index` or `queue_drain`. The formal judge does not pass root causes, hidden modes, latent coupling graphs, private simulator parameters, event channel maps, rewards or future observations.
