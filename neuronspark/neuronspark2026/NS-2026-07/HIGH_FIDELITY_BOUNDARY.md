
# HIGH_FIDELITY_BOUNDARY

- version_id: `NS-2026-07-r20260525-08-gpu-vision-stages`
- public_required_backend: `MuJoCo 3.3.7`
- public_optional_wrapper: `simulator/gymnasium_env.py`
- hidden_high_fidelity_status: `planned_disabled_until_production_evidence`

The public development environment is MuJoCo-backed and Gymnasium-style. Participants are not required to install Isaac Sim, Genesis, Omniverse, a GPU renderer, or any large robotics asset pack. r08 still uses a lightweight MuJoCo/proxy simulator rather than a full Isaac/ManiSkill GPU simulator, but it adds cross-task batched inference, 16x16 spatial-grid observations, 7x8x8 tactile images, staged goals and stricter hidden procedural OOD so the 4090D 24GB node becomes a meaningful model-inference boundary.

A future hidden high-fidelity layer may use Isaac Sim or Genesis only as a server-side audit/supplement. It must preserve the same `Agent.reset(task_info)` and `Agent.act(observation)` contract, map high-fidelity state/contact signals back into the public observation schema, and keep private calibration tasks, simulator assets, seeds and rollout traces out of public packages.

This r08 package does not claim that Isaac/Genesis high-fidelity hidden scoring is active. Activation requires separate production evidence: fixed image digest, offline assets, network-disabled execution, resource report, baseline score curve, replay redaction, and release-alignment update.
