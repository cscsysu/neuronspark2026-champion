# Getting Started

This package contains the public simulator, public tasks, weak demonstrations and the runtime contract for NS-2026-07.

## 1. Unpack The Public Simulator Package

Download `NS-2026-07-data.tar.xz` from the problem page and unpack it:

```bash
tar -xf NS-2026-07-data.tar.xz
cd NS-2026-07
```

The public simulator and tools are included in this directory. No hidden judge files, hidden seeds, private rollout parameters or official answers are included.

## 2. Install Public Simulator Dependencies

For local public MuJoCo simulation and public evaluation:

```bash
python -m pip install -r requirements-mujoco.txt
```

For optional Gymnasium-style local training:

```bash
python -m pip install -r requirements-dev.txt
```

If MuJoCo is unavailable, some public tools fall back to the lightweight proxy checker so that format checks remain usable. Official scoring is expected to run with MuJoCo installed.

## 3. Run Public Sanity Checks

```bash
python tools/check_format.py example_submission
python tools/run_public_eval.py example_submission --tasks tasks/valid_tasks.jsonl
python tools/gym_smoke.py --tasks tasks/valid_tasks.jsonl --index 0 --steps 8
python tools/render_replay.py example_submission --tasks tasks/valid_tasks.jsonl --index 0 --output-dir replay_debug
```

Public `valid_tasks.jsonl` and `test_tasks.jsonl` are local sanity/debug splits only. They are not leaderboard ranking feedback.

## 4. Reproduce The Official Runtime Contract

The official leaderboard runtime is `ns202607-r08-torch-cu128` or a functionally equivalent image:

- expected image tag: `aigame-eval-ns202607-r08-torch-cu128:20260526`
- expected offline image archive name, if registry pull is unavailable: `aigame-eval-ns202607-r08-torch-cu128-20260526.tar.zst`
- digest: announced on the platform problem page at launch
- ABI: Ubuntu 24.04.2 LTS, glibc 2.39, CPython 3.12.3, x86_64 Linux
- GPU target: 1 x RTX 4090 D, 24564 MiB VRAM
- packages: `torch==2.11.0+cu128`, `mujoco==3.3.7`, `numpy==2.4.6`, `scipy==1.17.1`, `tqdm==4.67.3`
- network: disabled during evaluation

At launch, the platform should provide either a registry pull command with digest or an offline image archive download. The image contains the runtime dependencies only; it does not contain hidden judge data, hidden tasks, hidden seeds or private rollout traces.

If you cannot use the official image directly, reproduce the ABI and package versions in `RUNTIME_ENVIRONMENT.md` and `requirements-runtime.txt`. Runtime `pip install`, model download and external API calls are not allowed during official evaluation, so all extra code and model assets must be bundled in the submitted zip.

## 5. Submit

Your submitted zip root must contain `agent.py`. Model weights and configs may be placed under `model/`, `models/` or an equivalent local directory. Include a `model_manifest.json` when using nontrivial model assets so that file hashes and resource requirements are clear.
