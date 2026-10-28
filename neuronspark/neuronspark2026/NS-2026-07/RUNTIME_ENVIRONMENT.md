# Official Runtime Environment

This file defines the public runtime contract for model submissions. It is the environment that participants should target when exporting `agent.py` plus offline model assets.

## Fixed Evaluation Contract

- runtime_name: `ns202607-r08-torch-cu128`
- equivalent_image_tag: `aigame-eval-ns202607-r08-torch-cu128:20260526`
- operating_system: `Ubuntu 24.04.2 LTS`
- libc: `glibc 2.39`
- architecture: `x86_64 Linux`
- python: `CPython 3.12.3`
- gpu: `1 x NVIDIA GeForce RTX 4090 D`, 24564 MiB VRAM
- nvidia_driver: `580.65.06`
- host_cuda_reported_by_nvidia_smi: `13.0`
- pytorch: `torch==2.11.0+cu128`
- pytorch_cuda_runtime: `12.8`
- mujoco: `mujoco==3.3.7`
- numpy: `numpy==2.4.6`
- scipy: `scipy==1.17.1`
- tqdm: `tqdm==4.67.3`
- network_policy: `none`

The exact production container digest may be announced by the platform at launch time, but it must be functionally equivalent to the contract above for leaderboard evaluation. The historical `aigame-eval-ns202607-mujoco:20260525` image is a CPU/MuJoCo smoke image and is not the r08 model-submission runtime contract.

## Obtaining The Runtime

The platform problem page should publish the launch-time image digest and one of the following access methods:

- a registry pull command, for example `docker pull aigame-eval-ns202607-r08-torch-cu128:20260526@sha256:<digest>`;
- an offline image archive named `aigame-eval-ns202607-r08-torch-cu128-20260526.tar.zst`, with sha256, for `docker load` after decompression.

This runtime image is only a dependency/ABI reproduction aid. It must not contain hidden judge code, hidden procedural seeds, private rollout parameters, official answers or internal scorer data. Participants do not need SSH access to the zxdmatrix evaluation node.

If the image cannot be used directly, reproduce the ABI and package versions listed in this file and `requirements-runtime.txt`. Official evaluation still runs with network disabled, so additional pure-Python dependencies, model weights and configs must be bundled in the submitted zip.

Before each official hidden evaluation, the remote node performs a bounded preflight check and recovery step. It verifies Docker availability, the runtime image, no-network/timeout configuration, storage roots, disk free watermark, `nvidia-smi`/GPU availability and initial GPU memory cleanliness; it may remove exited or stale evaluation containers labelled by the official evaluateapp. It does not kill arbitrary host processes, remove unlabelled historical containers, reset the GPU driver or reboot the node. The player-visible resource log may include redacted `preflight_status`, recovered-container counts, initial GPU memory and minimum disk-free summaries.

## Dependency Rules

The judge imports the submitted `agent.py` in a network-disabled environment. Runtime `pip install`, model download, external API calls and hosted inference services are not allowed.

Only the packages listed above are guaranteed. If a submission needs additional pure-Python code, it should be bundled in the submitted zip and imported locally. Native binary extensions or wheels outside the guaranteed package set are not guaranteed to work unless they are compatible with CPython 3.12, x86_64 Linux and glibc 2.39, and can be imported without network or installation steps.

`gymnasium` is optional for local training and the public dev wrapper. It is not required by the official `agent.py` interface and is not part of the guaranteed hidden runtime package set.

## Model Export Guidance

Recommended model formats are `state_dict` checkpoints, `.safetensors`, `.npz`, `.pt`/`.pth` files with source code included in the submission, and simple JSON/YAML config files. Avoid pickled full-model checkpoints that depend on training-time module paths not present in the final submission.

The model adapter should use `torch.cuda.is_available()` and `map_location` defensively, load all assets from paths relative to `agent.py`, and keep inference within the per-call timeouts described in `agent_api.md`.

## Resource Boundary

- submission zip: `<= 8192MB`
- extracted submission tree: `<= 16384MB`
- file count: `<= 8000`
- single file: `<= 8192MB`
- total model assets: `<= 12288MB`
- hidden wall time: `2700s`
- hidden control horizon: `128` steps
- max batched rollouts: `64`

Submissions may use the optional `reset_batch/act_batch` interface to amortize inference over the RTX 4090 D 24GB GPU.
