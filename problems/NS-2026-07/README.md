# NS-2026-07 触觉调度者：灵巧手接触丰富闭环仿真控制

> 题型：闭环仿真控制（提交 `agent.py`）｜满分 1500（竞赛分）｜难度：未记录｜最终线上得分 **1411 / 1500**（rawScore 347.87）

## 1. 题目介绍

- **任务**：灵巧手在 MuJoCo 开发仿真器里，用 **128 个控制步**完成三类任务：`nonprehensile_relocation`（非抓取搬运）、`tool_use`（工具使用）、`resource_sequence`（受限手指序列操作），可能含多阶段目标。隐藏评测会加入未见过的物理变体：mass、friction、fragility、action_delay、pose/tactile dropout、障碍物/走廊等。
- **接口**（见 [agent_api.md](../../neuronspark/neuronspark2026/NS-2026-07/agent_api.md)）：提交包根目录放 `agent.py`，定义 `Agent.reset(task_info)` 和 `Agent.act(observation) -> action`，可选 `reset_batch/act_batch`。
  - 观测：低维状态（到当前 stage / 最终目标的 dx、dy、dθ，contact、slip_risk 等）、位姿估计、逐指触觉、7×4 触觉热图、`tactile_image_7x8x8`、`vision_grid_16x16`、stage_context、sensor_status 等。
  - 动作（[action_schema.json](../../neuronspark/neuronspark2026/NS-2026-07/action_schema.json)）：`primitive ∈ {brace, push, drag, pivot, roll, lift_edge, tap, stabilize, wait, finish}`，`finger ∈ {thumb, index, middle, ring, pinky, palm, wrist}`，`force ∈ [0,1]`，`direction` 为二维向量。
- **数据**：`tasks/train_tasks.jsonl` 6000 条、`valid_tasks.jsonl` 360 条、`test_tasks.jsonl` 360 条；`demonstrations/weak_train_rollouts.jsonl` 1200 条弱规则 teacher 轨迹（成功率 9.6%）；`tactile_probes/` 若干触觉探针样例。公开 split 的 delay/dropout 全为 0，只用于本地 sanity。
- **评分**：每个任务得到 `score01`，失败任务封顶 `0.05 + 0.24 × objective`（writeup 从仿真器源码推导）。汇总为 `0.58 × task_mean + 0.24 × capability_tier_mean + 0.18 × worst_capability_tier`，再 ×1500。线上排名只看隐藏的程序化 rollout，反馈只返回总分。
- **运行限制**：Docker 镜像 `aigame-eval-ns202607-r08-torch-cu128:20260526`（CPython 3.12.3、torch 2.11 + cu128、mujoco 3.3.7、RTX 4090D），评测时断网。超时：`reset` 12s，单步 `act` 2s。

## 2. 我们的做法

**最终方案：纯规则、按物理参数调整的有限状态机控制器，不训练任何模型**。只依赖 Python 标准库 `math`，约 400 行。调参依据来自静读 `simulator/dexsim_core.py` 得到的物理常数和评分公式。

状态机流程为 **brace → tool → move → finish**：

1. **Brace 先建立接触**：palm 以 force 0.42 brace，直到 `contact ≥ 0.32` 或满 3 步。contact 偏低时后面的 push/tap 都会变弱。只改这一项，本地 valid 的 `tactile_slip_recovery` tier_min 就从 0.130 升到 0.191。
2. **Tool 阶段多敲几下**：利用失败任务的封顶公式，即使任务最终失败，也把 `tool_progress / tool_need` 推到 1.0，这样每个任务最多可拿 0.29 分。`taps_needed = ceil(tool_need × 1.35 / 0.048)`（tool_use）或 `× 1.45`（resource_sequence），并限制在一定范围内。
3. **resource_sequence 用 reserved 里最强的手指**：按 thumb 1.08 > index 1.00 > middle 0.95 > ring 0.72 > pinky 0.62 的力量系数选指。
4. **重物体用 roll 代替 pivot**：在 MuJoCo 的外力倍率中，push 4.8×、roll 4.4×（另有 0.42× 扭矩）、pivot 3.4×。`mass > 1.0` 时 pivot 几乎推不动物体，改用 roll 可以同时平移和旋转。
5. **Move 阶段的细节**：每隔 `2 + delay_hint` 步插一次 tap；stuck 检测（计数 +1/−2）触发后用 drag 脱困；`_avoid` 绕开障碍物和走廊边缘；`_force` 按 fragility 和 damage_risk 限力，并带在线 `force_boost` 自适应；finish 前要求 stage 完成度 ≥ 0.999，且 tap 数达标。
6. **sensor_status 自适应**：根据 delay、pose/tactile dropout 的提示，调整 brace 阈值、finish 容差和 tap 间隔。公开 split 里这些扰动全为 0，所以另写了 6 组扰动注入的 multi-variant proxy 做本地验证。v50 的 proxy 分是 348.5，隐藏实测 347.87。

**迭代过程**（来源：writeup §1、§5.5、§6）：
- 先试了学习类方案，最后都放弃：PPO 训练成功率 0%；GRU 行为克隆隐藏 321.32；CNN+GRU 全模态 BC 隐藏 320.81；CEM/MPPI 在公开仿真器上搜索，隐藏 321.07。主要原因是公开仿真器和隐藏仿真器的物理不一致。
- 规则线：v14 321.92 → v22（objective 救援）333.58 → v24（障碍物紧急避让）336.55 → v23（tap 系数 1.35/1.45 + 最强 reserved 手指）342.74 → **v50（brace 加长 + heavy roll）347.87**。
- 撤回或未采用的改动：opportunistic finish、更激进的 tap 系数（1.55/1.65 起）、轻物体也用 roll（−23）、更激进的 force_boost（−23）等。

## 3. 结果

| 来源 | 指标 | 数值 |
|---|---|---|
| 线上隐藏评测（writeup） | rawScore / 竞赛分 | **347.87 / 1411**（2026/5/27 21:31 提交 v50） |
| 本地 valid（[logs/eval_valid.log](../../neuronspark/neuronspark2026/writeup_submit/07/logs/eval_valid.log)） | score / raw_task_mean / success | 355.25 / 0.2564 / 0.142 |
| 本地 test（[logs/eval_test.log](../../neuronspark/neuronspark2026/writeup_submit/07/logs/eval_test.log)） | score / raw_task_mean / success | 449.03 / 0.2615 / 0.161 |
| 6 组扰动 proxy（writeup §5.4） | 平均分 | 348.5 |

隐藏评测资源：CPU 单进程，耗时 271.5 s，峰值内存约 198 MB，未用 GPU。

## 4. 目录与文件说明

题目目录：[neuronspark/neuronspark2026/NS-2026-07/](../../neuronspark/neuronspark2026/NS-2026-07/)（下表记为 `NS7/`）

| 路径 | 说明 |
|---|---|
| [neuronspark2026-answer/NS-2026-07-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-07-answer.zip) | **最终提交包**（只有 `agent.py`，19115 B），内容与 v50 一致 |
| [writeup_submit/07/agent.py](../../neuronspark/neuronspark2026/writeup_submit/07/agent.py) | **最终提交脚本**（v50），SHA256 `952df7ed…a024416` |
| [writeup_submit/07/NS-2026-07-answer.zip](../../neuronspark/neuronspark2026/writeup_submit/07/NS-2026-07-answer.zip) | 最终包副本，SHA256 `8c45b282…cf365e46` |
| [writeup_submit/07/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/07/writeup.md) | 完整解题报告 |
| [writeup_submit/07/run.sh](../../neuronspark/neuronspark2026/writeup_submit/07/run.sh)、[requirements.txt](../../neuronspark/neuronspark2026/writeup_submit/07/requirements.txt) | 一键复现脚本（格式检查 + valid/test 评测 + 哈希）及其依赖 |
| [writeup_submit/07/logs/](../../neuronspark/neuronspark2026/writeup_submit/07/logs/) | format_check / eval_valid / eval_test / hashes 日志 |
| 2026-07.zip<sup>†</sup> | 原始题目包（仿真器、任务、示范、工具） |
| [NS7/NS-2026-07-answer-v*.zip](../../neuronspark/neuronspark2026/NS-2026-07/) | 规则线历史版本 v11–v76。v50 = 最终版；v63–v76 是之后的实验，未作为最终提交 |
| [NS7/my_submission/](../../neuronspark/neuronspark2026/NS-2026-07/my_submission/) | 工作目录。当前 `agent.py` 与 v76 相同（**不是**最终版），另有 `agent_v25_backup.py` |
| `NS7/{gru,cnn_gru,bc,model}_submission/`、`NS-2026-07-answer-{gru,cnn-gru}.zip` | 已放弃的学习类方案（含 `.pt` 权重） |
| `NS7/{cem,search}_submission/`、`NS-2026-07-answer-search.zip` | 已放弃的 CEM/轨迹搜索方案 |
| `NS7/generate_data.py`、`train_ppo.py`、`train_bc_cpu.py`、`train_gru_hybrid.py`、`train_full_cnn_gru*.py` | 数据生成与训练脚本（只用于弃用方案） |
| `NS7/simulator/`、`NS7/tools/`、`NS7/tasks/`、`NS7/demonstrations/`、`NS7/tactile_probes/` | 官方仿真器、工具、任务 split、示范和探针 |
| `NS7/*.md`（GETTING_STARTED、agent_api、RUNTIME_ENVIRONMENT、DATA_*、TRAINING_DATA_GUIDE 等） | 官方说明，以及早期探索时整理的笔记 |

## 5. 如何运行

依赖：Python 3.12，`pip install mujoco==3.3.7 numpy==2.4.6`（即 [requirements.txt](../../neuronspark/neuronspark2026/writeup_submit/07/requirements.txt)；官方另有 `NS7/requirements-mujoco.txt`）。agent 本身不需要第三方库。

```bash
cd "neuronspark/neuronspark2026/NS-2026-07"      # 从仓库根目录出发
mkdir -p /tmp/ns07_sub && unzip -o ../../neuronspark2026-answer/NS-2026-07-answer.zip -d /tmp/ns07_sub

python tools/check_format.py /tmp/ns07_sub                                   # 格式检查（--tasks 默认 train）
python tools/run_public_eval.py /tmp/ns07_sub --tasks tasks/valid_tasks.jsonl # 本地 valid，--max-score 默认 1500
python tools/run_public_eval.py /tmp/ns07_sub --tasks tasks/test_tasks.jsonl
python tools/render_replay.py /tmp/ns07_sub --tasks tasks/valid_tasks.jsonl --index 0 --output-dir /tmp/replay  # 可选回放
```

一键复现脚本 `writeup_submit/07/run.sh`：
- `DATA_DIR` 默认硬编码为 `/Users/tristonchen/Desktop/neuronspark2026/NS-2026-07`，和本仓库的实际位置不一致，需要覆盖：
  `DATA_DIR="$PWD/neuronspark/neuronspark2026/NS-2026-07" bash neuronspark/neuronspark2026/writeup_submit/07/run.sh`
- 脚本会在 `writeup_submit/07/` 下创建 `my_submission/`，并覆盖 `logs/*.log`。

打包：在 `agent.py` 所在目录执行 `zip NS-2026-07-answer.zip agent.py`，zip 根目录只放 `agent.py`。

## 6. 参考

- [writeup.md](../../neuronspark/neuronspark2026/writeup_submit/07/writeup.md)：完整策略、消融表、提交记录和 AI 使用声明（A1）
- [GETTING_STARTED.md](../../neuronspark/neuronspark2026/NS-2026-07/GETTING_STARTED.md)、[agent_api.md](../../neuronspark/neuronspark2026/NS-2026-07/agent_api.md)、[RUNTIME_ENVIRONMENT.md](../../neuronspark/neuronspark2026/NS-2026-07/RUNTIME_ENVIRONMENT.md)

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
