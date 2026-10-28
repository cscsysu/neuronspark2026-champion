# NS-2026-05 声场巡检员：校园声景事件检测

> 难度：Challenging ｜ 总分：1200 ｜ 题型：音频多标签事件检测（gold-band 阈值满分）｜ AI 协助等级：A1-Open

## 1. 题目介绍

**背景**：夜间巡检机器人在校园中（教室、走廊、图书馆、食堂、实验室、校道等）连续录制 6 秒短音频，需要为每段音频输出一个或多个事件标签。

- **输入**：`train.csv`（720 段，含 `id,audio,duration_sec,site,device,labels`）、`test.csv`（243 段，无 labels）、`audio/train|test/*.wav`（6 s / 16 kHz / mono / 16-bit PCM）、`label_map.json`、官方轻量特征 `features/*_audio_features.csv`（rms、zcr、centroid、16 个频带均值/方差等）。
- **标签（9 类）**：footsteps、keyboard、door、alarm、vehicle、rain、applause、glass_break、ambient；其中 **door / alarm / vehicle / glass_break 为稀有/高优先级**。
- **输出**：`results.csv`，表头严格为 `id,labels`，多个标签用 `|` 分隔；无事件填 `ambient`，**ambient 不能与其它标签共存**，不得为空/重复/未知标签。压缩为 `NS-2026-05-answer.zip`，包内只有 `results.csv`。
- **评分**（来源：题面，记录于 conversation-2026-05-27-05.txt<sup>†</sup>）：

  | 项目 | 分值 |
  |---|---|
  | 宏平均 F1 | 400 |
  | 样本级 F1 | 250 |
  | 稀有事件召回（alarm/door/glass_break/vehicle） | 250 |
  | 隐藏高难样本 sample-F1 | 200 |
  | 格式合法性 | 100 |

  `raw = 100 + 400·macro_f1 + 250·sample_f1 + 250·rare_recall + 200·hard_sample_f1`；若 macro_f1≥0.72 且 sample_f1≥0.74 且 rare_recall≥0.84 且 hard_sample_f1≥0.72，则直接记 **1200 满分**。
- **规则限制**：最终标签必须由本地可复现的开源模型/传统 ML/规则流程生成，禁止闭源 API、商业 Agent、远程音频理解服务；使用外部预训练权重需在 writeup 中披露。

## 2. 我们的做法

**核心思路**：AudioSet 预训练的 **AST（Audio Spectrogram Transformer，`MIT/ast-finetuned-audioset-10-10-0.4593`，~87M）全参数微调** + 多标签头 + 逐类阈值 + ambient 互斥后处理。未使用官方 `features/*.csv`，直接从 WAV 端到端建模。

- **预处理**：`librosa.load(sr=16000, mono)` → pad/截断到 96000 采样点 → `ASTFeatureExtractor`（128-bin fbank）。
- **模型**：AST backbone 取 CLS 向量(768) → LayerNorm → Dropout(0.3) → Linear(768→256)+GELU → Dropout(0.2) → Linear(256→9)。
- **损失**：`BCEWithLogitsLoss`，稀有类 alarm/door/glass_break/vehicle 的 `pos_weight=2.0`，提升召回。
- **训练**：AdamW(lr=1e-4, wd=0.01)、CosineAnnealingLR(T_max=30)、batch 16、30 epoch、梯度裁剪 1.0。
- **阈值校准**：随机 80/20 划分（`np.random.seed(42)`，未按场景/设备分组）。每个 epoch 在验证集上对每类在 [0.10, 0.88]、步长 0.02 搜索使该类 F1 最大的阈值，按 `0.4·Macro-F1 + 0.25·Sample-F1 + 0.25·Rare-Recall` 选最佳 epoch 的阈值。
- **全量重训**：用全部 720 条训练数据重新从预训练权重训练 30 epoch，配合上一步阈值推理测试集（注意：第一阶段的 `best_model_state` 并未用于推理）。
- **后处理**：无标签激活 → 填 `ambient`；ambient 与其他标签同时激活 → 去掉 ambient。

最终阈值（epoch 11）：footsteps 0.10 / keyboard 0.50 / door 0.28 / alarm 0.10 / vehicle 0.60 / rain 0.10 / applause 0.20 / glass_break 0.10 / ambient 0.10。

**迭代过程**（据对话记录）：
1. 先评估了"官方特征+传统 ML / log-mel CNN / 预训练模型微调"几种方案，因有 RTX 3090 服务器直接选 AST 微调。
2. 尝试改用 ModelScope 下载权重失败（仓库 ID 不存在），回退为 `HF_ENDPOINT=https://hf-mirror.com` 镜像下载。
3. 修复 `CUDA error: invalid device ordinal`：通过 `CUDA_VISIBLE_DEVICES=1` 映射后脚本内统一使用 `cuda:0`。
4. 镜像重复联网检查导致卡住 → `run.sh` 加 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1` 使用本地缓存。
5. 一次跑通即达满分，未再迭代。

## 3. 结果

| 来源 | 指标 |
|---|---|
| 本地验证（80/20，最佳 epoch 11，[writeup](../../neuronspark/neuronspark2026/writeup_submit/05/writeup.md) 与训练日志） | Macro-F1 0.9385 / Sample-F1 0.9468 / Rare-Recall 0.9597（加权分 0.8520） |
| 线上（writeup） | **1200 / 1200（Gold-Band 满分）**，提交时间 2026/5/23 23:17:18 |

提交包 SHA256：`ec8257703012236ac892939b0fd768b59b4640e5b6a19970d0feb663e4e67999`（与仓库内 zip 一致）；`results.csv` 共 244 行（表头 + 243 条），writeup 记录多标签预测 74 条。

## 4. 目录与文件说明

| 路径 | 说明 |
|---|---|
| [neuronspark2026/NS-2026-05/train_ast.py](../../neuronspark/neuronspark2026/NS-2026-05/train_ast.py) | **最终提交脚本**：数据加载、AST 微调、阈值搜索、全量重训、测试推理、写 results.csv、调用格式检查并打包 zip |
| [neuronspark2026/NS-2026-05/run.sh](../../neuronspark/neuronspark2026/NS-2026-05/run.sh) | 一键运行：安装依赖、设置 HF 镜像/离线变量、`CUDA_VISIBLE_DEVICES=1` 运行训练脚本 |
| [neuronspark2026/NS-2026-05/requirements.txt](../../neuronspark/neuronspark2026/NS-2026-05/requirements.txt) | Python 依赖 |
| [neuronspark2026-answer/NS-2026-05-answer.zip](../../neuronspark/neuronspark2026-answer/NS-2026-05-answer.zip) | **最终提交包**（仅含 results.csv）；[neuronspark2026/NS-2026-05-answer.zip](../../neuronspark/neuronspark2026/NS-2026-05-answer.zip) 为相同副本 |
| neuronspark2026/2026-05.zip<sup>†</sup> | 原始题目数据包（train/test.csv、label_map.json、audio/、features/、example_submission/、tools/check_format.py、tools/make_features.py） |
| neuronspark2026/features/train_audio_features.csv<sup>†</sup>、test_audio_features.csv<sup>†</sup> | 官方预提取轻量特征（与 zip 内一致，本方案未使用） |
| [neuronspark2026/writeup_submit/05/](../../neuronspark/neuronspark2026/writeup_submit/05/) | writeup 提交目录：writeup.md + 与 NS-2026-05/ 相同的代码副本 |
| neuronspark2026/conversation-2026-05-27-05.txt<sup>†</sup> | 解题对话记录（含完整题面与训练日志） |

## 5. 如何运行

**环境**：Python 3.12，NVIDIA GPU（原环境 RTX 3090 24GB，显存约 6–8GB）。原环境训练耗时约 60–80 分钟（30+30 epoch），推理约 1–2 分钟。

```bash
pip install -r requirements.txt   # torch torchaudio transformers librosa scikit-learn numpy pandas soundfile
```

**数据放置**：把 `2026-05.zip` 解压到一个目录（即 `DATA_ROOT`），并把 `train_ast.py` 放进去，目录中需有 `train.csv`、`test.csv`、`label_map.json`、`audio/`、`tools/check_format.py`。

**需要修改的硬编码**：
- `train_ast.py` 中 `DATA_ROOT = "/data2/css2025/NS-2026-05"`（原服务器路径）→ 改成你的数据目录；所有输出（`ast_model.pt`、`results.csv`、`NS-2026-05-answer.zip`）都写到该目录。
- `run.sh` 中 `cd /data2/css2025/NS-2026-05` 与 `CUDA_VISIBLE_DEVICES=1` → 按本机修改。
- `run.sh` 设置了 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`，**首次运行**本地无缓存时需先去掉这两行，让模型从 HF（或 `HF_ENDPOINT=https://hf-mirror.com`）下载。

**训练 + 推理 + 打包**（一个脚本完成，无命令行参数）：

```bash
bash run.sh
# 或
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 python3 train_ast.py
```

**格式检查**（脚本末尾已自动调用，也可手动执行）：

```bash
python3 tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json
```

说明：日志中的 "5-fold CV" / "Loading from ModelScope" 字样为遗留打印，实际是 80/20 划分、从 HF 加载；训练未固定 PyTorch 随机种子，复现结果可能有小幅波动。

## 6. 参考

- Writeup：[writeup_submit/05/writeup.md](../../neuronspark/neuronspark2026/writeup_submit/05/writeup.md)
- 题面与训练日志：conversation-2026-05-27-05.txt<sup>†</sup>
- 预训练模型：MIT/ast-finetuned-audioset-10-10-0.4593（HuggingFace，BSD-3-Clause）

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
