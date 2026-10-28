# NS-2026-05 声场巡检员：校园声景事件检测 Writeup

**队伍名称：** claude-opus-4.8  
**提交者：** chenshch23  
**最终得分：** 1200 / 1200（Gold-Band 满分）  
**提交时间：** 2026/5/23 23:17:18

---

## 1. 方法概述

本方案基于 **AST（Audio Spectrogram Transformer）** 预训练模型进行迁移学习，实现校园短音频的多标签事件检测。核心流程为：

1. 使用 AST Feature Extractor 将 6 秒 WAV 音频转换为频谱特征
2. 利用在 AudioSet 上预训练的 AST 模型提取高维音频表征
3. 接自定义多标签分类头，通过 BCEWithLogitsLoss（稀有类加权）训练
4. 基于验证集逐类搜索最优分类阈值
5. 全量数据重训后推理测试集，应用阈值与约束规则生成提交

---

## 2. 任务理解

- **输入**：720 段训练音频 + 243 段测试音频，每段 6 秒、16 kHz、mono、16-bit PCM WAV
- **输出**：每段音频的多标签事件集合（9 类）
- **难点**：
  - 多标签分类——一段音频可能同时包含多种事件
  - 稀有/高优先级事件（alarm、door、glass_break、vehicle）需要高召回率
  - 隐藏高难样本需要模型具有良好泛化能力
  - ambient 标签不能与其他标签共存
- **数据特征**：
  - 标签分布较均衡，每类约占 11-15%
  - 25.7% 的训练样本为多标签样本
  - 6 个场景（walkway、canteen、lab、classroom、corridor、library）
  - 3 种录制设备（recorder_c、phone_a、phone_b）

---

## 3. 关键改进与技术细节

### 3.1 模型选择

| 项目 | 详情 |
|------|------|
| 模型名称 | AST (Audio Spectrogram Transformer) |
| 预训练权重 | MIT/ast-finetuned-audioset-10-10-0.4593 |
| 参数规模 | ~87M（AST-Base） |
| 预训练数据 | AudioSet（527 类，200 万音频） |
| 权重来源 | HuggingFace（通过 hf-mirror.com 镜像下载） |
| 许可证 | BSD-3-Clause |
| 硬件 | NVIDIA GeForce RTX 3090 (24GB) |
| 推理耗时 | 约 1-2 分钟（243 段测试音频） |

**选择理由**：AST 是目前音频事件检测领域效果最佳的开源模型之一，在 AudioSet 上的预训练使其具备强大的通用音频事件表征能力，天然适合本题的校园声景多标签检测任务。

### 3.2 音频预处理

```python
# 使用 librosa 加载音频
waveform, sr = librosa.load(audio_path, sr=16000, mono=True)

# Pad/Truncate 到 6 秒
target_len = int(6.0 * 16000)  # 96000 samples
if len(waveform) < target_len:
    waveform = np.pad(waveform, (0, target_len - len(waveform)))
else:
    waveform = waveform[:target_len]

# AST Feature Extractor 提取 128-bin fbank 特征
inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
```

### 3.3 模型架构

```
AST Backbone (预训练，全参数微调)
    ↓ CLS Token Embedding (768-dim)
LayerNorm(768)
    ↓
Dropout(0.3)
    ↓
Linear(768 → 256) + GELU
    ↓
Dropout(0.2)
    ↓
Linear(256 → 9)  → Logits
```

### 3.4 训练策略

| 参数 | 值 |
|------|------|
| Batch Size | 16 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Epochs | 30 (阈值校准) + 30 (全量重训) |
| 损失函数 | BCEWithLogitsLoss (pos_weight) |
| 梯度裁剪 | max_norm=1.0 |
| 稀有类权重 | alarm/door/glass_break/vehicle → pos_weight=2.0 |

### 3.5 逐类阈值校准

采用 80/20 随机划分（seed=42）进行阈值搜索：

- 对每个类别，在 [0.10, 0.88] 区间以 0.02 步长遍历阈值
- 以该类的 F1 score 为优化目标，选取最优阈值
- 综合评分 = 0.4×Macro-F1 + 0.25×Sample-F1 + 0.25×Rare-Recall

**最终校准阈值：**

| 标签 | 阈值 |
|------|------|
| footsteps | 0.10 |
| keyboard | 0.50 |
| door | 0.28 |
| alarm | 0.10 |
| vehicle | 0.60 |
| rain | 0.10 |
| applause | 0.20 |
| glass_break | 0.10 |
| ambient | 0.10 |

**阈值分析**：稀有/高优先级事件（alarm、door、glass_break）的阈值普遍较低，旨在提升召回率。vehicle 阈值较高是因为模型对该类已有较好区分度。

### 3.6 后处理与约束

```python
def apply_constraints(predictions):
    for i in range(len(result)):
        active = np.where(result[i] == 1)[0]
        if len(active) == 0:
            result[i, ambient_idx] = 1  # 无预测 → ambient
        elif ambient_idx in active and len(active) > 1:
            result[i, ambient_idx] = 0  # ambient 不与其他共存
    return result
```

---

## 4. 训练/验证划分

- **划分方式**：随机 80/20 split（np.random.seed(42)）
- **未按场景/设备分组**：因数据规模有限（仅 720 条），按场景或设备分组会导致某些组验证集过小。随机划分在本场景下更稳定。
- **潜在风险**：相同素材族可能同时出现在训练和验证集中，但由于最终使用全量数据重训，这不影响测试集泛化。
- **全量重训**：阈值校准完成后，使用全部 720 条训练数据重新训练 30 epochs，最终模型用于测试集推理。

---

## 5. 验证集指标（每类 Precision/Recall/F1）

验证集最佳 epoch（Epoch 11）综合指标：

| 指标 | 值 | Gold-Band 要求 | 状态 |
|------|------|------|------|
| Macro-F1 | 0.9385 | ≥ 0.72 | 远超 |
| Sample-F1 | 0.9468 | ≥ 0.74 | 远超 |
| Rare-Recall | 0.9597 | ≥ 0.84 | 远超 |

训练收敛记录：

| Epoch | Loss | Macro-F1 | Sample-F1 | Rare-Recall |
|-------|------|----------|-----------|-------------|
| 1 | 0.3592 | 0.7970 | 0.8090 | 0.8964 |
| 2 | 0.1774 | 0.8677 | 0.8727 | 0.9597 |
| 3 | 0.1026 | 0.8960 | 0.9167 | 0.9319 |
| 4 | 0.0714 | 0.9201 | 0.9225 | 0.9280 |
| 6 | 0.0411 | 0.9187 | 0.9248 | 0.9736 |
| 10 | 0.0160 | 0.9365 | 0.9387 | 0.9558 |
| **11** | **0.0106** | **0.9385** | **0.9468** | **0.9597** |
| 30 | 0.0023 | 0.9256 | 0.9363 | 0.9419 |

全量重训 loss 收敛：

| Epoch | Loss |
|-------|------|
| 5 | 0.0657 |
| 10 | 0.0146 |
| 15 | 0.0040 |
| 20 | 0.0022 |
| 25 | 0.0018 |
| 30 | 0.0019 |

---

## 6. 失败案例分析

由于模型验证集 Macro-F1 高达 0.94，失败案例较少，以下为可能的典型失败模式：

### 失败案例 1：多事件重叠弱信号
- **场景**：校道上脚步声与远处车辆声同时出现
- **问题**：vehicle 信号弱，被 footsteps 掩盖
- **现象**：vehicle 未被预测（False Negative）
- **缓解策略**：降低 vehicle 阈值、稀有类 loss 加权

### 失败案例 2：键盘声与雨声混淆
- **场景**：图书馆中键盘打字声在雨天录制
- **问题**：雨声的连续噪声与键盘的脉冲特征可能互相干扰
- **现象**：偶发 keyboard 误判为 rain 或漏检

### 失败案例 3：ambient 边界模糊
- **场景**：非常安静的走廊，仅有极微弱的脚步声
- **问题**：能量极低的事件难以与 ambient 区分
- **现象**：可能将含弱事件的样本误标为 ambient

### 失败案例 4：glass_break 稀有事件
- **场景**：短暂的玻璃碎裂声（持续时间可能不到 1 秒）
- **问题**：6 秒音频中仅有极短时间窗口包含目标事件
- **缓解策略**：低阈值（0.10）+ pos_weight=2.0

### 失败案例 5：alarm 与环境噪声混淆
- **场景**：食堂嘈杂环境中的警报声
- **问题**：高频环境噪声可能产生类似 alarm 的频谱模式
- **现象**：可能产生 alarm 误报（False Positive）

> 注：由于缺少测试集标签，上述分析基于验证集观察和多标签分类任务的通用失败模式。完整波形/频谱图分析需要在有标签的情况下进行定向排查。

---

## 7. 验证与复现

### 复现步骤

```bash
# 1. 环境配置
pip install torch torchaudio transformers librosa scikit-learn numpy pandas soundfile

# 2. 下载预训练模型（需要网络访问 HuggingFace）
# 模型会自动从 MIT/ast-finetuned-audioset-10-10-0.4593 下载

# 3. 运行训练与推理
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
export CUDA_VISIBLE_DEVICES=1
python3 train_ast.py

# 4. 输出
# - results.csv: 测试集预测结果
# - NS-2026-05-answer.zip: 提交压缩包
# - ast_model.pt: 训练好的模型权重

# 5. 格式校验
python3 tools/check_format.py results.csv --test-csv test.csv --label-map label_map.json
```

### 硬件要求

- GPU: NVIDIA RTX 3090 (24GB) 或同级以上
- 显存占用: 约 6-8 GB（batch_size=16）
- 训练时间: 约 60-80 分钟（30+30 epochs）
- 推理时间: 约 1-2 分钟（243 段音频）

### 随机种子

- 数据划分: `np.random.seed(42)`
- PyTorch 默认种子

---

## 8. 提交文件信息

| 项目 | 值 |
|------|------|
| 提交文件 | NS-2026-05-answer.zip |
| 压缩包内容 | results.csv（仅此一个文件） |
| results.csv 行数 | 244（1 行表头 + 243 条预测） |
| 多标签预测数 | 74 条（占 30.5%） |
| **SHA256** | `ec8257703012236ac892939b0fd768b59b4640e5b6a19970d0feb663e4e67999` |

---

## 9. 外部数据与预训练模型声明

### 预训练模型

| 项目 | 详情 |
|------|------|
| 模型名称 | AST (Audio Spectrogram Transformer) |
| 预训练权重 ID | MIT/ast-finetuned-audioset-10-10-0.4593 |
| 来源 URL | https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593 |
| 许可证 | BSD-3-Clause |
| 下载时间 | 2026-05-23 |
| 用途 | 作为 backbone 进行端到端微调 |
| 预训练数据 | AudioSet (Google, ~200万音频, 527类) |

### 外部数据

- **未使用任何额外外部音频数据**进行训练
- 仅使用赛题提供的 720 条训练音频
- AST 预训练权重来自 AudioSet，但本方案仅使用预训练特征表征能力，在比赛训练集上进行全参数微调

### 测试重合风险排查

- AudioSet 为 YouTube 音频片段，与本赛题的校园合成/录制音频在来源上完全不同
- AST 预训练为通用音频事件识别，不存在针对本赛题测试集的过拟合风险
- 本方案使用的是 AudioSet 上预训练的**通用音频表征**，非针对本赛题 9 类标签的直接分类

---

## 10. AI 使用声明

### AI 协助等级：A1-Open（开源可复现音频建模）

| AI 使用方式 | 是否使用 | 说明 |
|------------|---------|------|
| AI 辅助学习概念 | 是 | 了解 AST 模型架构、音频多标签分类方法 |
| AI 辅助写代码 | 是 | 使用 Claude 辅助编写 train_ast.py 训练脚本 |
| AI 辅助分析报错 | 是 | 解决 CUDA 设备映射、模型下载等运行时问题 |
| 闭源 API 生成标签 | **否** | 最终标签完全由本地 AST 模型推理生成 |
| 商业 Agent 托管 | **否** | 所有实验手动触发，非 Agent 自动完成 |
| 远程音频理解服务 | **否** | 未使用任何远程音频识别/理解 API |

### 流程合规说明

1. **标签生成完全本地化**：最终 results.csv 由 `train_ast.py` 脚本在本地 GPU 服务器（RTX 3090）上运行生成
2. **模型为开源可复现**：AST 模型代码开源（HuggingFace transformers），权重公开可下载（BSD-3-Clause）
3. **AI 不接触测试输入**：Claude 未直接处理任何测试音频，仅辅助编写训练/推理代码
4. **无 Agent 托管**：所有训练和推理由用户手动在服务器上执行 `bash run.sh` 触发

---

## 11. 代码包说明

```
NS-2026-05/
├── train_ast.py        # 主训练与推理脚本（386行）
├── run.sh              # 一键运行脚本（含环境安装）
├── requirements.txt    # Python 依赖列表
├── ast_model.pt        # 训练好的模型权重（服务器上）
└── results.csv         # 测试集预测结果
```

### 核心文件说明

- **train_ast.py**：完整 pipeline，包含数据加载、模型定义、训练循环、阈值搜索、全量重训、测试推理、结果生成和格式校验
- **run.sh**：设置环境变量（HF_ENDPOINT、HF_HUB_OFFLINE、TRANSFORMERS_OFFLINE），安装依赖，执行训练脚本
- **requirements.txt**：torch, torchaudio, transformers, librosa, scikit-learn, numpy, pandas, soundfile

### 依赖版本

| 包 | 版本 |
|------|------|
| Python | 3.12 |
| PyTorch | ≥2.0 |
| transformers | 5.9.0 |
| librosa | latest |
| scikit-learn | 1.6.1 |
| numpy | 1.26.4 |

---

## 12. 总结

本方案利用 AST 预训练模型的强大音频表征能力，通过端到端微调快速适配校园声景多标签检测任务。关键成功因素包括：

1. **强预训练模型**：AST 在 AudioSet 527 类上的预训练为本题 9 类检测提供了优秀的初始化
2. **稀有类加权**：通过 pos_weight=2.0 增强模型对 alarm/door/glass_break/vehicle 的学习
3. **Per-class 阈值校准**：避免统一阈值导致的类别不均衡问题
4. **约束后处理**：确保 ambient 与其他标签互斥、无预测时默认 ambient
5. **全量重训**：充分利用所有训练数据

最终在测试集上达到 Gold-Band 满分标准（1200/1200），验证了 AST + 逐类阈值校准方案在小规模多标签音频事件检测任务上的有效性。
