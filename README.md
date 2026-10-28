# NeuronSpark 2026 Champion Solutions

中山大学 AI 技术协会在 NeuronSpark 2026 AI 竞赛中的解题方案：10 道赛题（NS-2026-00 ~ 09）的解题代码、各版本提交包与 writeup。

> 本仓库为公开版：按各题数据协议，**不包含比赛原始数据**（题目数据包、train/test/valid 数据、模型权重等），也不包含解题过程的对话记录。复现时请从官方渠道获取题目数据。

每道题的详细说明（题目介绍 / 解题思路 / 结果 / 运行方式）见 [`problems/`](problems/) 下对应的 README。

## 赛题总览

| 题号 | 题目 | 方向 | 最终方案（概要） | 线上分数 |
|---|---|---|---|---|
| [00](problems/NS-2026-00/README.md) | 星图识别员：校园票据分类 | 图像分类 | EfficientNet-B2 迁移学习 | 500 / 500 |
| [01](problems/NS-2026-01/README.md) | 智慧饭堂：人流与备餐预测 | 表格时序回归 | Ridge 拟合乘法结构 + LightGBM 残差，5 seed 平均 | 489.92 / 500 |
| [02](problems/NS-2026-02/README.md) | 图书馆问答员：带引用的 RAG 问答 | RAG | 模板路由 + 正则字段抽取 + BM25 + 本地 Qwen3-4B (MLX) 精化 | 778.80 / 800 |
| [03](problems/NS-2026-03/README.md) | 防线之内：Prompt Injection 与内容安全检测 | 文本安全分类 | LLM few-shot（最高 752）；合规本地方案 BERT + Qwen3-8B LoRA 融合（641） | 752 / 800 * |
| [04](problems/NS-2026-04/README.md) | 分子炼金术：小分子性质预测 | 分子回归 | hashed fingerprint + 手工描述符，LightGBM 30 模型集成 | 800 / 800 |
| [05](problems/NS-2026-05/README.md) | 声场巡检员：校园声景事件检测 | 音频多标签 | AST (AudioSet) 全参数微调 + 逐类阈值 | 1200 / 1200 |
| [06](problems/NS-2026-06/README.md) | 标注净化师：噪声标签清洗与鲁棒分类 | 噪声标签学习 | source_group→label 映射 + LogisticRegression + 类别倍率 | 1061.73 / 1200 |
| [07](problems/NS-2026-07/README.md) | 触觉调度者：灵巧手闭环仿真控制 | 控制 / 仿真 | 规则状态机控制器（brace→tool→move→finish） | 1411 / 1500 |
| [08](problems/NS-2026-08/README.md) | 星云值班台：黑箱遥测世界模型闭环处置 | AIOps Agent | belief state + CSP 故障推断 + 探针 + 风险门控 | 1336.43 / 1500 |
| [09](problems/NS-2026-09/README.md) | 观测之环：隐变量世界模型反事实预测 | 世界模型 | 手写网格模拟器 + 280 种隐变量 profile 匹配 + 反事实 rollout | 1404.92 / 1500 |

\* 03 题 752 分方案调用闭源 API 逐条判别，规则更新后不再合规；合规路线最高 641。详见该题 README。

分数均来自各题 writeup / 解题过程记录，具体来源见各题 README。

## 快速开始

```bash
git clone https://github.com/cscsysu/neuronspark2026-champion.git
cd neuronspark2026-champion
# 查看某道题的说明，例如：
cat problems/NS-2026-09/README.md
```

每道题的 README 都给出了依赖、数据解压位置、训练 / 推理 / 打包 / 格式检查命令。通用约定：

- 原始题目数据（官方 `2026-XX.zip`）未包含在本仓库，获取后建议解压到独立目录，再按各题 README 修改脚本中的数据路径运行。
- 提交格式统一为 `NS-2026-XX-answer.zip`，各题 `tools/check_format.py` 可本地校验。
- Python 3.10+；表格 / 传统 ML 题依赖 `numpy pandas scikit-learn lightgbm`，深度学习题依赖 `torch transformers timm`，02 题在 Apple Silicon 上使用 `mlx-lm`，07 / 08 / 09 仅需标准库。

## 目录结构

```
.
├── README.md                     # 本文件
├── problems/NS-2026-XX/README.md # 每道题的说明文档
└── neuronspark/
    ├── neuronspark2026/          # 主工作区：各题代码与提交包
    │   ├── NS-2026-02|03|05|07|08|09/   # 对应题目的工作目录
    │   ├── 2026-00-data/, 2026-01/      # 00、01 题代码与提交结果
    │   ├── *.py                          # 根目录脚本混合了 00/03/04 等题，见各题 README
    │   └── writeup_submit/XX/            # 各题提交的 writeup
    ├── neuronspark2026-answer/   # 各题最终提交包 NS-2026-XX-answer.zip
    ├── 02/                       # 02 题最终方案（solution/）
    ├── 06/                       # 06 题最终方案与 writeup
    ├── 08/                       # 08 题题面、checker、最终 agent
    └── 09/                       # 09 题题面与最终 solver
```

## 注意事项

- 各题 README 中带 † 标注的文件（原始数据、权重、对话记录）未包含在公开版中。
- `.venv/` 未纳入版本库；02 题需按其 README 重新创建环境。
- 不少脚本硬编码了作者本机或服务器的绝对路径（如 `/Users/tristonchen/Desktop/...`、`/data2/css2025/...`），运行前需按各题 README 修改。
- 03 题调用 LLM 网关的脚本（`classify.py`、`run_pro.py`、`NS-2026-03/classify_*.py`、`plan_770.py`）不含任何凭证与内部地址，运行前需自行设置环境变量 `LLM_API_URL`（OpenAI 兼容的 chat completions 接口）和 `ACTOKEN`（访问凭证）。
- 数据集遵循各题 `DATA_NOTICE.md`：仅限 NeuronSpark 2026 竞赛、练习与 writeup 复现使用，请勿再分发。
