# NS-2026-03 防线之内：Prompt Injection 与内容安全检测

难度 Moderate ｜ 总分 800 ｜ 题型：Agent 安全 6 分类 + 风险分排序（比赛时间 2026/5/23 08:00 – 5/28 08:00）

> 本题没有 writeup，`neuronspark2026-answer/` 里也没有最终包。下文的赛题信息和分数来自对话记录 conversation-2026-05-25-03.txt<sup>†</sup>（里面贴了题面和平台 score details），做法来自代码。

## 1. 题目介绍

- **任务**：扮演 Agent 安全巡检员。给定 `system`（系统提示摘要）、`tools`（可用工具）、`history`（历史对话摘要）和 `user_input`，预测风险类别 `label` 并输出 `risk_score`∈[0,1]。
- **类别**（定义见 [label_schema.json](../../neuronspark/neuronspark2026/NS-2026-03/label_schema.json)）：`benign`、`prompt_injection`、`data_exfiltration`、`tool_abuse`、`jailbreak`、`ambiguous`。前 4 个非 benign 类是高风险类。schema 给了每类 risk_score 的建议区间（benign 0–0.30，ambiguous 0.35–0.65，高风险 0.70/0.75–1.00）。
- **数据**：train.jsonl<sup>†</sup> 960 条（benign 220 / prompt_injection 150 / data_exfiltration 150 / tool_abuse 150 / jailbreak 130 / ambiguous 160），test.jsonl<sup>†</sup> 430 条（无标签）。测试集里有新攻击模板、依赖上下文的样本、看起来像正常请求的样本、中英混合和混淆字符。训练集只有 5 种 system 场景，模板重复度很高。
- **评分**：Macro F1（6 类）×500 + 高风险 Recall（必须判到正确的高风险子类）×200 + AUC（高风险 vs 非高风险，用 risk_score 排序）×100。
- **提交**：`NS-2026-03-answer.zip`，里面只放 `results.json`。格式为 `[{"id","label","risk_score"}]`，每个测试 id 恰好出现一次，不能有多余字段。
- **规则变更（关键）**：比赛中途（5/25 看到的新版题面）加了限制：**禁止把测试样本上传到闭源 LLM、商业大模型 API 或托管 Agent 逐条判别**。允许规则系统、传统机器学习、开源预训练模型，也允许用 AI 生成增强样本（writeup 里必须说明）。AI 协助等级 A1。

## 2. 我们的做法

整个过程分三个阶段：

**阶段 A：闭源 LLM few-shot（规则变更前，5/23）**
- [classify.py](../../neuronspark/neuronspark2026/classify.py) / [run_pro.py](../../neuronspark/neuronspark2026/run_pro.py)：通过 OpenAI 兼容的 LLM 网关调用 DeepSeek V4。system prompt 里放 6 类定义、risk_score 区间、10 条判别要点和每类 2 个 few-shot 示例。每批送 5 条测试样本，模型输出 JSON 格式的 label + risk_score，用 `classify_progress.jsonl` 断点续跑。
- Flash 模型拿到 724，换成 Pro 模型拿到 **752**，也是平台显示的本题最高分。

**阶段 B：合规的本地开源模型（5/25–5/26，服务器 A40 + 3090）**
- v1 [run_quick.sh](../../neuronspark/neuronspark2026/NS-2026-03/run_quick.sh)：chinese-roberta-wwm-ext 做 5-fold 微调（[train_bert.py](../../neuronspark/neuronspark2026/NS-2026-03/train_bert.py)，把 system/tools/history/user_input 拼成输入），再加规则后处理，risk_score 用固定值，得 530。
- v2 [run_v2_bert.sh](../../neuronspark/neuronspark2026/NS-2026-03/run_v2_bert.sh) / [run_v2_qwen.sh](../../neuronspark/neuronspark2026/NS-2026-03/run_v2_qwen.sh)：关掉过于激进的规则，risk_score 改用高风险类概率之和（连续值）。另训 Qwen3-8B LoRA 5-fold（[train_qwen.py](../../neuronspark/neuronspark2026/NS-2026-03/train_qwen.py)），用 [infer.py](../../neuronspark/neuronspark2026/NS-2026-03/infer.py) 和 BERT 加权融合，得 602–611。
- v3 [recalibrate.py](../../neuronspark/neuronspark2026/NS-2026-03/recalibrate.py)：不重训，只调决策阈值：判 ambiguous 需要更高置信度，同时提高 benign 召回，得 618。
- v4 [run_v4.sh](../../neuronspark/neuronspark2026/NS-2026-03/run_v4.sh)：用本地 Qwen3-8B 生成增强样本（[generate_aug.py](../../neuronspark/neuronspark2026/NS-2026-03/generate_aug.py)，覆盖中英混合、混淆字符、隐晦表达、反讽、上下文依赖），重训 BERT 后和 Qwen3-8B 融合（0.4/0.6），得 **641**，是合规路线里分最高的一版。高风险 Recall 从 0.78 升到 0.88。
- v5 [infer_v5.py](../../neuronspark/neuronspark2026/NS-2026-03/infer_v5.py) 三路融合加 benign boost，得 640。v6 [gen_benign_boost.py](../../neuronspark/neuronspark2026/NS-2026-03/gen_benign_boost.py) 额外生成 300+ 条 benign，反而掉到 604。v7 [classify_zeroshot.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_zeroshot.py) 用 Qwen3-8B zero-shot，得 583（AUC 只有 0.83）。v8 [run_v8.sh](../../neuronspark/neuronspark2026/NS-2026-03/run_v8.sh) 全量 LoRA，分数未记录。[infer_qwen14b.py](../../neuronspark/neuronspark2026/NS-2026-03/infer_qwen14b.py) 和 [train_bert_v3.py](../../neuronspark/neuronspark2026/NS-2026-03/train_bert_v3.py) 也没有分数记录。
- 本地方案卡在 640 左右，原因是测试集模板跟训练集差别大：BERT 训练集 OOF 接近 0.97，到测试集 Macro F1 只有约 0.74；benign 预测数量远低于预期，遇到陌生表达就退回 ambiguous 或高风险。

**阶段 C：回到 API 冲分（5/26–5/27）**
- 先试 GLM 5.1（[classify_glm.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_glm.py) 得 678，[classify_glm_v2.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_glm_v2.py) 得 693），再试 DeepSeek Pro 长 prompt（[classify_dspro_v2.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_dspro_v2.py) 得 692）。两者都不如 v1 的简单 prompt。
- [plan_770.py](../../neuronspark/neuronspark2026/NS-2026-03/plan_770.py)：`--mode selfconsist` 做多次运行投票；`--mode ensemble` 以 DS Pro v1 为底，融合 GLM 和 DS v2 的结果；`--mode refined` 只在 752 prompt 上做最小改动（加强 ambiguous 和 benign 安全讨论的识别）。[run_hightemp.py](../../neuronspark/neuronspark2026/NS-2026-03/run_hightemp.py) 再加 2 次 temperature=0.5 的运行，做 5 票投票（run1 权重翻倍）。
- 5/27 生成的 rules-nb / nb / fusion / template / refined-fixedrisk / nodowngrade 几个结果，**对应脚本没有留在仓库里**。从文件内容看：nb 和 rules-nb 是"规则 + 朴素贝叶斯"类的结果，跟 DS Pro v1 的标签相差 183 和 238 条；其余几个都在 DS Pro v1 / refined 的标签上做少量改动（相差 11–19 条），risk_score 改成每类一个固定值（benign 0.14 / ambiguous 0.55 / PI 0.88 / TA 0.90 / DE 0.91 / JB 0.92）。

## 3. 结果

平台分数，来源是对话记录里的 score details，以及 [plan_770.py](../../neuronspark/neuronspark2026/NS-2026-03/plan_770.py) 注释（GLM v2 那一行）：

| 版本 | 提交包 | Macro F1 | HR Recall | AUC | 总分 |
|---|---|---|---|---|---|
| DS V4 Flash API few-shot | （已被 Pro 版覆盖） | 0.8695 | 0.9481 | 0.9910 | 724 |
| **DS V4 Pro API few-shot（平台最高）** | [NS-2026-03-answer.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer.zip) | – | – | – | **752** |
| BERT v1 + 规则 | [answer-bert.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-bert.zip) | 0.6335 | 0.7148 | 0.7041 | 530 |
| BERT v2 | [answer-bert-v2.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-bert-v2.zip) | 0.7099 | 0.7778 | 0.9458 | 605 |
| Qwen3-8B LoRA only | [answer-qwen-only.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-qwen-only.zip) | – | – | – | 602 |
| Ensemble v2 | [answer-ensemble-v2.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-ensemble-v2.zip) | 0.7253 | 0.7630 | 0.9623 | 611 |
| v3 重校准 | [answer-v3.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-v3.zip) | 0.7336 | 0.7778 | 0.9554 | 618 |
| **v4 增强 + 融合（合规最高）** | [answer-ensemble_v4.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-ensemble_v4.zip) | 0.7398 | 0.8778 | 0.9553 | **641** |
| v5 三路融合 | [answer-v5.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-v5.zip) | 0.7490 | 0.8630 | 0.9342 | 640 |
| v6 benign 增强 | [answer-v6.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-v6.zip) | 0.7230 | 0.7259 | 0.9634 | 604 |
| v7-fixed Qwen3-8B zero-shot | [answer-v7-fixed.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer-v7-fixed.zip) | 0.6968 | 0.7593 | 0.8302 | 583 |
| GLM 5.1 v1 / GLM v2 / DS Pro v2 | [glm](../../neuronspark/neuronspark2026/NS-2026-03/NS-2026-03-answer-glm.zip) / [glm-v2](../../neuronspark/neuronspark2026/NS-2026-03/NS-2026-03-answer-glm-v2.zip) / [dspro-v2](../../neuronspark/neuronspark2026/NS-2026-03/NS-2026-03-answer-dspro-v2.zip) | – | – | – | 678 / 693 / 692 |

其余提交包（bert_v4、qwen4b_*、plan_770 系列、5/27 系列）的分数都未记录。

**最终提交（推断）**：平台记录的最高分是 752，对应仓库根目录 [NS-2026-03-answer.zip](../../neuronspark/neuronspark2026/NS-2026-03-answer.zip)（5/23 12:01 打包，内容与 [results.json](../../neuronspark/neuronspark2026/results.json) 一致）。但这个方案违反了后来更新的规则。合规方案的最高分是 641（ensemble_v4）。仓库里最晚的产物是 [NS-2026-03-answer-nodowngrade.zip](../../neuronspark/neuronspark2026/NS-2026-03/NS-2026-03-answer-nodowngrade.zip)（5/27 11:36，与 [NS-2026-03/results.json](../../neuronspark/neuronspark2026/NS-2026-03/results.json) 一致），分数未记录，生成脚本也没留下。

## 4. 目录与文件说明

- 2026-03.zip<sup>†</sup>：原始题目包，包含 train/test/label_schema/example_submission/tools/check_format.py。
- [NS-2026-03/](../../neuronspark/neuronspark2026/NS-2026-03/)：数据和所有代码。
  - 环境：[setup_env.sh](../../neuronspark/neuronspark2026/NS-2026-03/setup_env.sh)、[download_models.py](../../neuronspark/neuronspark2026/NS-2026-03/download_models.py)
  - BERT：[train_bert.py](../../neuronspark/neuronspark2026/NS-2026-03/train_bert.py)、[train_bert_v3.py](../../neuronspark/neuronspark2026/NS-2026-03/train_bert_v3.py)
  - Qwen：[train_qwen.py](../../neuronspark/neuronspark2026/NS-2026-03/train_qwen.py)（LoRA，支持 `--full_train`）、[classify_zeroshot.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_zeroshot.py)、[infer_qwen14b.py](../../neuronspark/neuronspark2026/NS-2026-03/infer_qwen14b.py)
  - 推理和融合：[infer.py](../../neuronspark/neuronspark2026/NS-2026-03/infer.py)（有 `--use_bert/--use_qwen/--use_rules` 开关）、[infer_v5.py](../../neuronspark/neuronspark2026/NS-2026-03/infer_v5.py)、[recalibrate.py](../../neuronspark/neuronspark2026/NS-2026-03/recalibrate.py)
  - 数据增强：[generate_aug.py](../../neuronspark/neuronspark2026/NS-2026-03/generate_aug.py)、[gen_benign_boost.py](../../neuronspark/neuronspark2026/NS-2026-03/gen_benign_boost.py)
  - 一键脚本：`run_quick.sh`、`run_all.sh`、`run_qwen_only.sh`、`run_v2_*.sh`、`run_v3.sh`～`run_v8.sh`
  - API 方案：[classify_glm.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_glm.py)、[classify_glm_v2.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_glm_v2.py)、[classify_dspro_v2.py](../../neuronspark/neuronspark2026/NS-2026-03/classify_dspro_v2.py)、[plan_770.py](../../neuronspark/neuronspark2026/NS-2026-03/plan_770.py)、[run_hightemp.py](../../neuronspark/neuronspark2026/NS-2026-03/run_hightemp.py)
  - 结果文件：`results_*.json`、`*_progress.jsonl`（API 断点）、`sc_progress_run1-5.jsonl`（自洽投票）、`train_eval_progress.jsonl`（100 条训练样本的预测，和训练标签对比 95/100 一致，生成脚本未留存），以及 `NS-2026-03-answer-*.zip`。
- [classify.py](../../neuronspark/neuronspark2026/classify.py)、[run_pro.py](../../neuronspark/neuronspark2026/run_pro.py)、classify_progress.jsonl<sup>†</sup>：752 分方案的脚本和断点文件，放在 `neuronspark2026/` 根目录。
- `neuronspark2026/NS-2026-03-answer-*.zip`：阶段 B 各版本的提交包（从服务器 scp 回来的）。

## 5. 如何运行

**注意事项**
- API 脚本（`classify.py`、`run_pro.py`、`classify_glm*.py`、`classify_dspro_v2.py`、`plan_770.py`）通过 OpenAI 兼容的 chat completions 接口调用模型，接口地址与凭证均从环境变量读取，运行前先 `export LLM_API_URL=<接口地址> ACTOKEN=<你的 token>`。另外按更新后的规则，这条路线不合规。
- 本地方案的脚本按服务器路径写死：`/data2/css2025/NS-2026-03`（工作目录）、`/data2/css2025/models/Qwen/{Qwen3-8B,Qwen2.5-7B-Instruct,Qwen3-14B}`；GPU 映射为 `CUDA_VISIBLE_DEVICES=0` 对应 A40，`1` 对应 3090。换环境时要改 `--model_path/--qwen_base/QWEN_PATH` 和 `infer_qwen14b.py` 里的 `MODEL_PATH`。
- 训练产物（`bert_out/`、`bert_v2/`、`bert_v4/`、`qwen3_8b_out/`、`train_aug.jsonl`、`benign_extra.jsonl`、`infer_debug/`）都不在仓库里，需要重新跑。
- 各脚本都会覆盖当前目录下的 `results.json`。

**合规本地方案（复现 v4 / 641）**，在 `NS-2026-03/` 目录下按顺序执行：
```bash
bash setup_env.sh            # transformers/peft/accelerate/datasets/sklearn
bash run_quick.sh            # 从 ModelScope 下载 chinese-roberta-wwm-ext 到 models/，顺带跑 BERT v1
bash run_v2_bert.sh          # BERT v2 -> bert_v2/
bash run_v2_qwen.sh          # Qwen3-8B LoRA 5-fold -> qwen3_8b_out/，与 BERT v2 融合
bash run_v4.sh               # Qwen3-8B 生成 train_aug.jsonl -> 重训 bert_v4 -> 融合，输出 NS-2026-03-answer-ensemble_v4.zip
```

**API 方案（752，仅作记录）**：`classify.py` 从当前目录读 `train.jsonl/test.jsonl`，写出 `results.json` 和 `classify_progress.jsonl`。现在 `neuronspark2026/` 根目录下没有数据文件，需要在有数据的目录下运行，跑完后执行 `zip NS-2026-03-answer.zip results.json`。

**格式检查**（`tools/` 只在原始包里，可以先解压到 /tmp）：
```bash
unzip -o neuronspark/neuronspark2026/2026-03.zip tools/check_format.py -d /tmp/ns03
python /tmp/ns03/tools/check_format.py results.json --test-jsonl test.jsonl
```

## 6. 参考

- 对话记录（题面、各版本分数、规则变更）：conversation-2026-05-25-03.txt<sup>†</sup>
- 数据集：https://modelscope.cn/datasets/SteamedFresh/NS-2026-A-data （题面给出的地址）
- Writeup：无（`writeup_submit/` 里没有 03 目录）

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
