# NS-2026-02 图书馆问答员：带引用的 RAG 知识库问答

> 总分 800（答案 400 + 引用 250 + 拒答 100 + 格式 50）｜题型：中文 RAG 问答（带引用 / 拒答）｜最终线上 **778.80 / 800**｜难度：未记录

## 1. 题目介绍

- **任务**：给定校园制度知识库（图书馆借阅、实验室安全、设备借用、社团章程等），对每个问题输出简短答案 + 可验证的 `chunk_id` 引用；知识库无法回答时输出固定文本 `无法根据给定知识库回答` 且 citations 为空。
- **数据**（原始数据目录<sup>†</sup>）：
  - `corpus.jsonl`：90 个 chunk，字段 `chunk_id / doc_id / title / text`；含 2 个 prompt-injection 陷阱 chunk `SAFE-INJECT-001/002`
  - `train_qa.jsonl`：468 道带答案题（`id / question / answer / answerable / citations`，含 20 道不可答）
  - `test_questions.jsonl`：332 道（仅 `id / question`），覆盖事实、条件、多跳、否定、冲突、不可答、注入干扰
- **输出**：`results.json`，数组，每项恰好 `{id, answer, citations}`，citations ≤ 5（见 `schema.json`、`tools/check_format.py`）；打包为 `NS-2026-02-answer.zip`。
- **评分**（`tools/eval_public.py`）：
  - 答案：与 gold/aliases 的**字符级 F1**（或精确匹配）取最大 → 400 分
  - 引用：`|gold ∩ pred| / min(2, |gold|)`，只看前 5 个预测、最多需命中 2 个 gold → 250 分；不可答题 citations 为空才得分
  - 拒答：不可答题答案包含拒答文本的比例 → 100 分；格式 50 分
- **赛规**（据 writeup）：仅允许本地开源模型，生成式 ≤ 4B、embedding/rerank ≤ 1B，禁止闭源 API。

## 2. 我们的做法

最终方案位于 [`neuronspark/02/solution/`](../../neuronspark/02/solution/)：**规则化模板路由 + BM25 检索 + 三层拒答 + 本地 Qwen3-4B（mlx，4-bit）只精化非模板题答案**。

关键洞察（细节见 writeup）：
1. **评分偏好短答案**：char-F1 惩罚冗长，答案精炼到"数字+单位 / 地点 / 可以·不能"；citations 只给 top-2。
2. **语料高度模板化**：22 个 `*-ACCESS-*` chunk（面向对象/申请时限/地点/时长）+ 22 个 `*-CHECK-*` chunk（材料/取消/后果），正则 7 个字段 100% 命中。
3. **测试题 174/332（52%）是 9 个固定问法模板**（含 3 种多跳），且训练集同 (模板, 实体) 的答案完全一致 → 先查 train 表，未命中再按字段抽取。
4. **注入陷阱**：`SAFE-INJECT-001/002` 永久黑名单，不进检索和引用。

流水线（`src/pipeline.py` 的 `answer_question`）：
1. 实体覆盖率 < 0.34 → 拒答
2. `parse_template` 命中 → train 查表 / ACCESS·CHECK 字段抽取（`src/templates.py`、`src/template_extractor.py`）
3. 非模板题：BM25（jieba + char-bigram，`src/retriever.py`）→ 黑名单过滤 → 二次拒答（实体在 top-1 命中率低）→ 强抽检测（答案含"主要面向/提交时必须提供"等模板话术但问题没问 → 拒答）
4. 答案抽取：规则 `src/sentence_extractor.py`（是否/时间/单位数字/地点）；`--llm` 时由 `src/mlx_answerer.py` 调 Qwen3-4B（few-shot、关闭 thinking、max_tokens=40），异常或"无法回答"时回退规则

迭代（writeup 第 8 节，Test 为线上分）：

| 版本 | 改动 | Test |
|---|---|---|
| v0 | 数据画像/探索 | 686 |
| v1 | (template, entity) train 查表 | 700 |
| v2 | 9 模板 + 字段抽取 | 727 |
| v3 / v3.1 | 强抽检测抓 3 道漏拒；答案精简 | 741 / 744 |
| v4 | prompt 加 few-shot | 746 |
| v5 | + Qwen2.5-1.5B（mlx） | 769 |
| v6 | 换 Qwen3-4B-Instruct-2507 | 774 |
| v7 | Qwen3.5-4B（holdout 更高但线上下降，回退） | 771 |
| **v8** | **强化拒答（模板话术改 contains 判断）+ Qwen3-4B** | **778.80** |

> **另一套方案（未作为最终提交）**：[`neuronspark/neuronspark2026/NS-2026-02/`](../../neuronspark/neuronspark2026/NS-2026-02/) 是更早的 GPU 服务器方案：BM25 + 训练集相似题匹配 + Qwen2.5-3B / Qwen3.5-4B QLoRA 微调 + Self-Consistency，其 writeup 自报 702/800。其 `results.json` 与最终提交包内容不同。

## 3. 结果

| 来源 | 答案 | 引用 | 拒答 | 格式 | 总分 |
|---|---|---|---|---|---|
| 线上最终（writeup_submit/02） | 377.59 | 249.25 | 97.14 | 50 | **778.80** |
| train 全集自评（solution/README，早期纯规则版） | 374.3 | 250.0 | 100 | 50 | 774.3 |
| train holdout 80/20（solution/README，早期纯规则版） | — | — | — | — | 770.2 |
| train holdout（writeup，v8） | — | — | — | — | 797 |

最终提交包 SHA256 `835408a0…6318f0` 与 writeup 记录一致（已用 shasum 核对）。

## 4. 目录与文件说明

**最终方案** [`neuronspark/02/solution/`](../../neuronspark/02/solution/)
- [`run_pipeline.py`](../../neuronspark/02/solution/run_pipeline.py) — **最终推理入口**；默认纯规则写 `outputs/results.json`，`--llm` 写 `outputs/results_llm.json`
- [`src/`](../../neuronspark/02/solution/src/) — `data.py`（数据加载、拒答文本）、`templates.py`、`template_extractor.py`、`retriever.py`、`sentence_extractor.py`、`mlx_answerer.py`、`pipeline.py`
- [`validate_on_train.py`](../../neuronspark/02/solution/validate_on_train.py) — train 全集 + 80/20 holdout 自评（调用官方 eval）
- [`eval_full.py`](../../neuronspark/02/solution/eval_full.py) / [`eval_llm.py`](../../neuronspark/02/solution/eval_llm.py) — holdout 上规则 vs +LLM 对比
- [`README.md`](../../neuronspark/02/solution/README.md) — 方案原始说明（早期纯规则版）
- [`outputs/NS-2026-02-answer.zip`](../../neuronspark/02/solution/outputs/NS-2026-02-answer.zip) — **最终提交包**；`outputs/results.json` 与 `results_llm.json` 内容相同（v8 LLM 版）；`results_v8_rule.json` 为纯规则对照；`preds_*`、`ref_*`、`p4b.json`、`p15.json` 为验证/中间产物
- `neuronspark/02/solution.zip`<sup>†</sup> — 方案目录的压缩备份

**数据与提交**
- `neuronspark/02/1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/`<sup>†</sup> — 原始数据 + `tools/check_format.py`、`tools/eval_public.py`、`example_submission/`（同名 `.zip` 为原始压缩包）
- [`neuronspark/neuronspark2026-answer/NS-2026-02-answer.zip`](../../neuronspark/neuronspark2026-answer/NS-2026-02-answer.zip) — 归档的最终提交（与 solution/outputs 下的包逐字节相同）
- `neuronspark/neuronspark2026/2026-02.zip`<sup>†</sup> — 仅含示例提交 `results.json`（全拒答样例）

**备选方案（非最终）** [`neuronspark/neuronspark2026/NS-2026-02/`](../../neuronspark/neuronspark2026/NS-2026-02/)
- `solve_best.py`（Qwen2.5-3B）、`solve_final.py`（Qwen3.5-4B + LoRA）、`prepare_lora_data.py`、`train_lora.py`、`run_all.sh`、`setup_and_run.sh`、`writeup.md`（702 分版）、`NS-2026-02-answer.zip`
- 根目录 `neuronspark/neuronspark2026/NS-2026-02-answer.zip`、`NS-2026-02-answer(5).zip` 也是该线的历史包，非最终

## 5. 如何运行（最终方案）

环境：Apple Silicon Mac（mlx 需要 Metal），Python 3.13；writeup 记录 `jieba 0.42.1`、`mlx-lm 0.31.3`。纯规则路径只需 jieba。

```bash
cd "neuronspark/02/solution"
python3 -m venv .venv && source .venv/bin/activate
pip install jieba mlx-lm

# 数据路径由 src/data.py 自动解析为 ../1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/，无需移动
# 1) train 自评
python validate_on_train.py

# 2) 推理（最终版需 --llm，首次会下载 ~2.5GB 模型到 ~/.cache/huggingface）
NS_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit python run_pipeline.py --llm
#   → outputs/results_llm.json；纯规则：python run_pipeline.py → outputs/results.json

# 3) 格式检查
python ../1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/tools/check_format.py outputs/results_llm.json \
  --questions-jsonl ../1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/test_questions.jsonl

# 4) 打包（zip 内文件名须为 results.json）
cp outputs/results_llm.json outputs/results.json
cd outputs && zip NS-2026-02-answer.zip results.json
```

注意：
- `--llm` 输出到 `results_llm.json` 而非 `results.json`，打包前需复制/改名（最终包即如此得到）。
- `eval_full.py`、`eval_llm.py` 硬编码了 `/Users/tristonchen/Desktop/02/solution/.venv/bin/python`（旧路径，当前仓库不存在），运行前需改为本机 Python；各脚本 docstring 中的 `cd /Users/tristonchen/Desktop/02/solution` 同样是旧路径。
- 备选方案脚本将模型缓存硬编码到 `/data2/css2025/models`（ModelScope `snapshot_download`），需 CUDA GPU。

## 6. 参考

- 最终 writeup：[`neuronspark/neuronspark2026/writeup_submit/02/writeup.md`](../../neuronspark/neuronspark2026/writeup_submit/02/writeup.md)（778.80 分，含 prompt、拒答策略、案例分析）
- 方案说明：[`neuronspark/02/solution/README.md`](../../neuronspark/02/solution/README.md)
- 备选方案 writeup：[`neuronspark/neuronspark2026/NS-2026-02/writeup.md`](../../neuronspark/neuronspark2026/NS-2026-02/writeup.md)（702 分）

---

† 标注的文件为比赛原始数据或过程记录，按数据协议未包含在公开仓库中，请从官方渠道获取题目数据包。
