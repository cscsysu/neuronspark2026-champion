# NS-2026-02 图书馆问答员 RAG — 中山大学AI技术协会方案

## 📊 当前自评分（在官方 train_qa.jsonl 上）
```
答案准确性: 374.3 / 400  (93.6%)
引用命中率: 250.0 / 250  (100%)
拒答识别  : 100.0 / 100  (100%)
格式合法性: 50.0  / 50   (100%)
─────────────────────────────────
总分      : 774.3 / 800  (96.8%)

Holdout 80/20 评测: 770.2 / 800  (96.3%)
```

## 🎯 核心思路

这道题语料只有 90 个 chunk，看似探索型 RAG，其实**高度模板化**：

1. **44 个核心 chunk 是 ACCESS/CHECK 字段化文本**（22 + 22）
   - ACCESS: `主要面向X。预约或申请须Y完成，办理或使用地点为Z。单次使用或处理时长为W。`
   - CHECK : `提交时必须提供M。如需取消，须C办理。未按规则使用的后果是R。`
   - **44 个 chunk 全部 7 个字段在正则下 100% 命中**
2. **测试题 332 道里 174 道（52%）是 9 个固定模板**，对应 chunk_id 后缀完全确定：
   - `如需取消/未按规则后果/提交材料` → CHECK
   - `办理地点/提前多久申请/单次时长` → ACCESS
   - `办理前材料和服务对象` → CHECK + ACCESS（多跳）
3. **训练集中同 (template, entity) 的答案 100% 一致**
   - 对模板题，先在 train 中查表 → 不命中再走 corpus 字段抽取
4. **官方 eval_public.py 的字符级 char_f1**：答案不必精确到一个词，包含 gold 的关键字符就有部分分；citations 只看前 2 个 gold

## 🚀 GPU 是否需要？

**不需要 GPU**。整个流水线纯 Python + jieba + BM25 + 正则，332 道测试题在 MacBook 上 < 5 秒跑完。

如果你之后想再加几分，可挂上 `BAAI/bge-small-zh-v1.5`（95M，CPU 也能跑）做 dense rerank，但当前规则路径已经能拿到 770+ 分。

## 📁 项目结构
```
solution/
├── src/
│   ├── data.py                  # 数据加载、归一化、REFUSAL 文本
│   ├── templates.py             # 9 个题型模板 + entity 抽取
│   ├── template_extractor.py    # CHECK/ACCESS 字段正则抽取
│   ├── retriever.py             # BM25 (jieba 分词 + char-bigram)
│   ├── sentence_extractor.py    # 非模板题：单位/地点/是否/时间抽取
│   └── pipeline.py              # 主流水线 (拒答 → 模板 → BM25)
├── outputs/
│   ├── results.json             # 332 题预测
│   └── NS-2026-02-answer.zip    # 提交包 ✅ 已通过官方 check_format
├── run_pipeline.py              # 跑测试集
└── validate_on_train.py         # 在 train 上自评（含 holdout）
```

## 🧠 流水线
```
对每个问题 q:
  ① 实体覆盖率检查 (问题中的实体词在 corpus 中能否找到)
       命中率 < 1/3 → 拒答 "无法根据给定知识库回答" + []
  ② 模板路由 (parse_template)：
       a) 同 (template, entity) 在 train 中存在  → 复用答案 + citations
       b) 否则按 template→tag 抽取 corpus 字段:
          - ACCESS / CHECK / 多跳 (CHECK_MULTI / ACCESS_MULTI / CHECK_ACCESS_MULTI)
  ③ 非模板题 → BM25 top-k → 黑名单过滤 (SAFE-INJECT 陷阱) → 句子级抽取
       - 是否问题 → "可以"/"不能"
       - "几点" → 含目标语义的最大/最小时间值
       - "多久/几次/几本" → 按问题中的单位精确抽数字+单位
       - 地点 → 中心/办公室/窗口/系统 实体匹配
       - 介质/方式 → 通过/使用 X 转移/发送/结算
  ④ citations 统一 ≤ 2 个（评分只看前 2 个 gold）
```

## 🛡️ 已修的关键 trick
| 坑 | 处理 |
|---|---|
| `SAFE-INJECT-001/002` 是 prompt-injection 陷阱 chunk | 全程黑名单，永不进 citations |
| 拒答题（corpus 没有的实体） | 实体覆盖率 < 0.34 → 直接拒答；输出严格固定文本 |
| 多 chunk 多数字干扰（"续借2次/14天"） | 问题里识别"几次/几天"等显式单位，best_sentence 阶段强加 prior |
| "几点" 类问题取错时间 | 问题含"截止/归还/关闭/停止" → 取最大时间 |
| 训练里有同模板同 entity 重复 | 建 lookup 表，命中直接复用，避免重复抽取风险 |

## 🏃 运行步骤
```bash
# 1) 安装
pip install jieba

# 2) 在训练集上自评（看上限）
cd /Users/tristonchen/Desktop/02/solution
python3 validate_on_train.py
# 输出 ~774/800 (full) 和 ~770/800 (holdout)

# 3) 跑测试集 → outputs/results.json
python3 run_pipeline.py

# 4) 官方格式校验
python3 /Users/tristonchen/Desktop/02/1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/tools/check_format.py \
    outputs/results.json \
    --questions-jsonl /Users/tristonchen/Desktop/02/1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d/test_questions.jsonl
# 输出: OK: submission format is valid.

# 5) 提交包
ls outputs/NS-2026-02-answer.zip   # ✅ 已生成
```

## 📤 提交
直接上传 `outputs/NS-2026-02-answer.zip`。

## ✅ 满足赛规
| 规则 | 我们的方案 |
|---|---|
| 不得用闭源 API / 远程托管模型 | ✅ 纯本地 Python + jieba |
| 模型规模上限 (生成≤4B, embed/rerank≤1B) | ✅ 没用任何模型，全规则 |
| 可断网复现 | ✅ 仅依赖 jieba 词典 |
| 引用必须可验证 | ✅ 全部 citations 来自 corpus.jsonl 的真实 chunk_id |

## 🔧 如果还想冲分
1. 加 `BAAI/bge-small-zh-v1.5` (95M, CPU 可跑) 做 dense + BM25 RRF 融合，提升非模板题检索
2. 加 `BAAI/bge-reranker-v2-m3` (568M < 1B 合规) 重排 top-20→top-5
3. 在 train 上 fine-tune 几个还没覆盖的 corner case (复合题、否定题)
4. 上 `Qwen2.5-3B-Instruct` (合规 < 4B) 用严格 prompt 抽答案，理论可上 780+
   但需要 ~6GB 显存，量化后 Mac M 系列 也能跑
