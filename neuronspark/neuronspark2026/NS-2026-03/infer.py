"""
NS-2026-03: 推理 + Ensemble + 规则后处理
组合: Qwen2.5-7B LoRA (5 folds) + BERT (5 folds) + 规则
"""
import json
import os
import argparse
import re
import numpy as np
import torch

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

LABELS = ["benign", "prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak", "ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
HIGH_RISK = {"prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"}
RISK_SCORE_RANGE = {
    "benign": (0.05, 0.25),
    "prompt_injection": (0.78, 0.95),
    "data_exfiltration": (0.82, 0.97),
    "tool_abuse": (0.78, 0.95),
    "jailbreak": (0.82, 0.97),
    "ambiguous": (0.42, 0.60),
}

SYSTEM_PROMPT = """你是 Agent 安全分类器。根据给定的系统提示、可用工具、历史对话和当前用户输入，判断用户输入的安全风险类别，并给出 risk_score。

类别定义:
- benign: 正常请求，不要求绕过系统、不索取隐私、不越权调用工具
- prompt_injection: 试图覆盖、泄露、忽略系统指令或要求输出隐藏提示
- data_exfiltration: 诱导泄露令牌、密钥、隐私记录、内部日志、他人信息
- tool_abuse: 请求越权调用工具、跳过审批、绕过门禁、代他人执行操作
- jailbreak: 要求解除安全限制、扮演无约束角色、规避安全分类器
- ambiguous: 存在可疑意图但不足以判为高风险

risk_score 参考范围:
- benign: 0.00-0.30
- ambiguous: 0.35-0.65
- prompt_injection/tool_abuse: 0.70-1.00
- data_exfiltration/jailbreak: 0.75-1.00

只输出 JSON 格式: {"label": "...", "risk_score": 0.xx}"""


def build_qwen_prompt(tokenizer, d):
    tools_str = "\n".join([f"  - {t['name']}: {t['description']}" for t in d["tools"]])
    user_msg = f"""[系统提示]
{d['system']}

[可用工具]
{tools_str}

[历史对话]
{d['history']}

[当前用户输入]
{d['user_input']}

请输出 JSON。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_bert_text(d):
    tools_str = "; ".join([t["name"] + ":" + t["description"] for t in d["tools"]])
    return (
        f"[系统] {d['system']}\n"
        f"[工具] {tools_str}\n"
        f"[历史] {d['history']}\n"
        f"[输入] {d['user_input']}"
    )


def parse_json_output(text):
    text = text.strip()
    m = re.search(r'\{[^}]*"label"[^}]*\}', text, re.DOTALL)
    if not m:
        return {"label": "ambiguous", "risk_score": 0.5}
    try:
        obj = json.loads(m.group(0))
        label = obj.get("label", "ambiguous")
        if label not in LABELS:
            label = "ambiguous"
        rs = obj.get("risk_score", 0.5)
        try:
            rs = float(rs)
            rs = max(0.0, min(1.0, rs))
        except:
            rs = 0.5
        return {"label": label, "risk_score": rs}
    except:
        return {"label": "ambiguous", "risk_score": 0.5}


@torch.no_grad()
def run_qwen_folds(test_data, base_model_path, lora_dirs, gpu):
    """跑 N 个 Qwen LoRA fold，对每条样本投票得到 label 分布"""
    device = f"cuda:{gpu}"
    # 加载 base 一次
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map={"": gpu},
        trust_remote_code=True,
    )

    n_samples = len(test_data)
    # vote count per label
    vote_counts = np.zeros((n_samples, len(LABELS)), dtype=np.float32)
    risk_scores_acc = np.zeros(n_samples, dtype=np.float32)
    risk_count = np.zeros(n_samples, dtype=np.int32)

    for fold_dir in lora_dirs:
        print(f"\n=== Qwen LoRA: {fold_dir} ===")
        model = PeftModel.from_pretrained(base, fold_dir)
        model.eval()

        batch_size = 4
        for i in range(0, n_samples, batch_size):
            batch = test_data[i:i+batch_size]
            prompts = [build_qwen_prompt(tokenizer, d) for d in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=1024).to(device)
            out = model.generate(
                **enc, max_new_tokens=64, do_sample=False, num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )
            for j, ids in enumerate(out):
                gen = ids[enc["input_ids"].shape[1]:]
                text = tokenizer.decode(gen, skip_special_tokens=True)
                parsed = parse_json_output(text)
                idx = i + j
                vote_counts[idx, LABEL2ID[parsed["label"]]] += 1
                risk_scores_acc[idx] += parsed["risk_score"]
                risk_count[idx] += 1
            if (i // batch_size) % 20 == 0:
                print(f"  {i+len(batch)}/{n_samples}")

        # unload LoRA
        model = model.unload()
        del model
        torch.cuda.empty_cache()

    # normalize
    probs = vote_counts / max(1, len(lora_dirs))
    avg_risk = risk_scores_acc / np.maximum(1, risk_count)
    return probs, avg_risk


@torch.no_grad()
def run_bert_folds(test_data, bert_fold_dirs, gpu):
    """跑 N 个 BERT fold，平均 softmax 概率"""
    device = f"cuda:{gpu}"
    n_samples = len(test_data)
    all_probs = np.zeros((n_samples, len(LABELS)), dtype=np.float32)
    for fold_dir in bert_fold_dirs:
        print(f"\n=== BERT: {fold_dir} ===")
        tok = AutoTokenizer.from_pretrained(fold_dir)
        model = AutoModelForSequenceClassification.from_pretrained(fold_dir).to(device).eval()

        batch_size = 32
        for i in range(0, n_samples, batch_size):
            batch = test_data[i:i+batch_size]
            texts = [build_bert_text(d) for d in batch]
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=384).to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs[i:i+len(batch)] += p / len(bert_fold_dirs)

        del model
        torch.cuda.empty_cache()
    return all_probs


# ======== 规则后处理 ========
# 关键词规则：高置信度时强制覆盖
RULES = [
    # prompt_injection - 直接索取/泄露系统指令
    (r"(忽略|无视|ignore).{0,10}(以上|前面|之前|previous).{0,15}(指令|提示|规则|instruction)", "prompt_injection", 0.92),
    (r"(begin|end)[_\s]*system[_\s]*prompt", "prompt_injection", 0.93),
    (r"(输出|打印|show|reveal|display).{0,15}(系统|system).{0,15}(prompt|提示|消息)", "prompt_injection", 0.92),
    (r"(翻译|translate).{0,15}(系统|system).{0,15}(消息|message|prompt)", "prompt_injection", 0.90),
    (r"(隐藏的|hidden).{0,10}(policy|策略|提示|prompt)", "prompt_injection", 0.88),
    (r"以我这条为准", "prompt_injection", 0.78),
    (r"输出.{0,8}(完整|full).{0,8}(system|系统).{0,8}prompt", "prompt_injection", 0.94),
    (r"现在你是管理员", "prompt_injection", 0.88),

    # jailbreak - 要求解除限制/扮演无约束角色
    (r"(DAN\s*模式|进入\s*DAN|DAN mode)", "jailbreak", 0.95),
    (r"(safety\s*filters?\s*(are\s*)?disabled|pretend.{0,15}safety)", "jailbreak", 0.94),
    (r"(开发者|developer)\s*模式.{0,15}(无限制|unlimited|绕过)", "jailbreak", 0.92),
    (r"(扮演|act as).{0,15}(无限制|unlimited|no.{0,5}restrictions?|无约束)", "jailbreak", 0.90),
    (r"(解除|disable|绕过).{0,10}(安全|safety|限制|restriction|filter)", "jailbreak", 0.88),
    (r"内容安全分类器.{0,15}(敌人|绕过|规避)", "jailbreak", 0.92),
    (r"规避.{0,10}(检测|分类器|安全)", "jailbreak", 0.85),
    (r"安全策略无效", "jailbreak", 0.92),

    # data_exfiltration - 索取敏感数据
    (r"(API\s*key|api[_\s]*key|access[_\s]*token|密钥|令牌)", "data_exfiltration", 0.88),
    (r"(Authorization|authorization).{0,15}(字段|header|header)", "data_exfiltration", 0.92),
    (r"(密码|password|口令).{0,15}(数据库|database|表|table|导出|export)", "data_exfiltration", 0.92),
    (r"(打包|export|dump).{0,15}(发到|to|email|邮件|发给)", "data_exfiltration", 0.88),
    (r"(读取|read|查看).{0,15}(上一位|其他|另一|历史)\s*用户", "data_exfiltration", 0.90),
    (r"(发票|银行卡|身份证).{0,8}(数据库|号|信息|明细)", "data_exfiltration", 0.90),
    (r"(隐私|privacy|内部|internal).{0,8}(数据|信息|日志|log|记录)", "data_exfiltration", 0.85),
    (r"门禁(口令|密码|码)", "data_exfiltration", 0.92),

    # tool_abuse - 越权调用工具
    (r"(bypass|跳过|绕过).{0,8}(confirmation|审批|approval|审核)", "tool_abuse", 0.92),
    (r"(不用|无需|跳过).{0,5}审批.{0,15}(调用|直接)", "tool_abuse", 0.90),
    (r"直接调用.{0,15}(工具|tool|api|接口)", "tool_abuse", 0.85),
    (r"(代|帮)\s*(另一|其他|别).{0,8}(用户|user|人).{0,15}(提交|调用|执行)", "tool_abuse", 0.90),
    (r"submit.{0,10}as another user", "tool_abuse", 0.92),
    (r"(临时|temporarily).{0,8}(开|grant|提升).{0,10}(权限|permission|admin)", "tool_abuse", 0.88),
    (r"(不要|don't).{0,10}(留下|leave).{0,10}(记录|trace|log)", "tool_abuse", 0.85),

    # benign - 明显正常
    # 不主动覆盖，让模型决定
]

AMBIGUOUS_PATTERNS = [
    r"(如果|假如|if).{0,30}(哪些|什么|需要).{0,15}(审批|流程|步骤)",
    r"(测试环境|test\s*environment).{0,15}(模拟|simulate)",
    r"(改写建议|rephrase|rewrite).{0,15}(符合|comply|policy|政策)",
    r"^这句话.{0,30}是.{0,5}(违规|是否|不是)",
]


def apply_rules(d, model_label, model_probs, model_risk, use_rules=True):
    """规则后处理。返回 (final_label, final_risk_score, rule_hit)"""
    text = d["user_input"]

    # 计算更有区分度的 risk_score: 用高风险类别概率之和
    high_risk_prob_sum = sum(model_probs[LABEL2ID[l]] for l in HIGH_RISK)

    if use_rules:
        # 1. 强规则 (覆盖模型)
        for pattern, label, conf in RULES:
            if re.search(pattern, text, re.IGNORECASE):
                new_probs = model_probs.copy()
                new_probs[LABEL2ID[label]] = max(new_probs[LABEL2ID[label]], conf)
                new_probs = new_probs / new_probs.sum()
                final_label = LABELS[new_probs.argmax()]
                # risk_score 用高风险概率（连续，AUC 友好）
                final_risk = high_risk_prob_sum if final_label in HIGH_RISK else max(0.6, high_risk_prob_sum)
                if final_label not in HIGH_RISK:
                    final_risk = high_risk_prob_sum * 0.5  # 非高风险给低分
                return final_label, final_risk, f"rule:{pattern[:30]}->{label}"

        # 2. ambiguous 模式
        for pattern in AMBIGUOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                if model_label not in HIGH_RISK:
                    return "ambiguous", 0.5, f"amb:{pattern[:30]}"

    # 3. 不用规则 / 没命中：纯模型
    final_label = model_label
    # risk_score = 高风险概率之和。这样:
    # - benign 高置信 -> high_risk_prob_sum 接近 0
    # - 高风险 -> high_risk_prob_sum 接近 1
    # - ambiguous -> 居中
    if final_label == "benign":
        final_risk = high_risk_prob_sum * 0.5  # 压低
    elif final_label == "ambiguous":
        # ambiguous 自身概率 + 一半高风险概率（保持 AUC 区分度）
        final_risk = 0.3 + 0.4 * high_risk_prob_sum
    else:  # 高风险
        final_risk = max(high_risk_prob_sum, 0.7)

    final_risk = max(0.0, min(1.0, final_risk))
    return final_label, final_risk, "model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="test.jsonl")
    parser.add_argument("--qwen_base", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--qwen_lora_dir", default="qwen_out", help="contains fold_1...fold_5")
    parser.add_argument("--bert_dir", default="bert_out", help="contains fold_1...fold_5")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--qwen_weight", type=float, default=0.5)
    parser.add_argument("--bert_weight", type=float, default=0.5)
    parser.add_argument("--use_qwen", action="store_true")
    parser.add_argument("--use_bert", action="store_true")
    parser.add_argument("--use_rules", action="store_true", help="启用规则后处理")
    parser.add_argument("--debug_dir", default="infer_debug")
    args = parser.parse_args()

    os.makedirs(args.debug_dir, exist_ok=True)

    test_data = []
    with open(args.test_file) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Test: {len(test_data)}")

    n_samples = len(test_data)
    combined_probs = np.zeros((n_samples, len(LABELS)), dtype=np.float32)
    qwen_risk = np.zeros(n_samples, dtype=np.float32)

    # Qwen
    if args.use_qwen:
        lora_dirs = []
        for i in range(1, 6):
            d = os.path.join(args.qwen_lora_dir, f"fold_{i}")
            if os.path.isdir(d):
                lora_dirs.append(d)
        print(f"Qwen LoRA folds found: {len(lora_dirs)}")
        if lora_dirs:
            qprobs, qrisk = run_qwen_folds(test_data, args.qwen_base, lora_dirs, args.gpu)
            np.save(os.path.join(args.debug_dir, "qwen_probs.npy"), qprobs)
            np.save(os.path.join(args.debug_dir, "qwen_risk.npy"), qrisk)
            combined_probs += args.qwen_weight * qprobs
            qwen_risk = qrisk

    # BERT
    if args.use_bert:
        bert_dirs = []
        for i in range(1, 6):
            d = os.path.join(args.bert_dir, f"fold_{i}")
            if os.path.isdir(d):
                bert_dirs.append(d)
        print(f"BERT folds found: {len(bert_dirs)}")
        if bert_dirs:
            bprobs = run_bert_folds(test_data, bert_dirs, args.gpu)
            np.save(os.path.join(args.debug_dir, "bert_probs.npy"), bprobs)
            combined_probs += args.bert_weight * bprobs

    # normalize
    total_weight = (args.qwen_weight if args.use_qwen else 0) + (args.bert_weight if args.use_bert else 0)
    if total_weight > 0:
        combined_probs = combined_probs / total_weight
    np.save(os.path.join(args.debug_dir, "combined_probs.npy"), combined_probs)

    # final predictions
    results = []
    debug_records = []
    rule_hits = 0
    for i, d in enumerate(test_data):
        model_label = LABELS[combined_probs[i].argmax()]
        final_label, final_risk, rule_info = apply_rules(d, model_label, combined_probs[i], qwen_risk[i], use_rules=args.use_rules)
        if rule_info.startswith("rule:") or rule_info.startswith("amb:"):
            rule_hits += 1
        results.append({
            "id": d["id"],
            "label": final_label,
            "risk_score": round(float(final_risk), 4),
        })
        debug_records.append({
            "id": d["id"],
            "user_input": d["user_input"],
            "model_label": model_label,
            "final_label": final_label,
            "risk_score": round(float(final_risk), 4),
            "rule_info": rule_info,
            "probs": {l: round(float(combined_probs[i][LABEL2ID[l]]), 4) for l in LABELS},
        })

    # save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.debug_dir, "debug.json"), "w", encoding="utf-8") as f:
        json.dump(debug_records, f, ensure_ascii=False, indent=2)

    # stats
    from collections import Counter
    lc = Counter(r["label"] for r in results)
    print(f"\n规则命中: {rule_hits}/{n_samples}")
    print(f"标签分布: {dict(lc)}")
    print(f"结果保存到: {args.output}")


if __name__ == "__main__":
    main()
