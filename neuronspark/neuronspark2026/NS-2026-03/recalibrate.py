"""
v3 推理：使用已有模型概率，但调整决策逻辑
- ambiguous 需要更高置信度（避免回退）
- benign 召回提升
- risk_score 用模型最大类的概率(高风险) / (1-prob)(benign) 映射
"""
import json
import os
import argparse
import numpy as np

LABELS = ["benign", "prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak", "ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
HIGH_RISK = {"prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"}
HIGH_RISK_IDS = {LABEL2ID[l] for l in HIGH_RISK}
AMBIG_ID = LABEL2ID["ambiguous"]
BENIGN_ID = LABEL2ID["benign"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default="test.jsonl")
    parser.add_argument("--qwen_probs", default="infer_debug/qwen_probs.npy")
    parser.add_argument("--bert_probs", default="infer_debug/bert_probs.npy")
    parser.add_argument("--qwen_weight", type=float, default=0.55)
    parser.add_argument("--bert_weight", type=float, default=0.45)
    parser.add_argument("--ambig_threshold", type=float, default=0.5,
                        help="只有 ambiguous 概率超过这个才判 ambiguous")
    parser.add_argument("--output", default="results_v3.json")
    args = parser.parse_args()

    test_data = []
    with open(args.test_file) as f:
        for line in f:
            test_data.append(json.loads(line))
    n = len(test_data)
    print(f"Test: {n}")

    # 加载概率
    qwen_probs = np.load(args.qwen_probs) if os.path.exists(args.qwen_probs) else np.zeros((n, 6))
    bert_probs = np.load(args.bert_probs) if os.path.exists(args.bert_probs) else np.zeros((n, 6))

    print(f"Qwen probs: {qwen_probs.shape}, BERT probs: {bert_probs.shape}")

    combined = args.qwen_weight * qwen_probs + args.bert_weight * bert_probs
    total_w = args.qwen_weight + args.bert_weight
    combined = combined / total_w

    results = []
    label_counts = {l: 0 for l in LABELS}

    for i, d in enumerate(test_data):
        probs = combined[i]
        # 高风险类别概率之和
        high_risk_sum = sum(probs[LABEL2ID[l]] for l in HIGH_RISK)
        benign_p = probs[BENIGN_ID]
        ambig_p = probs[AMBIG_ID]

        # 决策：先看是不是明确高风险或明确 benign
        argmax = probs.argmax()
        argmax_label = LABELS[argmax]

        # 重新分类：避免过度判 ambiguous
        if argmax == AMBIG_ID:
            # 第二选项是什么？
            sorted_idx = np.argsort(probs)[::-1]
            second = sorted_idx[1]
            second_label = LABELS[second]
            second_p = probs[second]

            # 如果 ambiguous 概率不是显著最大，倾向第二选项
            margin = ambig_p - second_p
            if margin < 0.15 and (second in HIGH_RISK_IDS or second == BENIGN_ID):
                # 倾向更明确的判断
                if benign_p > 0.3:
                    final_label = "benign"
                elif high_risk_sum > 0.5:
                    final_label = second_label  # 选最高的高风险类别
                else:
                    final_label = "ambiguous"
            else:
                final_label = "ambiguous"
        else:
            final_label = argmax_label

        # risk_score 用连续概率
        if final_label == "benign":
            risk = high_risk_sum * 0.4 + ambig_p * 0.2  # 即使是 benign，保留区分度
        elif final_label == "ambiguous":
            risk = 0.35 + 0.3 * high_risk_sum  # 0.35-0.65
        else:  # 高风险
            risk = 0.7 + 0.3 * probs[LABEL2ID[final_label]]  # 0.7-1.0

        risk = max(0.0, min(1.0, risk))
        results.append({
            "id": d["id"],
            "label": final_label,
            "risk_score": round(float(risk), 4),
        })
        label_counts[final_label] += 1

    print(f"\n标签分布: {label_counts}")
    risks = [r["risk_score"] for r in results]
    print(f"risk_score: min={min(risks):.3f} max={max(risks):.3f} unique={len(set(round(r,3) for r in risks))}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"保存到: {args.output}")


if __name__ == "__main__":
    main()
