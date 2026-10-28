"""
NS-2026-03: BERT 微调 (5-fold CV)
模型: hfl/chinese-roberta-wwm-ext
输出: 6 类 softmax 概率
"""
import json
import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# 优先用 HF 镜像（hf-mirror.com 是稳定的国内镜像），失败回退 ModelScope
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["benign", "prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak", "ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
HIGH_RISK = {"prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"}


def build_text(d):
    """把样本拼成一段文本输入"""
    tools_str = "; ".join([t["name"] + ":" + t["description"] for t in d["tools"]])
    text = (
        f"[系统] {d['system']}\n"
        f"[工具] {tools_str}\n"
        f"[历史] {d['history']}\n"
        f"[输入] {d['user_input']}"
    )
    return text


class JsonlDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, with_label=True):
        self.data = data
        self.tok = tokenizer
        self.max_length = max_length
        self.with_label = with_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        text = build_text(d)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.with_label:
            item["labels"] = torch.tensor(LABEL2ID[d["label"]], dtype=torch.long)
        return item


def get_linear_schedule(optimizer, num_warmup, num_total):
    def lr_lambda(step):
        if step < num_warmup:
            return step / max(1, num_warmup)
        return max(0.0, (num_total - step) / max(1, num_total - num_warmup))
    return LambdaLR(optimizer, lr_lambda)


def train_one_fold(train_data, val_data, args, fold_idx, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS)
    ).to(device)

    train_ds = JsonlDataset(train_data, tokenizer, args.max_length)
    val_ds = JsonlDataset(val_data, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule(optimizer, int(0.1 * total_steps), total_steps)

    best_f1, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()

        # eval
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                lbl = batch.pop("labels").numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                labels.extend(lbl.tolist())
        f1 = f1_score(labels, preds, average="macro")
        print(f"[Fold {fold_idx} | Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f} macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    model.load_state_dict(best_state)

    # predict val probs
    model.eval()
    val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            val_probs.append(probs)
    val_probs = np.concatenate(val_probs, axis=0)

    return model, tokenizer, val_probs, best_f1


def predict(model, tokenizer, data, device, batch_size, max_length):
    ds = JsonlDataset(data, tokenizer, max_length, with_label=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--test_file", default="test.jsonl")
    parser.add_argument("--output_dir", default="bert_out")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    train_data = []
    with open(args.train_file) as f:
        for line in f:
            train_data.append(json.loads(line))

    test_data = []
    with open(args.test_file) as f:
        for line in f:
            test_data.append(json.loads(line))

    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    # 5-fold CV
    labels = np.array([LABEL2ID[d["label"]] for d in train_data])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_probs = np.zeros((len(train_data), len(LABELS)))
    oof_indices = np.zeros(len(train_data), dtype=int)
    test_probs = np.zeros((len(test_data), len(LABELS)))
    fold_f1s = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_data)), labels)):
        print(f"\n========== Fold {fold_idx+1}/{args.n_folds} ==========")
        tr = [train_data[i] for i in tr_idx]
        va = [train_data[i] for i in va_idx]

        model, tok, val_probs, best_f1 = train_one_fold(tr, va, args, fold_idx + 1, device)
        oof_probs[va_idx] = val_probs
        oof_indices[va_idx] = fold_idx
        fold_f1s.append(best_f1)

        # predict test
        tp = predict(model, tok, test_data, device, args.batch_size * 2, args.max_length)
        test_probs += tp / args.n_folds

        # save fold model
        save_dir = os.path.join(args.output_dir, f"fold_{fold_idx+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

        del model
        torch.cuda.empty_cache()

    # overall metrics
    oof_preds = oof_probs.argmax(-1)
    oof_labels = labels
    print(f"\n===== Overall CV Metrics =====")
    print(f"Fold F1s: {[f'{f:.4f}' for f in fold_f1s]}")
    print(f"OOF Macro F1: {f1_score(oof_labels, oof_preds, average='macro'):.4f}")
    print(f"\nClassification Report:")
    print(classification_report(oof_labels, oof_preds, target_names=LABELS, digits=4))
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(oof_labels, oof_preds)
    print("    " + "  ".join(f"{l[:6]:>6}" for l in LABELS))
    for i, row in enumerate(cm):
        print(f"{LABELS[i][:6]:>6}: " + "  ".join(f"{v:>6}" for v in row))

    # high-risk recall
    hr_ids = {LABEL2ID[l] for l in HIGH_RISK}
    hr_mask = np.isin(oof_labels, list(hr_ids))
    hr_correct = (oof_preds[hr_mask] == oof_labels[hr_mask]).sum()
    hr_recall = hr_correct / hr_mask.sum()
    print(f"\nHigh-risk Recall (predicting correct high-risk class): {hr_recall:.4f}")

    # save oof and test probs
    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_probs)
    np.save(os.path.join(args.output_dir, "oof_labels.npy"), oof_labels)

    # save oof predictions with ids for analysis
    oof_records = []
    for i, d in enumerate(train_data):
        oof_records.append({
            "id": d["id"],
            "true_label": d["label"],
            "pred_label": ID2LABEL[int(oof_preds[i])],
            "true_risk_score": d.get("risk_score"),
            "probs": {l: float(oof_probs[i][LABEL2ID[l]]) for l in LABELS},
        })
    with open(os.path.join(args.output_dir, "oof_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(oof_records, f, ensure_ascii=False, indent=2)

    # save test ids for inference
    test_ids = [d["id"] for d in test_data]
    with open(os.path.join(args.output_dir, "test_ids.json"), "w") as f:
        json.dump(test_ids, f)

    print(f"\n所有结果保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
