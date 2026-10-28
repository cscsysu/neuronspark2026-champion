"""
BERT v3: 针对 benign 召回不足的改进
- max_length 缩到 256（让 user_input 占主导）
- 仅用 user_input 作为输入，不拼接 system/tools/history
- 训练时对 user_input 做 mask 增强（防过拟合）
- 用 label_smoothing
"""
import json
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["benign", "prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak", "ambiguous"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
HIGH_RISK = {"prompt_injection", "data_exfiltration", "tool_abuse", "jailbreak"}


def build_text(d, mode="full"):
    """v3: 多种文本拼接方式"""
    if mode == "user_only":
        return f"输入: {d['user_input']}"
    elif mode == "history_user":
        return f"历史: {d['history']}\n输入: {d['user_input']}"
    elif mode == "tools_user":
        tool_names = ",".join([t["name"] for t in d["tools"]])
        return f"工具:[{tool_names}]\n输入: {d['user_input']}"
    else:  # full
        tools_str = "; ".join([t["name"] + ":" + t["description"] for t in d["tools"]])
        return (
            f"[系统] {d['system']}\n"
            f"[工具] {tools_str}\n"
            f"[历史] {d['history']}\n"
            f"[输入] {d['user_input']}"
        )


def random_word_mask(text, p=0.15, mask_token="[UNK]"):
    """对 token 做随机 mask 增强"""
    words = list(text)
    n = len(words)
    n_mask = max(1, int(n * p))
    idxs = random.sample(range(n), min(n_mask, n))
    for i in idxs:
        if random.random() < 0.8:
            words[i] = "?"
    return "".join(words)


class JsonlDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256, with_label=True, augment=False, mode="full"):
        self.data = data
        self.tok = tokenizer
        self.max_length = max_length
        self.with_label = with_label
        self.augment = augment
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        text = build_text(d, self.mode)
        if self.augment and random.random() < 0.3:
            text = random_word_mask(text, p=0.1)
        enc = self.tok(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
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
        args.model_name, num_labels=len(LABELS),
        hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2,  # 加大 dropout
    ).to(device)

    train_ds = JsonlDataset(train_data, tokenizer, args.max_length, augment=args.augment, mode=args.text_mode)
    val_ds = JsonlDataset(val_data, tokenizer, args.max_length, mode=args.text_mode)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule(optimizer, int(0.1 * total_steps), total_steps)

    # 用 label smoothing 的损失
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best_f1, best_state = 0.0, None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                lbl = batch.pop("labels").numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                labels_all.extend(lbl.tolist())
        f1 = f1_score(labels_all, preds, average="macro")
        print(f"[Fold {fold_idx} | Epoch {epoch+1}] loss={total_loss/len(train_loader):.4f} f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

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


def predict(model, tokenizer, data, device, batch_size, max_length, mode="full"):
    ds = JsonlDataset(data, tokenizer, max_length, with_label=False, mode=mode)
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
    parser.add_argument("--model_name", default="models/chinese-roberta-wwm-ext")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--test_file", default="test.jsonl")
    parser.add_argument("--output_dir", default="bert_v3")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--text_mode", default="full",
                        choices=["full", "user_only", "history_user", "tools_user"])
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | text_mode: {args.text_mode} | augment: {args.augment}")

    train_data = []
    with open(args.train_file) as f:
        for line in f:
            train_data.append(json.loads(line))
    test_data = []
    with open(args.test_file) as f:
        for line in f:
            test_data.append(json.loads(line))
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")

    labels = np.array([LABEL2ID[d["label"]] for d in train_data])
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    oof_probs = np.zeros((len(train_data), len(LABELS)))
    test_probs = np.zeros((len(test_data), len(LABELS)))
    fold_f1s = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_data)), labels)):
        print(f"\n========== Fold {fold_idx+1}/{args.n_folds} ==========")
        tr = [train_data[i] for i in tr_idx]
        va = [train_data[i] for i in va_idx]
        model, tok, val_probs, best_f1 = train_one_fold(tr, va, args, fold_idx + 1, device)
        oof_probs[va_idx] = val_probs
        fold_f1s.append(best_f1)
        tp = predict(model, tok, test_data, device, args.batch_size * 2, args.max_length, args.text_mode)
        test_probs += tp / args.n_folds

        save_dir = os.path.join(args.output_dir, f"fold_{fold_idx+1}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        del model
        torch.cuda.empty_cache()

    oof_preds = oof_probs.argmax(-1)
    print(f"\n===== OOF Macro F1: {f1_score(labels, oof_preds, average='macro'):.4f} =====")
    print(classification_report(labels, oof_preds, target_names=LABELS, digits=4))

    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_probs)


if __name__ == "__main__":
    main()
