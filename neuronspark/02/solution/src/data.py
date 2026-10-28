"""数据加载与基础工具"""
import json, os, re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "1fa3ed1b-c5a5-48e0-b0e5-09a27ac38e4d"

REFUSAL = "无法根据给定知识库回答"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus():
    return read_jsonl(DATA_DIR / "corpus.jsonl")


def load_train():
    return read_jsonl(DATA_DIR / "train_qa.jsonl")


def load_test():
    return read_jsonl(DATA_DIR / "test_questions.jsonl")


def normalize_question(q):
    """去掉问题里的固定前后缀，留下核心 query"""
    # 顶部填充词
    q = re.sub(r"^(请回答[：:]|按照给定知识库，?|根据知识库，?|制度中规定，?|请问|对于|关于|只根据本地文档，?)", "", q)
    # 中部 metafiller
    q = re.sub(r"只根据本地文档，?", "", q)
    q = re.sub(r"按照给定知识库，?", "", q)
    q = re.sub(r"根据知识库，?", "", q)
    q = re.sub(r"制度中规定，?", "", q)
    # 尾部
    q = re.sub(r"(请同时给出依据[。.]?|不要使用外部常识[。.]?|答案尽量简短[。.]?|请给出简短答案[。.]?)", "", q)
    return q.strip().rstrip("？?").rstrip("，").strip()
