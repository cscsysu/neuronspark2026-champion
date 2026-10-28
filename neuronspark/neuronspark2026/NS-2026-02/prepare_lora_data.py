"""
准备 LoRA 微调数据
将 train_qa.jsonl 转为 ChatML 格式的训练数据
"""
import json
import os
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_jsonl(p):
    with open(p, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

corpus = load_jsonl(os.path.join(DATA_DIR, 'corpus.jsonl'))
train_qa = load_jsonl(os.path.join(DATA_DIR, 'train_qa.jsonl'))

cmap = {c['chunk_id']: c for c in corpus}

SYSTEM_PROMPT = """你是图书馆问答员。根据知识库片段回答问题。
规则：
1. 答案从片段中直接摘录关键词/短语，不要改写
2. 答案简短（3-30字），只给核心信息
3. 数值/时间保持原文写法
4. 是否类只答"可以"或"不能"
5. 多个答案用"；"分隔
6. 片段中无相关信息时回答"无法根据给定知识库回答"
7. 不要输出片段编号"""

def build_context(citations):
    """根据citations构建知识库上下文"""
    parts = []
    seen_docs = set()
    for cid in citations:
        if cid in cmap:
            c = cmap[cid]
            parts.append(f"[{c['chunk_id']}] {c['text']}")
            # 加同doc其他chunk
            if c['doc_id'] not in seen_docs:
                seen_docs.add(c['doc_id'])
                for other in corpus:
                    if other['doc_id'] == c['doc_id'] and other['chunk_id'] != cid:
                        parts.append(f"[{other['chunk_id']}] {other['text']}")
    return "\n".join(parts[:6])

def build_context_for_unanswerable(question):
    """为不可答问题构建干扰上下文（随机选几个chunk）"""
    # 随机选3个不相关chunk作为干扰
    random.seed(hash(question) % 10000)
    selected = random.sample(corpus, min(3, len(corpus)))
    return "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in selected)

# 构建训练数据
train_data = []

for t in train_qa:
    question = t['question']
    answer = t['answer']
    answerable = t['answerable']
    citations = t['citations']
    
    if answerable:
        context = build_context(citations)
    else:
        context = build_context_for_unanswerable(question)
    
    # ChatML格式
    conversation = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"知识库片段：\n{context}\n\n问：{question}"},
            {"role": "assistant", "content": answer}
        ]
    }
    train_data.append(conversation)

# 打乱
random.seed(42)
random.shuffle(train_data)

# 保存
output_path = os.path.join(DATA_DIR, 'lora_train_data.json')
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"Generated {len(train_data)} training examples")
print(f"Saved to: {output_path}")
print(f"Answerable: {sum(1 for t in train_qa if t['answerable'])}")
print(f"Unanswerable: {sum(1 for t in train_qa if not t['answerable'])}")

# 也输出一个样例
print("\n=== Sample ===")
sample = train_data[0]
print(f"System: {sample['messages'][0]['content'][:50]}...")
print(f"User: {sample['messages'][1]['content'][:100]}...")
print(f"Assistant: {sample['messages'][2]['content']}")
