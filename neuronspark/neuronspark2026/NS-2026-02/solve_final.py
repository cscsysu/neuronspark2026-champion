"""
NS-2026-02 v8 - 回归v5框架 + 新LoRA训练数据
v5拿到702分的prompt不变，只改进：
1. 新LoRA数据（context格式和推理一致）
2. 训练2 epoch（减少过拟合）
3. 推理用v5的prompt和参数

CUDA_VISIBLE_DEVICES=0 python solve_final.py
"""
import json, re, os, time
import jieba
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_DIR = os.path.join(DATA_DIR, "lora_output")

# ========== Load data ==========
def load_jsonl(p):
    with open(p, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

corpus = load_jsonl(os.path.join(DATA_DIR, 'corpus.jsonl'))
train_qa = load_jsonl(os.path.join(DATA_DIR, 'train_qa.jsonl'))
test_qs = load_jsonl(os.path.join(DATA_DIR, 'test_questions.jsonl'))
print(f"Loaded: {len(corpus)} chunks, {len(train_qa)} train, {len(test_qs)} test")

# ========== BM25 ==========
STOP = set('的了是在有和与或者不为这那个一也都到就被从而但如果请问根据知识库简短答案规定按照给定请回答外部常识依据本地文档制度')
STOP.update(['请问','根据','知识库','简短','答案','规定','按照','给定','请回答','外部','常识','依据','本地','文档','制度','中','不要','使用'])

def tok(text):
    return [t for t in jieba.cut(text) if t.strip() and t not in STOP and len(t.strip())>0]

corpus_toks = [tok(c['title']+' '+c['text']) for c in corpus]
bm25 = BM25Okapi(corpus_toks)
train_toks = [tok(t['question']) for t in train_qa]
bm25_train = BM25Okapi(train_toks)

cmap = {c['chunk_id']: c for c in corpus}
dchunks = defaultdict(list)
for c in corpus: dchunks[c['doc_id']].append(c)

print("BM25 index ready.")

# ========== Model + LoRA ==========
print("Loading Qwen3.5-4B (4-bit) + LoRA adapter...")
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_DIR = snapshot_download("Qwen/Qwen3.5-4B", cache_dir="/data2/css2025/models")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True,
    quantization_config=bnb_config, device_map="auto"
)

if os.path.exists(LORA_DIR):
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    print(f"LoRA adapter loaded from {LORA_DIR}")
else:
    model = base_model
    print("WARNING: No LoRA adapter found, using base model")

model.eval()
print("Model ready on GPU.")

# ========== Prompt（和v5/702版完全一致） ==========
SYSTEM_PROMPT = """你是图书馆问答员。根据知识库片段回答问题。
规则：
1. 答案从片段中直接摘录关键词/短语，不要改写
2. 答案简短（3-30字），只给核心信息
3. 数值/时间保持原文写法
4. 是否类只答"可以"或"不能"
5. 多个答案用"；"分隔
6. 片段中无相关信息时回答"无法根据给定知识库回答"
7. 不要输出片段编号"""

@torch.no_grad()
def generate_answer(context, question, temperature=0.3, max_tokens=60):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"知识库片段：\n{context}\n\n问：{question}"}
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    
    out = model.generate(
        **inp, max_new_tokens=max_tokens,
        temperature=temperature, do_sample=True, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    response = tokenizer.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# ========== Retrieval ==========
def get_chunks(q, k=5):
    scores = bm25.get_scores(tok(q))
    ranked = sorted(enumerate(scores), key=lambda x:-x[1])
    return [{'id':corpus[i]['chunk_id'],'score':s,'text':corpus[i]['text'],
             'title':corpus[i]['title'],'doc':corpus[i]['doc_id']} for i,s in ranked[:k]]

def get_similar_train(q, k=5):
    scores = bm25_train.get_scores(tok(q))
    ranked = sorted(enumerate(scores), key=lambda x:-x[1])
    return [{'t':train_qa[i],'s':s} for i,s in ranked[:k]]

# ========== 拒答 ==========
def should_refuse(q, chunks, similar):
    if similar and similar[0]['s'] > 14 and not similar[0]['t']['answerable']:
        return True
    if not chunks or chunks[0]['score'] < 3.0:
        return True
    ents = set(w for w in tok(q) if len(w)>=2)
    if ents:
        ctoks = set(tok(chunks[0]['text']+' '+chunks[0]['title']))
        if len(ents & ctoks)==0 and chunks[0]['score']<5:
            return True
    return False

# ========== 后处理 ==========
def postprocess(answer):
    answer = answer.strip().split('\n')[0].strip()
    for p in ['根据知识库，','根据文档，','根据规定，','答案是：','答案：','答：',
              '根据给定知识库，','根据以上知识库，','根据知识库片段，','根据上述知识库，',
              '根据以上信息，','根据片段，','根据上述信息，']:
        if answer.startswith(p):
            answer = answer[len(p):]
    answer = answer.rstrip('。.！')
    # 过滤chunk_id
    answer = re.sub(r'\[?[A-Z]{2,}-[A-Z]*-?\d{0,3}\]?', '', answer)
    answer = re.sub(r'\[\]', '', answer)
    answer = re.sub(r'\[[A-Z].*?\]', '', answer)
    answer = re.sub(r'[；;]{2,}', '；', answer)
    answer = answer.strip('；;，, ')
    # 过滤乱输出
    if any(bp in answer for bp in ['不可类答','类答','须不可']):
        answer = "无法根据给定知识库回答"
    if '无法' in answer and '回答' in answer:
        answer = "无法根据给定知识库回答"
    # 长度
    if len(answer) > 33:
        parts = answer.split('；')
        if len(parts) > 1:
            result = parts[0]
            for p in parts[1:]:
                if len(result + '；' + p) <= 33:
                    result += '；' + p
                else:
                    break
            answer = result
        if len(answer) > 33:
            last = answer.rfind('，', 0, 33)
            if last > 15:
                answer = answer[:last]
            else:
                answer = answer[:33]
    if not answer or all(c in '，。；、！？ ' for c in answer):
        answer = "无法根据给定知识库回答"
    return answer

def pick_citations(answer, chunks):
    if answer == "无法根据给定知识库回答":
        return []
    ans_toks = set(tok(answer))
    scored = []
    for c in chunks:
        cid = c.get('chunk_id', c.get('id',''))
        text = c.get('text','')
        ctoks = set(tok(text))
        overlap = len(ans_toks & ctoks)
        bonus = 3 if answer in text else 0
        if not bonus:
            for part in answer.split('；'):
                if part.strip() and len(part.strip())>2 and part.strip() in text:
                    bonus = 2
                    break
        scored.append((cid, overlap + bonus))
    scored.sort(key=lambda x:-x[1])
    result = [cid for cid, sc in scored if sc > 0][:3]
    if not result and chunks:
        result = [chunks[0].get('chunk_id', chunks[0].get('id',''))]
    return result

# ========== Self-Consistency (3次，和v5一致) ==========
def self_consistency_answer(context, question, n=3):
    answers = []
    for i in range(n):
        temp = 0.1 if i == 0 else 0.3
        raw = generate_answer(context, question, temperature=temp)
        ans = postprocess(raw)
        answers.append(ans)
    
    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]
    if best_count == 1:
        best_answer = answers[0]
    return best_answer

# ========== Main ==========
def solve():
    results = []
    t0 = time.time()
    
    for i, tq in enumerate(test_qs):
        qid = tq['id']
        question = tq['question']
        
        chunks = get_chunks(question, 5)
        similar = get_similar_train(question, 5)
        
        # 拒答
        if should_refuse(question, chunks, similar):
            results.append({"id": qid, "answer": "无法根据给定知识库回答", "citations": []})
            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(test_qs)}] {time.time()-t0:.1f}s | REFUSED")
            continue
        
        # 训练集强匹配
        if similar[0]['s'] > 18 and similar[0]['t']['answerable']:
            top_cids = set(c['id'] for c in chunks[:5])
            best = similar[0]['t']
            if set(best['citations']) & top_cids:
                results.append({"id": qid, "answer": best['answer'], "citations": best['citations']})
                if (i+1) % 20 == 0:
                    print(f"  [{i+1}/{len(test_qs)}] {time.time()-t0:.1f}s | TRAIN: {best['answer'][:20]}")
                continue
        
        # LLM + Self-Consistency
        doc_id = chunks[0]['doc']
        ctx_chunks = list(dchunks[doc_id])
        seen = set(c['chunk_id'] for c in ctx_chunks)
        for c in chunks[1:4]:
            if c['id'] not in seen:
                ctx_chunks.append(cmap[c['id']])
                seen.add(c['id'])
            if len(ctx_chunks) >= 6:
                break
        
        context = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in ctx_chunks)
        
        answer = self_consistency_answer(context, question, n=3)
        
        if answer == "无法根据给定知识库回答":
            citations = []
        else:
            citations = pick_citations(answer, ctx_chunks)
        
        results.append({"id": qid, "answer": answer, "citations": citations[:5]})
        
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(test_qs)}] {time.time()-t0:.1f}s | {answer[:25]}")
    
    return results

if __name__ == '__main__':
    print("\n=== Running v8 inference ===")
    t_start = time.time()
    results = solve()
    total_time = time.time() - t_start
    
    out = os.path.join(DATA_DIR, 'results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    u = sum(1 for r in results if r['answer']=='无法根据给定知识库回答')
    print(f"\n=== Done in {total_time:.1f}s ===")
    print(f"Total: {len(results)}, Answered: {len(results)-u}, Refused: {u}")
    
    # 打包
    import zipfile
    zp = os.path.join(DATA_DIR, 'NS-2026-02-answer.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(out, 'results.json')
    print(f"Submit: {zp}")
