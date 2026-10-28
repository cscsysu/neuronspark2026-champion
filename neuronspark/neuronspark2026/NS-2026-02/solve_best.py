"""
NS-2026-02 v4 - 重点提升答案+恢复拒答
改进:
1. 去掉chunk QA复用（误报严重）
2. 只保留训练集强匹配(>18) + LLM生成
3. 让LLM自己判断拒答（比BM25阈值更准）
4. 改进prompt：更多few-shot + 严格格式约束
5. 后处理：过滤异常输出

CUDA_VISIBLE_DEVICES=0 python solve_best.py
"""
import json, re, os, sys, time
import jieba
from rank_bm25 import BM25Okapi
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

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

print("Index ready.")

# ========== Model ==========
print("Loading model from ModelScope...")
import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = snapshot_download("Qwen/Qwen2.5-3B-Instruct", cache_dir="/data2/css2025/models")
print(f"Model: {MODEL_DIR}")

tok_model = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
mdl = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True,
    torch_dtype=torch.float16, device_map="cuda:0")
mdl.eval()
print("Model ready on GPU.")

@torch.no_grad()
def ask_llm(prompt, max_tokens=60):
    msgs = [{"role":"user","content":prompt}]
    text = tok_model.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok_model(text, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inp, max_new_tokens=max_tokens, temperature=0.05,
                       do_sample=True, top_p=0.9, pad_token_id=tok_model.eos_token_id,
                       repetition_penalty=1.1)
    return tok_model.decode(out[0][inp['input_ids'].shape[1]:], skip_special_tokens=True).strip()

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

# ========== 拒答（恢复v3逻辑 + LLM辅助） ==========
def should_refuse(q, chunks, similar):
    """拒答判断 - 恢复v3保守策略"""
    # 训练集强匹配不可答
    if similar and similar[0]['s'] > 14 and not similar[0]['t']['answerable']:
        return True
    # BM25极低
    if not chunks or chunks[0]['score'] < 3.0:
        return True
    # 实体零覆盖
    ents = set(w for w in tok(q) if len(w)>=2)
    if ents:
        ctoks = set(tok(chunks[0]['text']+' '+chunks[0]['title']))
        if len(ents & ctoks)==0 and chunks[0]['score']<5:
            return True
    return False

# ========== 答案生成 ==========
def solve_one(question):
    chunks = get_chunks(question, 5)
    similar = get_similar_train(question, 5)
    
    # === 确定拒答 ===
    if should_refuse(question, chunks, similar):
        return "无法根据给定知识库回答", []
    
    # === 训练集匹配（带主题验证） ===
    # 检索到的文档ID集合
    top_doc_ids = set(c['doc'] for c in chunks[:3])
    
    for s in similar[:3]:
        if s['s'] > 16 and s['t']['answerable']:
            # 验证：训练答案的引用chunk所在文档 == 当前检索到的文档
            train_cites = s['t']['citations']
            match_doc = False
            for tcid in train_cites:
                if tcid in cmap and cmap[tcid]['doc_id'] in top_doc_ids:
                    match_doc = True
                    break
            if match_doc:
                return s['t']['answer'], s['t']['citations']
    
    # === LLM 生成（所有其他情况） ===
    return llm_generate_answer(question, chunks, similar)

def llm_generate_answer(question, chunks, similar):
    """用LLM生成答案"""
    # Context: top chunk所属doc全部chunks + 其他top chunks
    doc_id = chunks[0]['doc']
    ctx_chunks = list(dchunks[doc_id])
    seen_ids = set(c['chunk_id'] for c in ctx_chunks)
    for c in chunks[1:4]:
        if c['id'] not in seen_ids:
            ctx_chunks.append(cmap[c['id']])
            seen_ids.add(c['id'])
        if len(ctx_chunks) >= 6:
            break
    
    ctx = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in ctx_chunks)
    
    # Few-shot: 2个可答 + 1个拒答
    examples = []
    for s in similar[:4]:
        if s['t']['answerable'] and s['s'] > 4 and len(examples) < 2:
            examples.append(f"问：{s['t']['question']}\n答：{s['t']['answer']}")
    # 加拒答示例
    examples.append("问：校医院流感疫苗预约系统叫什么？\n答：无法根据给定知识库回答")
    
    ex_text = "\n\n".join(examples)
    
    prompt = f"""你是图书馆问答员。根据下面的知识库片段回答问题。

规则：
1. 答案必须从片段中直接摘录关键词/短语，不要自己编造
2. 片段中没有相关信息时，回答"无法根据给定知识库回答"
3. 答案简短（3-30字），只给最核心的信息
4. 数值/时间保持原文写法（如"2个工作日""提前3天""17:30"）
5. 是否类只答"可以"或"不能"
6. 多个答案用"；"分隔
7. 绝对不要输出片段编号（如LIB-BORROW-001）

知识库片段：
{ctx}

示例：
{ex_text}

问：{question}
答："""
    
    raw = ask_llm(prompt, max_tokens=50)
    answer = postprocess(raw)
    
    if answer == "无法根据给定知识库回答":
        return answer, []
    
    citations = pick_citations(answer, ctx_chunks)
    return answer, citations

def postprocess(answer):
    """后处理"""
    answer = answer.strip().split('\n')[0].strip()
    
    # 去前缀
    for p in ['根据知识库，','根据文档，','根据规定，','答案是：','答案：','答：',
              '根据给定知识库，','根据以上知识库，','根据知识库片段，','根据上述知识库，',
              '根据以上信息，','根据片段，']:
        if answer.startswith(p):
            answer = answer[len(p):]
    
    answer = answer.rstrip('。.！')
    
    # 过滤chunk_id和方括号引用 (如 [HEALTH-SLEEP-001], [SHUTTLE-], [] 等)
    answer = re.sub(r'\[?[A-Z]{2,}-[A-Z]*-?\d{0,3}\]?', '', answer)
    answer = re.sub(r'\[\]', '', answer)
    answer = re.sub(r'\[[A-Z].*?\]', '', answer)
    # 清理残留的分号和逗号
    answer = re.sub(r'[；;，,]\s*$', '', answer)
    answer = re.sub(r'^[；;，,]\s*', '', answer)
    answer = re.sub(r'[；;]{2,}', '；', answer)
    answer = answer.strip('；;，, ')
    
    # 过滤模型乱输出
    bad_patterns = ['不可类答', '类答', '不可根据', '根据给定知识库回答',
                    '须不可', '不可提前；须']
    for bp in bad_patterns:
        if bp in answer:
            answer = "无法根据给定知识库回答"
            return answer
    
    # 如果模型输出包含"无法"相关
    if '无法' in answer and '回答' in answer:
        answer = "无法根据给定知识库回答"
        return answer
    
    # 长度控制
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
            last_comma = answer.rfind('，', 0, 33)
            if last_comma > 15:
                answer = answer[:last_comma]
            else:
                answer = answer[:33]
    
    if not answer or all(c in '，。；、！？ ' for c in answer):
        answer = "无法根据给定知识库回答"
    
    return answer

def pick_citations(answer, chunks):
    """选择citations"""
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
    if not result:
        result = [chunks[0].get('chunk_id', chunks[0].get('id',''))]
    return result

# ========== Main ==========
def main():
    results = []
    t0 = time.time()
    
    for i, tq in enumerate(test_qs):
        answer, citations = solve_one(tq['question'])
        results.append({"id": tq['id'], "answer": answer, "citations": citations[:5]})
        
        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(test_qs)}] {elapsed:.1f}s | {answer[:30]}")
    
    return results

if __name__ == '__main__':
    print("\n=== Running inference v4 ===")
    t_start = time.time()
    results = main()
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
