"""主流水线：模板路由 + train 查表 + BM25 + 抽取 + 拒答"""
import re
import json
from collections import defaultdict

from .data import REFUSAL, normalize_question
from .templates import parse_template
from .template_extractor import answer_template, find_chunks_by_entity
from .retriever import BM25Retriever, tokenize_zh
from .sentence_extractor import extract_short_answer, best_sentence


# 陷阱 chunk：组委会放的 prompt injection 干扰，必须永远屏蔽
BLACKLIST_CHUNKS = {"SAFE-INJECT-001", "SAFE-INJECT-002"}


def filter_blacklist(chunks):
    return [c for c in chunks if c["chunk_id"] not in BLACKLIST_CHUNKS]


def build_train_lookup(train):
    """为 train 中模板题建索引：{(template, entity): (answer, citations)}"""
    lookup = {}
    for t in train:
        if not t.get("answerable", True):
            continue
        tmpl, tag, is_multi, ent = parse_template(t["question"])
        if tmpl and ent:
            lookup[(tmpl, ent)] = (t["answer"], list(t["citations"]))
    return lookup


def build_train_question_index(train):
    """近似问题查表：normalized question → train item（考虑同/类似问题）"""
    out = defaultdict(list)
    for t in train:
        nq = normalize_question(t["question"])
        out[nq].append(t)
    return out


def build_train_chunk_index(train):
    """按 chunk_id 索引训练 QA：{chunk_id: [train_item, ...]}（仅可答 + 非模板）"""
    out = defaultdict(list)
    for t in train:
        if not t.get("answerable", True):
            continue
        tmpl, _, _, _ = parse_template(t["question"])
        # 模板题已经走 (template, entity) lookup，这里专门服务非模板题
        if tmpl:
            continue
        for c in t["citations"]:
            out[c].append(t)
    return out


def jaccard_tokens(a_tokens, b_tokens):
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


# 高区分度关键词：必须在 test 和 train 问题里都出现/都不出现，否则不复用
DISTINCTIVE_TOKENS = {
    "本科生", "研究生", "教职工",
    "本科", "硕士", "博士",
    "普通图书", "特藏", "校史档案", "民国线装",
    "线上", "线下", "电子", "纸质",
    "白天", "夜间", "晚间", "早晨", "上午", "下午",
    "校内", "校外", "境外", "境内", "国际", "国内",
    "正常", "异常", "紧急", "一般", "重大", "轻微",
    "教学", "科研", "本月", "本周", "本学期",
    "B级", "A级", "BSL-1", "BSL-2",
}


def has_distinctive_mismatch(q_test, q_train):
    """检查 test 和 train 问题在关键名词上是否一致；若不一致返回 True"""
    a = set(t for t in DISTINCTIVE_TOKENS if t in q_test)
    b = set(t for t in DISTINCTIVE_TOKENS if t in q_train)
    return a != b


def lookup_train_by_chunk(question, primary_chunk_id, chunk_index):
    """在 primary chunk 关联的 train 非模板题里，找问题最相似的，返回 (answer, citations) 或 (None, None)"""
    candidates = chunk_index.get(primary_chunk_id, [])
    if not candidates:
        return None, None
    q_tokens = tokenize_zh(question)
    q_norm = normalize_question(question)
    best = None
    best_score = 0.0
    for t in candidates:
        # 关键名词不一致 → 不能复用
        if has_distinctive_mismatch(question, t["question"]):
            continue
        t_norm = normalize_question(t["question"])
        # 问题归一化后完全相同 → 直接命中
        if q_norm == t_norm:
            return t["answer"], list(t["citations"])
        t_tokens = tokenize_zh(t["question"])
        score = jaccard_tokens(q_tokens, t_tokens)
        if score > best_score:
            best_score = score
            best = t
    if best and best_score >= 0.55:
        return best["answer"], list(best["citations"])
    return None, None


def long_phrase_in_corpus(question, corpus, min_len=4):
    """检查问题中是否至少有一个 ≥min_len 字的中文连续短语在 corpus 出现。
    若问题里所有"长名词短语"都不在 corpus → 大概率不在知识库 → 拒答。"""
    nq = normalize_question(question)
    haystack = "\n".join(c["title"] + c["text"] for c in corpus)
    # 提取所有 ≥min_len 字的连续中文短语
    phrases = re.findall(r"[一-龥]{" + str(min_len) + r",}", nq)
    if not phrases:
        return True  # 没法判断，默认通过
    for p in phrases:
        # 滑窗匹配 min_len 字片段
        for i in range(len(p) - min_len + 1):
            if p[i:i+min_len] in haystack:
                return True
    return False


# ====== 拒答专用：核心实体存在性检查 ======
# 若问题的实体核心词在 corpus 中找不到，则拒答
QUESTION_STOPWORDS = {
    "请问", "请回答", "按照", "给定", "知识库", "根据", "制度", "中规定", "规定",
    "答案", "尽量", "简短", "请", "不要", "使用", "外部", "常识", "请同时",
    "给出", "依据", "对于", "关于", "如何", "怎样", "什么", "多少", "多久",
    "几", "几点", "哪里", "在哪", "能否", "是否", "可不可以",
    "有效", "时间", "时长", "费用", "价格", "售价", "怎么办", "提供",
    "服务", "开放", "营业", "周六", "周日", "周一", "周二", "周三", "周四", "周五",
    "本科生", "学生", "学校", "校园", "教师", "毕业",
    "有", "是", "的", "了", "在", "和", "为", "或",
}


def extract_question_entities(question):
    """从问题中抽出可能的实体短语 (jieba 分词后过滤)"""
    from .retriever import tokenize_zh
    toks = tokenize_zh(question)
    # 只保留 jieba 分出的 ≥2 字词，且非停用词
    out = []
    for t in toks:
        t = t.strip()
        if len(t) < 2:
            continue
        if t in QUESTION_STOPWORDS:
            continue
        if not re.search(r"[一-龥A-Za-z]", t):
            continue
        out.append(t)
    return out


def entity_coverage(question, corpus):
    """实体在 corpus 中的命中率：返回 0~1"""
    ents = extract_question_entities(question)
    if not ents:
        return 1.0
    haystack = "\n".join(c["title"] + c["text"] for c in corpus)
    hit = sum(1 for e in ents if e in haystack)
    return hit / max(1, len(ents))


def answer_question(q, corpus, retriever, train_lookup, train_qindex, refusal_threshold=2.0):
    """对单个问题返回 {"id": qid, "answer": ..., "citations": [...]}"""
    qid = q["id"]
    question = q["question"]

    # === 0. 强制拒答：实体在 corpus 中找不到 ===
    cov = entity_coverage(question, corpus)
    if cov < 0.34:  # 1/3 以下的实体能命中 → 大概率不在知识库
        return {"id": qid, "answer": REFUSAL, "citations": []}

    # === 1. 模板题路径 ===
    tmpl, tag, is_multi, entity = parse_template(question)
    if tmpl and entity:
        # 1a. 先查 train 是否有同 (template, entity)
        key = (tmpl, entity)
        if key in train_lookup:
            ans, cites = train_lookup[key]
            return {"id": qid, "answer": ans, "citations": cites[:5]}
        # 1b. 必须先确认 entity 在 corpus 里有对应的 ACCESS / CHECK chunk
        from .template_extractor import find_chunks_by_entity
        ent_chunks_check = find_chunks_by_entity(corpus, entity, "CHECK")
        ent_chunks_access = find_chunks_by_entity(corpus, entity, "ACCESS")
        # 检查 entity 主体（去除 metafiller 前后缀）能否在 chunk 中精确出现
        import re as _re
        ent_clean = _re.sub(r"\s+", "", entity)
        # 也检查 entity 的核心子串（去掉 "申请/服务/咨询" 等通用后缀）
        core_ent = _re.sub(r"(申请|服务|咨询|材料盖章|时长认定|资助报销|温度补助|训练)$", "", ent_clean)
        # 严格匹配：entity 整体或其 ≥4 字核心子串必须在 chunk 中
        def _has_full_match(chs):
            if not chs: return False
            for c in chs:
                hay = c["title"] + c["text"]
                if ent_clean in hay:
                    return True
                if len(core_ent) >= 4 and core_ent in hay:
                    return True
            return False
        has_check = _has_full_match(ent_chunks_check)
        has_access = _has_full_match(ent_chunks_access)
        if not (has_check or has_access):
            # entity 在 corpus 中不存在精确匹配 → 拒答
            return {"id": qid, "answer": REFUSAL, "citations": []}
        ans, cites = answer_template(question, tmpl, tag, entity, corpus)
        if ans and cites:
            cites = [c for c in cites if c not in BLACKLIST_CHUNKS]
            return {"id": qid, "answer": ans.strip(), "citations": cites[:5]}

    # === 2. 非模板：BM25 检索 (排除黑名单) ===
    bm25_top = retriever.topk(question, k=15)
    bm25_top = [(i, s) for i, s in bm25_top if corpus[i]["chunk_id"] not in BLACKLIST_CHUNKS]
    if not bm25_top:
        return {"id": qid, "answer": REFUSAL, "citations": []}

    top_chunks = [corpus[i] for i, _ in bm25_top[:5]]
    top_scores = [s for _, s in bm25_top[:5]]

    # 拒答检测：top-1 BM25 分数太低 → 拒答
    if top_scores[0] < refusal_threshold:
        return {"id": qid, "answer": REFUSAL, "citations": []}

    # 二次拒答：实体在 top-1 chunk 里完全没出现 + top1/top2 分数差异不显著 → 不可信
    ents = extract_question_entities(question)
    primary = top_chunks[0]
    primary_text = primary["title"] + primary["text"]
    if ents:
        ent_in_primary = sum(1 for e in ents if e in primary_text) / len(ents)
        # 极低实体命中 → 强制拒答 (即使 BM25 分高，仍然是误命中)
        if ent_in_primary < 0.2:
            return {"id": qid, "answer": REFUSAL, "citations": []}
        if ent_in_primary < 0.34 and top_scores[0] < refusal_threshold * 2.5:
            return {"id": qid, "answer": REFUSAL, "citations": []}

    # === 3. 抽答案 ===
    # 3a. 先尝试在 primary chunk 关联的 train 非模板题里查同 chunk + 相似问题
    if hasattr(answer_question, "_chunk_index"):
        chunk_index = answer_question._chunk_index
    else:
        chunk_index = {}
    train_ans, train_cites = lookup_train_by_chunk(question, primary["chunk_id"], chunk_index)
    if train_ans is not None:
        cites = [c for c in train_cites if c not in BLACKLIST_CHUNKS]
        return {"id": qid, "answer": train_ans, "citations": cites[:5]}

    # 3b. 规则抽取（默认）
    answer = extract_short_answer(primary["text"], question)
    if not answer or len(answer.strip()) == 0:
        answer = best_sentence(primary["text"], question, max_len=40)
    answer = answer.strip().rstrip("。").rstrip("，")
    if not answer:
        answer = primary["text"][:30]

    # 3c. 如果开启了 LLM 增强，用小模型重做答案（仅非模板题）
    if getattr(answer_question, "_use_llm", False):
        try:
            from .mlx_answerer import llm_answer
            secondary_text = top_chunks[1]["text"] if len(top_chunks) > 1 else None
            llm_ans = llm_answer(question, primary["text"], secondary_text)
            # 若 LLM 说"无法回答"，按拒答处理（仅当规则也无强信号时）
            if "无法回答" in llm_ans or "无法" in llm_ans:
                # 不直接拒答，回退规则
                pass
            elif llm_ans and 1 <= len(llm_ans) <= 60:
                answer = llm_ans
        except Exception as e:
            pass  # 失败时回退规则

    # === 4. 强抽嫌疑检测：仅对 *非模板题* 且答案完整包含模板字段语句（明显是从无关 chunk 强抽）→ 拒答
    # 但若问题本身包含此模板话术（如问"清单必须列明哪几项"），则不拒答。
    # train 上验证：无任何合法答案含这些话术 → 安全
    strong_template_phrases = [
        "未按规则使用的后果是",
        "清单必须列明",
        "提交时必须提供",
        "如需取消，须",
        "预约或申请须",
        "办理或使用地点为",
        "单次使用或处理时长为",
        "主要面向",
    ]
    for p in strong_template_phrases:
        if p in answer and p not in question:
            # 答案含模板话术但问题里没问 → 强抽嫌疑 → 拒答
            return {"id": qid, "answer": REFUSAL, "citations": []}

    # citations: 取 top-2
    cites = [c["chunk_id"] for c in top_chunks[:2]]
    return {"id": qid, "answer": answer, "citations": cites}


def run(test_questions, corpus, train, use_llm=False):
    retriever = BM25Retriever(corpus)
    train_lookup = build_train_lookup(train)
    train_qindex = build_train_question_index(train)
    chunk_index = build_train_chunk_index(train)
    answer_question._chunk_index = chunk_index
    answer_question._use_llm = use_llm
    out = []
    n = len(test_questions)
    import time
    t0 = time.time()
    for i, q in enumerate(test_questions):
        out.append(answer_question(q, corpus, retriever, train_lookup, train_qindex))
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{n}] {time.time()-t0:.1f}s", flush=True)
    return out
