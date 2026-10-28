"""模板答案抽取
对 22 个 CHECK chunk 和 22 个 ACCESS chunk，全部字段在正则下 100% 命中。
"""
import re

# ACCESS chunk 字段抽取
ACCESS_PATTERNS = {
    "object":     r"主要面向(.+?)。",
    "lead_time":  r"预约或申请须(.+?)完成",
    "place":      r"办理或使用地点为(.+?)。",
    "duration":   r"单次使用或处理时长为(.+?)。",
}

# CHECK chunk 字段抽取
CHECK_PATTERNS = {
    "material":     r"提交时必须提供(.+?)。",
    "cancel":       r"如需取消，?须(.+?)办理",
    "consequence":  r"未按规则使用的后果是(.+?)。",
}


def extract_access_fields(text):
    out = {}
    for k, p in ACCESS_PATTERNS.items():
        m = re.search(p, text)
        out[k] = m.group(1).strip() if m else None
    return out


def extract_check_fields(text):
    out = {}
    for k, p in CHECK_PATTERNS.items():
        m = re.search(p, text)
        out[k] = m.group(1).strip() if m else None
    return out


def find_chunks_by_entity(corpus, entity, suffix=None):
    """根据 entity 字符串和 chunk_id 后缀，找匹配的 chunk
    suffix: 'CHECK' / 'ACCESS' / None
    返回 list of chunk
    """
    if not entity:
        return []
    cands = corpus
    if suffix:
        cands = [c for c in corpus if f"-{suffix}-" in c["chunk_id"]]
    ent = re.sub(r"\s+", "", entity)
    if not ent:
        return []
    # 1) 整体包含 (title 或 text)
    exact = [c for c in cands if ent in (c["title"] + c["text"])]
    if exact:
        # 优先 text 开头出现的（说明 entity 是该 chunk 的主语）
        def score(c):
            text = c["text"]
            i = text.find(ent)
            return -i if i >= 0 else -len(text)
        exact.sort(key=score, reverse=True)
        return exact
    # 2) 模糊：最长子串重叠
    scored = []
    for c in cands:
        haystack = c["title"] + c["text"]
        best = 0
        for L in range(min(len(ent), 8), 1, -1):
            for i in range(len(ent) - L + 1):
                if ent[i:i+L] in haystack:
                    best = L
                    break
            if best:
                break
        if best >= 2:
            scored.append((c, best))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored]


def answer_template(question, template, tag, entity, corpus):
    """根据模板 tag 抽取答案 + citations
    返回 (answer_str, citations_list) 或 (None, None) 表示未能处理
    """
    if tag == "ACCESS":
        chunks = find_chunks_by_entity(corpus, entity, "ACCESS")
        if not chunks:
            return None, None
        c = chunks[0]
        f = extract_access_fields(c["text"])
        if "办理或使用地点在哪里" in question:
            return f.get("place"), [c["chunk_id"]]
        if "需要提前多久预约或申请" in question:
            return f.get("lead_time"), [c["chunk_id"]]
        if "单次使用或处理时长是多少" in question:
            return f.get("duration"), [c["chunk_id"]]

    elif tag == "CHECK":
        chunks = find_chunks_by_entity(corpus, entity, "CHECK")
        if not chunks:
            return None, None
        c = chunks[0]
        f = extract_check_fields(c["text"])
        if "如需取消须提前多久或如何办理" in question:
            return f.get("cancel"), [c["chunk_id"]]
        if "未按规则使用会有什么后果" in question:
            return f.get("consequence"), [c["chunk_id"]]
        if "提交时必须提供什么材料" in question:
            return f.get("material"), [c["chunk_id"]]

    elif tag == "ACCESS_MULTI":
        # 申请时限、办理地点和单次时长 三项分别是
        chunks = find_chunks_by_entity(corpus, entity, "ACCESS")
        if not chunks:
            return None, None
        c = chunks[0]
        f = extract_access_fields(c["text"])
        parts = [f.get("lead_time"), f.get("place"), f.get("duration")]
        if all(parts):
            return "；".join(parts), [c["chunk_id"]]

    elif tag == "CHECK_MULTI":
        # 撤销预约 + 未按规则的处理 两个答案
        chunks = find_chunks_by_entity(corpus, entity, "CHECK")
        if not chunks:
            return None, None
        c = chunks[0]
        f = extract_check_fields(c["text"])
        cancel = f.get("cancel")
        cons = f.get("consequence")
        if cancel and cons:
            # 训练答案格式: "提前10小时办理；30天内只能预约普通自习席"
            return f"{cancel}办理；{cons}", [c["chunk_id"]]

    elif tag == "CHECK_ACCESS_MULTI":
        # 办理前材料和服务对象分别是什么
        chunks_check = find_chunks_by_entity(corpus, entity, "CHECK")
        chunks_access = find_chunks_by_entity(corpus, entity, "ACCESS")
        if chunks_check and chunks_access:
            c1 = chunks_check[0]
            c2 = chunks_access[0]
            mat = extract_check_fields(c1["text"]).get("material")
            obj = extract_access_fields(c2["text"]).get("object")
            if mat and obj:
                # 训练答案格式: "材料：脚本提纲和版权确认表；对象：课程微课录制和学生项目展示"
                return f"材料：{mat}；对象：{obj}", [c1["chunk_id"], c2["chunk_id"]]

    return None, None
