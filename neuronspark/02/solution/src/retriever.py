"""检索器：BM25 + 简单规则匹配 (TF-IDF over jieba tokens)"""
import re
import math
from collections import Counter, defaultdict


def tokenize_zh(text):
    """jieba 分词，外加 char-bigram 增强（短文本/专名匹配更稳）"""
    try:
        import jieba
        toks = list(jieba.cut(text))
    except ImportError:
        toks = list(text)
    toks = [t.strip() for t in toks if t.strip() and not re.match(r"^\s+$", t)]
    # 加上 char bigram 提升专名匹配
    chars = re.sub(r"\s+", "", text)
    bigrams = [chars[i:i+2] for i in range(len(chars)-1)]
    return toks + bigrams


class BM25Retriever:
    """简易 BM25, k1=1.5, b=0.75"""
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus  # list of dict (chunk)
        self.k1 = k1
        self.b = b
        self.docs = [tokenize_zh(c["title"] + "。" + c["text"]) for c in corpus]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(1, self.N)
        # df
        self.df = defaultdict(int)
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        # idf
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for t, df in self.df.items()}
        # tf
        self.tf = [Counter(d) for d in self.docs]

    def score(self, query_tokens, idx):
        s = 0.0
        dl = len(self.docs[idx])
        for t in query_tokens:
            if t not in self.idf:
                continue
            f = self.tf[idx].get(t, 0)
            if f == 0:
                continue
            num = f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf[t] * num / den
        return s

    def topk(self, query, k=10):
        qtok = tokenize_zh(query)
        scores = [(i, self.score(qtok, i)) for i in range(self.N)]
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def entity_match_chunks(corpus, entity, tag_filter=None, topk=5):
    """根据 entity 字符串在 corpus 的 title+text 中匹配，可按 chunk_id 后缀过滤
    tag_filter: 'CHECK' or 'ACCESS' or None
    返回: list of (chunk, score)，score 是 entity 命中字符数
    """
    if not entity:
        return []
    # 先按 tag 过滤
    candidates = corpus
    if tag_filter:
        candidates = [c for c in corpus if f"-{tag_filter}-" in c["chunk_id"]]
    # 用 entity 与 (title + text) 做最长公共连续子串/子集匹配
    results = []
    ent_clean = re.sub(r"\s+", "", entity)
    if not ent_clean:
        return []
    for c in candidates:
        haystack = c["title"] + c["text"]
        # 尝试整体包含
        if ent_clean in haystack:
            results.append((c, 1000 + len(ent_clean)))
            continue
        # 否则用最长公共子串近似
        best = 0
        for L in range(min(len(ent_clean), 8), 1, -1):
            found = False
            for i in range(len(ent_clean) - L + 1):
                if ent_clean[i:i+L] in haystack:
                    best = max(best, L)
                    found = True
                    break
            if found:
                break
        if best >= 2:
            results.append((c, best))
    results.sort(key=lambda x: -x[1])
    return results[:topk]
