"""非模板题：BM25 检索 + 句子级抽取 (改进版)"""
import re
from .retriever import tokenize_zh


STOP = {"的", "了", "在", "和", "是", "问", "请", "什么", "多少", "多久", "几", "几点",
        "哪里", "如何", "能否", "是否", "需要", "怎样", "怎么", "可以", "知识库", "根据",
        "请问", "按照", "给定", "制度中", "规定", "答案", "尽量", "简短", "请回答", "对于",
        "如果", "或", "及", "至", "或者", "请同时", "给出", "依据", "不要", "使用",
        "外部", "常识", "请", "也", "需", "完成", "提交", "可不可以", "为", "时"}


def split_sentences(text):
    parts = re.split(r"(?<=[。！？\!\?；;])", text)
    return [s.strip() for s in parts if s.strip()]


def question_keywords(question):
    toks = tokenize_zh(question)
    out = set()
    for t in toks:
        t = t.strip()
        if not t or t in STOP or (len(t) == 1 and t not in "几"):
            continue
        # 排除单 char 中文
        if len(t) >= 2 and re.search(r"[一-龥A-Za-z0-9]", t):
            out.add(t)
    return out


def best_sentence(chunk_text, question, max_len=80):
    """找含问题关键词最多、且尽量短的句子；含 "借期/借用" 时优先含此词的句子；
    若问题指定单位 (几次/几天等)，强烈优先含 数字+该单位 的句子。"""
    sents = split_sentences(chunk_text)
    if not sents:
        return chunk_text[:max_len]
    qkw = question_keywords(question)
    # 强 priors
    priors = []
    for kw in ["借期", "借用", "续借", "介质", "时长", "费用", "上限", "尺寸",
               "几点", "停止", "截止", "关闭", "开放", "申请材料", "提交", "材料"]:
        if kw in question:
            priors.append(kw)

    # 强单位 prior: 问题里有"几次"则优先含 数字+次 的句子
    unit_prior = None
    for u_kw, unit in [("几次", "次"), ("几本", "本"), ("几天", "天"), ("几人", "人"),
                        ("几小时", "小时"), ("几分钟", "分钟"),
                        ("多少次", "次"), ("多少本", "本"), ("多少天", "天"),
                        ("多少元", "元"), ("多少人", "人"), ("多少分", "分"),
                        ("多少分钟", "分钟"), ("多少小时", "小时")]:
        if u_kw in question:
            unit_prior = unit
            break

    best = sents[0]
    best_score = -1e9
    for s in sents:
        score = sum(2 for kw in qkw if kw in s)
        score += sum(5 for p in priors if p in s)
        if unit_prior:
            if re.search(r"\d+\s*" + re.escape(unit_prior), s):
                score += 20  # 巨大加分
        score -= len(s) * 0.02
        if score > best_score:
            best = s
            best_score = score
    return best


# 单位映射
TIME_UNITS_LONG = ["个工作日", "个月", "个学期"]
TIME_UNITS_SHORT = ["工作日", "分钟", "小时", "天", "秒", "周", "月", "年"]
COUNT_UNITS = ["元", "人", "本", "台", "分", "件", "份", "页", "张", "次", "GB", "MB", "TB", "kg", "克", "千克", "ml", "升", "米", "公里", "kmH"]


def detect_target_unit(question):
    """从问题里识别想要的单位类型 → 返回单位列表（按优先匹配顺序）"""
    q = question
    units = []
    # 显式单位词 - 最高优先级
    explicit_pairs = [
        ("多少天", ["天"]),
        ("几天", ["天"]),
        ("多少次", ["次"]),
        ("几次", ["次"]),
        ("多少本", ["本"]),
        ("几本", ["本"]),
        ("多少元", ["元"]),
        ("多少分钟", ["分钟"]),
        ("几分钟", ["分钟"]),
        ("多少小时", ["小时"]),
        ("几小时", ["小时"]),
        ("多少人", ["人"]),
        ("几人", ["人"]),
        ("多少台", ["台"]),
        ("多少件", ["件"]),
        ("多少份", ["份"]),
        ("多少页", ["页"]),
        ("多少张", ["张"]),
        ("多少分", ["分"]),
    ]
    for kw, us in explicit_pairs:
        if kw in q:
            return us
    # 含 "借期/借用" 倾向时间单位
    if "借期" in q or "借用" in q or "外借" in q:
        return ["天", "小时", "周"]
    # 模糊
    if "多久" in q or "多长时间" in q or "时长" in q or "时限" in q:
        return TIME_UNITS_LONG + TIME_UNITS_SHORT
    if "几点" in q or "几时" in q or "什么时段" in q or "截止" in q or ("停止" in q and "入场" in q) or "关闭" in q:
        return ["TIME_OF_DAY"]
    if "什么时候" in q or "什么时间" in q:
        # 可能是 "当天/次日/事先" 这种词
        return ["WHEN"]
    if "多少" in q or "几" in q:
        return COUNT_UNITS
    return []


def extract_with_units(text, units):
    """优先按 units 列表匹配"""
    for u in units:
        if u == "TIME_OF_DAY":
            m = re.search(r"\d{1,2}[:：]\d{2}", text)
            if m:
                return m.group(0).replace("：", ":")
            continue
        if u == "WHEN":
            # 词性时间词
            for kw in ["当天", "当日", "次日", "次工作日", "事先", "提前", "事发后", "返校后", "活动结束后"]:
                if kw in text:
                    # 试图带上紧挨着的时间数字
                    m = re.search(re.escape(kw) + r"\s*\d+\s*(?:小时|分钟|天|个工作日|工作日)?", text)
                    if m:
                        return m.group(0).strip()
                    return kw
            continue
        # 数字 + 单位
        pat = r"((?:至少|最长|最少|最多|累计|每次|每日|每天|每月|每年|提前|发车前|入校前|事发后|提交前|返校后|活动结束后|投放前|培训前|任务开始前|后续|超过)?\s*\d+(?:\.\d+)?\s*" + re.escape(u) + r")"
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()
    return None


def extract_yes_no(text):
    """返回 ("可以"/"不能"/"不能外借"/etc.) 或 None"""
    if re.search(r"(不得|不能|禁止|不予|不办理|不接受|不允许|不外借|不再|不准)", text):
        if "外借" in text and "不办理外借" in text:
            return "不能外借"
        return "不能"
    return "可以"


def extract_place(text, question):
    """匹配地点短语：X中心/办公室/窗口/服务台/系统..."""
    # 优先匹配较长完整短语
    pats = [
        r"([一-龥A-Za-z0-9]{1,15}?(?:学生|学校|学院|学籍|教学|后勤|财务|国际|研究生|健康|体育|艺术|科研|创客|工程|图书|信息|宿舍|餐饮|交通|物业)?(?:服务中心|发展中心|资助中心|管理中心|支持中心|咨询中心|训练中心|活动中心|物流中心|服务台|管理处|办公室|管理系统|登记岗|窗口|入口|平台|站|室\b))",
        r"([一-龥A-Za-z0-9]{1,12}?(?:中心|办公室|窗口|服务台|系统|平台|入口|登记岗|加工间|物资间|车辆调度|管理处|站|室|楼|区|柜|场|台|岗|门))",
    ]
    for pat in pats:
        ms = re.findall(pat, text)
        if ms:
            return ms[0]
    return None


def extract_short_answer(chunk_text, question):
    """改进版：先 best_sentence，再按问题想要的单位精确抽"""
    sent = best_sentence(chunk_text, question, max_len=80)
    qtxt = question

    # 0. 否定/特殊：含 "不得/禁止/不允许" 关键句应优先识别
    # "是否仍完全受限" / "是否能共享给校外人员" / "是否允许"  → 看 chunk 里相关句子
    # 也包含 "X吗？" 单 clause 是非问
    is_yes_no = ("是否" in qtxt or "能否" in qtxt or "可不可以" in qtxt
                 or "可以吗" in qtxt or "可以用" in qtxt)
    # "X吗？" — 必须是结尾的"吗" + 单 clause（去末尾"吗?"后整问无其他"?"或"？"，且不含"几点/什么/哪里"等多疑问）
    if "吗" in qtxt and not is_yes_no:
        # 看是否单 clause 是非问
        body = qtxt.rstrip("？?。 ")
        # 多问句切分：含多个 "?" 或 "？"
        n_qs = qtxt.count("？") + qtxt.count("?")
        is_compound = n_qs >= 2
        # 含其他疑问词
        has_other_q = any(w in body for w in ["几点", "几时", "什么时间", "什么时候",
                                                "什么材料", "什么资料", "哪里", "在哪",
                                                "几次", "几天", "几人", "几本", "为什么",
                                                "如何", "怎样", "怎么"])
        if not is_compound and not has_other_q:
            is_yes_no = True

    if is_yes_no:
        has_neg = False
        for s in split_sentences(chunk_text):
            qkw = question_keywords(qtxt)
            if any(k in s for k in qkw) and re.search(r"(不得|不能|禁止|不予|不办理|不接受|不允许|不外借|不准|不再受理)", s):
                has_neg = True
                break
        return "不能" if has_neg else "可以"

    # 0b. "X是多少" 类常量答案（A2, 35元, 等）
    m = re.search(r"(?:为|是)([A-Za-z0-9一-龥]{1,8})(?:[。，；]|$)", sent)
    if m and ("是多少" in qtxt or "尺寸" in qtxt):
        cand = m.group(1)
        if len(cand) <= 6:
            return cand

    # 0c. 即时/无延迟答案
    if ("多久" in qtxt or "什么时候" in qtxt or "多少时间" in qtxt) and ("即时" in chunk_text or "立即" in chunk_text or "实时" in chunk_text):
        # 检查 best_sentence 中是否含"即时生成"等
        m = re.search(r"(即时\s*\w*|立即\s*\w*|实时\s*\w*)", sent)
        if m and (m.group(1) in qtxt or "生成" in qtxt or "完成" in qtxt or "处理" in qtxt or "出"  in qtxt):
            return m.group(1).strip()

    # 1. 单位识别 → 精确抽取
    units = detect_target_unit(qtxt)
    if units:
        ans = extract_with_units(sent, units)
        if ans:
            return ans
        ans = extract_with_units(chunk_text, units)
        if ans:
            return ans

    # 2. 地点类
    if "哪里" in qtxt or "什么地方" in qtxt or "在哪" in qtxt or "什么窗口" in qtxt or "交至" in qtxt or "提交至" in qtxt or "在何处" in qtxt:
        place = extract_place(sent, qtxt)
        if place:
            return place
        place = extract_place(chunk_text, qtxt)
        if place:
            return place

    # 3. 什么材料 / 什么方式
    if "什么材料" in qtxt or "什么资料" in qtxt or "提交什么" in qtxt:
        m = re.search(r"提供(.+?)(?:。|；)", sent)
        if m: return m.group(1)
        m = re.search(r"(?:须|需)(?:提交|提供)(.+?)(?:。|；)", sent)
        if m: return m.group(1)
    if "什么方式" in qtxt or "什么介质" in qtxt or "通过什么" in qtxt or "用什么" in qtxt or "应使用什么" in qtxt:
        m = re.search(r"使用([^，。；]+?)(?:转移|发送|结算)", sent)
        if m: return m.group(1)
        m = re.search(r"通过([^，。；]+?)(?:结算|发送|转移|认证)", sent)
        if m: return m.group(1)
        # "应使用什么转移" → 抽 "登记过的加密U盘转移" 或 "登记过的加密U盘"
        m = re.search(r"使用(.{2,15}?)(?=[。，；]|不得)", sent)
        if m: return m.group(1)

    # 4. "几点" / 时间点 — 找最相关的时间点（按问题词倾向）
    if "几点" in qtxt or "几时" in qtxt:
        # 偏向于"晚点"语义：归还/截止/关闭/停止/结束 → 取最大时间
        late_kw = ["归还", "截止", "关闭", "停止", "结束", "退场", "下班"]
        is_late = any(k in qtxt for k in late_kw)
        sents = split_sentences(chunk_text)
        qkw = question_keywords(qtxt)
        prior_sents = sorted(sents, key=lambda s: -sum(1 for k in qkw if k in s))
        for s in prior_sents:
            times = re.findall(r"\d{1,2}[:：]\d{2}", s)
            if times:
                times = [t.replace("：", ":") for t in times]
                if is_late:
                    # 取最大
                    times.sort(key=lambda t: int(t.split(":")[0]) * 60 + int(t.split(":")[1]))
                    return times[-1]
                return times[0]

    # 5. "谁/由谁处理" 类
    if "谁" in qtxt:
        m = re.search(r"(?:交给|移交|由)([^，。；]+?)(?:处理|审核|确认)", sent)
        if m: return m.group(0).strip()

    # 5b. "应做什么/如何处理/怎么办/怎么处理/应当如何/排班如何处理" 动作类
    if any(k in qtxt for k in ["应做什么", "应当如何", "如何处理", "怎么处理",
                                  "怎么办", "应该怎么", "应如何", "需要做什么",
                                  "排班如何", "怎样处理"]):
        m = re.search(r"[时，,]([^，。；]{2,30})$", sent.rstrip("。"))
        if m:
            after = m.group(1).strip()
            after = re.sub(r"^(使用人|项目组|学生|借用人|读者|申请人|预约人|主办方|主办社团|社团)?(?:要|须|应|需|应当)", "", after).strip()
            if after:
                return after

    # 5c. "X 还是 Y" 选择疑问
    if "还是" in qtxt:
        m = re.search(r"([^，。；?？]+?)还是([^，。；?？]+)", qtxt)
        if m:
            opt_a, opt_b = m.group(1).strip(), m.group(2).strip()
            # 在 chunk 中找哪个出现在动作或结果位置
            for s in split_sentences(chunk_text):
                if opt_b in s and opt_a not in s.split(opt_b)[0][-15:]:
                    return opt_b
                if opt_a in s and opt_b not in s.split(opt_a)[0][-15:]:
                    return opt_a

    # 5d. "位于哪里" 类（chunk 里有 "X位于Y"）
    if "位于" in qtxt:
        m = re.search(r"位于([^，。；]+)", sent)
        if m:
            place = m.group(1).strip().rstrip("。")
            if place: return place

    # 5e. "需要额外提交什么" / "需要提交什么" / "上传什么" → 抽 后短语
    if "提交什么" in qtxt or "提交哪些" in qtxt or "上传什么" in qtxt or "提供什么" in qtxt or "额外提交" in qtxt:
        m = re.search(r"(?:额外)?(?:须|需|要|应)?(?:提交|提供|上传)([^，。；]+)", sent)
        if m:
            ans = m.group(1).strip().rstrip("。")
            if ans and len(ans) <= 30:
                return ans

    # 5f. "X会怎样/什么处理/会受到" 后果类
    if "会怎样" in qtxt or "怎么样" in qtxt or "会受到" in qtxt or "如何处罚" in qtxt or "处罚" in qtxt or "什么处理" in qtxt or "怎样处罚" in qtxt or "会怎么" in qtxt:
        m = re.search(r"(?:将|会|须|应|爽约|逾期|超期|未按|超过)?(?:暂停|扣减|减少|增加|冻结|取消|撤销|补正|拒绝|终止|延期|顺延|视为|降低|处罚)([^。；]*)", sent)
        if m and len(m.group(0)) <= 25:
            return m.group(0)

    # 5g. "开放多久/开放期" → 抽 "开放于X" 或 "开放期为X"
    if "开放多久" in qtxt or "开放期" in qtxt or "开放时段" in qtxt:
        m = re.search(r"开放[于至到为]([^，。；]+)", sent)
        if m:
            ans = m.group(1).strip()
            if ans and len(ans) <= 25:
                return ans

    # 6. 兜底：含关键词的最短分句
    sub_parts = re.split(r"[，,；;]", sent)
    qkw = question_keywords(qtxt)
    if qkw:
        ranked = sorted(sub_parts, key=lambda s: (-(sum(1 for k in qkw if k in s)), len(s)))
        if ranked and any(k in ranked[0] for k in qkw):
            return ranked[0].strip()
    return sent
