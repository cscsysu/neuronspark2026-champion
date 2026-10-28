"""模板识别 + entity 抽取

题型完全模板化：识别模板 + entity 后，可以直接查 train 表或锁定 corpus chunk。
"""
import re
from .data import normalize_question


# (模板字符串, 标签, 是否多跳)
# 注意：长模板必须排在短模板前，避免短的先匹配
# 模板要尽量长，把"的申请时限、"和"如果既要撤销预约..."这种 metafiller 都吃掉
TEMPLATES = [
    # 多跳/复合题（长模板）
    ("如果既要撤销预约又想知道未按规则的处理，两个答案分别是什么", "CHECK_MULTI", True),
    ("既要撤销预约又想知道未按规则的处理，两个答案分别是什么", "CHECK_MULTI", True),
    ("未按规则的处理，两个答案分别是什么", "CHECK_MULTI", True),
    ("办理前材料和服务对象分别是什么", "CHECK_ACCESS_MULTI", True),
    ("申请时限、办理地点和单次时长三项分别是什么", "ACCESS_MULTI", True),
    ("办理地点和单次时长三项分别是什么", "ACCESS_MULTI", True),
    # 单跳模板
    ("如需取消须提前多久或如何办理", "CHECK", False),
    ("未按规则使用会有什么后果", "CHECK", False),
    ("提交时必须提供什么材料", "CHECK", False),
    ("办理或使用地点在哪里", "ACCESS", False),
    ("需要提前多久预约或申请", "ACCESS", False),
    ("单次使用或处理时长是多少", "ACCESS", False),
]

# 在解析 entity 时，去除常见 entity 后缀 metafiller
ENTITY_TRAILING = ["的申请时限、", "的", "，如果", "，"]


def parse_template(question):
    """
    返回 (template_str, tag, is_multi, entity) 或 (None, None, False, None)
    entity 是问题中模板前的实体名（如"星轨激光切割机"）
    """
    nq = normalize_question(question)
    for tmpl, tag, is_multi in TEMPLATES:
        if tmpl in nq:
            entity = nq.split(tmpl)[0]
            # 剥离前缀
            for pre in ["对于", "关于", "针对"]:
                if entity.startswith(pre):
                    entity = entity[len(pre):]
            entity = entity.strip()
            # 反复剥离常见尾缀
            changed = True
            while changed:
                changed = False
                for suf in ENTITY_TRAILING:
                    if entity.endswith(suf):
                        entity = entity[:-len(suf)].strip()
                        changed = True
                # 标点
                while entity and entity[-1] in "，,、；;":
                    entity = entity[:-1]
                    changed = True
            return tmpl, tag, is_multi, entity
    return None, None, False, None


def is_template_question(question):
    return parse_template(question)[0] is not None
