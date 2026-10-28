"""mlx-lm 小模型答案抽取器
仅在非模板可答题上调用本地小模型
对模板题/拒答题一律不调用，保持 v4 路径

可选模型 (赛规上限 4B)：
- mlx-community/Qwen2.5-1.5B-Instruct-4bit  (~1GB,  快, F1 ~0.95)
- mlx-community/Qwen2.5-3B-Instruct-4bit    (~1.7GB)
- mlx-community/Qwen3-4B-Instruct-2507-4bit (~2.5GB, 最新最强)
"""
import os, time

# 默认 model: 通过环境变量 NS_MODEL 切换
MODEL_REPO = os.environ.get("NS_MODEL", "mlx-community/Qwen3-4B-Instruct-2507-4bit")

_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        print(f"[mlx] Loading {MODEL_REPO} ...", flush=True)
        t0 = time.time()
        _model, _tokenizer = load(MODEL_REPO)
        print(f"[mlx] Loaded in {time.time()-t0:.1f}s", flush=True)
    return _model, _tokenizer


SYSTEM_PROMPT = """你是图书馆问答员。从知识库片段中**抽取最短答案**。

铁律：
- 答案必须是片段原文中的短语，不要改写
- 优先 2-8 字，绝不超过 30 字
- 不要重复问题里的主语 (如问"普通图书续借几次"，答"2次"，不要答"最多可以续借2次")
- 数字+单位：只给数字和单位 → "14天"、"21:30"、"3次"、"30元"
- 是否能否：只输出"可以"或"不能"
- 地点：只给地点短语 → "学生支持中心"
- 多步问题：用"；"分隔，每段尽量短
- 片段无法回答 → "无法回答"
- 不要写"答案是"、"根据"等前缀

参考样例：
Q: 普通图书最多可以续借几次？  →  2次
Q: 体育场地一个月内未到场2次会暂停多久权限？  →  7天
Q: 移动投影仪需要提前多久申请？  →  3个工作日
Q: 数据导出应使用什么介质？  →  登记过的加密U盘
Q: 装液氮容器能否进电梯？  →  不能"""


def llm_answer(question, primary_text, secondary_text=None, max_tokens=40):
    """喂给小模型抽答案"""
    model, tokenizer = get_model()
    from mlx_lm import generate
    ctx = f"片段1：{primary_text}"
    if secondary_text and secondary_text != primary_text:
        ctx += f"\n片段2：{secondary_text}"
    user_msg = f"{ctx}\n\n问：{question}\n\n直接输出最短答案（原文短语，2-8字最佳）："
    # Qwen3 / Qwen3.5 默认启用 thinking，必须关闭以加速 + 避免输出 thinking 文本
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_msg}],
            enable_thinking=False, **kwargs,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": user_msg}],
            **kwargs,
        )
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return _post_process(out)


def _post_process(s):
    s = s.strip()
    # Qwen3.5 / Qwen3 thinking 残留
    if "</think>" in s:
        s = s.split("</think>")[-1].strip()
    if s.startswith("<think>"):
        # 没正确关闭 — 取最后一行兜底
        s = s.split("\n")[-1].strip()
    # 去引号
    s = s.strip('"\'""''「」『』【】 ')
    # 去前缀
    for p in ["答案是：", "答案是", "答：", "答:", "回答：", "根据片段，", "根据知识库，",
             "根据原文，", "答案：", "原文：", "片段中："]:
        if s.startswith(p):
            s = s[len(p):].strip()
    # 去末尾标点
    s = s.rstrip("。，；,;.! ")
    # 取第一行
    s = s.split("\n")[0].strip()
    # 去掉中文数字+空格 ("2 次" → "2次")
    s = re.sub(r"(\d)\s+(?=[一-龥])", r"\1", s)
    return s


import re
