"""意图识别：判断一条消息是不是在让机器人画图 / 改图。

为什么需要这个模块：
LLM 工具调用依赖模型自身能力，小模型、关闭工具的人设、或者中转接口不支持
function calling 时，「画只猫」就只会变成一句聊天回复。这里用规则做一层兜底，
保证「画画 / 生图」这类明确诉求一定能触发。

原则：宁可漏判（还有 LLM 工具兜底），也不要误判（用户只是在聊画画，
机器人突然甩一张图会很突兀）。所以强触发词直接命中，弱信号一律排除。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Decision:
    action: str = "none"      # none | draw | edit
    prompt: str = ""
    reason: str = ""          # 命中原因，写日志用


# ---------------------------------------------------------------- 词表

# 本插件只出静态图，命中这些词一律不接（视频交给专门的插件）
VIDEO_WORDS = ("视频", "短片", "动画", "运镜", "动起来", "让它动", "mp4", "剪辑")

EDIT_WORDS = (
    "改成", "变成", "换成", "转成", "风格化", "重绘", "重画", "修图", "p成", "p一下",
    "加个", "加上", "去掉", "去掉", "背景换", "换背景", "上色", "动漫化", "手办化",
    "q版化", "卡通化", "水彩", "油画", "素描", "线稿", "换脸", "抠图",
)

# 命中这些词说明用户在「聊画画」而不是「要画」
NEGATIVE_WORDS = (
    "怎么画", "如何画", "为什么", "为啥", "区别", "教程", "教学", "学画", "教我画",
    "原理", "画饼", "画师", "画家", "识别", "分析", "看看", "看下", "描述",
    "评价", "打分", "解读", "重点", "红线", "圈出", "标注", "流程图", "思维导图",
)

POLITE_PREFIX = ("帮我", "给我", "替我", "请", "麻烦", "能不能", "可不可以", "可以", "能")

_DRAW_VERB = r"(?:画|绘制|生成|设计|做|出|整|来|搞)"
_QUANT = r"(?:一|两|几|半)?\s*(?:张|幅|个|只|条|份|位|座|朵|片|棵|辆|架|艘|轮|块|群|堆|套|组)"
_OBJECT = (
    r"(?:图|图片|图像|插画|海报|头像|壁纸|logo|标志|表情包|封面|立绘|原画|配图|"
    r"banner|宣传图|插画图|概念图|设计图|示意图)"
)

# 强触发：动词 + 量词，或明确的关键词
_STRONG_PATTERNS = [
    re.compile(rf"(?:帮我|给我|替我|请|麻烦|能不能|可不可以|可以|能)?\s*{_DRAW_VERB}\s*{_QUANT}"),
    re.compile(r"(?:帮我|给我|替我|请|麻烦)\s*(?:画|绘制|生成|设计)\s*[^\s]{1,40}"),
    re.compile(rf"(?:生成|设计|做|出|整|搞)\s*[^\s]{{0,12}}{_OBJECT}"),
    re.compile(r"(?:生图|文生图|ai绘画|ai画图|ai绘图|画图|出图|来张图|来一张图|整张图)"),
    re.compile(rf"(?:来|整|搞)\s*{_QUANT}\s*(?:图|画)"),
    re.compile(rf"(?:我想要|想要|我要|要)\s*{_QUANT}?\s*[^\s]{{1,24}}{_OBJECT}"),
]

_EDIT_PATTERN = re.compile("|".join(re.escape(w) for w in EDIT_WORDS), re.I)
_VIDEO_PATTERN = re.compile("|".join(re.escape(w) for w in VIDEO_WORDS), re.I)

# 极短消息里裸动词：「画猫」「画只柴犬」
_SHORT_DRAW = re.compile(rf"^{_DRAW_VERB}\s*[^\s，。！？,\.!?]{{1,12}}$")


def detect(
    text: str, has_image: bool = False, extra_keywords: tuple[str, ...] = ()
) -> Decision:
    """识别文本意图。has_image 表示用户这条消息/引用里带了图。

    extra_keywords 是用户在配置里补充的触发词，命中即视为画图。
    """
    raw = (text or "").strip()
    if not raw:
        return Decision()

    text_l = raw.lower()

    # 1. 视频需求交给视频插件，别抢
    if _VIDEO_PATTERN.search(text_l):
        return Decision(reason="视频/动画类需求，不处理")

    # 2. 带图 + 修改指令 = 改图（优先级最高，"把这张改成水彩" 不该重新画）
    if has_image and _EDIT_PATTERN.search(text_l):
        return Decision("edit", clean_prompt(raw), "带图 + 修改指令")

    # 3. 明确在讨论/学习画画，不生成
    for word in NEGATIVE_WORDS:
        if word in text_l:
            return Decision(reason=f"命中排除词「{word}」")

    # 4. 疑问句不是命令：
    #    "你会画图吗" / "能不能画一张" 这类能力询问直接跳过；
    #    "帮我画一张猫好吗" 这种带明确请求前缀的仍放行。
    is_question = bool(re.search(r"[?？]\s*$|[吗嘛]\s*[?？]?\s*$", raw))
    capability_question = bool(
        re.search(r"(?:会不会|能不能|可不可以|可以|能|支持)\s*(?:画|生图|画画|绘图|作图)(?!\s*(?:一|两|几)?\s*(?:张|幅|个|只|条))", raw)
    )
    if capability_question and is_question:
        return Decision(reason="能力询问")
    if is_question and not raw.startswith(POLITE_PREFIX):
        return Decision(reason="疑问句")

    # 5. 自定义触发词
    for word in extra_keywords:
        if word and word.lower() in text_l:
            return Decision("draw", clean_prompt(raw), f"命中自定义触发词「{word}」")

    # 6. 强触发
    for pattern in _STRONG_PATTERNS:
        if pattern.search(raw):
            subject = clean_prompt(raw)
            if not subject:
                return Decision(reason="只有指令没有画面描述")
            return Decision("draw", subject, f"命中规则 {pattern.pattern[:24]}")

    # 7. 短句裸动词：「画猫」
    if len(raw) <= 14 and _SHORT_DRAW.match(raw):
        subject = clean_prompt(raw)
        if subject:
            return Decision("draw", subject, "短句裸动词")

    return Decision(reason="未命中画图意图")


def clean_prompt(text: str) -> str:
    """把「帮我画一张赛博朋克风的猫，谢谢」洗成「赛博朋克风的猫」。"""
    s = (text or "").strip()
    s = re.sub(r"^[\[\(（]?@?\S{1,20}[\]\)）]?\s*", "", s) if s.startswith("@") else s

    # 去掉开头的礼貌语 + 动词 + 数量词（可能套两层：「能不能给我画张…」）
    for _ in range(2):
        s = re.sub(
            r"^(?:帮我|给我|替我|请|麻烦|能不能|可不可以|可以|能|我想要|想要|我要|想|要)\s*",
            "", s,
        )
        s = re.sub(
            rf"^{_DRAW_VERB}\s*(?:一|两|几|半|点)?\s*(?:张|幅|个|只|条|份|位|座|朵|片|棵|辆|架|艘|轮|块|群|堆|套|组)?\s*",
            "", s,
        )
    s = re.sub(r"^(?:的)?(?:图片|图像|插画|画|图)[:：,，]?\s*", "", s)
    s = re.sub(r"^(?:内容|画面)(?:是|为)[:：,，]?\s*", "", s)
    # 「一张猫咪」「只猫」→ 去掉落单的量词
    stripped = re.sub(
        r"^(?:一|两|几|半)?\s*(?:张|幅|个|只|条|份|位|座|朵|片|棵|辆|架|艘|轮|块|群|堆|套|组)\s*",
        "", s,
    )
    if stripped:
        s = stripped
    # 「猫的图」「一张猫咪图片」→ 去掉尾部的「图」
    s = re.sub(r"的(?:图|图片|画)$", "", s)
    s = re.sub(r"(?:图片|图像|插画)$", "", s)

    # 去掉结尾的语气词
    s = re.sub(r"(?:谢谢|多谢|拜托|呗|吧|呀|啊|哦|嘛|吗|呢|哈|了)\s*[。.!！?？~～]*\s*$", "", s)
    s = s.strip(" \t\r\n，。！？、,.!?~～\"'“”")

    return s


def extract_prompt(message_str: str, command_names: tuple[str, ...] = ()) -> str:
    """从 「/画图 一只猫」 里取出 「一只猫」。

    不用 AstrBot 的位置参数，是因为提示词里通常有空格，按空格切会丢内容。
    """
    s = (message_str or "").strip()
    s = re.sub(r"^[/!！.．。]", "", s)
    for name in sorted(command_names, key=len, reverse=True):
        if s.lower().startswith(name.lower()):
            s = s[len(name):]
            break
    else:
        # 没匹配到命令名时，去掉第一个 token，避免把 "/draw" 当提示词
        s = re.sub(r"^\S+\s*", "", s)
    return s.strip()
