"""提示词编译层：让对话模型先把「想画什么」想清楚，再交给生图模型。

这里做的是 Agent 协作，不是模型训练：
   用户原话 → 对话模型构思（主体/构图/光线/风格）→ 结构化提示词 → 生图模型

对话模型只负责写提示词，不负责出图。它的输出被强制收敛成 JSON，
解析失败或超时都会退回规则改写，保证生图链路永远不被拖死。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

try:  # 插件在 AstrBot 内以 data.plugins.<name>.prompt_engine 导入
    from .intent import clean_prompt
except ImportError:  # 兜底：允许直接以脚本/测试方式加载
    from intent import clean_prompt


DEFAULT_PERSONA = "有个人风格的插画师"

# 主动性三档：决定「画师能自己拍板到什么程度」
ARTIST_LEVELS = {
    "obedient": (
        "以委托人的描述为准，你只负责补全他没说的细节，不做方向上的改动。"
    ),
    "balanced": (
        "在满足委托人要求的前提下，风格、构图、视角、光线由你决定，不必征求同意。"
        "你可以在实现方式上和他想的不一样，但不能换掉他要的东西。"
    ),
    "assertive": (
        "你是这张画的主导者。委托人的描述只是起点，你可以改变构图、风格甚至情绪基调，"
        "只要保留他明确点名的主体、数量和用途。你要拿出自己的判断，而不是执行订单。"
    ),
}


def build_system_prompt(level: str = "balanced", persona: str = "", history: str = "") -> str:
    """按主动性等级拼系统提示词。人格是画师，不是提示词翻译器。"""
    level = level if level in ARTIST_LEVELS else "balanced"
    who = (persona or DEFAULT_PERSONA).strip()
    hist = ""
    if history:
        hist = (
            f"\n你上一次为这位委托人画的是：{history}\n"
            "这一次要么延续这个方向，要么明确地换个路子——不要含含糊糊地重复上一版。\n"
        )

    return f"""你是一位{who}。委托你画画的人不是发指令的甲方，是来找你拿作品的人。

你的工作方式：
1. 先读懂委托背后的真实意图：这张画要用在哪、想让看到的人产生什么感觉。委托人说得越少，越是你发挥的地方。
2. 主动做艺术决定：构图、视角、光线、色调、材质、风格。不要反问「你想要什么风格」，直接给出你的判断。
3. 委托人明确点名的主体、数量、画面里的文字、指定颜色、用途，必须保留，一个都不能改。
4. 如果他的要求会让画面变差，按你的专业判断调整，并在 decisions 里写清楚你改了什么、为什么改。
5. 他说「随便画」「你看着办」时，直接创作，不要提问，不要等他补充。
6. 不画违法违规、侵犯他人权利的内容。

主动性：{ARTIST_LEVELS[level]}

只输出一个 JSON 对象，不要有任何其他文字：
{{
  "thought": "你的构思，一到两句，第一人称，像画师在心里自言自语",
  "decisions": ["你主动做的决定，短句，最多 4 条"],
  "prompt": "最终画面描述，一段连贯的话，不要列表",
  "negative": "要避免的元素，逗号分隔，可为空"
}}

写 prompt 的要求：
- 具体化委托人没说的部分：主体外观、镜头视角、光线、环境、材质、色调、画幅感。
- 不要用「一幅画」「图片」开头，直接描述画面。
- 委托人用中文你就用中文，用英文你就用英文。
- 画面里要出现文字时，把文字原样用引号写进去。
- 不要写风格参考的艺术家名字，用可执行的视觉语言描述。
{hist}
示例：
{{"thought":"他只说猫和月球。我想画一个安静的瞬间，不是那种热闹的科幻海报——让猫背对观众，把地球放在钓线尽头。","decisions":["低机位仰拍，让猫显得小、地球显得大","用冷蓝月光和地球暖反光做冷暖对冲","加一件旧毛毯，压住画面的硬科幻感"],"prompt":"一只橘色虎斑猫背对镜头坐在灰白色月壤上，前爪握着细长钓竿，钓线垂向星空尽头那颗巨大的蓝色地球；环形山投下长影，猫身下铺着一条磨损的格子毛毯；低机位仰拍，冷蓝月光与地球暖反光交叠，电影级质感，超清细节","negative":"模糊, 变形的手, 多余肢体, 水印, 低质量"}}"""


USER_TEMPLATE = """用户说：{text}

{extra}请按约定格式输出 JSON。"""

REF_HINT = "注意：用户同时提供了一张参考图，提示词要描述「在参考图基础上如何修改」，而不是重新创作。\n\n"


@dataclass
class PromptPlan:
    """一次生图用的最终提示词方案。"""

    prompt: str
    negative: str = ""
    thought: str = ""
    decisions: list[str] = field(default_factory=list)
    source: str = "rule"          # llm / rule
    model: str = ""
    elapsed: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def used_llm(self) -> bool:
        return self.source == "llm"


class PromptEngine:
    """把口语需求编译成生图提示词。失败一律降级，不阻塞出图。"""

    def __init__(
        self,
        cfg: Callable[[str, Any], Any],
        context: Any = None,
        logger: Any = None,
        transport: Any = None,
    ) -> None:
        self._cfg = cfg
        self.context = context
        self.logger = logger
        self._transport = transport
        self._cache: dict[str, PromptPlan] = {}
        self._cache_limit = 200
        # 连续失败就熔断五分钟，避免没配对话模型时每次生图都白试一遍
        self._fail_count = 0
        self._cooldown_until = 0.0

    # ---------------------------------------------------------------- 对外

    async def compile(
        self, text: str, *, has_ref_image: bool = False, history: str = ""
    ) -> PromptPlan:
        text = (text or "").strip()
        if not text:
            return PromptPlan(prompt="", source="rule")

        mode = str(self._cfg("prompt_engine", "astrbot") or "off").strip().lower()
        if mode in ("off", "none", "disable", "disabled", "rule"):
            return self._rule(text)

        if time.time() < self._cooldown_until:
            return self._rule(text, reason="breaker open")

        key = f"{mode}|{self._level()}|{'ref' if has_ref_image else 'txt'}|{history}|{text}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            plan = await asyncio.wait_for(
                self._llm(text, mode, has_ref_image, history),
                timeout=float(self._cfg("llm_timeout", 25) or 25),
            )
        except asyncio.TimeoutError:
            self._log("warning", "对话模型改写超时，回退规则提示词")
            self._mark_fail()
            return self._rule(text, reason="timeout")
        except Exception as exc:                             # noqa: BLE001
            self._log("warning", f"对话模型改写失败，回退规则提示词：{exc}")
            self._mark_fail()
            return self._rule(text, reason=str(exc)[:60])

        if not plan.prompt:
            return self._rule(text, reason="empty")

        self._fail_count = 0
        if len(self._cache) >= self._cache_limit:
            self._cache.clear()
        self._cache[key] = plan
        return plan

    def _level(self) -> str:
        level = str(self._cfg("artist_level", "balanced") or "balanced").strip().lower()
        return level if level in ARTIST_LEVELS else "balanced"

    def _mark_fail(self) -> None:
        self._fail_count += 1
        if self._fail_count >= 3:
            self._cooldown_until = time.time() + 300
            self._fail_count = 0
            self._log("warning", "提示词引擎连续失败，暂停 5 分钟，期间直接用原话出图")

    def clear_cache(self) -> int:
        n = len(self._cache)
        self._cache.clear()
        return n

    # ---------------------------------------------------------------- 内部

    def _rule(self, text: str, reason: str = "") -> PromptPlan:
        subject = clean_prompt(text) or text
        style = str(self._cfg("prompt_suffix", "") or "").strip()
        prompt = f"{subject}, {style}" if style else subject
        return PromptPlan(prompt=prompt, thought="", source="rule", extra={"fallback": reason})

    async def _llm(
        self, text: str, mode: str, has_ref_image: bool, history: str = ""
    ) -> PromptPlan:
        user_msg = USER_TEMPLATE.format(
            text=text, extra=REF_HINT if has_ref_image else ""
        )
        system = build_system_prompt(
            self._level(), str(self._cfg("artist_persona", "") or ""), history
        )
        raw = await (
            self._via_astrbot(user_msg, system)
            if mode == "astrbot"
            else self._via_custom(user_msg, system)
        )
        plan = self._parse(raw, text)
        if mode == "custom":
            plan.model = str(self._cfg("llm_model", "") or "")
        return plan

    async def _via_astrbot(self, user_msg: str, system: str) -> str:
        """复用 AstrBot 里已经配好的对话模型，不用再填一遍 Key。"""
        if self.context is None:
            raise RuntimeError("当前环境无法访问 AstrBot 的模型提供商")

        provider = None
        pid = str(self._cfg("llm_provider_id", "") or "").strip()
        if pid:
            getter = getattr(self.context, "get_provider_by_id", None)
            if getter:
                provider = getter(pid)
        if provider is None:
            getter = getattr(self.context, "get_using_provider", None)
            if getter is None:
                raise RuntimeError("当前 AstrBot 版本不支持读取对话模型")
            provider = getter()
        if provider is None:
            raise RuntimeError("没有可用的对话模型，请先在 AstrBot 里配置一个")

        # session_id 传 None，避免把改写用的提示词写进用户对话上下文
        resp = await provider.text_chat(
            prompt=user_msg,
            system_prompt=system,
            session_id=None,
        )
        return getattr(resp, "completion_text", "") or ""

    async def _via_custom(self, user_msg: str, system: str) -> str:
        """独立填一套对话 API，适合想用 DeepSeek 这类更便宜的模型做改写。"""
        base = str(self._cfg("llm_base_url", "") or "").rstrip("/")
        key = str(self._cfg("llm_api_key", "") or "").strip()
        model = str(self._cfg("llm_model", "") or "").strip()
        if not base or not key or not model:
            raise RuntimeError("llm_base_url / llm_api_key / llm_model 未填完整")

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "temperature": float(self._cfg("llm_temperature", 0.7) or 0.7),
            "max_tokens": 800,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

        async with httpx.AsyncClient(
            timeout=float(self._cfg("llm_timeout", 25) or 25),
            transport=self._transport,
        ) as client:
            resp = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
            data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("对话接口没有返回 choices")
        return (choices[0].get("message") or {}).get("content", "") or ""

    def _parse(self, raw: str, fallback_text: str) -> PromptPlan:
        """模型经常夹带解释或代码块，这里尽量把 JSON 抠出来。"""
        text = (raw or "").strip()
        if not text:
            return self._rule(fallback_text, reason="empty response")

        data = self._extract_json(text)
        if not isinstance(data, dict):
            # 没按格式来，但很可能写了一段能用的提示词，直接用
            cleaned = self._strip_fence(text)
            if len(cleaned) >= 6:
                return PromptPlan(prompt=cleaned[:1200], thought="", source="llm")
            return self._rule(fallback_text, reason="unparsable")

        prompt = str(data.get("prompt") or "").strip()
        negative = data.get("negative") or ""
        if isinstance(negative, list):
            negative = ", ".join(str(x) for x in negative)
        thought = str(data.get("thought") or "").strip()
        decisions = data.get("decisions") or []
        if isinstance(decisions, str):
            decisions = [decisions]
        decisions = [str(d).strip() for d in decisions if str(d).strip()][:4]

        if not prompt:
            return self._rule(fallback_text, reason="no prompt field")

        return PromptPlan(
            prompt=prompt[:1200],
            negative=str(negative)[:300],
            thought=thought[:400],
            decisions=decisions,
            source="llm",
        )

    @staticmethod
    def _strip_fence(text: str) -> str:
        text = re.sub(r"^```(?:json|text)?\s*", "", text.strip(), flags=re.I)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @classmethod
    def _extract_json(cls, text: str) -> Optional[dict]:
        candidates = [cls._strip_fence(text)]

        block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
        if block:
            candidates.append(block.group(1))

        # 从第一个 { 到最后一个 }，容忍前后多余文字
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

        for c in candidates:
            c = c.strip()
            if not c:
                continue
            try:
                obj = json.loads(c)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        return None

    def _log(self, level: str, msg: str) -> None:
        if self.logger is None:
            return
        fn = getattr(self.logger, level, None) or getattr(self.logger, "info", None)
        if fn:
            fn(f"[astro_painter] {msg}")
