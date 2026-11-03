"""AstroPainter：给 AstrBot 接上任一生图模型。

三种触发方式，保证「想画图」这件事不会漏：
1. LLM 工具（generate_image / edit_image）—— 模型自己判断，最自然；
2. 指令 /画图、/生图、/draw —— 不依赖模型能力，最稳；
3. 全消息意图检测（auto_detect）—— 模型不支持 function calling 时兜底，
   「帮我画只猫」直接出图，不用加指令。

作者: qianwangshengpaerna
版本: 1.4.0
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.event.filter import EventMessageType
from astrbot.api.star import Context, Star

try:  # 插件在 AstrBot 内以 data.plugins.<name>.main 导入
    from .image_client import ImageAPIError, ImageClient, ImageRequest
    from .intent import VIDEO_WORDS, clean_prompt, detect, extract_prompt
    from .prompt_engine import PromptEngine
except ImportError:  # 兜底：允许直接以脚本/测试方式加载
    from image_client import ImageAPIError, ImageClient, ImageRequest
    from intent import VIDEO_WORDS, clean_prompt, detect, extract_prompt
    from prompt_engine import PromptEngine

PLUGIN_NAME = "astrbot_plugin_astro_painter"
COMMAND_NAMES = ("画图", "draw", "生图", "画画", "ai画图", "绘图", "作图")


class AstroPainter(Star):
    """AstroPainter 生图插件

    用法：
    - /画图 一只在月球上钓鱼的猫   → 文生图
    - 先发一张图，再 /画图 改成水彩风格 → 图生图
    - 直接说「帮我画个赛博朋克风的头像」 → 自动识别并出图
    - /生图状态 → 查看当前模型、额度和配置
    """

    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self._client: Optional[ImageClient] = None
        self._sem: Optional[asyncio.Semaphore] = None
        self._cooldown: dict[str, float] = {}
        self._prompts = PromptEngine(self._cfg, context=context, logger=logger)

    # ------------------------------------------------------------ 配置

    def _cfg(self, key: str, default=None):
        try:
            value = self.config.get(key, default)
        except Exception:                                    # noqa: BLE001
            return default
        return default if value is None else value

    def _output_dir(self) -> Path:
        try:
            from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

            base = Path(get_astrbot_plugin_data_path()) / PLUGIN_NAME
        except Exception:                                    # noqa: BLE001
            base = Path(tempfile.gettempdir()) / PLUGIN_NAME
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _get_client(self) -> ImageClient:
        if self._client is None:
            extra = self._cfg("extra_keys", "") or ""
            self._client = ImageClient(
                base_url=str(self._cfg("base_url", "https://api.openai.com/v1")),
                api_key=str(self._cfg("api_key", "")),
                model=str(self._cfg("model", "gpt-image-1")),
                api_mode=str(self._cfg("api_mode", "images")),
                timeout=int(self._cfg("timeout", 180)),
                max_retries=int(self._cfg("max_retries", 2)),
                extra_keys=tuple(k for k in re.split(r"[\s,;，；]+", extra) if k),
                output_dir=self._output_dir(),
            )
        return self._client

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(max(1, int(self._cfg("max_concurrent", 2))))
        return self._sem

    # ------------------------------------------------------------ 指令

    @filter.command("画图", alias={"draw", "生图", "画画", "ai画图", "绘图", "作图"})
    async def cmd_draw(self, event: AstrMessageEvent):
        """画图 <描述>：生成图片；先发图再用本指令，则按描述改图。"""
        prompt = extract_prompt(event.message_str or "", COMMAND_NAMES)
        if prompt and not clean_prompt(prompt):
            prompt = ""
        if not prompt:
            yield event.plain_result(
                "用法：/画图 一只在月球上钓鱼的猫\n"
                "改图：先发一张图，再发「/画图 改成水彩风格」"
            )
            return

        # 本插件只做静态图，视频需求不接（避免误扣费）
        if any(w in prompt for w in VIDEO_WORDS):
            yield event.plain_result("本插件只生成静态图片，不做视频。想要视频请用视频插件。")
            return

        image_path = await self._image_path(self._pick_image(event))

        # 先做限额/冷却检查，避免先回「正在画」再回「额度用完」
        allowed, reason = await self._guard(event, "command")
        if not allowed:
            yield event.plain_result(reason)
            return

        if self._cfg("show_thought", False):
            # 先把构思发出去，让用户看到它「怎么想」，不用干等
            preview = await self._preview_thought(event, prompt, bool(image_path))
            yield event.plain_result(preview)
        else:
            yield event.plain_result("🎨 收到，正在画，稍等十几秒…")
        path, error = await self._generate(
            event, prompt, image_path, source="command", skip_guard=True
        )
        if error:
            yield event.plain_result(error)
            return

        event.set_extra("astro_painter_handled", True)
        yield event.image_result(path)

    @filter.command("生图状态")
    async def cmd_status(self, event: AstrMessageEvent):
        """生图状态：查看当前模型、接口模式和今日剩余额度。"""
        client = self._get_client()
        key = client._current_key if client._keys else ""          # noqa: SLF001
        masked = f"{key[:6]}…{key[-4:]}" if len(key) > 12 else ("已配置" if key else "未配置")
        used = int(await self.get_kv_data(self._quota_key(event.get_sender_id()), 0) or 0)
        quota = int(self._cfg("daily_quota", 0))
        pool_quota = int(self._cfg("global_daily_quota", 5))

        if pool_quota > 0:
            pool_used = int(await self.get_kv_data(self._pool_key(event), 0) or 0)
            where = {"group": "本群", "user": "你"}.get(self._scope(event), "全站")
            quota_line = f"今日剩余：{where} {max(0, pool_quota - pool_used)} / {pool_quota}"
            if quota > 0:
                quota_line += f"（你还有 {max(0, quota - used)} 张）"
        else:
            quota_line = f"今日剩余：{'不限' if quota <= 0 else max(0, quota - used)}"

        canvas = self._max_canvas()
        canvas_text = f"{canvas[0]}x{canvas[1]}" if canvas else "不限"

        yield event.plain_result(
            f"模型：{self._cfg('model')}\n"
            f"接口：{self._cfg('api_mode')} · {self._cfg('base_url')}\n"
            f"Key：{masked}\n"
            f"{quota_line}\n"
            f"画布上限：{canvas_text}\n"
            f"提示词引擎：{self._engine_label()}\n"
            f"自动识别：{'开' if self._cfg('auto_detect', True) else '关'}"
        )

    # ------------------------------------------------------------ LLM 工具

    @filter.llm_tool(name="generate_image")
    async def tool_generate_image(
        self, event: AstrMessageEvent, prompt: str, size: str = "auto"
    ) -> str:
        """根据文字描述生成图片（AI 画画 / 生图）。

        当用户想要一张图时**必须**调用本工具，不要只用文字回复。触发场景包括：
        - 明确要求：「画一张」「帮我画」「生成一张图」「来个…的图」「生图」
        - 要具体产物：插画、海报、头像、壁纸、logo、表情包、封面、立绘、配图
        - 描述性请求：「画只猫」「来个赛博朋克风的城市」「做张极简海报」

        用户描述通常很简略，请先补全提示词再调用：主体 + 场景 + 风格 + 光线 + 构图，
        英文提示词效果通常更好。例如「画只猫」应补成
        「一只橘猫坐在窗台上，午后逆光，浅景深，写实摄影」。

        Args:
            prompt(string): 补全后的完整画面描述，不要包含「帮我画」这类指令词。
            size(string): 画面比例，可选 auto / 1:1 / 16:9 / 9:16，默认 auto（按用户说的横屏竖屏判断）
        """
        if not (prompt or "").strip():
            return "缺少提示词，无法生成图片。"

        image_path = await self._image_path(self._pick_image(event))
        path, error = await self._generate(
            event, clean_prompt(prompt), image_path, source="tool", size_hint=size
        )
        if error:
            return error

        event.set_extra("astro_painter_handled", True)
        await self._send_image(event, path, prompt)
        return "图片已生成并发送给用户。"

    @filter.llm_tool(name="edit_image")
    async def tool_edit_image(self, event: AstrMessageEvent, instruction: str) -> str:
        """修改用户发来的图片（改图 / 图生图 / P图）。

        当用户发了一张图并希望改动时调用：换风格（水彩、动漫、油画）、换背景、
        加/去元素、上色、Q版化、手办化、修图。用户没有附图时不要调用本工具。

        Args:
            instruction(string): 修改要求，例如「保持构图，改成水彩风格」「背景换成海边」
        """
        image_path = await self._image_path(self._pick_image(event))
        if not image_path:
            return "用户这条消息里没有图片，无法改图。请让用户先发送或引用一张图片。"

        path, error = await self._generate(
            event, (instruction or "restyle the image").strip(), image_path, source="tool"
        )
        if error:
            return error

        event.set_extra("astro_painter_handled", True)
        await self._send_image(event, path, instruction)
        return "改图完成并已发送给用户。"

    # ------------------------------------------------------------ 自动识别

    @filter.event_message_type(EventMessageType.ALL)
    async def auto_draw(self, event: AstrMessageEvent):
        """兜底：不依赖 function calling，直接识别「画图 / 生图」意图。"""
        if not self._cfg("auto_detect", True):
            return
        if event.get_extra("astro_painter_handled"):
            return

        text = (event.message_str or "").strip()
        if not text or text.startswith(("/", "!", "！", ".", "。")):
            return

        image = self._pick_image(event)
        extra_words = tuple(
            w.strip()
            for w in re.split(r"[,，;；\s]+", str(self._cfg("extra_trigger_words", "") or ""))
            if w.strip()
        )
        decision = detect(text, has_image=image is not None, extra_keywords=extra_words)
        if decision.action == "none":
            return

        image_path = await self._image_path(image)
        path, error = await self._generate(
            event, decision.prompt, image_path, source="auto"
        )
        if not path:
            logger.info(f"[{PLUGIN_NAME}] 自动生图跳过：{error or decision.reason}")
            # 冷却时 _guard 返回空串，保持安静；额度/报错则明确告知，避免用户干等
            if error:
                yield event.plain_result(error)
            return

        event.set_extra("astro_painter_handled", True)
        event.stop_event()
        yield event.image_result(path)

    # ------------------------------------------------------------ 内部

    async def _generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        image_path: Optional[str],
        *,
        source: str,
        size_hint: str = "auto",
        skip_guard: bool = False,
    ) -> tuple[Optional[str], str]:
        uid = str(event.get_sender_id())
        if not skip_guard:
            allowed, reason = await self._guard(event, source)
            if not allowed:
                return None, reason

        # 对话模型先把需求想清楚，再交给生图模型
        history = await self._load_history(uid)
        plan = await self._prompts.compile(
            prompt, has_ref_image=bool(image_path), history=history
        )
        final_prompt = plan.prompt or prompt
        event.set_extra("astro_painter_plan", plan)
        if plan.source == "llm":
            logger.info(
                f"[{PLUGIN_NAME}] 提示词改写完成（{plan.model or 'astrbot'}）：{final_prompt[:80]}"
            )

        size = self._resolve_size(prompt, size_hint)
        try:
            async with self._semaphore():
                result = await self._get_client().generate(
                    ImageRequest(
                        prompt=final_prompt,
                        size=size,
                        model=str(self._cfg("model", "gpt-image-1")),
                        image_path=image_path,
                        max_canvas=self._max_canvas(),
                        negative=plan.negative,
                    )
                )
        except ImageAPIError as exc:
            logger.warning(f"[{PLUGIN_NAME}] 生成失败：{exc}")
            return None, f"画失败了：{exc}"
        except Exception as exc:                             # noqa: BLE001
            logger.exception(f"[{PLUGIN_NAME}] 未预期的错误：{exc}")
            return None, "画图时出了点意外，已经记到日志里了，稍后再试一次。"

        self._cooldown[uid] = time.time()
        await self._save_history(uid, plan)
        await self._add_quota(event)
        logger.info(f"[{PLUGIN_NAME}] 生成成功 source={source} size={size} path={result.path}")
        return result.path, ""

    async def _preview_thought(self, event: AstrMessageEvent, prompt: str, has_ref: bool) -> str:
        """先算一遍提示词，把画师的构思发出去。结果会命中 compile 的缓存，不会重复计费。"""
        history = await self._load_history(str(event.get_sender_id()))
        plan = await self._prompts.compile(prompt, has_ref_image=has_ref, history=history)
        if not plan.thought and not plan.decisions:
            return "🎨 收到，正在画，稍等十几秒…"

        lines = ["🎨 我想这么画："]
        if plan.thought:
            lines.append(plan.thought)
        for d in plan.decisions[:4]:
            lines.append(f"· {d}")
        lines.append("")
        lines.append("正在画，稍等…")
        return "\n".join(lines)

    # ------------------------------------------------------------ 画师记忆

    @staticmethod
    def _last_key(uid: str) -> str:
        return f"last:{uid}"

    async def _load_history(self, uid: str) -> str:
        """取这个用户上一张画的构思，让画师知道「我上次给他画了什么」。"""
        if not self._cfg("remember_last", True):
            return ""
        try:
            raw = await self.get_kv_data(self._last_key(uid), "")
        except Exception:                                    # noqa: BLE001
            return ""
        if not raw:
            return ""
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(data, dict):
            return ""
        parts = [str(data.get(k) or "").strip() for k in ("thought", "prompt")]
        return "；".join(p for p in parts if p)[:300]

    async def _save_history(self, uid: str, plan) -> None:
        if not self._cfg("remember_last", True):
            return
        payload = {
            "thought": (plan.thought or "")[:200],
            "decisions": list(plan.decisions or [])[:4],
            "prompt": (plan.prompt or "")[:500],
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            await self.put_kv_data(
                self._last_key(uid), json.dumps(payload, ensure_ascii=False)
            )
        except Exception:                                    # noqa: BLE001
            pass

    async def _guard(self, event: AstrMessageEvent, source: str) -> tuple[bool, str]:
        uid = str(event.get_sender_id())
        cooldown = int(self._cfg("cooldown_seconds", 20))
        if cooldown > 0:
            left = cooldown - (time.time() - self._cooldown.get(uid, 0))
            if left > 0:
                # 自动识别被冷却时保持安静，避免群聊刷屏
                if source == "auto":
                    return False, ""
                return False, f"慢一点，{int(left) + 1} 秒后再试。"

        # 1) 单用户每日额度（默认不限，由额度池统一封顶）
        quota = int(self._cfg("daily_quota", 0))
        if quota > 0:
            used = int(await self.get_kv_data(self._quota_key(uid), 0) or 0)
            if used >= quota:
                return False, f"今天的 {quota} 张额度用完了，明天再来。"

        # 2) 额度池（默认全站共 5 张，作用范围由 quota_scope 决定）
        pool_quota = int(self._cfg("global_daily_quota", 5))
        if pool_quota > 0:
            used_pool = int(await self.get_kv_data(self._pool_key(event), 0) or 0)
            if used_pool >= pool_quota:
                where = {"group": "本群", "user": "你"}.get(self._scope(event), "全站")
                return False, f"今天{where}的 {pool_quota} 张额度已经用完了，明天再来。"
        return True, ""

    async def _add_quota(self, event: AstrMessageEvent) -> None:
        uid = str(event.get_sender_id())
        key = self._quota_key(uid)
        used = int(await self.get_kv_data(key, 0) or 0)
        await self.put_kv_data(key, used + 1)

        pkey = self._pool_key(event)
        p_used = int(await self.get_kv_data(pkey, 0) or 0)
        await self.put_kv_data(pkey, p_used + 1)

    @staticmethod
    def _quota_key(uid: str) -> str:
        return f"quota:{datetime.now().strftime('%Y-%m-%d')}:u:{uid}"

    def _engine_label(self) -> str:
        mode = str(self._cfg("prompt_engine", "astrbot") or "off").strip().lower()
        if mode in ("off", "none", "disable", "disabled", "rule"):
            return "关闭（直接用原话）"
        if mode == "custom":
            model = str(self._cfg("llm_model", "") or "未填模型")
            return f"独立接口 · {model}"
        return "复用 AstrBot 对话模型"

    def _scope(self, event: AstrMessageEvent) -> str:
        """额度池的记账粒度：site（全站）/ group（按群）/ user（按人）。"""
        scope = str(self._cfg("quota_scope", "site") or "site").strip().lower()
        return scope if scope in ("site", "group", "user") else "site"

    def _pool_key(self, event: AstrMessageEvent) -> str:
        """额度池的 KV 键。QQ 群场景下 group 会带群号，每个群各自计数。"""
        scope = self._scope(event)
        if scope == "group":
            try:
                gid = str(event.get_group_id() or "")
            except Exception:                                # noqa: BLE001
                gid = ""
            # 私聊没有群号，退化成每个私聊独立计数
            bucket = f"g:{gid}" if gid else f"p:{event.get_sender_id()}"
        elif scope == "user":
            bucket = f"u:{event.get_sender_id()}"
        else:
            bucket = "__global__"
        return f"quota:{datetime.now().strftime('%Y-%m-%d')}:{bucket}"

    def _resolve_size(self, text: str, hint: str = "auto") -> str:
        mapping = {
            "1:1": str(self._cfg("size_square", "720x720")),
            "16:9": str(self._cfg("size_landscape", "1280x720")),
            "9:16": str(self._cfg("size_portrait", "720x1280")),
        }
        if hint in mapping:
            return mapping[hint]
        if re.search(r"竖屏|竖版|9[:：]16|手机壁纸", text):
            return mapping["9:16"]
        if re.search(r"横屏|横版|16[:：]9|宽屏|banner", text, re.I):
            return mapping["16:9"]
        if re.search(r"方图|正方形|1[:：]1|头像|logo|图标", text, re.I):
            return mapping["1:1"]
        return str(self._cfg("size", "720x720"))

    def _max_canvas(self) -> Optional[tuple[int, int]]:
        """输出画布上限，默认 1280x720（720p）。填 off 关闭。"""
        raw = str(self._cfg("max_canvas", "1280x720") or "").strip().lower()
        if raw in ("", "off", "none", "0", "unlimited", "no"):
            return None
        match = re.fullmatch(r"(\d{2,5})\s*[x×*]\s*(\d{2,5})", raw)
        if not match:
            logger.warning(f"[{PLUGIN_NAME}] max_canvas 配置无法解析（{raw}），回退到 1280x720")
            return (1280, 720)
        return (int(match.group(1)), int(match.group(2)))

    @staticmethod
    def _pick_image(event: AstrMessageEvent):
        try:
            messages = event.get_messages()
        except Exception:                                    # noqa: BLE001
            return None
        for comp in messages:
            if isinstance(comp, Comp.Image):
                return comp
        # 引用消息里的图
        for comp in messages:
            for sub in getattr(comp, "chain", None) or []:
                if isinstance(sub, Comp.Image):
                    return sub
        return None

    async def _image_path(self, image) -> Optional[str]:
        if image is None:
            return None
        try:
            return await image.convert_to_file_path()
        except Exception as exc:                             # noqa: BLE001
            logger.warning(f"[{PLUGIN_NAME}] 读取图片失败：{exc}")
            return None

    async def _send_image(self, event: AstrMessageEvent, path: str, prompt: str) -> None:
        chain = MessageChain()
        if self._cfg("reply_with_prompt", True):
            chain.message(f"🎨 {prompt[:80]}")
        chain.file_image(str(path))
        await event.send(chain)

    async def terminate(self) -> None:
        self._client = None
        self._sem = None
        logger.info(f"[{PLUGIN_NAME}] 已卸载")
