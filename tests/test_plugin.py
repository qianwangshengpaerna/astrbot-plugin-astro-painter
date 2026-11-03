"""插件集成测试：在真实 AstrBot 4.x 环境中加载插件，用假事件跑完整链路。"""

import base64
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

# 必须在 import astrbot 之前设置，否则 AstrBot 会在当前目录生成 data/
os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="astrbot_test_root_"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as plugin_main  # noqa: E402
from astrbot.api.message_components import Image  # noqa: E402
from image_client import ImageClient  # noqa: E402

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode()


def solid_png(width: int, height: int) -> str:
    """生成指定尺寸的纯色 PNG，用来验证画布上限。"""
    import io

    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), (30, 60, 120)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeEvent:
    def __init__(self, text: str = "", images=(), uid: str = "10001", group_id: str = ""):
        self.message_str = text
        self._images = list(images)
        self._uid = uid
        self._group_id = group_id
        self.extras: dict = {}
        self.stopped = False
        self.sent: list = []

    def get_sender_id(self):
        return self._uid

    def get_group_id(self):
        return self._group_id

    def get_sender_name(self):
        return "tester"

    def get_messages(self):
        return self._images

    def plain_result(self, text):
        return ("plain", text)

    def image_result(self, path):
        return ("image", path)

    def chain_result(self, chain):
        return ("chain", chain)

    async def send(self, chain):
        self.sent.append(chain)

    def set_extra(self, key, value):
        self.extras[key] = value

    def get_extra(self, key=None, default=None):
        return self.extras.get(key, default)

    def stop_event(self):
        self.stopped = True

    @property
    def unified_msg_origin(self):
        return "test:10001"


class MockAPI:
    def __init__(self, payload: str = ""):
        self.calls: list[httpx.Request] = []
        self.payload = payload or PNG_B64

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if request.url.path.endswith("/images/edits"):
            return httpx.Response(200, json={"data": [{"b64_json": self.payload}]})
        if request.url.path.endswith("/images/generations"):
            return httpx.Response(200, json={"data": [{"b64_json": self.payload}]})
        return httpx.Response(404)


class FakePromptEngine:
    """记录 compile 调用，返回预设方案。"""

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    async def compile(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        return self.plan


async def collect(generator):
    items = []
    async for item in generator:
        items.append(item)
    return items


class PluginTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api = MockAPI()
        self.tmp = Path(tempfile.mkdtemp(prefix="astro_plugin_test_"))
        self.config = {
            "api_mode": "images",
            "base_url": "https://api.test/v1",
            "api_key": "sk-test",
            "model": "test-model",
            "size": "720x720",
            "size_square": "720x720",
            "size_landscape": "1280x720",
            "size_portrait": "720x1280",
            "max_canvas": "1280x720",
            "timeout": 30,
            "max_retries": 1,
            "cooldown_seconds": 0,
            "max_concurrent": 2,
            "auto_detect": True,
            "prompt_engine": "off",   # 这组用例只测生图链路，不测对话模型改写
            "extra_trigger_words": "",
            "reply_with_prompt": True,
        }
        self.plugin = plugin_main.AstroPainter(context=object(), config=self.config)
        self.plugin.plugin_id = "astrbot_plugin_astro_painter"
        self.plugin._client = ImageClient(
            base_url="https://api.test/v1",
            api_key="sk-test",
            model="test-model",
            output_dir=self.tmp,
            transport=httpx.MockTransport(self.api),
        )
        self._store: dict = {}

        async def get_kv(key, default=None):
            return self._store.get(key, default)

        async def put_kv(key, value):
            self._store[key] = value

        self.plugin.get_kv_data = get_kv
        self.plugin.put_kv_data = put_kv

    # ------------------------------------------------ 指令

    async def test_command_draw(self):
        event = FakeEvent("画图 一只在月球上钓鱼的猫")
        results = await collect(self.plugin.cmd_draw(event))

        self.assertEqual(results[0][0], "plain")
        self.assertEqual(results[-1][0], "image")
        self.assertTrue(Path(results[-1][1]).exists())
        self.assertTrue(any("images/generations" in str(c.url) for c in self.api.calls))

    async def test_artist_thought_shown_with_decisions(self):
        from prompt_engine import PromptPlan

        self.config["show_thought"] = True
        self.plugin._prompts = FakePromptEngine(
            PromptPlan(
                prompt="一只猫背对镜头坐在月壤上，低机位仰拍",
                thought="我只画一个安静的瞬间。",
                decisions=["低机位仰拍", "冷蓝月光配地球暖反光"],
                source="llm",
            )
        )
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 猫在月球钓鱼")))
        self.assertIn("我想这么画", results[0][1])
        self.assertIn("安静的瞬间", results[0][1])
        self.assertIn("低机位仰拍", results[0][1])
        self.assertIn("冷蓝月光", results[0][1])
        self.assertEqual(results[-1][0], "image")

    async def test_generated_prompt_comes_from_artist(self):
        from prompt_engine import PromptPlan

        self.plugin._prompts = FakePromptEngine(
            PromptPlan(prompt="画师自己决定的完整提示词", source="llm")
        )
        await collect(self.plugin.cmd_draw(FakeEvent("画图 猫")))
        body = json.loads(self.api.calls[-1].content)
        self.assertEqual(body["prompt"], "画师自己决定的完整提示词")

    async def test_remembers_last_artwork(self):
        from prompt_engine import PromptPlan

        engine = FakePromptEngine(PromptPlan(prompt="一只猫在月球钓鱼", thought="安静的一瞬"))
        self.plugin._prompts = engine

        await collect(self.plugin.cmd_draw(FakeEvent("画图 猫在月球钓鱼")))
        key = self.plugin._last_key("10001")
        self.assertIn(key, self._store)
        saved = json.loads(self._store[key])
        self.assertIn("月球", saved["prompt"])

        # 第二次应把上一版作为参考传给画师
        engine.plan = PromptPlan(prompt="第二版", thought="换个方向")
        await collect(self.plugin.cmd_draw(FakeEvent("画图 再来一张")))
        self.assertIn("月球", engine.calls[-1].get("history", ""))

    async def test_remember_last_can_be_disabled(self):
        from prompt_engine import PromptPlan

        self.config["remember_last"] = False
        engine = FakePromptEngine(PromptPlan(prompt="一只猫", thought="想法"))
        self.plugin._prompts = engine

        await collect(self.plugin.cmd_draw(FakeEvent("画图 猫")))
        self.assertNotIn(self.plugin._last_key("10001"), self._store)
        await collect(self.plugin.cmd_draw(FakeEvent("画图 再画一次")))
        self.assertEqual(engine.calls[-1].get("history"), "")

    async def test_command_without_prompt_shows_usage(self):
        event = FakeEvent("画图")
        results = await collect(self.plugin.cmd_draw(event))
        self.assertEqual(len(results), 1)
        self.assertIn("用法", results[0][1])

    async def test_cooldown(self):
        self.plugin.config["cooldown_seconds"] = 30
        event = FakeEvent("画图 猫")
        await collect(self.plugin.cmd_draw(event))

        event2 = FakeEvent("画图 狗")
        results = await collect(self.plugin.cmd_draw(event2))
        self.assertEqual(len(results), 1)
        self.assertIn("慢一点", results[0][1])
        self.assertEqual(len(self.api.calls), 1)

    async def test_daily_quota(self):
        self.plugin.config["daily_quota"] = 1
        self.plugin.config["global_daily_quota"] = 0
        await collect(self.plugin.cmd_draw(FakeEvent("画图 猫")))
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 狗")))
        self.assertIn("额度", results[0][1])
        self.assertEqual(len(self.api.calls), 1)

    async def test_quota_scope_group(self):
        """按群限额：两个 QQ 群各自 5 张，互不占用。"""
        self.plugin.config["quota_scope"] = "group"
        for i in range(5):
            results = await collect(
                self.plugin.cmd_draw(FakeEvent(f"画图 猫{i}", uid=f"u{i}", group_id="111"))
            )
            self.assertEqual(results[-1][0], "image", f"群 A 第 {i + 1} 张应成功")

        # 群 A 额度用尽
        blocked = await collect(
            self.plugin.cmd_draw(FakeEvent("画图 狗", uid="u8", group_id="111"))
        )
        self.assertIn("本群", blocked[0][1])

        # 群 B 仍有 5 张
        ok = await collect(
            self.plugin.cmd_draw(FakeEvent("画图 猫", uid="u9", group_id="222"))
        )
        self.assertEqual(ok[-1][0], "image")

    async def test_quota_scope_site_shares_across_groups(self):
        """默认全站限额：不同群抢夺同一个池子。"""
        self.plugin.config.pop("quota_scope", None)
        for i in range(5):
            await collect(
                self.plugin.cmd_draw(FakeEvent(f"画图 猫{i}", uid=f"u{i}", group_id=f"g{i}"))
            )
        blocked = await collect(
            self.plugin.cmd_draw(FakeEvent("画图 狗", uid="u9", group_id="new"))
        )
        self.assertIn("全站", blocked[0][1])
        self.assertEqual(len(self.api.calls), 5)

    async def test_quota_scope_user_cross_group(self):
        """按人限额：同一个人换群也共用 5 张。"""
        self.plugin.config["quota_scope"] = "user"
        for i in range(5):
            await collect(
                self.plugin.cmd_draw(FakeEvent(f"画图 猫{i}", uid="same", group_id=f"g{i}"))
            )
        blocked = await collect(
            self.plugin.cmd_draw(FakeEvent("画图 狗", uid="same", group_id="another"))
        )
        self.assertIn("你", blocked[0][1])

    async def test_default_global_quota_is_five(self):
        """默认全站每天总共 5 张，第 6 张被拦——换个人也一样。"""
        for i in range(5):
            results = await collect(
                self.plugin.cmd_draw(FakeEvent(f"画图 猫{i}", uid=f"u{i}"))
            )
            self.assertEqual(results[-1][0], "image", f"第 {i + 1} 张应成功")

        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 第六张", uid="u9")))
        self.assertEqual(len(results), 1)
        self.assertIn("5 张额度", results[0][1])
        self.assertEqual(len(self.api.calls), 5)

    async def test_global_quota_blocks_everyone(self):
        """全站额度用完，换个人也画不了。"""
        self.plugin.config["daily_quota"] = 0        # 放开单人限制
        self.plugin.config["global_daily_quota"] = 2
        for i in range(2):
            await collect(self.plugin.cmd_draw(FakeEvent(f"画图 猫{i}", uid=f"u{i}")))
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 狗", uid="u9")))
        self.assertIn("2 张额度", results[0][1])
        self.assertEqual(len(self.api.calls), 2)

    async def test_status_shows_quota_and_canvas(self):
        results = await collect(self.plugin.cmd_status(FakeEvent("生图状态")))
        text = results[0][1]
        self.assertIn("今日剩余：全站", text)
        self.assertIn("1280x720", text)

    async def test_status_command(self):
        results = await collect(self.plugin.cmd_status(FakeEvent("生图状态")))
        text = results[0][1]
        self.assertIn("test-model", text)
        self.assertIn("今日剩余", text)

    # ------------------------------------------------ 自动识别

    async def test_auto_detect_draws(self):
        event = FakeEvent("帮我画一只橘猫")
        results = await collect(self.plugin.auto_draw(event))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "image")
        self.assertTrue(event.stopped)
        self.assertTrue(event.get_extra("astro_painter_handled"))

    async def test_auto_detect_ignores_chat(self):
        event = FakeEvent("你好，今天天气不错")
        results = await collect(self.plugin.auto_draw(event))
        self.assertEqual(results, [])
        self.assertFalse(event.stopped)
        self.assertEqual(self.api.calls, [])

    async def test_auto_detect_skips_commands(self):
        event = FakeEvent("/画图 猫")
        results = await collect(self.plugin.auto_draw(event))
        self.assertEqual(results, [])

    async def test_auto_detect_can_be_disabled(self):
        self.plugin.config["auto_detect"] = False
        results = await collect(self.plugin.auto_draw(FakeEvent("帮我画一只橘猫")))
        self.assertEqual(results, [])

    async def test_auto_detect_edit_with_image(self):
        ref = self.tmp / "ref.png"
        ref.write_bytes(PNG)
        event = FakeEvent("把这张图改成水彩风格", images=[Image.fromFileSystem(str(ref))])

        results = await collect(self.plugin.auto_draw(event))
        self.assertEqual(results[0][0], "image")
        self.assertTrue(any("images/edits" in str(c.url) for c in self.api.calls))
        self.assertIn(b'name="image"', self.api.calls[0].read())

    # ------------------------------------------------ LLM 工具

    async def test_llm_tool_generate(self):
        event = FakeEvent("画只橘猫")
        reply = await self.plugin.tool_generate_image(event, "一只橘猫，午后逆光")

        self.assertIn("已生成", reply)
        self.assertEqual(len(event.sent), 1)
        from astrbot.api.message_components import Image as CompImage

        self.assertTrue(any(isinstance(c, CompImage) for c in event.sent[0].chain))

    async def test_llm_tool_edit_requires_image(self):
        event = FakeEvent("改一下")
        reply = await self.plugin.tool_edit_image(event, "改成水彩")
        self.assertIn("没有图片", reply)
        self.assertEqual(self.api.calls, [])

    async def test_llm_tool_edit_with_image(self):
        ref = self.tmp / "ref.png"
        ref.write_bytes(PNG)
        event = FakeEvent("改成水彩", images=[Image.fromFileSystem(str(ref))])
        reply = await self.plugin.tool_edit_image(event, "保持构图，改成水彩风格")

        self.assertIn("改图完成", reply)
        self.assertTrue(any("images/edits" in str(c.url) for c in self.api.calls))

    # ------------------------------------------------ 尺寸

    async def test_size_detection(self):
        self.assertEqual(self.plugin._resolve_size("来张竖屏壁纸"), "720x1280")
        self.assertEqual(self.plugin._resolve_size("横屏海报"), "1280x720")
        self.assertEqual(self.plugin._resolve_size("做个头像"), "720x720")
        self.assertEqual(self.plugin._resolve_size("随便画", hint="16:9"), "1280x720")

    async def test_defaults_are_720p(self):
        """不传 size 配置时，默认尺寸也不能超过 720p。"""
        plugin = plugin_main.AstroPainter(context=object(), config={"model": "x"})
        plugin.plugin_id = "astrbot_plugin_astro_painter"
        for text, hint in (("画个方图", "auto"), ("随便", "16:9"), ("随便", "9:16")):
            size = plugin._resolve_size(text, hint=hint)
            width, height = (int(v) for v in size.split("x"))
            self.assertLessEqual(max(width, height), 1280, size)
            self.assertLessEqual(min(width, height), 720, size)
        self.assertEqual(plugin._max_canvas(), (1280, 720))

    async def test_canvas_is_downscaled(self):
        """厂商返回 1536x1024，落盘必须压到 1280x720 以内且保持比例。"""
        from PIL import Image as PILImage

        self.api.payload = solid_png(1536, 1024)
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 横屏海报")))
        path = results[-1][1]

        with PILImage.open(path) as im:
            width, height = im.size
        self.assertLessEqual(width, 1280)
        self.assertLessEqual(height, 720)
        # 3:2 → 1080x720，比例不变
        self.assertAlmostEqual(width / height, 1536 / 1024, places=2)

    async def test_canvas_small_image_untouched(self):
        """本来就小于 720p 的图不该被二次压缩。"""
        from PIL import Image as PILImage

        self.api.payload = solid_png(640, 480)
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 小猫")))
        with PILImage.open(results[-1][1]) as im:
            self.assertEqual(im.size, (640, 480))

    async def test_size_unsupported_falls_back(self):
        """模型不认 1280x720 时，去掉 size 重试一次，并仍然压到画布上限。"""
        from PIL import Image as PILImage

        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(
                    400, json={"error": {"message": "Invalid value for size: 1280x720"}}
                )
            return httpx.Response(200, json={"data": [{"b64_json": solid_png(1024, 1024)}]})

        self.plugin._client = ImageClient(
            base_url="https://api.test/v1",
            api_key="sk-test",
            model="test-model",
            output_dir=self.tmp,
            transport=httpx.MockTransport(handler),
        )
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 横屏海报")))
        self.assertEqual(results[-1][0], "image")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("size", json.loads(calls[1].content))
        with PILImage.open(results[-1][1]) as im:
            self.assertLessEqual(max(im.size), 1280)

    async def test_video_request_rejected(self):
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 生成一段视频：宇航员在沙漠走")))
        self.assertEqual(len(results), 1)
        self.assertIn("不做视频", results[0][1])
        self.assertEqual(len(self.api.calls), 0)

    async def test_error_is_friendly(self):
        self.plugin._client = ImageClient(
            base_url="https://api.test/v1",
            api_key="sk-test",
            model="test-model",
            output_dir=self.tmp,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, json={"error": {"message": "bad key"}})
            ),
        )
        results = await collect(self.plugin.cmd_draw(FakeEvent("画图 猫")))
        self.assertIn("鉴权", results[-1][1])


class PackagingTest(unittest.TestCase):
    """发布前的静态检查：配置项、元数据、许可证文件。"""

    def test_schema_covers_all_config_keys(self):
        import re

        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        used = set(re.findall(r'self\._cfg\(\s*"([a-z_]+)"', source))
        missing = used - set(schema)
        self.assertFalse(missing, f"以下配置项缺少 schema: {missing}")

    def test_metadata(self):
        import yaml

        meta = yaml.safe_load((ROOT / "metadata.yaml").read_text(encoding="utf-8"))
        self.assertEqual(meta["name"], ROOT.name)
        self.assertRegex(meta["version"], r"^\d+\.\d+\.\d+$")
        self.assertTrue(meta["desc"].strip())
        self.assertTrue(meta["repo"].startswith("https://github.com/"))

    def test_required_files(self):
        for name in ("main.py", "metadata.yaml", "_conf_schema.json", "requirements.txt", "README.md"):
            self.assertTrue((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
