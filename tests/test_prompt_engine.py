"""提示词引擎的离线用例：解析、降级、缓存、两条调用路径。"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ASTRBOT_ROOT", tempfile.mkdtemp(prefix="astrbot_test_root_"))

import httpx

from prompt_engine import ARTIST_LEVELS, build_system_prompt, PromptEngine


def make_cfg(**over):
    base = {
        "prompt_engine": "custom",
        "llm_base_url": "https://api.test/v1",
        "llm_api_key": "sk-test",
        "llm_model": "deepseek-chat",
        "llm_temperature": 0.7,
        "llm_timeout": 5,
        "prompt_suffix": "",
    }
    base.update(over)
    return lambda key, default=None: base.get(key, default)


GOOD_WITH_DECISIONS = {
    "thought": "他只说猫和月球，我想画一个安静的瞬间。",
    "decisions": ["低机位仰拍", "冷蓝月光配地球暖反光", "加一条旧毛毯压住硬科幻感"],
    "prompt": "一只橘色虎斑猫背对镜头坐在月壤上，低机位仰拍，冷蓝月光，电影级质感",
    "negative": "模糊, 水印",
}

GOOD = {
    "thought": "把猫放在低重力月面，用地球反光做主光源。",
    "prompt": "一只橘色虎斑猫坐在月壤上握着钓竿，背景是巨大的蓝色地球，广角低机位，冷蓝主光",
    "negative": "模糊, 多余肢体, 水印",
}


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


class TestParse(unittest.TestCase):
    def setUp(self):
        self.engine = PromptEngine(make_cfg())

    def test_plain_json(self):
        plan = self.engine._parse(json.dumps(GOOD, ensure_ascii=False), "原话")
        self.assertEqual(plan.source, "llm")
        self.assertIn("橘色虎斑猫", plan.prompt)
        self.assertIn("模糊", plan.negative)
        self.assertTrue(plan.thought)

    def test_json_in_code_fence(self):
        raw = "```json\n" + json.dumps(GOOD, ensure_ascii=False) + "\n```"
        plan = self.engine._parse(raw, "原话")
        self.assertEqual(plan.source, "llm")
        self.assertIn("钓竿", plan.prompt)

    def test_json_with_chatter(self):
        raw = "好的，我来写提示词：\n" + json.dumps(GOOD, ensure_ascii=False) + "\n希望满意。"
        plan = self.engine._parse(raw, "原话")
        self.assertEqual(plan.source, "llm")
        self.assertIn("地球", plan.prompt)

    def test_negative_list_joined(self):
        data = dict(GOOD, negative=["模糊", "水印"])
        plan = self.engine._parse(json.dumps(data, ensure_ascii=False), "原话")
        self.assertEqual(plan.negative, "模糊, 水印")

    def test_plain_text_fallback_as_prompt(self):
        plan = self.engine._parse("一只猫坐在窗台，午后逆光，35mm 浅景深", "原话")
        self.assertEqual(plan.source, "llm")
        self.assertIn("窗台", plan.prompt)

    def test_empty_returns_rule(self):
        plan = self.engine._parse("   ", "画只猫")
        self.assertEqual(plan.source, "rule")
        self.assertTrue(plan.prompt)

    def test_missing_prompt_field_falls_back(self):
        plan = self.engine._parse('{"thought":"想了"}', "画只猫")
        self.assertEqual(plan.source, "rule")

    def test_garbage_falls_back(self):
        plan = self.engine._parse("{不是合法 json", "画只猫")
        # 长度够就会被当成提示词用，不能是空
        self.assertTrue(plan.prompt)


class TestArtistPrompt(unittest.TestCase):
    def test_levels_differ(self):
        texts = {lvl: build_system_prompt(lvl) for lvl in ARTIST_LEVELS}
        self.assertEqual(len(set(texts.values())), 3)
        self.assertIn("主导者", texts["assertive"])
        self.assertIn("以委托人的描述为准", texts["obedient"])

    def test_unknown_level_falls_back(self):
        self.assertEqual(
            build_system_prompt("whatever"), build_system_prompt("balanced")
        )

    def test_persona_injected(self):
        text = build_system_prompt("balanced", "偏爱胶片质感的街头摄影师")
        self.assertIn("偏爱胶片质感的街头摄影师", text)

    def test_history_injected(self):
        text = build_system_prompt("balanced", "", "一只猫在月球钓鱼")
        self.assertIn("一只猫在月球钓鱼", text)
        self.assertIn("不要含含糊糊地重复上一版", text)

    def test_no_history_no_noise(self):
        self.assertNotIn("上一次", build_system_prompt("balanced"))

    def test_decisions_parsed(self):
        engine = PromptEngine(make_cfg())
        plan = engine._parse(json.dumps(GOOD_WITH_DECISIONS, ensure_ascii=False), "原话")
        self.assertEqual(len(plan.decisions), 3)
        self.assertIn("低机位仰拍", plan.decisions)
        self.assertEqual(plan.source, "llm")

    def test_decisions_string_wrapped(self):
        data = dict(GOOD_WITH_DECISIONS, decisions="只做了一条决定")
        plan = PromptEngine(make_cfg())._parse(json.dumps(data, ensure_ascii=False), "原话")
        self.assertEqual(plan.decisions, ["只做了一条决定"])

    def test_decisions_capped(self):
        data = dict(GOOD_WITH_DECISIONS, decisions=[f"决定{i}" for i in range(10)])
        plan = PromptEngine(make_cfg())._parse(json.dumps(data, ensure_ascii=False), "原话")
        self.assertEqual(len(plan.decisions), 4)


class TestCustomAPI(unittest.TestCase):
    def test_success(self):
        seen = []

        def handler(request):
            seen.append(request)
            body = json.loads(request.content)
            self.assertEqual(body["model"], "deepseek-chat")
            self.assertEqual(body["messages"][0]["role"], "system")
            return chat_response(json.dumps(GOOD, ensure_ascii=False))

        engine = PromptEngine(
            make_cfg(), transport=httpx.MockTransport(handler)
        )
        plan = asyncio.run(engine.compile("画一只在月球钓鱼的猫"))
        self.assertEqual(plan.source, "llm")
        self.assertIn("钓竿", plan.prompt)
        self.assertEqual(len(seen), 1)

    def test_reference_image_hint(self):
        captured = {}

        def handler(request):
            captured["body"] = json.loads(request.content)
            return chat_response(json.dumps(GOOD, ensure_ascii=False))

        engine = PromptEngine(make_cfg(), transport=httpx.MockTransport(handler))
        asyncio.run(engine.compile("改成水彩", has_ref_image=True))
        user_msg = captured["body"]["messages"][1]["content"]
        self.assertIn("参考图", user_msg)

    def test_http_error_falls_back_to_rule(self):
        def handler(request):
            return httpx.Response(401, text="invalid key")

        engine = PromptEngine(make_cfg(), transport=httpx.MockTransport(handler))
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "rule")
        self.assertEqual(plan.prompt, "猫")
        self.assertIn("401", plan.extra.get("fallback", ""))

    def test_timeout_falls_back(self):
        async def slow(request):
            await asyncio.sleep(3)
            return chat_response("{}")

        engine = PromptEngine(
            make_cfg(llm_timeout=1),
            transport=httpx.MockTransport(slow),
        )
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "rule")
        self.assertIn("timeout", plan.extra.get("fallback", ""))

    def test_cache_hits(self):
        calls = []

        def handler(request):
            calls.append(1)
            return chat_response(json.dumps(GOOD, ensure_ascii=False))

        engine = PromptEngine(make_cfg(), transport=httpx.MockTransport(handler))

        async def run():
            return [await engine.compile("画只猫") for _ in range(3)]

        plans = asyncio.run(run())
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(p.source == "llm" for p in plans))

    def test_history_changes_cache_key(self):
        calls = []

        def handler(request):
            calls.append(1)
            return chat_response(json.dumps(GOOD, ensure_ascii=False))

        engine = PromptEngine(make_cfg(), transport=httpx.MockTransport(handler))

        async def run():
            await engine.compile("画只猫", history="上次画的是狗")
            await engine.compile("画只猫", history="上次画的是猫")

        asyncio.run(run())
        self.assertEqual(len(calls), 2)

    def test_mode_off_never_calls_api(self):
        calls = []

        def handler(request):
            calls.append(1)
            return chat_response("{}")

        engine = PromptEngine(
            make_cfg(prompt_engine="off"), transport=httpx.MockTransport(handler)
        )
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "rule")
        self.assertEqual(len(calls), 0)

    def test_missing_config_falls_back(self):
        engine = PromptEngine(make_cfg(llm_api_key=""))
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "rule")


class FakeProvider:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def text_chat(self, prompt=None, system_prompt=None, session_id=None, contexts=None, **kw):
        self.calls.append(
            {"prompt": prompt, "system_prompt": system_prompt, "session_id": session_id}
        )

        class R:
            completion_text = self.content

        return R()


class FakeContext:
    def __init__(self, provider):
        self.provider = provider

    def get_using_provider(self, umo=None):
        return self.provider

    def get_provider_by_id(self, pid):
        return self.provider


class TestAstrbotProvider(unittest.TestCase):
    def test_uses_configured_provider(self):
        provider = FakeProvider(json.dumps(GOOD, ensure_ascii=False))
        engine = PromptEngine(
            make_cfg(prompt_engine="astrbot"),
            context=FakeContext(provider),
        )
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "llm")
        self.assertEqual(len(provider.calls), 1)
        self.assertIsNone(provider.calls[0]["session_id"])
        self.assertIn("委托你画画", provider.calls[0]["system_prompt"])

    def test_no_provider_falls_back(self):
        engine = PromptEngine(
            make_cfg(prompt_engine="astrbot"), context=FakeContext(None)
        )
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "rule")

    def test_no_context_falls_back(self):
        engine = PromptEngine(make_cfg(prompt_engine="astrbot"), context=None)
        plan = asyncio.run(engine.compile("画只猫"))
        self.assertEqual(plan.source, "rule")


if __name__ == "__main__":
    unittest.main()
