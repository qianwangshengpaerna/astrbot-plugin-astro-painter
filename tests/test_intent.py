"""意图识别的回归测试：这些用例就是「敏感度」的边界。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent import clean_prompt, detect, extract_prompt


class DetectTest(unittest.TestCase):
    def test_should_draw(self):
        cases = [
            "帮我画一只橘猫",
            "画只猫",
            "画一只在月球上钓鱼的猫",
            "生成一张赛博朋克风格的海报",
            "来张图：雨夜街头",
            "做个头像",
            "帮我设计一个咖啡店的 logo",
            "生图：雪山日出",
            "给我生成一张竖屏壁纸",
            "能帮我画一张头像吗",
            "画猫",
            "我想要一张猫咪图片",
            "我想要只猫的图片",
            "可以画一张猫吗",
            "来点二次元的图",
            "生成图片，内容是雪山",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(detect(text).action, "draw", text)

    def test_should_not_draw(self):
        cases = [
            "你好呀",
            "你会画图吗",
            "怎么画一只猫",
            "我想学画画",
            "给我来杯咖啡",
            "画个重点",
            "这张图画得真好",
            "帮我分析一下这张图",
            "生成一段视频：宇航员在沙漠走",
            "你能画吗",
            "讲讲 AI 绘画的原理",
            "画一张",
            "一只猫",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(detect(text).action, "none", text)

    def test_edit_needs_image(self):
        self.assertEqual(detect("把这张图改成水彩风格", has_image=True).action, "edit")
        # 没有图时，改图指令应该退化成普通聊天，而不是凭空画一张
        self.assertEqual(detect("把这张图改成水彩风格", has_image=False).action, "none")

    def test_extra_keywords(self):
        self.assertEqual(detect("出图 雪山", extra_keywords=("出图",)).action, "draw")
        self.assertEqual(detect("纯聊天", extra_keywords=("出图",)).action, "none")

    def test_prompt_cleanup(self):
        self.assertEqual(clean_prompt("帮我画一张赛博朋克风的猫，谢谢"), "赛博朋克风的猫")
        self.assertEqual(clean_prompt("生成一张海报"), "海报")
        self.assertTrue(clean_prompt("画只猫"))
        self.assertEqual(clean_prompt("画一张"), "")
        self.assertEqual(clean_prompt("我想要一张猫咪图片"), "猫咪")

    def test_extract_prompt(self):
        self.assertEqual(extract_prompt("/画图 一只橘猫", ("画图",)), "一只橘猫")
        self.assertEqual(extract_prompt("/draw a cat on the moon", ("draw",)), "a cat on the moon")
        self.assertEqual(extract_prompt("/生图 雨夜 霓虹 街道", ("生图",)), "雨夜 霓虹 街道")


if __name__ == "__main__":
    unittest.main()
