"""HTTP 客户端测试：用 httpx.MockTransport 模拟各家接口，不产生真实计费请求。"""

import base64
import sys
import tempfile
import unittest
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from image_client import ImageAPIError, ImageClient, ImageRequest

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG).decode()


class MockAPI:
    """记录请求并返回预设响应。"""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.fail_times = 0
        self.status = 200
        self.chat_content = "画好了：![img](https://cdn.test/a.png)"
        self.payload: dict | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if self.fail_times > 0:
            self.fail_times -= 1
            return httpx.Response(429, json={"error": {"message": "rate limit"}})

        if path.endswith("/images/generations"):
            self.payload = httpx.Response  # noop
            return httpx.Response(200, json={"data": [{"b64_json": PNG_B64, "revised_prompt": "ok"}]})
        if path.endswith("/images/edits"):
            return httpx.Response(200, json={"data": [{"b64_json": PNG_B64}]})
        if path.endswith("/chat/completions"):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": self.chat_content}}]}
            )
        if request.url.host == "cdn.test":
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
        if path.endswith("/broken"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404, json={"error": {"message": "not found"}})


def build(handler, **kwargs) -> tuple[ImageClient, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="astro_client_test_"))
    client = ImageClient(
        base_url="https://api.test/v1",
        api_key="sk-primary",
        model="test-model",
        output_dir=tmp,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )
    return client, tmp


class ClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_b64(self):
        api = MockAPI()
        client, tmp = build(api)
        result = await client.generate(ImageRequest("一只橘猫", size="1024x1024"))

        self.assertTrue(Path(result.path).exists())
        self.assertEqual(Path(result.path).read_bytes()[:8], PNG[:8])
        self.assertEqual(result.revised_prompt, "ok")

        req = api.requests[0]
        self.assertEqual(req.headers["authorization"], "Bearer sk-primary")
        self.assertTrue(req.url.path.endswith("/images/generations"))
        body = req.read().decode()
        self.assertIn("一只橘猫", body)
        self.assertIn("1024x1024", body)

    async def test_download_url(self):
        api_response = {"data": [{"url": "https://cdn.test/a.png"}]}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "cdn.test":
                return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
            return httpx.Response(200, json=api_response)

        client, _ = build(handler)
        result = await client.generate(ImageRequest("猫"))
        self.assertEqual(Path(result.path).read_bytes(), PNG)

    async def test_retry_on_429(self):
        api = MockAPI()
        api.fail_times = 2
        client, _ = build(api, max_retries=2)
        result = await client.generate(ImageRequest("猫"))
        self.assertTrue(Path(result.path).exists())
        self.assertEqual(len(api.requests), 3)

    async def test_key_rotation(self):
        api = MockAPI()
        api.fail_times = 1
        tmp = Path(tempfile.mkdtemp(prefix="astro_client_test_"))
        client = ImageClient(
            base_url="https://api.test/v1",
            api_key="sk-primary",
            extra_keys=("sk-backup",),
            model="m",
            max_retries=2,
            output_dir=tmp,
            transport=httpx.MockTransport(api),
        )
        await client.generate(ImageRequest("猫"))
        auths = [r.headers["authorization"] for r in api.requests]
        self.assertEqual(auths, ["Bearer sk-primary", "Bearer sk-backup"])

    async def test_auth_error(self):
        client, _ = build(lambda request: httpx.Response(401, json={"error": {"message": "bad key"}}))
        with self.assertRaises(ImageAPIError) as ctx:
            await client.generate(ImageRequest("猫"))
        self.assertIn("鉴权", str(ctx.exception))

    async def test_bad_request_message(self):
        client, _ = build(
            lambda request: httpx.Response(400, json={"error": {"message": "unknown model"}})
        )
        with self.assertRaises(ImageAPIError) as ctx:
            await client.generate(ImageRequest("猫"))
        self.assertIn("400", str(ctx.exception))
        self.assertIn("unknown model", str(ctx.exception))

    async def test_chat_mode_markdown(self):
        api = MockAPI()
        client, _ = build(api, api_mode="chat")
        result = await client.generate(ImageRequest("猫"))
        self.assertEqual(Path(result.path).read_bytes(), PNG)
        self.assertTrue(api.requests[0].url.path.endswith("/chat/completions"))

    async def test_chat_mode_data_uri(self):
        api = MockAPI()
        api.chat_content = f"data:image/png;base64,{PNG_B64}"
        client, _ = build(api, api_mode="chat")
        result = await client.generate(ImageRequest("猫"))
        self.assertEqual(Path(result.path).read_bytes(), PNG)

    async def test_edit_uses_multipart(self):
        api = MockAPI()
        client, _ = build(api)
        ref = Path(tempfile.mkdtemp()) / "ref.png"
        ref.write_bytes(PNG)

        result = await client.generate(ImageRequest("改成水彩", image_path=str(ref)))
        self.assertTrue(Path(result.path).exists())

        req = api.requests[0]
        self.assertTrue(req.url.path.endswith("/images/edits"))
        self.assertIn("multipart/form-data", req.headers["content-type"])
        self.assertIn(b'name="image"', req.read())

    async def test_empty_result(self):
        client, _ = build(lambda request: httpx.Response(200, json={"data": []}))
        with self.assertRaises(ImageAPIError):
            await client.generate(ImageRequest("猫"))

    async def test_missing_key(self):
        _client, _ = build(MockAPI())
        client = ImageClient(base_url="https://api.test/v1", api_key="", model="m")
        with self.assertRaises(ImageAPIError) as ctx:
            await client.generate(ImageRequest("猫"))
        self.assertIn("API Key", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
