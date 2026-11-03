"""生图 API 客户端：屏蔽各家接口差异，只暴露 `generate()`。

支持两种主流形态：
1. api_mode = "images"：OpenAI Images 协议（/images/generations、/images/edits）。
   硅基流动、智谱、302/one-api 中转、OpenAI 官方、xAI 等大多是这套。
2. api_mode = "chat"：Chat Completions 协议，图片以 markdown / URL / base64
   的形式藏在回复文本里（Gemini 兼容层、部分聚合站是这套）。

只依赖 httpx（AstrBot 自带），不引入额外库。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import httpx


class ImageAPIError(Exception):
    """可以直接展示给用户的错误。"""


class _Retryable(Exception):
    """限流 / 5xx / 网络抖动，值得重试。"""


class _SizeUnsupported(Exception):
    """模型不认这个尺寸，需要去掉 size 让厂商用自己的默认值。"""


@dataclass
class ImageRequest:
    prompt: str
    size: str = "1024x1024"
    model: str = ""
    image_path: Optional[str] = None      # 有值 = 改图
    max_canvas: Optional[tuple[int, int]] = None   # 输出画布上限 (宽, 高)
    negative: str = ""                    # 需要避免的元素，由提示词层给出


@dataclass
class ImageResult:
    path: str
    prompt: str
    model: str
    revised_prompt: Optional[str] = None


_DATA_URI = re.compile(r"data:image/[\w.+-]+;base64,([A-Za-z0-9+/=\s]+)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_URL = re.compile(r"https?://[^\s'\"<>)]+\.(?:png|jpe?g|webp|gif)(?:\?[^\s'\"<>)]*)?", re.I)


class ImageClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        api_mode: str = "images",
        timeout: float = 180,
        max_retries: int = 2,
        extra_keys: tuple[str, ...] = (),
        output_dir: str | Path = ".",
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_mode = api_mode if api_mode in ("images", "chat") else "images"
        self.timeout = max(10, int(timeout))
        self.max_retries = max(0, int(max_retries))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._transport = transport

        self._keys = [k.strip() for k in (api_key, *extra_keys) if k and k.strip()]
        self._key_index = 0

    # ------------------------------------------------------------ 对外

    async def generate(self, req: ImageRequest) -> ImageResult:
        if not self.base_url:
            raise ImageAPIError("还没有配置接口地址（base_url），请到插件配置里填写。")
        if not self._keys:
            raise ImageAPIError("还没有配置 API Key，请到插件配置里填写 api_key。")
        if not req.prompt.strip():
            raise ImageAPIError("提示词是空的。")

        # 多数 OpenAI 兼容接口没有独立的 negative 参数，拼进提示词最通用
        if req.negative.strip():
            req = ImageRequest(
                prompt=f"{req.prompt}. Avoid: {req.negative.strip()}",
                size=req.size,
                model=req.model,
                image_path=req.image_path,
                max_canvas=req.max_canvas,
            )

        attempts = self.max_retries + 1
        last_error: Exception | None = None
        size_fallback = False   # 模型不认自定义尺寸时，只回退一次，避免死循环

        for attempt in range(attempts):
            try:
                payload, revised = await self._request(req)
                return ImageResult(
                    path=str(self._save(payload, req.max_canvas)),
                    prompt=req.prompt,
                    model=req.model or self.model,
                    revised_prompt=revised,
                )
            except _SizeUnsupported as exc:
                # 有的模型只接受固定尺寸（如 gpt-image-1 的 1024x1024），
                # 去掉 size 用厂商默认值，再靠本地缩放压到画布上限内。
                if size_fallback or not req.size:
                    last_error = exc
                else:
                    size_fallback = True
                    req = replace(req, size="")
                    continue
            except _Retryable as exc:
                last_error = exc
                self._rotate_key()
                if attempt < attempts - 1:
                    await asyncio.sleep(min(0.6 * (2**attempt), 6))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    await asyncio.sleep(min(0.6 * (2**attempt), 6))

        raise ImageAPIError(f"生成失败（已重试 {attempts} 次）：{_short(last_error)}")

    # ------------------------------------------------------------ 请求

    async def _request(self, req: ImageRequest) -> tuple[bytes, Optional[str]]:
        headers = {"Authorization": f"Bearer {self._current_key}"}
        timeout = httpx.Timeout(self.timeout, connect=15)
        async with httpx.AsyncClient(
            timeout=timeout, transport=self._transport, follow_redirects=True
        ) as client:
            if self.api_mode == "chat":
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=self._chat_payload(req),
                )
                _check(resp)
                return await self._from_chat(client, resp.json())

            if req.image_path:
                with open(req.image_path, "rb") as fh:
                    files = {"image": (Path(req.image_path).name, fh.read(), _mime(req.image_path))}
                data = {"model": self.model, "prompt": req.prompt, "size": req.size}
                resp = await client.post(
                    f"{self.base_url}/images/edits", headers=headers, data=data, files=files
                )
            else:
                resp = await client.post(
                    f"{self.base_url}/images/generations",
                    headers=headers,
                    json=self._images_payload(req),
                )
            _check(resp)
            return await self._from_images(client, resp.json())

    def _images_payload(self, req: ImageRequest) -> dict[str, Any]:
        payload = {"model": self.model, "prompt": req.prompt, "n": 1}
        if req.size and req.size != "auto":
            payload["size"] = req.size
        return payload

    def _chat_payload(self, req: ImageRequest) -> dict[str, Any]:
        content: Any = req.prompt
        if req.image_path:
            content = [
                {"type": "text", "text": req.prompt},
                {"type": "image_url", "image_url": {"url": _to_data_uri(req.image_path)}},
            ]
        return {"model": self.model, "messages": [{"role": "user", "content": content}], "stream": False}

    async def _from_images(self, client: httpx.AsyncClient, data: dict) -> tuple[bytes, Optional[str]]:
        items = data.get("data")
        if isinstance(items, dict):
            items = [items]
        if not items:
            # 有些中转把结果放在 images / output 里
            items = data.get("images") or data.get("output")
        if not isinstance(items, list) or not items:
            raise ImageAPIError(f"接口没有返回图片：{_short(data)}")

        item = items[0]
        if isinstance(item, str):
            item = {"url": item}
        if not isinstance(item, dict):
            raise ImageAPIError(f"无法解析接口返回：{_short(item)}")

        revised = item.get("revised_prompt") or item.get("prompt")
        if item.get("b64_json"):
            return _b64decode(item["b64_json"]), revised
        if item.get("url"):
            return await self._download(client, item["url"]), revised
        if item.get("image_base64"):
            return _b64decode(item["image_base64"]), revised
        raise ImageAPIError(f"返回里既没有 url 也没有 b64：{_short(item)}")

    async def _from_chat(self, client: httpx.AsyncClient, data: dict) -> tuple[bytes, Optional[str]]:
        choices = data.get("choices") or []
        if not choices:
            raise ImageAPIError(f"接口没有返回内容：{_short(data)}")
        message = choices[0].get("message", {})

        # 1) 一些聚合站会把图片放在 message.images 里
        images = message.get("images")
        if isinstance(images, list) and images:
            first = images[0]
            if isinstance(first, str):
                url = first
            else:
                url = first.get("url")
                if not url:
                    image_url = first.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if url:
                return await self._download(client, url), None

        content = message.get("content")
        if isinstance(content, list):      # 多模态数组
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in content
            )
        text = content or ""

        match = _DATA_URI.search(text)
        if match:
            return _b64decode(match.group(1)), None
        match = _MD_IMAGE.search(text) or _URL.search(text)
        if match:
            return await self._download(client, match.group(1)), None

        raise ImageAPIError(f"模型回复里没有找到图片：{_short(text)}")

    async def _download(self, client: httpx.AsyncClient, url: str) -> bytes:
        if url.startswith("data:"):
            match = _DATA_URI.search(url)
            if not match:
                raise ImageAPIError("图片 data URI 解析失败")
            return _b64decode(match.group(1))
        resp = await client.get(url)
        if resp.status_code >= 400:
            raise ImageAPIError(f"下载图片失败：HTTP {resp.status_code}")
        return resp.content

    # ------------------------------------------------------------ 存储

    def _save(self, data: bytes, max_canvas: Optional[tuple[int, int]] = None) -> Path:
        if not data:
            raise ImageAPIError("接口返回了空文件")
        data = _fit_canvas(data, max_canvas)
        suffix = _sniff_suffix(data)
        path = self.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}{suffix}"
        path.write_bytes(data)
        return path

    # ------------------------------------------------------------ Key 轮换

    @property
    def _current_key(self) -> str:
        return self._keys[self._key_index % len(self._keys)]

    def _rotate_key(self) -> None:
        if len(self._keys) > 1:
            self._key_index = (self._key_index + 1) % len(self._keys)


# ---------------------------------------------------------------- 工具函数

def _check(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    detail = _extract_error(resp)
    if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
        raise _Retryable(f"HTTP {resp.status_code} {detail}")
    if resp.status_code in (401, 403):
        raise ImageAPIError(f"鉴权失败（HTTP {resp.status_code}），请检查 API Key。{detail}")
    if resp.status_code == 404:
        raise ImageAPIError(f"接口地址不存在（404），请检查 base_url 是否包含 /v1。{detail}")
    if resp.status_code == 400:
        if re.search(r"\bsize\b|尺寸|分辨率|resolution|width|height", detail, re.I):
            raise _SizeUnsupported(detail)
        raise ImageAPIError(f"请求被拒绝（400），常见原因是模型名或尺寸不支持。{detail}")
    raise ImageAPIError(f"接口报错 HTTP {resp.status_code}：{detail}")


def _extract_error(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:                                       # noqa: BLE001
        return _short(resp.text)
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return _short(err.get("message") or err)
        if err:
            return _short(err)
        return _short(data.get("message") or data)
    return _short(data)


def _short(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text[:limit]


def _b64decode(raw: str) -> bytes:
    cleaned = raw.split(",", 1)[-1] if raw.startswith("data:") else raw
    cleaned = re.sub(r"\s+", "", cleaned)
    try:
        return base64.b64decode(cleaned, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ImageAPIError("图片 base64 解码失败") from exc


def _fit_canvas(data: bytes, max_canvas: Optional[tuple[int, int]]) -> bytes:
    """把图片等比缩到画布上限内；本来就不超限则原样返回。

    厂商返回的尺寸不受我们控制（有的直接给 1536x1024，有的给 2K），
    所以这里做最后一道兜底，保证落到磁盘上的文件一定 ≤ max_canvas。
    """
    if not max_canvas:
        return data
    max_w, max_h = max_canvas
    if max_w <= 0 or max_h <= 0:
        return data
    try:
        from PIL import Image
    except Exception:                                        # noqa: BLE001
        return data          # 没装 Pillow 就跳过，宁可尺寸超限也不丢图

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            width, height = im.size
            if width <= max_w and height <= max_h:
                return data
            scale = min(max_w / width, max_h / height)
            target = (max(1, round(width * scale)), max(1, round(height * scale)))

            fmt = (im.format or "PNG").upper()
            if fmt not in ("PNG", "JPEG", "WEBP"):
                fmt = "PNG"
            resized = im.resize(target, Image.LANCZOS)
            if fmt == "JPEG" and resized.mode not in ("RGB", "L"):
                resized = resized.convert("RGB")

            buf = io.BytesIO()
            if fmt == "JPEG":
                resized.save(buf, "JPEG", quality=92, optimize=True)
            elif fmt == "WEBP":
                resized.save(buf, "WEBP", quality=92)
            else:
                resized.save(buf, "PNG", optimize=True)
            return buf.getvalue()
    except Exception:                                        # noqa: BLE001
        return data


def _sniff_suffix(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"\xff\xd8"):
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    return ".png"


def _mime(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".gif": "image/gif",
    }.get(suffix, "image/png")


def _to_data_uri(path: str) -> str:
    data = Path(path).read_bytes()
    return f"data:{_mime(path)};base64,{base64.b64encode(data).decode()}"
