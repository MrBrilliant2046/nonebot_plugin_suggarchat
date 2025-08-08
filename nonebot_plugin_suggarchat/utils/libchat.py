from __future__ import annotations
from collections import OrderedDict
from time import time
from collections.abc import Iterable
from copy import deepcopy
import base64
import mimetypes
from urllib.parse import urlparse

import nonebot
import openai
from nonebot import logger
from nonebot.adapters.onebot.v11 import (
    Bot,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)

try:
    import httpx  # 用于下载图片为字节
except Exception:  # pragma: no cover
    httpx = None

from ..chatmanager import chat_manager
from ..config import config_manager
from .functions import remove_think_tag
from .protocol import AdapterManager, ModelAdapter

# 确保 httpx 可用（未安装也不影响运行，会降级保留原 URL）
if "httpx" not in globals():
    try:
        import httpx  # 用于下载图片
    except Exception:
        httpx = None
try:
    from PIL import Image
except Exception:
    Image = None
# 缓存/策略常量：先定义，供后续函数使用
_BAD_URL_TTL = globals().get("_BAD_URL_TTL", 30 * 60)           # 坏链接缓存 30 分钟
_DATA_URL_TTL = globals().get("_DATA_URL_TTL", 30 * 60)         # data URL 缓存 30 分钟
_DATA_URL_CACHE_MAX = globals().get("_DATA_URL_CACHE_MAX", 128) # 最多 128 条
_SKIP_KNOWN_BAD_IN_URL_MODE = globals().get("_SKIP_KNOWN_BAD_IN_URL_MODE", True)

# LRU 与坏链缓存
_bad_url_cache: dict[str, float] = {}
_data_url_cache: "OrderedDict[str, tuple[float, str]]" = OrderedDict()

# 限制与目标
_MAX_IMAGE_BYTES = 1 * 1024 * 1024          # 单图最大内联字节（已存在可保留/调整）
_MAX_TOTAL_DATA_URL_BYTES = 1 * 1024 * 1024 # 本次请求内联图片总量上限（已存在可保留/调整）
_MAX_EMBED_IMAGE_COUNT = 3                  # 本次请求最多内联图片数量（已存在可保留/调整）

# 压缩策略参数
_EMBED_MAX_DIM = 1280            # 最大宽高
_EMBED_TARGET_BYTES = 800 * 1024 # 目标大小 ~800KB（会尽量逼近，不保证一定达到）
_EMBED_JPEG_QUALITY_START = 75   # JPEG 起始质量
_EMBED_JPEG_QUALITY_MIN = 50     # JPEG 最低质量
def _collect_image_urls(messages) -> list[str]:
    urls = []
    for m in messages:
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        iu = p.get("image_url")
                        if isinstance(iu, str):
                            urls.append(iu)
                        elif isinstance(iu, dict):
                            u = iu.get("url")
                            if isinstance(u, str):
                                urls.append(u)
    return urls

def _keep_only_latest_user_images(messages) -> list:
    """
    仅保留“最新一条用户消息”中的图片分片，其他消息里的图片分片剥离；
    同时对该消息内的图片 URL 去重，并过滤已知坏 URL。
    """
    cleaned = []
    # 找到最后一条 user 消息的索引
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            last_user_idx = i
            break
    seen = set()
    for idx, m in enumerate(messages):
        if isinstance(m, dict):
            content = m.get("content")
            if isinstance(content, list):
                new_parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        if idx != last_user_idx:
                            # 非最新用户消息中的图片一律剥离
                            continue
                        iu = p.get("image_url")
                        url = iu if isinstance(iu, str) else (iu.get("url") if isinstance(iu, dict) else None)
                        # 去重 + 跳过已知坏链
                        if isinstance(url, str):
                            if url in seen:
                                continue
                            if _is_http_url(url) and _cache_is_bad_url(url):
                                # 用占位避免消息内容为空
                                new_parts.append({"type": "text", "text": "[image omitted: known-bad-url]"})
                                continue
                            seen.add(url)
                    new_parts.append(p)
                m = {**m, "content": new_parts}
        cleaned.append(m)
    return cleaned

def _debug_log_image_urls(tag: str, messages) -> None:
    try:
        from nonebot import logger
        urls = _collect_image_urls(messages)
        if urls:
            logger.info(f"{tag}: 即将发送的图片URL共 {len(urls)} 条：")
            for u in urls:
                logger.info(f"{tag}: {u}")
        else:
            logger.info(f"{tag}: 本次请求不包含图片URL")
    except Exception:
        pass
def _is_http_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def _is_data_url(s: str) -> bool:
    return isinstance(s, str) and s.startswith("data:")

def _bytes_to_data_url(data: bytes, ctype: str) -> str:
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{ctype};base64,{b64}"
def _cache_cleanup_now() -> None:
    now = time()
    # 清理坏链接
    expired = [u for u, ts in _bad_url_cache.items() if ts <= now]
    for u in expired:
        _bad_url_cache.pop(u, None)
    # 清理 data URL
    dead = [u for u, (ts, _) in _data_url_cache.items() if ts <= now]
    for u in dead:
        _data_url_cache.pop(u, None)

def _cache_is_bad_url(url: str) -> bool:
    _cache_cleanup_now()
    ts = _bad_url_cache.get(url)
    if ts and ts > time():
        return True
    if ts:
        _bad_url_cache.pop(url, None)
    return False

def _cache_mark_bad_url(url: str, ttl: int = _BAD_URL_TTL) -> None:
    _bad_url_cache[url] = time() + ttl

def _cache_get_data_url(url: str) -> str | None:
    _cache_cleanup_now()
    val = _data_url_cache.get(url)
    if not val:
        return None
    ts, data_url = val
    if ts <= time():
        _data_url_cache.pop(url, None)
        return None
    # 触发 LRU：移动到末尾
    _data_url_cache.move_to_end(url, last=True)
    return data_url

def _cache_set_data_url(url: str, data_url: str, ttl: int = _DATA_URL_TTL) -> None:
    _cache_cleanup_now()
    _data_url_cache[url] = (time() + ttl, data_url)
    _data_url_cache.move_to_end(url, last=True)
    # LRU 逐出
    while len(_data_url_cache) > _DATA_URL_CACHE_MAX:
        _data_url_cache.popitem(last=False)
def _has_alpha(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA"):
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False

def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= _EMBED_MAX_DIM:
        return img
    scale = _EMBED_MAX_DIM / float(m)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

def _compress_image_bytes(data: bytes, src_content_type: str | None) -> tuple[bytes, str]:
    """
    尝试压缩图片到目标体积，返回 (compressed_bytes, content_type)。
    未安装 Pillow 或解码失败时，返回原始数据与推断的 content-type。
    """
    # 无 Pillow 或无法解码时原样返回
    if Image is None:
        ctype = src_content_type or "application/octet-stream"
        guessed = mimetypes.guess_type(f"_.{ctype.split('/')[-1]}")[0]
        return data, (ctype or guessed or "application/octet-stream")

    try:
        img = Image.open(BytesIO(data))
        img.load()
    except Exception:
        ctype = src_content_type or mimetypes.guess_type("_.bin")[0] or "application/octet-stream"
        return data, ctype

    img = _resize_if_needed(img)

    has_alpha = _has_alpha(img)
    # 优先无损信息：如果没有 alpha，用 JPEG；有 alpha 用 WebP（支持 alpha），回退 PNG。
    # 先设定一个目标大小
    target = _EMBED_TARGET_BYTES

    # 尝试编码并逐步逼近 target
    if not has_alpha:
        # 转 RGB
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        quality = _EMBED_JPEG_QUALITY_START
        best_buf = None
        best_size = 1 << 60
        while quality >= _EMBED_JPEG_QUALITY_MIN:
            buf = BytesIO()
            try:
                img.save(
                    buf,
                    format="JPEG",
                    quality=quality,
                    optimize=True,
                    progressive=True,
                    subsampling="4:2:0",
                )
            except OSError:
                # 某些环境关闭 optimize 再试
                buf = BytesIO()
                img.save(
                    buf,
                    format="JPEG",
                    quality=quality,
                    progressive=True,
                )
            size = buf.tell()
            # 记录最小者
            if size < best_size:
                best_size = size
                best_buf = buf.getvalue()
            if size <= target:
                return buf.getvalue(), "image/jpeg"
            quality -= 5
        # 未达到 target，返回最小者
        return best_buf or data, "image/jpeg"
    else:
        # 有透明，先试 WebP（有 alpha）
        try:
            # 先质量 80，若仍偏大再降到 60
            for q in (80, 60):
                buf = BytesIO()
                img.save(buf, format="WEBP", quality=q, method=4)
                if buf.tell() <= target:
                    return buf.getvalue(), "image/webp"
                best_webp = buf.getvalue()
            # 若仍偏大，保留最小 webp
            return best_webp, "image/webp"
        except Exception:
            # 回退 PNG（通常比 webp 大，但保真）
            buf = BytesIO()
            try:
                img.save(buf, format="PNG", optimize=True)
            except Exception:
                img.save(buf, format="PNG")
            return buf.getvalue(), "image/png"

async def _download_to_data_url_global(url: str) -> str | None:
    """下载图片并（必要时）压缩为 data URL。失败或仍过大返回 None。"""
    if not httpx:
        logger.warning("未安装 httpx，跳过将图片URL转换为 data URL（保留原URL）。")
        return None
    try:
        timeout = config_manager.config.llm_config.llm_timeout
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            content_length = None
            try:
                h = await client.head(url)
                if h.status_code // 100 == 2:
                    cl = h.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        content_length = int(cl)
            except Exception:
                pass

            if content_length is not None and content_length > (10 * _MAX_IMAGE_BYTES):
                # 过于巨大直接放弃，避免占用带宽
                logger.warning(f"图片过大（{content_length} 字节），放弃转换为 data URL：{url}")
                return None

            r = await client.get(url)
            if r.status_code // 100 != 2:
                logger.warning(f"下载图片失败，HTTP {r.status_code}：{url}")
                return None

            data = r.content
            ctype = r.headers.get("Content-Type") or mimetypes.guess_type(url)[0] or "application/octet-stream"

            # 若超出上限或超过 512KB，尝试压缩
            if len(data) > _MAX_IMAGE_BYTES or len(data) > (512 * 1024):
                comp_data, comp_ctype = _compress_image_bytes(data, ctype)
            else:
                comp_data, comp_ctype = data, ctype

            if len(comp_data) > _MAX_IMAGE_BYTES:
                logger.warning(f"压缩后仍超出单图上限（{len(comp_data)} 字节）放弃内联：{url}")
                return None

            return _bytes_to_data_url(comp_data, comp_ctype)
    except Exception as e:
        logger.warning(f"下载/转换图片为 data URL 失败：{e}")
        return None
def _filter_bad_image_urls(messages: list) -> list:
    """在 URL 模式下，跳过已知坏图片 URL（可减少上游无谓抓取和日志噪音）。"""
    if not _SKIP_KNOWN_BAD_IN_URL_MODE:
        return messages
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        iu = p.get("image_url")
                        url = iu if isinstance(iu, str) else (iu.get("url") if isinstance(iu, dict) else None)
                        if isinstance(url, str) and _is_http_url(url) and _cache_is_bad_url(url):
                            # 用占位文本替代，避免结构为空
                            new_parts.append({"type": "text", "text": "[image omitted: known-bad-url]"})
                            continue
                    new_parts.append(p)
                msg = {**msg, "content": new_parts}
        cleaned.append(msg)
    return cleaned
def _normalize_content_parts_global(parts: list) -> list:
    """将消息的多模态分片规范为 OpenAI 要求的结构（与适配器保持一致）"""
    normalized = []
    for part in parts:
        if isinstance(part, str):
            normalized.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            normalized.append({"type": "text", "text": str(part)})
            continue
        t = part.get("type")
        if t == "input_image":
            url = part.get("url") or part.get("image_url")
            if isinstance(url, str):
                normalized.append({"type": "image_url", "image_url": {"url": url}})
            elif isinstance(url, dict) and url.get("url"):
                normalized.append({"type": "image_url", "image_url": url})
            else:
                normalized.append({"type": "text", "text": ""})
            continue
        if t == "image_url":
            iu = part.get("image_url")
            if isinstance(iu, str):
                normalized.append({"type": "image_url", "image_url": {"url": iu}})
            elif isinstance(iu, dict) and iu.get("url"):
                normalized.append(part)
            else:
                normalized.append({"type": "text", "text": ""})
            continue
        if t == "text":
            text = part.get("text")
            if isinstance(text, str):
                normalized.append({"type": "text", "text": text})
            else:
                fallback = part.get("content") or ""
                normalized.append({"type": "text", "text": str(fallback)})
            continue
        fallback = part.get("text") or part.get("content") or ""
        normalized.append({"type": "text", "text": str(fallback)})
    return normalized

def _strip_image_parts_global(messages) -> list:
    """移除消息中的所有 image_url 分片；若消息因此为空，则放入一个空文本分片。"""
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                new_parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        continue
                    new_parts.append(p)
                if not new_parts:
                    new_parts = [{"type": "text", "text": ""}]
                msg = {**msg, "content": new_parts}
        cleaned.append(msg)
    return cleaned

async def _convert_images_to_data_urls_in_messages(
    messages: Iterable[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """将消息中 http(s) 的图片 URL 转为 data URL；命中缓存；对已知坏 URL 直接以占位文本替代。"""
    converted: list[ChatCompletionMessageParam] = []
    total_embedded = 0
    embedded_count = 0

    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, list):
                new_parts: list = []
                for p in content:
                    if (
                        isinstance(p, dict)
                        and p.get("type") == "image_url"
                        and embedded_count < _MAX_EMBED_IMAGE_COUNT
                        and total_embedded < _MAX_TOTAL_DATA_URL_BYTES
                    ):
                        iu = p.get("image_url")
                        url: str | None = None
                        if isinstance(iu, str):
                            url = iu
                        elif isinstance(iu, dict):
                            url = iu.get("url") if isinstance(iu.get("url"), str) else None

                        if url and _is_http_url(url) and not _is_data_url(url):
                            # 已知坏 URL：不再尝试，直接用占位文本（相当于提前“剥离该图”）
                            if _cache_is_bad_url(url):
                                new_parts.append({"type": "text", "text": "[image omitted: known-bad-url]"})
                                continue

                            # 命中 data URL 缓存
                            cached = _cache_get_data_url(url)
                            if cached:
                                data_url = cached
                            else:
                                data_url = await _download_to_data_url_global(url)

                            if data_url:
                                if total_embedded + len(data_url) <= _MAX_TOTAL_DATA_URL_BYTES:
                                    if isinstance(iu, dict):
                                        new_iu = {**iu, "url": data_url}
                                        p = {"type": "image_url", "image_url": new_iu}
                                    else:
                                        p = {"type": "image_url", "image_url": {"url": data_url}}
                                    total_embedded += len(data_url)
                                    embedded_count += 1
                                else:
                                    # 超过总上限：改用占位文本
                                    p = {"type": "text", "text": "[image omitted: size-budget]"}
                            else:
                                # 本次下载失败：直接占位（避免继续在 data 模式下走 URL）
                                p = {"type": "text", "text": "[image omitted: download-failed]"}
                    new_parts.append(p)
                msg = {**msg, "content": new_parts}
        converted.append(msg)
    return converted


async def tools_caller(
    messages: list,
    tools: list,
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
) -> ChatCompletionMessage:
    if not tool_choice:
        tool_choice = (
            "required"
            if (
                config_manager.config.llm_config.tools.require_tools and len(tools) > 1
            )
            else "auto"
        )
    config = config_manager.config
    preset_list = [config.preset, *deepcopy(config.preset_extension.backup_preset_list)]
    err: None | Exception = None
    if not preset_list:
        preset_list = ["default"]
    for name in preset_list:
        try:
            preset = await config_manager.get_preset(name)

            if preset.protocol not in ("__main__", "openai"):
                continue
            base_url = preset.base_url
            key = preset.api_key
            model = preset.model

            logger.debug(f"开始获取 {preset.model} 的带有工具的对话")
            logger.debug(f"预设：{name}")
            logger.debug(f"密钥：{preset.api_key[:7]}...")
            logger.debug(f"协议：{preset.protocol}")
            logger.debug(f"API地址：{preset.base_url}")

            client = openai.AsyncOpenAI(
                base_url=base_url, api_key=key, timeout=config.llm_config.llm_timeout
            )

            # 1) 规范化多模态分片，得到 URL 模式（未内联）的消息
            url_mode_messages: list[ChatCompletionMessageParam] = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, list):
                        msg = {**msg, "content": _normalize_content_parts_global(content)}
                url_mode_messages.append(msg)
            url_mode_messages = _keep_only_latest_user_images(url_mode_messages)

            _debug_log_image_urls("tools_caller(URL-first)", url_mode_messages)
            async def _do_create(req_messages: list[ChatCompletionMessageParam]) -> ChatCompletion:
                return await client.chat.completions.create(
                    model=model,
                    messages=req_messages,
                    stream=False,
                    tool_choice=tool_choice,
                    tools=tools,
                )

            # 2) 优先使用 URL 模式
            try:
                completion: ChatCompletion = await _do_create(url_mode_messages)
            except Exception as e:
                msg = str(e)

                # 如果 URL 模式因为图片下载/鉴权问题失败 → 尝试 base64 data URL 模式
                if (
                    "invalid_image_url" in msg
                    or "Timeout while downloading" in msg
                    or "fail to get image from url" in msg
                    or "count_token_messages_failed" in msg
                ):
                    logger.warning("tools_caller: URL 模式因图片失败，改用 data URL 模式重试")
                    try:
                        dataurl_messages = await _convert_images_to_data_urls_in_messages(url_mode_messages)
                    except Exception as ce:
                        logger.warning(f"tools_caller: data URL 转换异常，跳过转换：{ce}")
                        dataurl_messages = url_mode_messages  # 退回 URL 模式以便进入后续剥离逻辑

                    try:
                        completion = await _do_create(dataurl_messages)
                    except Exception as e2:
                        msg2 = str(e2)
                        # data URL 仍失败 → 剥离图片
                        if (
                            "invalid_image_url" in msg2
                            or "Timeout while downloading" in msg2
                            or "fail to get image from url" in msg2
                            or "count_token_messages_failed" in msg2
                            or "Error code: 413" in msg2
                            or "status 413" in msg2
                            or " param': '413" in msg2
                        ):
                            logger.warning("tools_caller: data URL/体积问题，剥离图片后重试")
                            no_img = _strip_image_parts_global(url_mode_messages)
                            try:
                                completion = await _do_create(no_img)
                            except Exception as e3:
                                # 仍失败（可能是 413 或其他体积相关）→ 裁剪文本兜底
                                if "Error code: 413" in str(e3) or "status 413" in str(e3):
                                    logger.warning("tools_caller: 仍 413，裁剪文本后最终重试")
                                    shrunk = _shrink_messages_by_chars(no_img, max_chars=8000)
                                    completion = await _do_create(shrunk)
                                else:
                                    raise
                        else:
                            raise

                # 如果 URL 模式直接 413（多为文本上下文过大） → 裁剪文本后重试
                elif "Error code: 413" in msg or "status 413" in msg or " param': '413" in msg:
                    logger.warning("tools_caller: URL 模式 413，裁剪文本后重试")
                    shrunk = _shrink_messages_by_chars(url_mode_messages, max_chars=8000)
                    completion = await _do_create(shrunk)

                else:
                    # 其他错误直接抛出
                    raise

            return completion.choices[0].message

        except Exception as e:
            logger.warning(f"[OpenAI] {name} 模型调用失败: {e}")
            err = e
            continue
    logger.warning("Tools调用因为没有OPENAI协议模型而失败")
    if err is not None:
        raise err
    return ChatCompletionMessage(role="assistant", content="")


async def get_chat(
    messages: list,
    bot: Bot | None = None,
    tokens: int = 0,
) -> str:
    """获取聊天响应"""
    # 获取最大token数量
    if bot is None:
        nb_bot = nonebot.get_bot()
        assert isinstance(nb_bot, Bot)
    else:
        nb_bot = bot
    presets = [
        config_manager.config.preset,
        *config_manager.config.preset_extension.backup_preset_list,
    ]
    err: Exception | None = None
    for pname in presets:
        preset = await config_manager.get_preset(pname)
        # 根据预设选择API密钥和基础URL
        is_thought_chain_model = preset.thought_chain_model
        if adapter := AdapterManager().safe_get_adapter(preset.protocol):
            # 如果适配器存在，使用它
            logger.debug(f"使用适配器 {adapter.__name__} 处理协议 {preset.protocol}")
        else:
            raise ValueError(f"未定义的协议适配器：{preset.protocol}")
        # 记录日志
        logger.debug(f"开始获取 {preset.model} 的对话")
        logger.debug(f"预设：{config_manager.config.preset}")
        logger.debug(f"密钥：{preset.api_key[:7]}...")
        logger.debug(f"协议：{preset.protocol}")
        logger.debug(f"API地址：{preset.base_url}")
        logger.debug(f"当前对话Tokens:{tokens}")
        response = ""
        # 调用适配器获取聊天响应
        try:
            for index in range(1, config_manager.config.llm_config.max_retries + 1):
                e = None
                try:
                    processer = adapter(preset, config_manager.config)
                    response = await processer.call_api(messages)
                    break  # 成功获取响应，跳出重试循环
                except Exception as e:
                    logger.warning(f"发生错误: {e}")
                    if index == config_manager.config.llm_config.max_retries:
                        logger.warning(
                            f"请检查API Key和API base_url！获取对话时发生错误: {e}"
                        )
                        raise e
                    logger.info(f"开始第 {index + 1} 次重试")
                    continue
                finally:
                    if (
                        err is not None
                        and not config_manager.config.llm_config.auto_retry
                    ):
                        raise err
        except Exception as e:
            logger.warning(f"调用适配器失败{e}")
            err = e
            continue

        if chat_manager.debug:
            logger.debug(response)
        return remove_think_tag(response) if is_thought_chain_model else response
    if err is not None:
        raise err
    return ""


class OpenAIAdapter(ModelAdapter):
    """OpenAI协议适配器"""

    @staticmethod
    def _normalize_content_parts(parts):
        """将消息的多模态分片规范为 OpenAI 要求的结构"""
        normalized = []
        for part in parts:
            # 纯字符串转文本分片
            if isinstance(part, str):
                normalized.append({"type": "text", "text": part})
                continue
            if not isinstance(part, dict):
                normalized.append({"type": "text", "text": str(part)})
                continue
            t = part.get("type")
            # 兼容旧写法 input_image/url
            if t == "input_image":
                url = part.get("url") or part.get("image_url")
                if isinstance(url, str):
                    normalized.append({"type": "image_url", "image_url": {"url": url}})
                elif isinstance(url, dict) and url.get("url"):
                    normalized.append({"type": "image_url", "image_url": url})
                else:
                    # 无法识别则降级为空文本，避免 400
                    normalized.append({"type": "text", "text": ""})
                continue
            # 正确写法 image_url 对象
            if t == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, str):
                    normalized.append({"type": "image_url", "image_url": {"url": iu}})
                elif isinstance(iu, dict) and iu.get("url"):
                    normalized.append(part)
                else:
                    normalized.append({"type": "text", "text": ""})
                continue
            # 文本分片规范化
            if t == "text":
                text = part.get("text")
                if isinstance(text, str):
                    normalized.append({"type": "text", "text": text})
                else:
                    fallback = part.get("content") or ""
                    normalized.append({"type": "text", "text": str(fallback)})
                continue
            # 其他未知类型，尽量转文本
            fallback = part.get("text") or part.get("content") or ""
            normalized.append({"type": "text", "text": str(fallback)})
        return normalized

    @staticmethod
    def _strip_image_parts(messages: Iterable[ChatCompletionMessageParam]) -> list[ChatCompletionMessageParam]:
        """移除消息中的所有 image_url 分片；若消息因此为空，则放入一个空文本分片，避免 400。"""
        cleaned: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    new_parts = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "image_url":
                            # 跳过图片分片
                            continue
                        new_parts.append(p)
                    if not new_parts:
                        new_parts = [{"type": "text", "text": ""}]
                    msg = {**msg, "content": new_parts}
            cleaned.append(msg)
        return cleaned

    async def _download_to_data_url(self, url: str) -> str | None:
        return await _download_to_data_url_global(url)

    async def _convert_images_to_data_urls(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        """将消息中 http(s) 的图片 URL 转换为 data URL；已是 data: 的跳过。"""
        converted: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    new_parts: list = []
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "image_url":
                            iu = p.get("image_url")
                            if isinstance(iu, str):
                                url = iu
                                if _is_http_url(url) and not _is_data_url(url):
                                    data_url = await self._download_to_data_url(url)
                                    if data_url:
                                        p = {"type": "image_url", "image_url": {"url": data_url}}
                            elif isinstance(iu, dict):
                                url = iu.get("url")
                                if isinstance(url, str) and _is_http_url(url) and not _is_data_url(url):
                                    data_url = await self._download_to_data_url(url)
                                    if data_url:
                                        # 保留 detail 等其他字段
                                        new_iu = {**iu, "url": data_url}
                                        p = {"type": "image_url", "image_url": new_iu}
                        new_parts.append(p)
                    msg = {**msg, "content": new_parts}
            converted.append(msg)
        return converted

    async def call_api(self, messages: Iterable[ChatCompletionMessageParam]) -> str:
        """调用OpenAI API获取聊天响应"""
        preset = self.preset
        config = self.config
        client = openai.AsyncOpenAI(
            base_url=preset.base_url,
            api_key=preset.api_key,
            timeout=config.llm_config.llm_timeout,
        )

        # 规范化多模态消息结构，得到 URL 模式消息
        url_mode_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    msg = {**msg, "content": self._normalize_content_parts(content)}
            url_mode_messages.append(msg)
        url_mode_messages = _keep_only_latest_user_images(url_mode_messages)
        _debug_log_image_urls("call_api(URL-first)", url_mode_messages)
        completion: ChatCompletion | openai.AsyncStream[ChatCompletionChunk] | None = None

        async def _do_create(req_messages: list[ChatCompletionMessageParam]):
            return await client.chat.completions.create(
                model=preset.model,
                messages=req_messages,
                max_tokens=config.llm_config.max_tokens,
                stream=config.llm_config.stream,
            )

        # 先 URL 模式
        try:
            completion = await _do_create(url_mode_messages)
        except Exception as e:
            msg = str(e)

            # URL 模式因图片问题失败 → data URL 模式
            if (
                "invalid_image_url" in msg
                or "Timeout while downloading" in msg
                or "fail to get image from url" in msg
                or "count_token_messages_failed" in msg
            ):
                logger.warning("call_api: URL 模式因图片失败，改用 data URL 模式重试")
                try:
                    dataurl_messages = await self._convert_images_to_data_urls(url_mode_messages)
                except Exception as ce:
                    logger.warning(f"call_api: data URL 转换异常，跳过转换：{ce}")
                    dataurl_messages = url_mode_messages

                try:
                    completion = await _do_create(dataurl_messages)
                except Exception as e2:
                    msg2 = str(e2)
                    # data URL 仍失败 → 剥离图片
                    if (
                        "invalid_image_url" in msg2
                        or "Timeout while downloading" in msg2
                        or "fail to get image from url" in msg2
                        or "count_token_messages_failed" in msg2
                        or "Error code: 413" in msg2
                        or "status 413" in msg2
                        or " param': '413" in msg2
                    ):
                        logger.warning("call_api: data URL/体积问题，剥离图片后重试")
                        no_img = self._strip_image_parts(url_mode_messages)
                        try:
                            completion = await _do_create(no_img)
                        except Exception as e3:
                            if "Error code: 413" in str(e3) or "status 413" in str(e3):
                                logger.warning("call_api: 仍 413，裁剪文本后最终重试")
                                shrunk = _shrink_messages_by_chars(no_img, max_chars=8000)
                                completion = await _do_create(shrunk)
                            else:
                                raise
                    else:
                        raise

            # URL 模式直接 413 → 裁剪文本
            elif "Error code: 413" in msg or "status 413" in msg or " param': '413" in msg:
                logger.warning("call_api: URL 模式 413，裁剪文本后重试")
                shrunk = _shrink_messages_by_chars(url_mode_messages, max_chars=8000)
                completion = await _do_create(shrunk)

            else:
                raise

        # 处理响应
        response: str = ""
        if config.llm_config.stream and isinstance(completion, openai.AsyncStream):
            async for chunk in completion:
                try:
                    if chunk.choices[0].delta.content is not None:
                        response += chunk.choices[0].delta.content
                        if chat_manager.debug:
                            logger.debug(chunk.choices[0].delta.content)
                except IndexError:
                    break
        else:
            if chat_manager.debug:
                logger.debug(completion)
            if isinstance(completion, ChatCompletion):
                response = (
                    completion.choices[0].message.content
                    if completion.choices[0].message.content is not None
                    else ""
                )
            else:
                raise RuntimeError("收到意外的响应类型")

        return response if response is not None else ""
    @staticmethod
    def get_adapter_protocol() -> tuple[str, ...]:
        return "openai", "__main__"
