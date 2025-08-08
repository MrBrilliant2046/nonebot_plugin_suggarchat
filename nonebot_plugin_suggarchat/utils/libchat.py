from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

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

from ..chatmanager import chat_manager
from ..config import config_manager
from .functions import remove_think_tag
from .protocol import AdapterManager, ModelAdapter


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
            )  # 排除默认工具
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
            completion: ChatCompletion = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                tool_choice=tool_choice,
                tools=tools,
            )
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

    async def call_api(self, messages: Iterable[ChatCompletionMessageParam]) -> str:
        """调用OpenAI API获取聊天响应"""
        preset = self.preset
        config = self.config
        client = openai.AsyncOpenAI(
            base_url=preset.base_url,
            api_key=preset.api_key,
            timeout=config.llm_config.llm_timeout,
        )

        # 规范化多模态消息结构，避免 image_url 为字符串等无效格式
        norm_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    msg = {**msg, "content": self._normalize_content_parts(content)}
            norm_messages.append(msg)

        completion: ChatCompletion | openai.AsyncStream[ChatCompletionChunk] | None = None

        async def _do_create(req_messages: list[ChatCompletionMessageParam]):
            return await client.chat.completions.create(
                model=preset.model,
                messages=req_messages,
                max_tokens=config.llm_config.max_tokens,
                stream=config.llm_config.stream,
            )

        try:
            completion = await _do_create(norm_messages)
        except Exception as e:
            # 上游统计 token 或抓取图片失败时的常见报错，移除图片后重试一次
            msg = str(e)
            if ("fail to get image from url" in msg) or ("count_token_messages_failed" in msg):
                fallback_messages = self._strip_image_parts(norm_messages)
                completion = await _do_create(fallback_messages)
            else:
                raise

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
