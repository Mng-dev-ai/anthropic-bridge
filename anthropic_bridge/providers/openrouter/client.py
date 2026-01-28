import json
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ...cache import get_reasoning_cache
from ...transform import (
    convert_anthropic_messages_to_openai,
    convert_anthropic_tool_choice_to_openai,
    convert_anthropic_tools_to_openai,
)
from .registry import ProviderRegistry

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/Mng-dev-ai/claudex",
    "X-Title": "Claudex",
}


class OpenRouterProvider:
    def __init__(self, target_model: str, api_key: str):
        self.target_model = target_model.removeprefix("openrouter/")
        self.api_key = api_key
        self.provider_registry = ProviderRegistry(self.target_model)
        self._is_gemini = (
            "gemini" in target_model.lower() or "google/" in target_model.lower()
        )

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        provider = self.provider_registry.get_provider()
        if hasattr(provider, "reset"):
            provider.reset()

        messages = await self._convert_messages_async(payload)
        tools = convert_anthropic_tools_to_openai(payload.get("tools"))

        openrouter_payload: dict[str, Any] = {
            "model": self.target_model,
            "messages": messages,
            "temperature": payload.get("temperature", 1),
            "stream": True,
            "max_tokens": payload.get("max_tokens", 16000),
            "stream_options": {"include_usage": True},
        }

        if tools:
            openrouter_payload["tools"] = tools
            tool_choice = convert_anthropic_tool_choice_to_openai(
                payload.get("tool_choice")
            )
            if tool_choice:
                openrouter_payload["tool_choice"] = tool_choice

        if payload.get("thinking"):
            openrouter_payload["include_reasoning"] = True

        provider.prepare_request(openrouter_payload, payload)

        async for event in self._stream_openrouter(openrouter_payload, provider):
            yield event

    def _convert_messages(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        system = payload.get("system")
        if isinstance(system, list):
            system = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )

        messages = convert_anthropic_messages_to_openai(
            payload.get("messages", []), system
        )

        return messages

    async def _convert_messages_async(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        messages = self._convert_messages(payload)

        if self._is_gemini:
            await self._inject_gemini_reasoning(messages, payload.get("messages", []))

        if "grok" in self.target_model.lower() or "x-ai" in self.target_model.lower():
            instruction = (
                "IMPORTANT: When calling tools, you MUST use the OpenAI tool_calls format with JSON. "
                "NEVER use XML format like <xai:function_call>."
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += "\n\n" + instruction
            else:
                messages.insert(0, {"role": "system", "content": instruction})

        return messages

    async def _inject_gemini_reasoning(
        self,
        openai_messages: list[dict[str, Any]],
        anthropic_messages: list[dict[str, Any]],
    ) -> None:
        cache = get_reasoning_cache()
        for msg in openai_messages:
            if msg.get("role") != "assistant" or not msg.get("tool_calls"):
                continue

            for tc in msg.get("tool_calls", []):
                tool_id = tc.get("id", "")
                cached = await cache.get(tool_id)
                if cached:
                    if "reasoning_details" not in msg:
                        msg["reasoning_details"] = []
                    msg["reasoning_details"].extend(cached)

    async def _stream_openrouter(
        self, payload: dict[str, Any], provider: Any
    ) -> AsyncIterator[str]:
        msg_id = f"msg_{int(time.time())}_{self._random_id()}"

        yield self._sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.target_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        yield self._sse("ping", {"type": "ping"})

        text_started = False
        text_idx = -1
        thinking_started = False
        thinking_idx = -1
        cur_idx = 0
        saw_tool_use = False
        tools: dict[object, dict[str, Any]] = {}
        usage: dict[str, Any] | None = None
        current_reasoning_details: list[dict[str, Any]] = []

        async with (
            httpx.AsyncClient(timeout=300.0) as client,
            client.stream(
                "POST",
                OPENROUTER_API_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    **OPENROUTER_HEADERS,
                },
                json=payload,
            ) as response,
        ):
            if response.status_code != 200:
                error_text = await response.aread()
                yield self._sse(
                    "error",
                    {
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": error_text.decode(errors="replace"),
                        },
                    },
                )
                yield self._sse("message_stop", {"type": "message_stop"})
                yield "data: [DONE]\n\n"
                return

            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                lines = buffer.split("\n")
                buffer = lines.pop()

                done = False
                for line in lines:
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue

                    data_str = line[6:]
                    if data_str == "[DONE]":
                        done = True
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if data.get("usage"):
                        usage = data["usage"]

                    delta = data.get("choices", [{}])[0].get("delta", {})

                    if self._is_gemini and delta.get("reasoning_details"):
                        current_reasoning_details.extend(delta["reasoning_details"])

                    reasoning = delta.get("reasoning") or ""
                    content = delta.get("content") or ""

                    if reasoning:
                        if not thinking_started:
                            thinking_idx = cur_idx
                            cur_idx += 1
                            yield self._sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": thinking_idx,
                                    "content_block": {
                                        "type": "thinking",
                                        "thinking": "",
                                        "signature": "",
                                    },
                                },
                            )
                            thinking_started = True

                        if thinking_started:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_idx,
                                    "delta": {
                                        "type": "thinking_delta",
                                        "thinking": reasoning,
                                    },
                                },
                            )

                    if content:
                        if thinking_started:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_idx,
                                    "delta": {"type": "signature_delta", "signature": ""},
                                },
                            )
                            yield self._sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": thinking_idx},
                            )
                            thinking_started = False

                        result = provider.process_text_content(content, "")
                        clean_text = result.cleaned_text

                        if clean_text:
                            if not text_started:
                                text_idx = cur_idx
                                cur_idx += 1
                                yield self._sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_idx,
                                        "content_block": {"type": "text", "text": ""},
                                    },
                                )
                                text_started = True

                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": clean_text},
                                },
                            )

                        for tc in result.extracted_tool_calls:
                            if text_started:
                                yield self._sse(
                                    "content_block_stop",
                                    {"type": "content_block_stop", "index": text_idx},
                                )
                                text_started = False

                            saw_tool_use = True
                            tool_idx = cur_idx
                            cur_idx += 1
                            yield self._sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": tool_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tc.id,
                                        "name": tc.name,
                                    },
                                },
                            )
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": tool_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(tc.arguments),
                                    },
                                },
                            )
                            yield self._sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": tool_idx},
                            )
                            if self._is_gemini and current_reasoning_details:
                                await get_reasoning_cache().set(
                                    tc.id, current_reasoning_details.copy()
                                )

                    tool_calls = delta.get("tool_calls", [])
                    for tc in tool_calls:
                        key: object
                        if "index" in tc:
                            key = tc["index"]
                        else:
                            key = tc.get("id") or f"tool_{int(time.time())}_{self._random_id()}"

                        if key not in tools:
                            if text_started:
                                yield self._sse(
                                    "content_block_stop",
                                    {"type": "content_block_stop", "index": text_idx},
                                )
                                text_started = False

                            tools[key] = {
                                "id": tc.get("id") or f"tool_{int(time.time())}",
                                "name": tc.get("function", {}).get("name", ""),
                                "block_idx": cur_idx,
                                "started": False,
                                "closed": False,
                            }
                            cur_idx += 1
                            saw_tool_use = True

                        t = tools[key]
                        fn = tc.get("function", {})

                        if fn.get("name") and not t["started"]:
                            t["name"] = fn["name"]
                            yield self._sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": t["block_idx"],
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": t["id"],
                                        "name": t["name"],
                                    },
                                },
                            )
                            t["started"] = True

                        if fn.get("arguments") and t["started"]:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": t["block_idx"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": fn["arguments"],
                                    },
                                },
                            )

                    finish = data.get("choices", [{}])[0].get("finish_reason")
                    if finish == "tool_calls":
                        for t in tools.values():
                            if t["started"] and not t["closed"]:
                                yield self._sse(
                                    "content_block_stop",
                                    {
                                        "type": "content_block_stop",
                                        "index": t["block_idx"],
                                    },
                                )
                                t["closed"] = True
                                if self._is_gemini and current_reasoning_details:
                                    await get_reasoning_cache().set(
                                        t["id"], current_reasoning_details.copy()
                                    )
                        done = True
                if done:
                    break

        if thinking_started:
            yield self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": thinking_idx,
                    "delta": {"type": "signature_delta", "signature": ""},
                },
            )
            yield self._sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": thinking_idx},
            )

        if text_started:
            yield self._sse(
                "content_block_stop", {"type": "content_block_stop", "index": text_idx}
            )

        for t in tools.values():
            if t["started"] and not t["closed"]:
                yield self._sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": t["block_idx"]},
                )
                if self._is_gemini and current_reasoning_details:
                    await get_reasoning_cache().set(t["id"], current_reasoning_details.copy())

        yield self._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if saw_tool_use else "end_turn",
                    "stop_sequence": None,
                },
                "usage": {
                    "input_tokens": usage.get("prompt_tokens", 0) if usage else 0,
                    "output_tokens": usage.get("completion_tokens", 0) if usage else 0,
                },
            },
        )
        yield self._sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
