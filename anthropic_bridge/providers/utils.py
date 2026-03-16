import json
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any

import tiktoken

_encoding: tiktoken.Encoding | None = None

DEFAULT_USAGE = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
}


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def random_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def estimate_input_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> int:
    enc = _get_encoding()
    total = 0
    for msg in messages:
        total += 4
        content = msg.get("content")
        if isinstance(content, str):
            total += len(enc.encode(content))
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or part.get("content") or ""
                    if text:
                        total += len(enc.encode(text))
                elif isinstance(part, str):
                    total += len(enc.encode(part))
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name"):
                total += len(enc.encode(fn["name"]))
            if fn.get("arguments"):
                total += len(enc.encode(fn["arguments"]))
    if tools:
        total += len(enc.encode(json.dumps(tools)))
    total += 2
    return total


async def yield_error_events(
    message: str, model: str
) -> AsyncIterator[str]:
    msg_id = f"msg_{int(time.time())}_{random_id()}"
    yield sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": dict(DEFAULT_USAGE),
            },
        },
    )
    yield sse(
        "error",
        {"type": "error", "error": {"type": "api_error", "message": message}},
    )
    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": dict(DEFAULT_USAGE),
        },
    )
    yield sse("message_stop", {"type": "message_stop"})


def map_reasoning_effort(
    budget_tokens: int | None, model_id: str | None = None
) -> str | None:
    if not budget_tokens:
        return None

    if budget_tokens >= 32000:
        if _model_supports_xhigh(model_id):
            return "xhigh"
        return "high"
    elif budget_tokens >= 15000:
        return "high"
    elif budget_tokens >= 10000:
        return "medium"
    elif budget_tokens >= 1:
        return "low"

    return None


def normalize_model_id(model_id: str) -> str:
    lower = model_id.lower()
    if "/" in lower:
        lower = lower.split("/", 1)[1]
    return lower


def first_choice(data: dict[str, Any]) -> dict[str, Any] | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    return choice


def _model_supports_xhigh(model_id: str | None) -> bool:
    if not model_id:
        return False

    lower = normalize_model_id(model_id)

    return lower.startswith(("gpt-5.1-codex-max", "gpt-5.2", "gpt-5.3", "gpt-5.4"))


class AnthropicSSEEmitter:
    def __init__(self, model: str, estimated_input: int):
        self.model = model
        self.msg_id = f"msg_{int(time.time())}_{random_id()}"
        self._text_started = False
        self._text_idx = -1
        self._thinking_started = False
        self._thinking_idx = -1
        self._cur_idx = 0
        self._tools: dict[str | int, dict[str, Any]] = {}
        self.had_content = False
        self.estimated_input = estimated_input

    def message_start(self) -> list[str]:
        return [
            sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        **DEFAULT_USAGE,
                        "input_tokens": self.estimated_input,
                        "output_tokens": 1,
                    },
                },
            }),
            sse("ping", {"type": "ping"}),
        ]

    def thinking_delta(self, text: str) -> list[str]:
        self.had_content = True
        events: list[str] = []
        if not self._thinking_started:
            self._thinking_idx = self._cur_idx
            self._cur_idx += 1
            events.append(sse("content_block_start", {
                "type": "content_block_start",
                "index": self._thinking_idx,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }))
            self._thinking_started = True
        events.append(sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self._thinking_idx,
            "delta": {"type": "thinking_delta", "thinking": text},
        }))
        return events

    def close_thinking(self, signature: str = "") -> list[str]:
        if not self._thinking_started:
            return []
        self._thinking_started = False
        return [
            sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._thinking_idx,
                "delta": {"type": "signature_delta", "signature": signature},
            }),
            sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self._thinking_idx,
            }),
        ]

    def text_delta(self, text: str) -> list[str]:
        self.had_content = True
        events: list[str] = []
        if not self._text_started:
            self._text_idx = self._cur_idx
            self._cur_idx += 1
            events.append(sse("content_block_start", {
                "type": "content_block_start",
                "index": self._text_idx,
                "content_block": {"type": "text", "text": ""},
            }))
            self._text_started = True
        events.append(sse("content_block_delta", {
            "type": "content_block_delta",
            "index": self._text_idx,
            "delta": {"type": "text_delta", "text": text},
        }))
        return events

    def close_text(self) -> list[str]:
        if not self._text_started:
            return []
        self._text_started = False
        return [sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self._text_idx,
        })]

    def register_tool(self, tool_key: str | int, tool_id: str) -> list[str]:
        """Reserve a block index for a tool. Closes text block if open."""
        events = self.close_text()
        idx = self._cur_idx
        self._cur_idx += 1
        self._tools[tool_key] = {
            "id": tool_id,
            "name": "",
            "block_idx": idx,
            "started": False,
            "closed": False,
        }
        return events

    def start_tool(self, tool_key: str | int, name: str) -> list[str]:
        t = self._tools.get(tool_key)
        if not t or t["started"]:
            return []
        t["name"] = name
        t["started"] = True
        self.had_content = True
        return [sse("content_block_start", {
            "type": "content_block_start",
            "index": t["block_idx"],
            "content_block": {
                "type": "tool_use",
                "id": t["id"],
                "name": name,
                "input": {},
            },
        })]

    def add_tool(self, tool_key: str | int, tool_id: str, name: str) -> list[str]:
        """Register and immediately start a tool block."""
        events = self.register_tool(tool_key, tool_id)
        events.extend(self.start_tool(tool_key, name))
        return events

    def tool_delta(self, tool_key: str | int, partial_json: str) -> list[str]:
        t = self._tools.get(tool_key)
        if not t or not t["started"]:
            return []
        return [sse("content_block_delta", {
            "type": "content_block_delta",
            "index": t["block_idx"],
            "delta": {"type": "input_json_delta", "partial_json": partial_json},
        })]

    def close_tool(self, tool_key: str | int) -> list[str]:
        t = self._tools.get(tool_key)
        if not t or t["closed"]:
            return []
        t["closed"] = True
        return [sse("content_block_stop", {
            "type": "content_block_stop",
            "index": t["block_idx"],
        })]

    def get_tool(self, tool_key: str | int) -> dict[str, Any] | None:
        return self._tools.get(tool_key)

    @property
    def has_tools(self) -> bool:
        return bool(self._tools)

    @property
    def thinking_started(self) -> bool:
        return self._thinking_started

    @property
    def text_started(self) -> bool:
        return self._text_started

    @property
    def tool_keys(self) -> list[str | int]:
        return list(self._tools)

    def finish(self, usage: dict[str, int], signature: str = "") -> list[str]:
        events: list[str] = []
        events.extend(self.close_thinking(signature))
        events.extend(self.close_text())
        for key in list(self._tools):
            events.extend(self.close_tool(key))

        stop_reason = "tool_use" if self._tools else "end_turn"
        events.append(sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": usage,
        }))
        events.append(sse("message_stop", {"type": "message_stop"}))
        return events

    def error_and_finish(self, message: str) -> list[str]:
        return [
            sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": message},
            }),
            *self.finish(dict(DEFAULT_USAGE)),
        ]
