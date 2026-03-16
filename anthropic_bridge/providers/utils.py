import json
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any

import tiktoken

_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def estimate_input_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate input token count from OpenAI-format messages using tiktoken.

    Uses cl100k_base encoding which is a reasonable approximation for most models.
    Adds per-message overhead similar to OpenAI's token counting rules.
    """
    enc = _get_encoding()
    total = 0
    for msg in messages:
        total += 4  # every message has <|im_start|>role\n ... <|im_end|>\n
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
        # Count tool call arguments
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name"):
                total += len(enc.encode(fn["name"]))
            if fn.get("arguments"):
                total += len(enc.encode(fn["arguments"]))
    total += 2  # assistant priming
    return total


async def yield_error_events(
    message: str, model: str
) -> AsyncIterator[str]:
    """Yield a complete SSE error sequence so the SDK always gets a valid message."""
    msg_id = f"msg_{int(time.time())}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=12))}"
    usage: dict[str, int] = {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
    }
    yield _sse(
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
                "usage": usage,
            },
        },
    )
    yield _sse(
        "error",
        {"type": "error", "error": {"type": "api_error", "message": message}},
    )
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": usage,
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


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
    """Strip provider prefix and lowercase a model ID (e.g. 'openai/gpt-5.4' → 'gpt-5.4')."""
    lower = model_id.lower()
    if "/" in lower:
        lower = lower.split("/", 1)[1]
    return lower


def _model_supports_xhigh(model_id: str | None) -> bool:
    if not model_id:
        return False

    lower = normalize_model_id(model_id)

    # OpenAI docs: xhigh is supported for gpt-5.1-codex-max and for models
    # after gpt-5.1-codex-max (e.g., gpt-5.2, gpt-5.3, gpt-5.4 and their codex variants).
    return lower.startswith(("gpt-5.1-codex-max", "gpt-5.2", "gpt-5.3", "gpt-5.4"))
