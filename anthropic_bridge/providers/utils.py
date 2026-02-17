import json
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any


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


def _model_supports_xhigh(model_id: str | None) -> bool:
    if not model_id:
        return False

    lower = model_id.lower()
    if "/" in lower:
        lower = lower.split("/", 1)[1]

    # OpenAI docs: xhigh is supported for gpt-5.1-codex-max and for models
    # after gpt-5.1-codex-max (e.g., gpt-5.2, gpt-5.3 and their codex variants).
    return lower.startswith(("gpt-5.1-codex-max", "gpt-5.2", "gpt-5.3"))
