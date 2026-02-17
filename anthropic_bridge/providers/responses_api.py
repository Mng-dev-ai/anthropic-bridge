import json
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx


def build_responses_input(
    payload: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    system = payload.get("system", "")
    if isinstance(system, list):
        system = "\n\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in system
        )
    if not system:
        system = "You are a helpful assistant."

    input_messages: list[dict[str, Any]] = []
    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            input_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            pending_text: list[str] = []

            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        pending_text.append(item.get("text", ""))
                    elif item.get("type") == "tool_result":
                        if pending_text:
                            input_messages.append(
                                {"role": role, "content": "\n".join(pending_text)}
                            )
                            pending_text.clear()
                        result_content = item.get("content", "")
                        if not isinstance(result_content, str):
                            result_content = json.dumps(result_content)
                        input_messages.append(
                            {
                                "type": "function_call_output",
                                "call_id": item.get("tool_use_id", ""),
                                "output": result_content,
                            }
                        )
                    elif item.get("type") == "tool_use":
                        if pending_text:
                            input_messages.append(
                                {"role": role, "content": "\n".join(pending_text)}
                            )
                            pending_text.clear()
                        input_messages.append(
                            {
                                "type": "function_call",
                                "call_id": item.get("id", ""),
                                "name": item.get("name", ""),
                                "arguments": json.dumps(item.get("input", {})),
                            }
                        )

            if pending_text:
                input_messages.append(
                    {"role": role, "content": "\n".join(pending_text)}
                )

    return system, input_messages


def convert_tools_for_responses(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return []
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        }
        for tool in tools
    ]


async def stream_responses_api(
    endpoint: str,
    headers: dict[str, str],
    request_body: dict[str, Any],
    target_model: str,
) -> AsyncIterator[str]:
    msg_id = f"msg_{int(time.time())}_{_random_id()}"

    yield _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": target_model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    yield _sse("ping", {"type": "ping"})

    text_started = False
    text_idx = -1
    thinking_started = False
    thinking_idx = -1
    cur_idx = 0
    tools: dict[str, dict[str, Any]] = {}
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise RuntimeError(
                        f"API error ({response.status_code}): {error_text.decode()}"
                    )

                current_event = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("event: "):
                        current_event = line[7:]
                        continue

                    if not line.startswith("data: "):
                        continue

                    data_line = line[6:]
                    if not data_line or not current_event:
                        continue

                    try:
                        event_data = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue

                    event_type = current_event

                    if event_type == "response.output_text.delta":
                        delta = event_data.get("delta", "")
                        if delta:
                            if thinking_started:
                                yield _sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": thinking_idx,
                                        "delta": {
                                            "type": "signature_delta",
                                            "signature": "",
                                        },
                                    },
                                )
                                yield _sse(
                                    "content_block_stop",
                                    {
                                        "type": "content_block_stop",
                                        "index": thinking_idx,
                                    },
                                )
                                thinking_started = False

                            if not text_started:
                                text_idx = cur_idx
                                cur_idx += 1
                                yield _sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": text_idx,
                                        "content_block": {
                                            "type": "text",
                                            "text": "",
                                        },
                                    },
                                )
                                text_started = True

                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": delta},
                                },
                            )

                    elif event_type in (
                        "response.reasoning_summary_text.delta",
                        "response.reasoning.delta",
                    ):
                        delta = event_data.get("delta", "")
                        if delta:
                            if not thinking_started:
                                thinking_idx = cur_idx
                                cur_idx += 1
                                yield _sse(
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
                                yield _sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": thinking_idx,
                                        "delta": {
                                            "type": "thinking_delta",
                                            "thinking": delta,
                                        },
                                    },
                                )

                    elif event_type == "response.output_item.added":
                        item = event_data.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get(
                                "call_id", f"tool_{int(time.time())}"
                            )
                            name = item.get("name", "")
                            tools[call_id] = {
                                "id": call_id,
                                "name": name,
                                "block_idx": cur_idx,
                                "started": True,
                                "closed": False,
                            }
                            cur_idx += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": tools[call_id]["block_idx"],
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": call_id,
                                        "name": name,
                                    },
                                },
                            )

                    elif event_type == "response.function_call_arguments.delta":
                        call_id = event_data.get("call_id", "")
                        delta = event_data.get("delta", "")
                        if call_id in tools and delta:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": tools[call_id]["block_idx"],
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": delta,
                                    },
                                },
                            )

                    elif event_type == "response.output_item.done":
                        item = event_data.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id", "")
                            if call_id in tools and not tools[call_id]["closed"]:
                                args = item.get("arguments", "")
                                if args:
                                    yield _sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": tools[call_id]["block_idx"],
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": args,
                                            },
                                        },
                                    )
                                yield _sse(
                                    "content_block_stop",
                                    {
                                        "type": "content_block_stop",
                                        "index": tools[call_id]["block_idx"],
                                    },
                                )
                                tools[call_id]["closed"] = True

                    elif event_type == "response.completed":
                        resp = event_data.get("response", {})
                        resp_usage = resp.get("usage", {})
                        usage = {
                            "input_tokens": resp_usage.get("input_tokens", 0),
                            "output_tokens": resp_usage.get("output_tokens", 0),
                        }
                        break

    except Exception as e:
        yield _sse(
            "error",
            {"type": "error", "error": {"type": "api_error", "message": str(e)}},
        )
        yield _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        )
        yield _sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"
        return

    if thinking_started:
        yield _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": thinking_idx,
                "delta": {"type": "signature_delta", "signature": ""},
            },
        )
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": thinking_idx},
        )

    if text_started:
        yield _sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": text_idx},
        )

    for t in tools.values():
        if t["started"] and not t["closed"]:
            yield _sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": t["block_idx"]},
            )

    stop_reason = "tool_use" if tools else "end_turn"
    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
            },
            "usage": usage,
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})
    yield "data: [DONE]\n\n"


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _random_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
