import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx

from ..transform import normalize_system_message
from .utils import DEFAULT_USAGE, AnthropicSSEEmitter, estimate_input_tokens


def build_responses_input(
    payload: dict[str, Any],
) -> tuple[str | None, list[dict[str, Any]]]:
    system = normalize_system_message(payload.get("system"))
    input_messages: list[dict[str, Any]] = []
    for msg in payload.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            input_messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        pending_text: list[str] = []

        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "text":
                pending_text.append(item.get("text", ""))
                continue

            if pending_text:
                input_messages.append(
                    {"role": role, "content": "\n".join(pending_text)}
                )
                pending_text.clear()

            if item_type == "tool_result":
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
            elif item_type == "tool_use":
                input_messages.append(
                    {
                        "type": "function_call",
                        "call_id": item.get("id", ""),
                        "name": item.get("name", ""),
                        "arguments": json.dumps(item.get("input", {})),
                    }
                )

        if pending_text:
            input_messages.append({"role": role, "content": "\n".join(pending_text)})

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


def convert_tool_choice_for_responses(
    tool_choice: dict[str, Any] | None,
) -> str | dict[str, Any] | None:
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "none":
        return "none"
    if choice_type == "any":
        return "required"
    if choice_type == "auto":
        return "auto"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "name": tool_choice["name"]}
    return "auto"


def _estimate_responses_input_tokens(
    input_messages: list[dict[str, Any]],
    instructions: str | None,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for item in input_messages:
        if item.get("role"):
            messages.append({"role": item["role"], "content": item.get("content", "")})
        elif item.get("type") == "function_call_output":
            messages.append({"role": "tool", "content": item.get("output", "")})
        elif item.get("type") == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", ""),
                            }
                        }
                    ],
                }
            )
    return estimate_input_tokens(messages, tools)


async def stream_responses_api(
    endpoint: str,
    headers: dict[str, str],
    request_body: dict[str, Any],
    target_model: str,
) -> AsyncIterator[str]:
    estimated_input = await asyncio.to_thread(
        _estimate_responses_input_tokens,
        request_body.get("input", []),
        request_body.get("instructions", ""),
        request_body.get("tools"),
    )

    emitter = AnthropicSSEEmitter(target_model, estimated_input)
    for _e in emitter.message_start():
        yield _e

    reasoning_event_mode: Literal["summary", "reasoning"] | None = None
    arguments_streamed: dict[str, bool] = {}
    usage = dict(DEFAULT_USAGE)

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", endpoint, headers=headers, json=request_body,
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

                    if current_event == "response.output_text.delta":
                        delta = event_data.get("delta", "")
                        if delta:
                            for _e in emitter.close_thinking():
                                yield _e
                            for _e in emitter.text_delta(delta):
                                yield _e

                    elif current_event in {
                        "response.reasoning_summary_text.delta",
                        "response.reasoning.delta",
                    }:
                        delta = event_data.get("delta", "")
                        event_mode: Literal["summary", "reasoning"] = (
                            "summary"
                            if current_event == "response.reasoning_summary_text.delta"
                            else "reasoning"
                        )
                        # Stick to the first reasoning stream to avoid duplicated thinking deltas
                        if delta and (
                            reasoning_event_mode is None
                            or reasoning_event_mode == event_mode
                        ):
                            reasoning_event_mode = event_mode
                            for _e in emitter.thinking_delta(delta):
                                yield _e

                    elif current_event == "response.output_item.added":
                        item = event_data.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id", f"tool_{int(time.time())}")
                            name = item.get("name", "")
                            for _e in emitter.add_tool(call_id, call_id, name):
                                yield _e

                    elif current_event == "response.function_call_arguments.delta":
                        call_id = event_data.get("call_id", "")
                        delta = event_data.get("delta", "")
                        if call_id and delta:
                            arguments_streamed[call_id] = True
                            for _e in emitter.tool_delta(call_id, delta):
                                yield _e

                    elif current_event == "response.output_item.done":
                        item = event_data.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id", "")
                            args = item.get("arguments", "")
                            if args and not arguments_streamed.get(call_id):
                                for _e in emitter.tool_delta(call_id, args):
                                    yield _e
                            for _e in emitter.close_tool(call_id):
                                yield _e

                    elif current_event == "response.completed":
                        resp = event_data.get("response", {})
                        resp_usage = resp.get("usage", {})
                        usage = {
                            "input_tokens": resp_usage.get("input_tokens", 0),
                            "cache_creation_input_tokens": resp_usage.get("cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": resp_usage.get("cache_read_input_tokens", 0),
                            "output_tokens": resp_usage.get("output_tokens", 0),
                        }
                        break

    except Exception as e:
        for _e in emitter.error_and_finish(str(e)):
            yield _e
        return

    for _e in emitter.finish(usage):
        yield _e
