import json
from collections.abc import AsyncIterator
from typing import Any

from .providers.utils import estimate_input_tokens
from .transform import (
    convert_anthropic_messages_to_openai,
    convert_anthropic_tools_to_openai,
)

_DEFAULT_USAGE = {
    "input_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 0,
}

_BINARY_TYPES = {"base64", "image", "document", "file"}


async def iter_sse_events(
    chunks: AsyncIterator[str],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    buffer = ""
    current_event = ""

    async for chunk in chunks:
        buffer += chunk
        lines = buffer.split("\n")
        buffer = lines.pop()

        for raw_line in lines:
            line = raw_line.rstrip("\r")
            if not line:
                continue
            if line.startswith("event: "):
                current_event = line[7:]
                continue
            if not current_event or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if not data_str or data_str == "[DONE]":
                continue

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if isinstance(data, dict):
                yield current_event, data


async def collect_anthropic_response(
    chunks: AsyncIterator[str],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    message: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    tool_input_chunks: dict[int, list[str]] = {}
    error: dict[str, Any] | None = None

    async for event, data in iter_sse_events(chunks):
        if event == "message_start":
            start_message = data.get("message", {})
            message = {
                "id": start_message.get("id", ""),
                "type": start_message.get("type", "message"),
                "role": start_message.get("role", "assistant"),
                "content": [],
                "model": start_message.get("model"),
                "stop_reason": start_message.get("stop_reason"),
                "stop_sequence": start_message.get("stop_sequence"),
                "usage": dict(start_message.get("usage", _DEFAULT_USAGE)),
            }
            continue

        if event == "content_block_start":
            index = data.get("index")
            if isinstance(index, int):
                block = data.get("content_block", {})
                if isinstance(block, dict):
                    blocks[index] = dict(block)
                    if block.get("type") == "tool_use":
                        tool_input_chunks[index] = []
            continue

        if event == "content_block_delta":
            index = data.get("index")
            if not isinstance(index, int):
                continue
            block = blocks.setdefault(index, {})
            delta = data.get("delta", {})
            if not isinstance(delta, dict):
                continue

            delta_type = delta.get("type")
            if delta_type == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta_type == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get(
                    "thinking", ""
                )
            elif delta_type == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get(
                    "signature", ""
                )
            elif delta_type == "input_json_delta":
                tool_input_chunks.setdefault(index, []).append(
                    delta.get("partial_json", "")
                )
            continue

        if event == "message_delta" and message is not None:
            delta = data.get("delta", {})
            if isinstance(delta, dict):
                message["stop_reason"] = delta.get("stop_reason")
                message["stop_sequence"] = delta.get("stop_sequence")
            usage = data.get("usage")
            if isinstance(usage, dict):
                message["usage"] = dict(usage)
            continue

        if event == "error" and error is None:
            raw_error = data.get("error", {})
            if isinstance(raw_error, dict):
                error = {
                    "type": "error",
                    "error": {
                        "type": raw_error.get("type", "api_error"),
                        "message": raw_error.get("message", "Upstream provider error"),
                    },
                }

    if message is None:
        return None, error

    ordered_blocks = []
    for index in sorted(blocks):
        block = dict(blocks[index])
        if block.get("type") == "tool_use":
            block["input"] = _parse_tool_input(tool_input_chunks.get(index, []))
        if block.get("type") == "thinking" and not block.get("signature"):
            block.pop("signature", None)
        ordered_blocks.append(block)

    message["content"] = ordered_blocks
    return message, error


def estimate_anthropic_input_tokens(payload: dict[str, Any]) -> int:
    messages = convert_anthropic_messages_to_openai(
        _normalize_messages_for_estimate(payload.get("messages", [])),
        _normalize_system_for_estimate(payload.get("system")),
    )
    tools = convert_anthropic_tools_to_openai(payload.get("tools"))
    return estimate_input_tokens(messages, tools)


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(_strip_binary_payload(value), sort_keys=True)


def _normalize_messages_for_estimate(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    return [
        {
            **message,
            "content": _normalize_content_for_estimate(message.get("content", "")),
        }
        for message in messages
        if isinstance(message, dict)
    ]


def _normalize_content_for_estimate(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return _stringify_value(content)
    return [_normalize_block_for_estimate(block) for block in content]


def _normalize_block_for_estimate(block: Any) -> dict[str, Any]:
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)}

    block_type = block.get("type")
    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}
    if block_type == "thinking":
        return {"type": "text", "text": block.get("thinking", "")}
    if block_type in {"image", "document"}:
        return {"type": "text", "text": _describe_media_block(block)}
    if block_type == "tool_use":
        return {**block, "input": _strip_binary_payload(block.get("input", {}))}
    if block_type == "tool_result":
        return {**block, "content": _strip_binary_payload(block.get("content", ""))}
    return {"type": "text", "text": _stringify_value(block)}


def _normalize_system_for_estimate(system: Any) -> str | None:
    if system is None or isinstance(system, str):
        return system
    if not isinstance(system, list):
        return _stringify_value(system)
    return "\n\n".join(
        item["text"]
        if isinstance(item, dict) and isinstance(item.get("text"), str)
        else _stringify_value(item)
        for item in system
    )


def _describe_media_block(block: dict[str, Any]) -> str:
    block_type = block.get("type", "media")
    source = block.get("source", {})
    if not isinstance(source, dict):
        return f"[{block_type}]"

    media_type = source.get("media_type")
    source_type = source.get("type")
    details = " ".join(
        str(part) for part in (block_type, media_type, source_type) if part
    )
    return f"[{details or block_type}]"


def _strip_binary_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_binary_payload(item) for item in value]

    if not isinstance(value, dict):
        return value

    is_binary_container = (
        "media_type" in value or value.get("type") in _BINARY_TYPES
    )
    sanitized: dict[str, Any] = {}
    for key, item in value.items():
        if key == "data" and is_binary_container:
            sanitized[key] = "[binary omitted]"
        else:
            sanitized[key] = _strip_binary_payload(item)
    return sanitized


def _parse_tool_input(deltas: list[str]) -> Any:
    if not deltas:
        return {}

    candidate = "".join(deltas)
    if not candidate:
        return {}

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}
