import json
from typing import Any

DROP_KEYS = [
    "n",
    "presence_penalty",
    "frequency_penalty",
    "best_of",
    "logit_bias",
    "seed",
    "stream_options",
    "logprobs",
    "top_logprobs",
    "user",
    "response_format",
    "service_tier",
    "parallel_tool_calls",
    "functions",
    "function_call",
    "developer",
    "strict",
    "reasoning_effort",
]


def remove_uri_format(schema: Any) -> Any:
    if not schema or not isinstance(schema, dict):
        return schema

    if schema.get("type") == "string" and schema.get("format") == "uri":
        return {k: v for k, v in schema.items() if k != "format"}

    if isinstance(schema, list):
        return [remove_uri_format(item) for item in schema]

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "properties" and isinstance(value, dict):
            result[key] = {k: remove_uri_format(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = remove_uri_format(value)
        elif key == "additionalProperties" and isinstance(value, dict):
            result[key] = remove_uri_format(value)
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            result[key] = [remove_uri_format(item) for item in value]
        else:
            result[key] = (
                remove_uri_format(value) if isinstance(value, (dict, list)) else value
            )
    return result


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif block.get("content"):
                    text_parts.append(extract_text_content(block["content"]))
        return "\n".join(text_parts)

    if isinstance(content, dict):
        if content.get("text"):
            return str(content["text"])
        elif content.get("content"):
            return extract_text_content(content["content"])

    return json.dumps(content)


def sanitize_anthropic_request(req: dict[str, Any]) -> list[str]:
    dropped: list[str] = []

    if "stop" in req:
        stop = req.pop("stop")
        req["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    if "user" in req:
        req["metadata"] = {**req.get("metadata", {}), "user_id": req.pop("user")}
        dropped.append("user")

    for key in DROP_KEYS:
        if key in req:
            dropped.append(key)
            del req[key]

    if req.get("max_tokens") is None:
        req["max_tokens"] = 4096

    return dropped


def convert_anthropic_tools_to_openai(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return []

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": remove_uri_format(tool.get("input_schema", {})),
            },
        }
        for tool in tools
    ]


def convert_anthropic_tool_choice_to_openai(
    tool_choice: dict[str, Any] | None,
) -> str | dict[str, Any] | None:
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type == "none":
        return "none"
    elif choice_type == "any":
        return "required"
    elif choice_type == "auto":
        return "auto"
    elif choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return "auto"


def convert_anthropic_messages_to_openai(
    messages: list[dict[str, Any]], system: str | list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    openai_messages: list[dict[str, Any]] = []

    if system:
        if isinstance(system, list):
            system_text = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )
        else:
            system_text = system
        openai_messages.append({"role": "system", "content": system_text})

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            if isinstance(content, list):
                content_parts: list[dict[str, Any]] = []
                seen_tool_ids: set[str] = set()

                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        content_parts.append(
                            {"type": "text", "text": block.get("text", "")}
                        )
                    elif block_type == "image":
                        source = block.get("source", {})
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{source.get('media_type')};base64,{source.get('data')}"
                                },
                            }
                        )
                    elif block_type == "tool_result":
                        tool_id = block.get("tool_use_id")
                        if tool_id and tool_id not in seen_tool_ids:
                            if content_parts:
                                openai_messages.append({"role": "user", "content": list(content_parts)})
                                content_parts.clear()
                            seen_tool_ids.add(tool_id)
                            result_content = block.get("content", "")
                            if not isinstance(result_content, str):
                                result_content = json.dumps(result_content)
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "content": result_content,
                                    "tool_call_id": tool_id,
                                }
                            )

                if content_parts:
                    openai_messages.append({"role": "user", "content": content_parts})
            else:
                openai_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            if isinstance(content, list):
                text_parts: list[str] = []
                tool_calls: list[dict[str, Any]] = []
                seen_ids: set[str] = set()

                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        text_parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        block_id = block.get("id")
                        if block_id and block_id not in seen_ids:
                            seen_ids.add(block_id)
                            tool_calls.append(
                                {
                                    "id": block_id,
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name"),
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                }
                            )

                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = " ".join(text_parts)
                elif tool_calls:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                if assistant_msg.get("content") is not None or assistant_msg.get(
                    "tool_calls"
                ):
                    openai_messages.append(assistant_msg)
            else:
                openai_messages.append({"role": "assistant", "content": content})

    return openai_messages
