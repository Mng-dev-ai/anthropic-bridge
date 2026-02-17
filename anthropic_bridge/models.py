from typing import Any, Literal, TypedDict


class ContentBlock(TypedDict, total=False):
    type: Literal["text", "image", "tool_use", "tool_result", "thinking"]
    text: str
    id: str
    name: str
    input: dict[str, Any]
    tool_use_id: str
    content: str | list[Any]
    source: dict[str, str]
    thinking: str


class AnthropicMessage(TypedDict, total=False):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlock]


class AnthropicRequest(TypedDict, total=False):
    model: str
    messages: list[AnthropicMessage]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    system: str | list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_choice: dict[str, Any]
    thinking: dict[str, Any]
    stop_sequences: list[str]
    metadata: dict[str, Any]


class OpenRouterMessage(TypedDict, total=False):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str


class OpenRouterRequest(TypedDict, total=False):
    model: str
    messages: list[OpenRouterMessage]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool
    tools: list[dict[str, Any]]
    tool_choice: str | dict[str, Any]
    stream_options: dict[str, bool]
    include_reasoning: bool
    thinking: dict[str, Any]
