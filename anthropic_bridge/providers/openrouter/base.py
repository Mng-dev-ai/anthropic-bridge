from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResult:
    cleaned_text: str
    extracted_tool_calls: list[ToolCall] = field(default_factory=list)
    was_transformed: bool = False
