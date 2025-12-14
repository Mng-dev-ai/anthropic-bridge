from abc import ABC, abstractmethod
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


class BaseProvider(ABC):
    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        pass

    @abstractmethod
    def should_handle(self, model_id: str) -> bool:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        return request

    def reset(self) -> None:  # noqa: B027
        pass


class DefaultProvider(BaseProvider):
    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        return ProviderResult(cleaned_text=text_content)

    def should_handle(self, model_id: str) -> bool:
        return False

    def get_name(self) -> str:
        return "DefaultProvider"
