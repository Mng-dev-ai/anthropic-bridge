from typing import Any

from ..utils import map_reasoning_effort, normalize_model_id
from .base import BaseProvider, ProviderResult


def _needs_developer_role(model_id: str) -> bool:
    """GPT-5+ and o-series models use 'developer' role instead of 'system'."""
    lower = normalize_model_id(model_id)
    return (
        lower.startswith(("o1", "o3", "o4"))
        or lower.startswith("gpt-5")
    )


class OpenRouterOpenAIProvider(BaseProvider):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._use_developer_role = _needs_developer_role(model_id)

    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        return ProviderResult(cleaned_text=text_content)

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        reasoning_active = False

        if original_request.get("thinking"):
            budget = original_request["thinking"].get("budget_tokens", 0)
            effort = map_reasoning_effort(budget, request.get("model"))
            if effort:
                request["reasoning_effort"] = effort
                reasoning_active = True
            request.pop("thinking", None)

        # Temperature is incompatible with reasoning_effort != "none"
        if reasoning_active:
            request.pop("temperature", None)

        # GPT-5+ and o-series use 'developer' role instead of 'system'
        if self._use_developer_role:
            for msg in request.get("messages", []):
                if msg.get("role") == "system":
                    msg["role"] = "developer"

        return request

    def should_handle(self, model_id: str) -> bool:
        lower = model_id.lower()
        return "openai/" in lower or lower.startswith(("o1", "o3", "o4"))

    def get_name(self) -> str:
        return "OpenRouterOpenAIProvider"
