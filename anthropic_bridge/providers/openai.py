from typing import Any

from .base import BaseProvider, ProviderResult


class OpenAIProvider(BaseProvider):
    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        return ProviderResult(cleaned_text=text_content)

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        if original_request.get("thinking"):
            budget = original_request["thinking"].get("budget_tokens", 0)
            # map budget to reasoning_effort: minimal/low/medium/high
            if budget >= 32000:
                effort = "high"
            elif budget >= 16000:
                effort = "medium"
            elif budget >= 8000:
                effort = "low"
            else:
                effort = "minimal"
            request["reasoning_effort"] = effort
            request.pop("thinking", None)
        return request

    def should_handle(self, model_id: str) -> bool:
        lower = model_id.lower()
        return "openai/" in lower or lower.startswith("o1") or lower.startswith("o3")

    def get_name(self) -> str:
        return "OpenAIProvider"
