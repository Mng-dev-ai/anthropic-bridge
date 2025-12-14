from typing import Any

from .base import BaseProvider, ProviderResult


class QwenProvider(BaseProvider):
    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        return ProviderResult(cleaned_text=text_content)

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        if original_request.get("thinking"):
            budget = original_request["thinking"].get("budget_tokens", 0)
            request["enable_thinking"] = True
            request["thinking_budget"] = budget
            request.pop("thinking", None)
        return request

    def should_handle(self, model_id: str) -> bool:
        return "qwen" in model_id.lower()

    def get_name(self) -> str:
        return "QwenProvider"
