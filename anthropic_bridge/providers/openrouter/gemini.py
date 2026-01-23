from typing import Any

from .base import BaseProvider, ProviderResult


class GeminiProvider(BaseProvider):
    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        return ProviderResult(cleaned_text=text_content)

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        if original_request.get("thinking"):
            budget = original_request["thinking"].get("budget_tokens", 0)
            if "gemini-3" in self.model_id.lower():
                request["thinking_level"] = "high" if budget >= 16000 else "low"
            else:
                request["thinking_budget"] = min(budget, 24000)
            request.pop("thinking", None)
        return request

    def should_handle(self, model_id: str) -> bool:
        return "gemini" in model_id.lower() or "google/" in model_id.lower()

    def get_name(self) -> str:
        return "GeminiProvider"
