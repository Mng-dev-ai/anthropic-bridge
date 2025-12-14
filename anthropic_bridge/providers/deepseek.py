from typing import Any

from .base import BaseProvider, ProviderResult


class DeepSeekProvider(BaseProvider):
    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        return ProviderResult(cleaned_text=text_content)

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        # DeepSeek R1 auto-manages reasoning, strip thinking params to avoid API errors
        request.pop("thinking", None)
        return request

    def should_handle(self, model_id: str) -> bool:
        return "deepseek" in model_id.lower()

    def get_name(self) -> str:
        return "DeepSeekProvider"
