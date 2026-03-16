from typing import Any

from ..utils import map_reasoning_effort, normalize_model_id
from .base import ProviderResult
from .grok import GrokProvider


def _needs_developer_role(model_id: str) -> bool:
    lower = normalize_model_id(model_id)
    return lower.startswith(("o1", "o3", "o4")) or lower.startswith("gpt-5")


class ProviderRegistry:
    def __init__(self, model_id: str):
        self.model_id = model_id
        lower = model_id.lower()
        self._grok = GrokProvider(model_id) if ("grok" in lower or "x-ai/" in lower) else None
        self._is_gemini = "gemini" in lower or "google/" in lower
        self._is_openai = "openai/" in lower or lower.startswith(("o1", "o3", "o4"))
        self._is_deepseek = "deepseek" in lower
        self._is_minimax = "minimax" in lower
        self._is_qwen = "qwen" in lower
        self._use_developer_role = self._is_openai and _needs_developer_role(model_id)

    def process_text_content(self, text: str, accumulated: str) -> ProviderResult:
        if self._grok:
            return self._grok.process_text_content(text, accumulated)
        return ProviderResult(cleaned_text=text)

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> None:
        thinking = original_request.get("thinking")

        if self._grok:
            if thinking:
                if "mini" in self.model_id.lower():
                    budget = thinking.get("budget_tokens", 0)
                    request["reasoning_effort"] = "high" if budget >= 20000 else "low"
                request.pop("thinking", None)
            return

        if self._is_gemini:
            if thinking:
                budget = thinking.get("budget_tokens", 0)
                if "gemini-3" in self.model_id.lower():
                    request["thinking_level"] = "high" if budget >= 16000 else "low"
                else:
                    request["thinking_budget"] = min(budget, 24000)
                request.pop("thinking", None)
            return

        if self._is_openai:
            reasoning_active = False
            if thinking:
                budget = thinking.get("budget_tokens", 0)
                effort = map_reasoning_effort(budget, request.get("model"))
                if effort:
                    request["reasoning_effort"] = effort
                    reasoning_active = True
                request.pop("thinking", None)
            if reasoning_active:
                request.pop("temperature", None)
            if self._use_developer_role:
                for msg in request.get("messages", []):
                    if msg.get("role") == "system":
                        msg["role"] = "developer"
            return

        if self._is_deepseek:
            request.pop("thinking", None)
            return

        if self._is_minimax:
            if thinking:
                request["reasoning_split"] = True
                request.pop("thinking", None)
            return

        if self._is_qwen:
            if thinking:
                budget = thinking.get("budget_tokens", 0)
                request["enable_thinking"] = True
                request["thinking_budget"] = budget
                request.pop("thinking", None)
            return

    def reset(self) -> None:
        if self._grok:
            self._grok.reset()
