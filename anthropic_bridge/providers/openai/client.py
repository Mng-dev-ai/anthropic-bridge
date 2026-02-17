from collections.abc import AsyncIterator
from typing import Any

from ..responses_api import (
    build_responses_input,
    convert_tools_for_responses,
    stream_responses_api,
)
from ..utils import map_reasoning_effort, yield_error_events
from .auth import get_auth

CODEX_API_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"


class OpenAIProvider:
    def __init__(self, target_model: str):
        self.target_model = target_model.removeprefix("openai/")
        self._access_token: str | None = None
        self._account_id: str | None = None
        self._expires_at: float = 0

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        try:
            self._access_token, self._account_id, self._expires_at = await get_auth(
                self._access_token, self._account_id, self._expires_at
            )

            instructions, input_messages = build_responses_input(payload)
            tools = convert_tools_for_responses(payload.get("tools"))

            request_body: dict[str, Any] = {
                "model": self.target_model,
                "instructions": instructions,
                "input": input_messages,
                "stream": True,
                "store": False,
            }

            if payload.get("temperature") is not None:
                request_body["temperature"] = payload["temperature"]

            if tools:
                request_body["tools"] = tools

            if payload.get("thinking"):
                effort = map_reasoning_effort(
                    payload["thinking"].get("budget_tokens"), self.target_model
                )
                if effort:
                    request_body["reasoning"] = {"effort": effort, "summary": "auto"}

        except Exception as e:
            async for event in yield_error_events(str(e), self.target_model):
                yield event
            return

        async for event in stream_responses_api(
            CODEX_API_ENDPOINT,
            self._build_headers(),
            request_body,
            self.target_model,
        ):
            yield event

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._access_token}",
        }
        if self._account_id:
            headers["ChatGPT-Account-Id"] = self._account_id
        return headers
