import json
import random
import re
import string
import time
from typing import Any

from .base import BaseProvider, ProviderResult, ToolCall


class GrokProvider(BaseProvider):
    def __init__(self, model_id: str):
        super().__init__(model_id)
        self._xml_buffer = ""

    def process_text_content(
        self, text_content: str, accumulated_text: str
    ) -> ProviderResult:
        self._xml_buffer += text_content

        xml_pattern = re.compile(
            r'<xai:function_call name="([^"]+)">(.*?)</xai:function_call>', re.DOTALL
        )
        matches = list(xml_pattern.finditer(self._xml_buffer))

        if not matches:
            if "<xai:function_call" in self._xml_buffer:
                return ProviderResult(cleaned_text="")

            result = ProviderResult(cleaned_text=self._xml_buffer)
            self._xml_buffer = ""
            return result

        tool_calls = [
            ToolCall(
                id=f"grok_{int(time.time())}_{self._random_id()}",
                name=match.group(1),
                arguments=self._parse_xml_params(match.group(2)),
            )
            for match in matches
        ]

        cleaned = self._xml_buffer
        for match in matches:
            cleaned = cleaned.replace(match.group(0), "")

        self._xml_buffer = ""
        return ProviderResult(
            cleaned_text=cleaned.strip(),
            extracted_tool_calls=tool_calls,
            was_transformed=True,
        )

    def prepare_request(
        self, request: dict[str, Any], original_request: dict[str, Any]
    ) -> dict[str, Any]:
        if original_request.get("thinking"):
            if "mini" in self.model_id.lower():
                budget = original_request["thinking"].get("budget_tokens", 0)
                request["reasoning_effort"] = "high" if budget >= 20000 else "low"
            request.pop("thinking", None)
        return request

    def should_handle(self, model_id: str) -> bool:
        return "grok" in model_id.lower() or "x-ai/" in model_id.lower()

    def get_name(self) -> str:
        return "GrokProvider"

    def reset(self) -> None:
        self._xml_buffer = ""

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=9))

    def _parse_xml_params(self, xml_content: str) -> dict[str, Any]:
        params: dict[str, Any] = {}
        param_pattern = re.compile(
            r'<xai:parameter name="([^"]+)">([^<]*)</xai:parameter>'
        )

        for match in param_pattern.finditer(xml_content):
            name, value = match.group(1), match.group(2)
            try:
                params[name] = json.loads(value)
            except json.JSONDecodeError:
                params[name] = value

        return params
