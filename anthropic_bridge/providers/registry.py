from .base import BaseProvider, DefaultProvider
from .deepseek import DeepSeekProvider
from .gemini import GeminiProvider
from .grok import GrokProvider
from .minimax import MiniMaxProvider
from .openai import OpenAIProvider
from .qwen import QwenProvider


class ProviderRegistry:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self._providers: list[BaseProvider] = [
            GrokProvider(model_id),
            GeminiProvider(model_id),
            OpenAIProvider(model_id),
            QwenProvider(model_id),
            MiniMaxProvider(model_id),
            DeepSeekProvider(model_id),
        ]
        self._default = DefaultProvider(model_id)

    def get_provider(self) -> BaseProvider:
        for provider in self._providers:
            if provider.should_handle(self.model_id):
                return provider
        return self._default
