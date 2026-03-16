from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import get_reasoning_cache
from .protocol import collect_anthropic_response, estimate_anthropic_input_tokens
from .providers import CopilotProvider, OpenAIProvider, OpenRouterProvider
from .providers.openai.auth import auth_file_exists

ProviderType = Literal["openrouter", "copilot", "openai"]


@dataclass
class ProxyConfig:
    openrouter_api_key: str | None = None
    copilot_token: str | None = None


class AnthropicBridge:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = FastAPI(title="Anthropic Bridge")
        self._openrouter_clients: dict[str, OpenRouterProvider] = {}
        self._openai_clients: dict[str, OpenAIProvider] = {}
        self._copilot_clients: dict[str, CopilotProvider] = {}
        self._setup_routes()
        self._setup_cors()
        get_reasoning_cache()

    def _setup_cors(self) -> None:
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        @self.app.get("/")
        async def root() -> dict[str, str]:
            return {"status": "ok", "message": "Anthropic Bridge"}

        @self.app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.post("/v1/messages/count_tokens")
        async def count_tokens(request: Request) -> JSONResponse:
            body = await request.json()
            return JSONResponse({"input_tokens": estimate_anthropic_input_tokens(body)})

        @self.app.post("/v1/messages", response_model=None)
        async def messages(request: Request) -> StreamingResponse | JSONResponse:
            body = await request.json()
            model = body.get("model", "")

            provider = self._get_provider(model)
            if provider is None:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": self._get_provider_error_message(model),
                        }
                    },
                )

            if body.get("stream") is not True:
                message, error = await collect_anthropic_response(provider.handle(body))
                if error:
                    return JSONResponse(status_code=502, content=error)
                if message is None:
                    return JSONResponse(
                        status_code=502,
                        content={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Provider returned no message.",
                            },
                        },
                    )
                return JSONResponse(message)

            return StreamingResponse(
                provider.handle(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _get_provider(
        self, model: str
    ) -> CopilotProvider | OpenAIProvider | OpenRouterProvider | None:
        requested_provider = self._get_requested_provider(model)

        if requested_provider:
            requested_model = self._model_for_provider(model, requested_provider)
            return self._make_provider(requested_model, requested_provider)

        for provider_type in ("openrouter", "copilot", "openai"):
            provider_model = self._model_for_provider(model, provider_type)
            provider = self._make_provider(provider_model, provider_type)
            if provider:
                return provider

        return None

    def _get_provider_error_message(self, model: str) -> str:
        requested_provider = self._get_requested_provider(model)
        if requested_provider is None:
            return (
                "No provider configured. Set OPENROUTER_API_KEY, "
                "GITHUB_COPILOT_TOKEN, or configure OpenAI auth."
            )

        messages = {
            "openai": "OpenAI auth not configured. Run 'codex login' to use openai/* models.",
            "copilot": "GitHub Copilot token not configured. Set GITHUB_COPILOT_TOKEN to use copilot/* models.",
            "openrouter": "OpenRouter API key not configured. Set OPENROUTER_API_KEY to use openrouter/* models.",
        }
        return messages[requested_provider]

    def _get_requested_provider(self, model: str) -> ProviderType | None:
        if model.startswith("openai/"):
            return "openai"
        if model.startswith("copilot/"):
            return "copilot"
        if model.startswith("openrouter/"):
            return "openrouter"
        return None

    def _model_for_provider(self, model: str, provider_type: ProviderType) -> str:
        model_name = model.split("/", 1)[1] if "/" in model else model
        if provider_type == "openrouter":
            return f"openrouter/{model_name}"
        if provider_type == "copilot":
            return f"copilot/{model_name}"
        return f"openai/{model_name}"

    def _make_provider(
        self, model: str, provider_type: ProviderType
    ) -> CopilotProvider | OpenAIProvider | OpenRouterProvider | None:
        if provider_type == "openrouter" and self.config.openrouter_api_key:
            if model not in self._openrouter_clients:
                self._openrouter_clients[model] = OpenRouterProvider(
                    model, self.config.openrouter_api_key
                )
            return self._openrouter_clients[model]

        if provider_type == "copilot" and self.config.copilot_token:
            if model not in self._copilot_clients:
                self._copilot_clients[model] = CopilotProvider(
                    model, self.config.copilot_token
                )
            return self._copilot_clients[model]

        if provider_type == "openai" and auth_file_exists():
            if model not in self._openai_clients:
                self._openai_clients[model] = OpenAIProvider(model)
            return self._openai_clients[model]

        return None


def create_app(
    openrouter_api_key: str | None = None,
    copilot_token: str | None = None,
) -> FastAPI:
    config = ProxyConfig(
        openrouter_api_key=openrouter_api_key,
        copilot_token=copilot_token,
    )
    return AnthropicBridge(config).app
