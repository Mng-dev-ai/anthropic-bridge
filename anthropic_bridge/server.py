import json
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import get_reasoning_cache
from .providers import OpenAIProvider, OpenRouterProvider


@dataclass
class ProxyConfig:
    openrouter_api_key: str | None = None


class AnthropicBridge:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = FastAPI(title="Anthropic Bridge")
        self._openrouter_clients: dict[str, OpenRouterProvider] = {}
        self._openai_clients: dict[str, OpenAIProvider] = {}
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
            text = json.dumps(body)
            return JSONResponse({"input_tokens": len(text) // 4})

        @self.app.post("/v1/messages", response_model=None)
        async def messages(request: Request) -> StreamingResponse | JSONResponse:
            body = await request.json()
            model = body.get("model", "")

            provider = self._get_provider(model)
            if provider is None:
                if model.startswith("openrouter/"):
                    error_msg = f"OPENROUTER_API_KEY required for model '{model}'."
                else:
                    error_msg = (
                        f"Unknown model prefix '{model}'. "
                        "Use openai/* or openrouter/* models."
                    )
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": error_msg,
                        }
                    },
                )

            return StreamingResponse(
                provider.handle(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _get_provider(self, model: str) -> OpenAIProvider | OpenRouterProvider | None:
        if model.startswith("openai/"):
            if model not in self._openai_clients:
                self._openai_clients[model] = OpenAIProvider(model)
            return self._openai_clients[model]

        if model.startswith("openrouter/"):
            if not self.config.openrouter_api_key:
                return None
            if model not in self._openrouter_clients:
                self._openrouter_clients[model] = OpenRouterProvider(
                    model, self.config.openrouter_api_key
                )
            return self._openrouter_clients[model]

        return None


def create_app(openrouter_api_key: str | None = None) -> FastAPI:
    config = ProxyConfig(openrouter_api_key=openrouter_api_key)
    return AnthropicBridge(config).app
