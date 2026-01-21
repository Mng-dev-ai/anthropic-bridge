import json
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import get_reasoning_cache
from .client import OpenRouterClient
from .providers.codex import CodexClient


@dataclass
class ProxyConfig:
    openrouter_api_key: str | None = None


class AnthropicBridge:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = FastAPI(title="Anthropic Bridge")
        self._clients: dict[str, OpenRouterClient] = {}
        self._codex_clients: dict[str, CodexClient] = {}
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

            if self._should_use_codex(model):
                handler = self._get_codex_client(model).handle(body)
            elif not self.config.openrouter_api_key:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "authentication_error",
                            "message": f"OPENROUTER_API_KEY required for model '{model}'. "
                            "Use codex/* models or set the API key.",
                        }
                    },
                )
            else:
                handler = self._get_client(model).handle(body)

            return StreamingResponse(
                handler,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _should_use_codex(self, model: str) -> bool:
        return model.startswith("codex/")

    def _get_codex_client(self, model: str) -> CodexClient:
        if model not in self._codex_clients:
            self._codex_clients[model] = CodexClient(model)
        return self._codex_clients[model]

    def _get_client(self, model: str) -> OpenRouterClient:
        if model not in self._clients:
            self._clients[model] = OpenRouterClient(
                model, self.config.openrouter_api_key or ""
            )
        return self._clients[model]


def create_app(openrouter_api_key: str | None = None) -> FastAPI:
    config = ProxyConfig(openrouter_api_key=openrouter_api_key)
    return AnthropicBridge(config).app
