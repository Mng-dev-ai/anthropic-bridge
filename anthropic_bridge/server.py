import json
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .cache import get_reasoning_cache
from .client import OpenRouterClient


@dataclass
class ProxyConfig:
    openrouter_api_key: str


class AnthropicBridge:
    def __init__(self, config: ProxyConfig):
        self.config = config
        self.app = FastAPI(title="Anthropic Bridge")
        self._clients: dict[str, OpenRouterClient] = {}
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

        @self.app.post("/v1/messages")
        async def messages(request: Request) -> StreamingResponse:
            body = await request.json()
            model = body.get("model", "")
            client = self._get_client(model)

            return StreamingResponse(
                client.handle(body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

    def _get_client(self, model: str) -> OpenRouterClient:
        if model not in self._clients:
            self._clients[model] = OpenRouterClient(
                model, self.config.openrouter_api_key
            )
        return self._clients[model]


def create_app(openrouter_api_key: str) -> FastAPI:
    config = ProxyConfig(openrouter_api_key=openrouter_api_key)
    return AnthropicBridge(config).app
