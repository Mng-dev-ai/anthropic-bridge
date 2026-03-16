# anthropic-bridge

A proxy server that exposes an Anthropic Messages API-compatible endpoint while routing requests to various LLM providers.

## Features

- Anthropic Messages API compatible (`/v1/messages`)
- JSON responses by default, SSE when `stream: true`
- Tool/function calling support
- Multi-round conversations
- Support for multiple providers: OpenAI, GitHub Copilot, OpenRouter (Gemini, Grok, DeepSeek, Qwen, MiniMax, etc.)
- Extended thinking/reasoning support for compatible models
- Reasoning cache for Gemini models across tool call rounds
- Approximate `/v1/messages/count_tokens` support

## Installation

```bash
pip install anthropic-bridge
```

For development:

```bash
git clone https://github.com/michaelgendy/anthropic-bridge.git
cd anthropic-bridge
pip install -e ".[test,dev]"
```

## Usage

Start the bridge server:

```bash
anthropic-bridge --port 8080
```

All providers are configured via environment variables. The server is designed to run inside managed environments (e.g. claudex sandboxes) where tokens are injected automatically.

### Provider Examples

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="not-used",
    base_url="http://localhost:8080"
)

# OpenAI (via ChatGPT subscription)
response = client.messages.create(
    model="openai/gpt-5.2",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# GitHub Copilot
response = client.messages.create(
    model="copilot/gpt-5.3-codex",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)

# OpenRouter
response = client.messages.create(
    model="openrouter/google/gemini-3-pro-preview",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

`client.messages.create(...)` works as a normal non-streaming Anthropic request. If you send `stream=True`, the bridge returns Anthropic-compatible SSE events.

### Thinking/Reasoning

Use the `thinking` parameter to control reasoning effort (supported on OpenAI and compatible models):

```python
response = client.messages.create(
    model="openai/gpt-5.2",
    max_tokens=1024,
    thinking={"budget_tokens": 15000},  # Maps to "high" effort
    messages=[{"role": "user", "content": "Solve this problem..."}]
)
```

| Budget Tokens | Reasoning Effort |
|---------------|------------------|
| 1 - 9,999 | low |
| 10,000 - 14,999 | medium |
| 15,000 - 31,999 | high |
| 32,000+ | xhigh |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health check |
| `/v1/messages` | POST | Anthropic Messages API. Returns JSON unless `stream: true` is set. |
| `/v1/messages/count_tokens` | POST | Approximate token counting for structured Anthropic inputs. |

## Configuration

All providers are configured exclusively through environment variables:

| Environment Variable | Required | Description |
|---------------------|----------|-------------|
| `OPENROUTER_API_KEY` | No | OpenRouter API key (required for `openrouter/*` models) |
| `GITHUB_COPILOT_TOKEN` | No | GitHub Copilot OAuth token (required for `copilot/*` models) |

OpenAI models (`openai/*`) authenticate via the Codex CLI auth file (`~/.codex/auth.json`), which is set up externally.

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--port` | 8080 | Port to run on |
| `--host` | 127.0.0.1 | Host to bind to |

### Model Routing

- `openai/*` → Direct OpenAI API (via Codex CLI auth)
- `copilot/*` → GitHub Copilot API (via `GITHUB_COPILOT_TOKEN`)
- `openrouter/*` → OpenRouter API (via `OPENROUTER_API_KEY`)
- Any other model → Falls back in this order: OpenRouter, Copilot, OpenAI

Explicit prefixes are strict. If you request `openai/*`, `copilot/*`, or `openrouter/*` and that backend is not configured, the bridge returns an authentication/configuration error instead of silently rerouting to another provider.

## Architecture

- `server.py` owns provider selection, Anthropic route handling, SSE passthrough, and non-stream aggregation.
- `providers/openrouter/client.py` handles OpenRouter chat-completions streaming and provider-specific request tweaks.
- `providers/openai/client.py` and `providers/copilot/client.py` translate Anthropic requests onto Responses API or chat-completions style upstream APIs.
- `transform.py` converts Anthropic messages, tools, and tool choice into the upstream shapes used by the providers.

## Provider Notes

This project does not keep a frozen catalog of upstream model IDs. Use the provider prefixes with the model IDs currently offered by that provider.

Provider-specific optimizations currently exist for:

- **Google Gemini** (`openrouter/google/*`) - Reasoning detail caching
- **OpenAI** (`openrouter/openai/*`) - Extended thinking support
- **xAI Grok** (`openrouter/x-ai/*`) - XML tool call parsing
- **DeepSeek** (`openrouter/deepseek/*`)
- **Qwen** (`openrouter/qwen/*`)
- **MiniMax** (`openrouter/minimax/*`)

## Development

`pytest tests/ -v` runs the deterministic test suite by default. Live upstream tests are opt-in and require `ANTHROPIC_BRIDGE_LIVE_TESTS=1`.

## License

MIT
