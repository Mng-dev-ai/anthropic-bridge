# anthropic-bridge

A proxy server that exposes an Anthropic Messages API-compatible endpoint while routing requests to various LLM providers.

## Features

- Anthropic Messages API compatible (`/v1/messages`)
- Streaming SSE responses
- Tool/function calling support
- Multi-round conversations
- Support for multiple providers: OpenAI, GitHub Copilot, OpenRouter (Gemini, Grok, DeepSeek, Qwen, MiniMax, etc.)
- Extended thinking/reasoning support for compatible models
- Reasoning cache for Gemini models across tool call rounds

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
    model="openai/gpt-5.3-codex",
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
    model="openrouter/google/gemini-2.5-pro-preview",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Thinking/Reasoning

Use the `thinking` parameter to control reasoning effort (supported on OpenAI and compatible models):

```python
response = client.messages.create(
    model="openai/gpt-5.3-codex",
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
| `/v1/messages` | POST | Anthropic Messages API |
| `/v1/messages/count_tokens` | POST | Token counting (approximate) |

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
- Any other model → Falls back to OpenRouter

## Supported Models

### OpenAI (via ChatGPT subscription)

- `openai/gpt-5.3-codex` - Codex 5.3
- `openai/gpt-5.2-codex` - Codex 5.2

### GitHub Copilot (via GitHub Copilot subscription)

- `copilot/gpt-5.3-codex` - Codex 5.3
- `copilot/claude-opus-4.6` - Claude Opus 4.6
- `copilot/gemini-3-pro` - Gemini 3 Pro

### OpenRouter

Any model available on OpenRouter can be used with the `openrouter/` prefix. Provider-specific optimizations exist for:

- **Google Gemini** (`openrouter/google/*`) - Reasoning detail caching
- **OpenAI** (`openrouter/openai/*`) - Extended thinking support
- **xAI Grok** (`openrouter/x-ai/*`) - XML tool call parsing
- **DeepSeek** (`openrouter/deepseek/*`)
- **Qwen** (`openrouter/qwen/*`)
- **MiniMax** (`openrouter/minimax/*`)

## License

MIT
