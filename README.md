# anthropic-bridge

A proxy server that exposes an Anthropic Messages API-compatible endpoint while routing requests to various LLM providers.

## Features

- Anthropic Messages API compatible (`/v1/messages`)
- Streaming SSE responses
- Tool/function calling support
- Multi-round conversations
- Support for multiple providers: OpenAI, Gemini, Grok, DeepSeek, Qwen, MiniMax
- Extended thinking/reasoning support for compatible models
- Reasoning cache for Gemini models across tool call rounds
- OpenAI integration via ChatGPT subscription (uses Codex CLI for auth)

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

### With OpenAI (ChatGPT Subscription)

First, authenticate with Codex CLI using your ChatGPT subscription:

```bash
codex login
```

Then start the bridge:

```bash
anthropic-bridge --port 8080
```

Use `openai/` prefixed models:

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="not-used",
    base_url="http://localhost:8080"
)

response = client.messages.create(
    model="openai/gpt-5.2-codex",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### OpenAI Models with Thinking

Use the `thinking` parameter to control reasoning effort:

```python
response = client.messages.create(
    model="openai/gpt-5.2-codex",
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

### With OpenRouter

Set your OpenRouter API key and start the server:

```bash
export OPENROUTER_API_KEY=your_key
anthropic-bridge --port 8080 --host 127.0.0.1
```

Use `openrouter/` prefixed models:

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="not-used",
    base_url="http://localhost:8080"
)

response = client.messages.create(
    model="openrouter/google/gemini-2.5-pro-preview",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health check |
| `/v1/messages` | POST | Anthropic Messages API |
| `/v1/messages/count_tokens` | POST | Token counting (approximate) |

## Configuration

| Environment Variable | Required | Description |
|---------------------|----------|-------------|
| `OPENROUTER_API_KEY` | No* | Your OpenRouter API key (*required for `openrouter/*` models) |

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--port` | 8080 | Port to run on |
| `--host` | 127.0.0.1 | Host to bind to |

### Model Routing

- `openai/*` models → Direct OpenAI API (via Codex CLI auth)
- `openrouter/*` models → OpenRouter API (requires `OPENROUTER_API_KEY`)

## Supported Models

### OpenAI (via ChatGPT subscription)

- `openai/gpt-5.2-codex` - GPT-5.2 Codex
- `openai/gpt-5.2` - GPT-5.2

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
