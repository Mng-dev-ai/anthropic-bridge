# anthropic-bridge

A proxy server that exposes an Anthropic Messages API-compatible endpoint while routing requests to various LLM providers through OpenRouter and Codex CLI.

## Features

- Anthropic Messages API compatible (`/v1/messages`)
- Streaming SSE responses
- Tool/function calling support
- Multi-round conversations
- Support for multiple providers: Gemini, OpenAI, Grok, DeepSeek, Qwen, MiniMax
- Extended thinking/reasoning support for compatible models
- Reasoning cache for Gemini models across tool call rounds
- **Codex CLI integration** - Use OpenAI's Codex models with your ChatGPT subscription

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

### With OpenRouter

Set your OpenRouter API key and start the server:

```bash
export OPENROUTER_API_KEY=your_key
anthropic-bridge --port 8080 --host 127.0.0.1
```

Then point your Anthropic SDK client to `http://localhost:8080`:

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="not-used",
    base_url="http://localhost:8080"
)

response = client.messages.create(
    model="google/gemini-2.5-pro-preview",  # Any OpenRouter model
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### With Codex CLI

First, authenticate with Codex CLI using your ChatGPT subscription:

```bash
codex login
```

Then start the bridge (no API key needed for Codex models):

```bash
anthropic-bridge --port 8080
```

Use `codex/` prefixed models:

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="not-used",
    base_url="http://localhost:8080"
)

response = client.messages.create(
    model="codex/gpt-5.2-codex",  # Codex model
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Codex Models with Reasoning Levels

Append reasoning level suffix to control reasoning effort:

| Model | Description |
|-------|-------------|
| `codex/gpt-5.2-codex` | Default reasoning |
| `codex/gpt-5.2-codex:low` | Low reasoning effort |
| `codex/gpt-5.2-codex:medium` | Medium reasoning effort |
| `codex/gpt-5.2-codex:high` | High reasoning effort |
| `codex/gpt-5.2-codex:xhigh` | Extra high reasoning effort |
| `codex/gpt-5.2` | GPT-5.2 base model |

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
| `OPENROUTER_API_KEY` | No* | Your OpenRouter API key (*required for non-Codex models) |

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--port` | 8080 | Port to run on |
| `--host` | 127.0.0.1 | Host to bind to |

### Model Routing

- Models prefixed with `codex/` are routed to Codex CLI
- All other models are routed to OpenRouter (requires `OPENROUTER_API_KEY`)

## Supported Models

### Codex CLI (via ChatGPT subscription)

- **Codex** (`codex/*`) - GPT-5.2, GPT-5.2-Codex with reasoning levels

### OpenRouter

Any model available on OpenRouter can be used. Provider-specific optimizations exist for:

- **Google Gemini** (`google/*`) - Reasoning detail caching
- **OpenAI** (`openai/*`) - Extended thinking support
- **xAI Grok** (`x-ai/*`) - XML tool call parsing
- **DeepSeek** (`deepseek/*`)
- **Qwen** (`qwen/*`)
- **MiniMax** (`minimax/*`)

## License

MIT
