# anthropic-bridge

A proxy server that exposes an Anthropic Messages API-compatible endpoint while routing requests to various LLM providers through OpenRouter.

## Features

- Anthropic Messages API compatible (`/v1/messages`)
- Streaming SSE responses
- Tool/function calling support
- Multi-round conversations
- Support for multiple providers: Gemini, OpenAI, Grok, DeepSeek, Qwen, MiniMax
- Extended thinking/reasoning support for compatible models
- Reasoning cache for Gemini models across tool call rounds

## Installation

```bash
pip install -e .

# With development dependencies
pip install -e ".[test,dev]"
```

## Usage

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
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key |

| CLI Flag | Default | Description |
|----------|---------|-------------|
| `--port` | 8080 | Port to run on |
| `--host` | 127.0.0.1 | Host to bind to |

## Supported Models

Any model available on OpenRouter can be used. Provider-specific optimizations exist for:

- **Google Gemini** (`google/*`) - Reasoning detail caching
- **OpenAI** (`openai/*`) - Extended thinking support
- **xAI Grok** (`x-ai/*`) - XML tool call parsing
- **DeepSeek** (`deepseek/*`)
- **Qwen** (`qwen/*`)
- **MiniMax** (`minimax/*`)

## License

MIT
