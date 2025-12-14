# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

anthropic-bridge is a proxy server that translates Anthropic Messages API requests into OpenRouter API format, enabling use of various LLM providers (Gemini, OpenAI, Grok, DeepSeek, Qwen, MiniMax) through an Anthropic-compatible interface.

## Commands

```bash
# Install dependencies
pip install -e ".[test,dev]"

# Run server (requires OPENROUTER_API_KEY env var)
OPENROUTER_API_KEY=your_key anthropic-bridge --port 8080 --host 127.0.0.1

# Lint
ruff check anthropic_bridge/ tests/

# Type check
mypy anthropic_bridge/

# Run tests (requires OPENROUTER_API_KEY env var)
OPENROUTER_API_KEY=your_key pytest tests/ -v
```

## Architecture

**Request Flow**: Anthropic API request → `server.py` → `client.py` → OpenRouter API → SSE stream converted back to Anthropic format

**Core Components**:
- `server.py` - FastAPI app exposing `/v1/messages` endpoint that accepts Anthropic API format
- `client.py` - `OpenRouterClient` handles request transformation and streams OpenRouter responses back as Anthropic SSE events
- `transform.py` - Converts Anthropic messages/tools/tool_choice to OpenAI format for OpenRouter

**Provider System** (`providers/`):
- `BaseProvider` - Abstract base defining `process_text_content()`, `should_handle()`, and `prepare_request()` hooks
- `ProviderRegistry` - Selects appropriate provider based on model ID
- Provider implementations (Grok, Gemini, OpenAI, etc.) handle model-specific quirks like XML tool call parsing (Grok) or reasoning detail injection (Gemini)

**Caching** (`cache.py`):
- `ReasoningCache` persists Gemini reasoning details between tool call rounds to `~/.anthropic_bridge/cache/`
