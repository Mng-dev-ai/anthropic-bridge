# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

# Run deterministic tests
pytest tests/ -v

# Run live upstream tests
ANTHROPIC_BRIDGE_LIVE_TESTS=1 OPENROUTER_API_KEY=your_key pytest tests/test_integration.py -v
```

## Architecture

**Request Flow**: Anthropic API request → `server.py` → provider client → upstream API → Anthropic JSON or SSE response

**Core Components**:
- `server.py` - FastAPI app exposing `/v1/messages` and `/v1/messages/count_tokens`, with strict provider-prefix routing and non-stream aggregation
- `transform.py` - Converts Anthropic messages/tools/tool_choice into upstream request shapes
- `protocol.py` - Parses Anthropic-style SSE, aggregates non-stream responses, and estimates approximate token counts

**Provider System** (`providers/`):
- `openrouter/client.py` - OpenRouter chat-completions streaming adapter
- `openai/client.py` - OpenAI Responses API adapter via Codex auth
- `copilot/client.py` - GitHub Copilot adapter for Responses API and chat-completions style models
- `openrouter/registry.py` - Selects provider-specific request/response tweaks based on model ID
- Provider implementations (Grok, Gemini, OpenAI, etc.) handle model-specific quirks like XML tool call parsing or reasoning options

**Caching** (`cache.py`):
- `ReasoningCache` persists Gemini reasoning details between tool call rounds to `~/.anthropic_bridge/cache/`
