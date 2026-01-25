import asyncio
import json
import time
from pathlib import Path
from typing import Any

import aiofiles

DEFAULT_CACHE_DIR = Path.home() / ".anthropic_bridge" / "cache"
DEFAULT_TTL_DAYS = 30


class ReasoningCache:
    def __init__(self, cache_dir: Path | None = None, ttl_days: int = DEFAULT_TTL_DAYS):
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_file = self._cache_dir / "reasoning_details.json"
        self._ttl_seconds = ttl_days * 24 * 60 * 60
        self._lock = asyncio.Lock()
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._lock:
            if self._loaded:  # Double-checked locking for async safety
                return  # type: ignore[unreachable]
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            if self._cache_file.exists():
                try:
                    async with aiofiles.open(self._cache_file, "r") as f:
                        content = await f.read()
                    data = json.loads(content)
                    self._memory_cache = data if isinstance(data, dict) else {}
                except (json.JSONDecodeError, OSError):
                    self._memory_cache = {}
            self._loaded = True

    async def _save(self) -> None:
        try:
            async with aiofiles.open(self._cache_file, "w") as f:
                await f.write(json.dumps(self._memory_cache, indent=2))
        except OSError:
            pass

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            k
            for k, v in self._memory_cache.items()
            if now - v.get("timestamp", 0) > self._ttl_seconds
        ]
        for k in expired:
            del self._memory_cache[k]

    async def get(self, tool_call_id: str) -> list[dict[str, Any]] | None:
        await self._ensure_loaded()
        entry = self._memory_cache.get(tool_call_id)
        if not entry:
            return None
        if time.time() - entry.get("timestamp", 0) > self._ttl_seconds:
            async with self._lock:
                self._memory_cache.pop(tool_call_id, None)
                await self._save()
            return None
        return entry.get("data")

    async def set(self, tool_call_id: str, reasoning_details: list[dict[str, Any]]) -> None:
        await self._ensure_loaded()
        async with self._lock:
            self._memory_cache[tool_call_id] = {
                "timestamp": time.time(),
                "data": reasoning_details,
            }
            self._cleanup_expired()
            await self._save()

    async def clear(self) -> None:
        async with self._lock:
            self._memory_cache = {}
            if self._cache_file.exists():
                self._cache_file.unlink()


# global instance
_cache: ReasoningCache | None = None


def get_reasoning_cache() -> ReasoningCache:
    global _cache
    if _cache is None:
        _cache = ReasoningCache()
    return _cache
