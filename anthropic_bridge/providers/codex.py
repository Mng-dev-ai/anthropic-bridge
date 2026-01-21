import asyncio
import json
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any


class CodexClient:
    def __init__(self, target_model: str):
        model_str = target_model.removeprefix("codex/")
        # Parse reasoning level suffix (e.g., "gpt-5.2-codex:high")
        if ":" in model_str:
            model, level = model_str.rsplit(":", 1)
            self.target_model = model
            self.reasoning_level: str | None = level
        else:
            self.target_model = model_str
            self.reasoning_level = None

    async def handle(self, payload: dict[str, Any]) -> AsyncIterator[str]:
        prompt = self._extract_prompt(payload)
        msg_id = f"msg_{int(time.time())}_{self._random_id()}"

        yield self._sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self.target_model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        yield self._sse("ping", {"type": "ping"})

        text_started = False
        text_idx = 0
        thinking_started = False
        thinking_idx = -1
        cur_idx = 0
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        tool_blocks: dict[str, int] = {}

        cmd = [
            "codex",
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-m",
            self.target_model,
        ]
        if self.reasoning_level:
            cmd.extend(["-c", f"reasoning_effort={self.reasoning_level}"])
        cmd.append(prompt)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            async for line in self._read_lines(proc.stdout):
                if not line.strip():
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "item.started":
                    item = event.get("item", {})
                    item_type = item.get("type", "")

                    if item_type == "reasoning" and not thinking_started:
                        thinking_idx = cur_idx
                        cur_idx += 1
                        yield self._sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": thinking_idx,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "",
                                },
                            },
                        )
                        thinking_started = True

                    elif item_type in ("command_execution", "file_change", "mcp_tool_call"):
                        item_id = item.get("id", "")
                        tool_idx = cur_idx
                        cur_idx += 1
                        tool_blocks[item_id] = tool_idx
                        tool_name = self._get_tool_name(item_type, item)
                        yield self._sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_idx,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": f"tool_{item_id or self._random_id()}",
                                    "name": tool_name,
                                    "input": {},
                                },
                            },
                        )

                elif event_type == "item.updated":
                    item = event.get("item", {})
                    item_type = item.get("type", "")
                    text = item.get("text", "")

                    if item_type == "reasoning" and thinking_started and text:
                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": thinking_idx,
                                "delta": {"type": "thinking_delta", "thinking": text},
                            },
                        )

                    elif item_type == "agent_message" and text:
                        if thinking_started:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_idx,
                                    "delta": {"type": "signature_delta", "signature": ""},
                                },
                            )
                            yield self._sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": thinking_idx},
                            )
                            thinking_started = False

                        if not text_started:
                            text_idx = cur_idx
                            cur_idx += 1
                            yield self._sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": text_idx,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            text_started = True

                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )

                elif event_type == "item.completed":
                    item = event.get("item", {})
                    item_type = item.get("type", "")
                    item_id = item.get("id", "")
                    text = item.get("text", "")

                    if item_type in ("command_execution", "file_change", "mcp_tool_call"):
                        tool_idx = tool_blocks.get(item_id, -1)
                        if tool_idx >= 0:
                            tool_input = self._build_tool_input(item_type, item)
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": tool_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": json.dumps(tool_input),
                                    },
                                },
                            )
                            yield self._sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": tool_idx},
                            )

                    elif item_type == "agent_message" and text:
                        if thinking_started:
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": thinking_idx,
                                    "delta": {"type": "signature_delta", "signature": ""},
                                },
                            )
                            yield self._sse(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": thinking_idx},
                            )
                            thinking_started = False

                        if not text_started:
                            text_idx = cur_idx
                            cur_idx += 1
                            yield self._sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": text_idx,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )
                            text_started = True

                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )

                elif event_type == "turn.completed":
                    turn_usage = event.get("usage", {})
                    usage["input_tokens"] = turn_usage.get("input_tokens", 0)
                    usage["output_tokens"] = turn_usage.get("output_tokens", 0)

                elif event_type == "error" or event_type == "turn.failed":
                    error_msg = event.get("message") or event.get("error", {}).get(
                        "message", "Unknown error"
                    )
                    if not text_started:
                        text_idx = cur_idx
                        cur_idx += 1
                        yield self._sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": text_idx,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                        text_started = True

                    yield self._sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": text_idx,
                            "delta": {
                                "type": "text_delta",
                                "text": f"Error: {error_msg}",
                            },
                        },
                    )

        finally:
            if proc.returncode is None:
                proc.terminate()
                await proc.wait()

        if thinking_started:
            yield self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": thinking_idx,
                    "delta": {"type": "signature_delta", "signature": ""},
                },
            )
            yield self._sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": thinking_idx},
            )

        if text_started:
            yield self._sse(
                "content_block_stop", {"type": "content_block_stop", "index": text_idx}
            )

        yield self._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": usage,
            },
        )
        yield self._sse("message_stop", {"type": "message_stop"})
        yield "data: [DONE]\n\n"

    def _extract_prompt(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages", [])
        system = payload.get("system", "")

        if isinstance(system, list):
            system = "\n\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system
            )

        parts: list[str] = []
        if system:
            parts.append(f"System: {system}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = "\n".join(text_parts)

            if content:
                parts.append(f"{role.capitalize()}: {content}")

        return "\n\n".join(parts)

    async def _read_lines(
        self, stream: asyncio.StreamReader | None
    ) -> AsyncIterator[str]:
        if stream is None:
            return

        while True:
            line = await stream.readline()
            if not line:
                break
            yield line.decode("utf-8", errors="replace")

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))

    def _get_tool_name(self, item_type: str, item: dict[str, Any]) -> str:
        if item_type == "command_execution":
            return "bash"
        elif item_type == "file_change":
            return "file_editor"
        elif item_type == "mcp_tool_call":
            name = item.get("name")
            return str(name) if name else "mcp_tool"
        return "unknown_tool"

    def _build_tool_input(self, item_type: str, item: dict[str, Any]) -> dict[str, Any]:
        if item_type == "command_execution":
            return {
                "command": item.get("command", ""),
                "output": item.get("aggregated_output", ""),
                "exit_code": item.get("exit_code", 0),
            }
        elif item_type == "file_change":
            return {
                "path": item.get("path", ""),
                "action": item.get("action", "modify"),
            }
        elif item_type == "mcp_tool_call":
            args = item.get("arguments")
            return dict(args) if isinstance(args, dict) else {}
        return {}
