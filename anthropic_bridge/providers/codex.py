import asyncio
import json
import logging
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)


class CodexClient:
    def __init__(self, target_model: str):
        model_str = target_model.removeprefix("codex/")
        if ":" in model_str:
            model, level = model_str.rsplit(":", 1)
            self.target_model = model
            self.reasoning_level: str | None = level
        else:
            self.target_model = model_str
            self.reasoning_level = None
        self._tool_counter = 0
        self._active_tools: dict[str, str] = {}

    def _next_tool_id(self, prefix: str) -> str:
        self._tool_counter += 1
        return f"{prefix}_{self._tool_counter}_{self._random_id()}"

    def _tool_marker(self, marker_type: str, payload: dict[str, Any]) -> str:
        return f"<!--CODEX_TOOL_{marker_type}:{json.dumps(payload)}-->"

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
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse Codex JSON line: %s", e)
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

                    elif item_type == "command_execution":
                        command = item.get("command", "")
                        item_id = item.get("id", "")
                        tool_id = self._next_tool_id("codex_cmd")
                        self._active_tools[item_id] = tool_id
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
                        marker = self._tool_marker(
                            "START",
                            {"id": tool_id, "name": "CodexCommand", "input": {"command": command}},
                        )
                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": marker},
                            },
                        )

                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id", "")
                        tool_name = item.get("name", "mcp_tool")
                        args = item.get("arguments", {})
                        tool_id = self._next_tool_id("codex_mcp")
                        self._active_tools[item_id] = tool_id
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
                        if not isinstance(args, dict):
                            logger.warning("MCP tool args is not a dict: %s", type(args))
                        mcp_input = dict(args) if isinstance(args, dict) else {}
                        marker = self._tool_marker(
                            "START",
                            {"id": tool_id, "name": tool_name, "input": mcp_input},
                        )
                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": marker},
                            },
                        )

                    elif item_type == "web_search":
                        item_id = item.get("id", "")
                        query = item.get("query", "")
                        tool_id = self._next_tool_id("codex_search")
                        self._active_tools[item_id] = tool_id
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
                        marker = self._tool_marker(
                            "START",
                            {"id": tool_id, "name": "WebSearch", "input": {"query": query}},
                        )
                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": marker},
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
                    text = item.get("text", "")

                    if item_type == "command_execution":
                        item_id = item.get("id", "")
                        if item_id in self._active_tools:
                            tool_id = self._active_tools.pop(item_id)
                            output = item.get("aggregated_output", "")
                            exit_code = item.get("exit_code", 0)
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
                            marker = self._tool_marker(
                                "RESULT",
                                {"id": tool_id, "output": output, "exit_code": exit_code},
                            )
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": marker},
                                },
                            )

                    elif item_type == "file_change":
                        # file_change has changes array: [{"path": "...", "kind": "add|update|delete"}]
                        changes = item.get("changes", [])
                        for change in changes:
                            if not isinstance(change, dict):
                                continue
                            path = change.get("path", "")
                            kind = change.get("kind", "update")
                            tool_id = self._next_tool_id("codex_file")
                            # Map kind to existing tool: add → Write, update → Edit
                            tool_name = "Write" if kind == "add" else "Edit"
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
                            start_marker = self._tool_marker(
                                "START",
                                {"id": tool_id, "name": tool_name, "input": {"file_path": path}},
                            )
                            result_marker = self._tool_marker("RESULT", {"id": tool_id})
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": start_marker + result_marker},
                                },
                            )

                    elif item_type == "mcp_tool_call":
                        item_id = item.get("id", "")
                        if item_id in self._active_tools:
                            tool_id = self._active_tools.pop(item_id)
                            output = item.get("aggregated_output", "")
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
                            marker = self._tool_marker(
                                "RESULT",
                                {"id": tool_id, "output": output},
                            )
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": marker},
                                },
                            )

                    elif item_type == "web_search":
                        item_id = item.get("id", "")
                        if item_id in self._active_tools:
                            tool_id = self._active_tools.pop(item_id)
                            results = item.get("results", [])
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
                            marker = self._tool_marker(
                                "RESULT",
                                {"id": tool_id, "output": results},
                            )
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": marker},
                                },
                            )

                    elif item_type == "view_image":
                        path = item.get("path", "")
                        tool_id = self._next_tool_id("codex_img")
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
                        start_marker = self._tool_marker(
                            "START",
                            {"id": tool_id, "name": "Read", "input": {"file_path": path}},
                        )
                        result_marker = self._tool_marker("RESULT", {"id": tool_id})
                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": start_marker + result_marker},
                            },
                        )

                    elif item_type == "todo_list":
                        items = item.get("items", [])
                        tool_id = self._next_tool_id("codex_todo")
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
                        todo_items = list(items) if isinstance(items, list) else []
                        start_marker = self._tool_marker(
                            "START",
                            {"id": tool_id, "name": "TodoWrite", "input": {"todos": todo_items}},
                        )
                        result_marker = self._tool_marker("RESULT", {"id": tool_id})
                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": text_idx,
                                "delta": {"type": "text_delta", "text": start_marker + result_marker},
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
