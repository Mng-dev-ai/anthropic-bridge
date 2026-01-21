import asyncio
import contextlib
import json
import logging
import os
import pty
import random
import string
import time
from collections.abc import AsyncIterator
from typing import Any, Callable

logger = logging.getLogger(__name__)

class PtyLineReader:
    def __init__(self, queue: "asyncio.Queue[bytes]") -> None:
        self._queue = queue
        self._buffer = b""

    async def readline(self) -> bytes:
        while True:
            newline = self._buffer.find(b"\n")
            if newline != -1:
                line = self._buffer[: newline + 1]
                self._buffer = self._buffer[newline + 1 :]
                return line

            chunk = await self._queue.get()
            if not chunk:
                if self._buffer:
                    line = self._buffer
                    self._buffer = b""
                    return line
                return b""
            self._buffer += chunk


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
        self._request_id = 0

    def _next_tool_id(self, prefix: str) -> str:
        self._tool_counter += 1
        return f"{prefix}_{self._tool_counter}_{self._random_id()}"

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

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

        master_fd, slave_fd = pty.openpty()
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "app-server",
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)

        output_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)

        async def read_master() -> None:
            loop = asyncio.get_running_loop()
            try:
                while True:
                    data = await loop.run_in_executor(None, os.read, master_fd, 4096)
                    if not data:
                        break
                    await output_queue.put(data)
            finally:
                await output_queue.put(b"")

        reader_task = asyncio.create_task(read_master())
        line_reader = PtyLineReader(output_queue)
        writer: Callable[[bytes], None] = lambda payload: os.write(master_fd, payload)

        try:
            init_id = self._next_request_id()
            await self._send_request(writer, init_id, "initialize", {
                "clientInfo": {"name": "anthropic_bridge", "version": "1.0.0"}
            })
            await self._read_response(line_reader, init_id)
            await self._send_notification(writer, "initialized", {})

            thread_id_req = self._next_request_id()
            thread_params: dict[str, Any] = {
                "approvalPolicy": "never",
                "sandbox": "danger-full-access",
            }
            if self.target_model and self.target_model.lower() != "default":
                thread_params["model"] = self.target_model
            if self.reasoning_level:
                thread_params["effort"] = self.reasoning_level

            await self._send_request(writer, thread_id_req, "thread/start", thread_params)
            thread_resp = await self._read_response(line_reader, thread_id_req)
            thread_id = thread_resp.get("result", {}).get("thread", {}).get("id")

            if not thread_id:
                raise RuntimeError(f"Failed to start thread: {thread_resp}")

            await self._read_until_method(line_reader, "thread/started")

            turn_id_req = self._next_request_id()
            await self._send_request(writer, turn_id_req, "turn/start", {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}]
            })

            async for msg in self._read_notifications(line_reader):
                method = msg.get("method", "")
                params = msg.get("params", {})

                if method == "item/agentMessage/delta":
                    delta = params.get("delta", "")
                    if delta:
                        if not text_started:
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
                                "delta": {"type": "text_delta", "text": delta},
                            },
                        )

                elif method == "item/reasoning/summaryTextDelta":
                    delta = params.get("delta", "")
                    if delta:
                        if not thinking_started:
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

                        yield self._sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": thinking_idx,
                                "delta": {"type": "thinking_delta", "thinking": delta},
                            },
                        )

                elif method == "item/started":
                    item = params.get("item", {})
                    item_type = item.get("type", "")
                    item_id = item.get("id", "")

                    if item_type == "commandExecution":
                        command = item.get("command", "")
                        tool_id = self._next_tool_id("codex_cmd")
                        self._active_tools[item_id] = tool_id
                        if not text_started:
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
                        yield self._sse("ping", {"type": "ping"})

                    elif item_type == "fileChange":
                        changes = item.get("changes", [])
                        for change in changes:
                            if not isinstance(change, dict):
                                continue
                            path = change.get("path", "")
                            kind = change.get("kind", "update")
                            tool_id = self._next_tool_id("codex_file")
                            self._active_tools[f"{item_id}:{path}"] = tool_id
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
                            marker = self._tool_marker(
                                "START",
                                {"id": tool_id, "name": tool_name, "input": {"file_path": path}},
                            )
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": marker},
                                },
                            )
                            yield self._sse("ping", {"type": "ping"})

                    elif item_type == "mcpToolCall":
                        tool_name = item.get("tool", "mcp_tool")
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
                        yield self._sse("ping", {"type": "ping"})

                    elif item_type == "webSearch":
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
                        yield self._sse("ping", {"type": "ping"})

                elif method == "item/completed":
                    item = params.get("item", {})
                    item_type = item.get("type", "")
                    item_id = item.get("id", "")

                    if item_type == "commandExecution":
                        if item_id in self._active_tools:
                            tool_id = self._active_tools.pop(item_id)
                            output = item.get("aggregatedOutput", "")
                            exit_code = item.get("exitCode", 0)
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
                            yield self._sse("ping", {"type": "ping"})

                    elif item_type == "fileChange":
                        changes = item.get("changes", [])
                        for change in changes:
                            if not isinstance(change, dict):
                                continue
                            path = change.get("path", "")
                            key = f"{item_id}:{path}"
                            if key in self._active_tools:
                                tool_id = self._active_tools.pop(key)
                                marker = self._tool_marker("RESULT", {"id": tool_id})
                                yield self._sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": text_idx,
                                        "delta": {"type": "text_delta", "text": marker},
                                    },
                                )
                                yield self._sse("ping", {"type": "ping"})

                    elif item_type == "mcpToolCall":
                        if item_id in self._active_tools:
                            tool_id = self._active_tools.pop(item_id)
                            result = item.get("result", {})
                            output = result.get("content", "") if result else ""
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
                            yield self._sse("ping", {"type": "ping"})

                    elif item_type == "webSearch":
                        if item_id in self._active_tools:
                            tool_id = self._active_tools.pop(item_id)
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
                                {"id": tool_id, "output": "search completed"},
                            )
                            yield self._sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": text_idx,
                                    "delta": {"type": "text_delta", "text": marker},
                                },
                            )
                            yield self._sse("ping", {"type": "ping"})

                elif method == "turn/completed":
                    break

                elif method == "thread/tokenUsage/updated":
                    token_usage = params.get("tokenUsage", {})
                    usage["input_tokens"] = token_usage.get("inputTokens", 0)
                    usage["output_tokens"] = token_usage.get("outputTokens", 0)

                elif method == "error":
                    error_msg = params.get("error", {}).get("message", "Unknown error")
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
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
            with contextlib.suppress(OSError):
                os.close(master_fd)
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

    async def _send_request(
        self,
        write_fn: Callable[[bytes], None],
        req_id: int,
        method: str,
        params: dict[str, Any],
    ) -> None:
        msg = {"method": method, "id": req_id, "params": params}
        line = json.dumps(msg) + "\n"
        await asyncio.to_thread(write_fn, line.encode())

    async def _send_notification(
        self,
        write_fn: Callable[[bytes], None],
        method: str,
        params: dict[str, Any],
    ) -> None:
        msg = {"method": method, "params": params}
        line = json.dumps(msg) + "\n"
        await asyncio.to_thread(write_fn, line.encode())

    async def _read_response(
        self,
        reader: PtyLineReader,
        expected_id: int,
    ) -> dict[str, Any]:
        while True:
            line = await reader.readline()
            if not line:
                return {}
            try:
                msg: dict[str, Any] = json.loads(line.decode())
                if msg.get("id") == expected_id:
                    return msg
            except json.JSONDecodeError:
                continue

    async def _read_until_method(
        self,
        reader: PtyLineReader,
        target_method: str,
    ) -> dict[str, Any]:
        while True:
            line = await reader.readline()
            if not line:
                return {}
            try:
                msg: dict[str, Any] = json.loads(line.decode())
                if msg.get("method") == target_method:
                    return msg
            except json.JSONDecodeError:
                continue

    async def _read_notifications(
        self,
        reader: PtyLineReader,
    ) -> AsyncIterator[dict[str, Any]]:
        while True:
            try:
                line = await asyncio.wait_for(reader.readline(), timeout=300)
                if not line:
                    break
                msg = json.loads(line.decode())
                yield msg
                if msg.get("method") == "turn/completed":
                    break
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for Codex response")
                break
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse Codex JSON: %s", e)
                continue

    def _extract_prompt(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages", [])
        parts: list[str] = []

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

    def _sse(self, event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _random_id(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
