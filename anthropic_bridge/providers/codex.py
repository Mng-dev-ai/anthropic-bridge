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

        proc = await asyncio.create_subprocess_exec(
            "codex", "app-server",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            # Initialize handshake
            init_id = self._next_request_id()
            await self._send_request(proc, init_id, "initialize", {
                "clientInfo": {"name": "anthropic_bridge", "version": "1.0.0"}
            })
            await self._read_response(proc, init_id)
            await self._send_notification(proc, "initialized", {})

            # Start thread
            thread_id_req = self._next_request_id()
            thread_params: dict[str, Any] = {
                "approvalPolicy": "never",
                "sandbox": "danger-full-access",
            }
            if self.target_model and self.target_model.lower() != "default":
                thread_params["model"] = self.target_model
            if self.reasoning_level:
                thread_params["effort"] = self.reasoning_level

            await self._send_request(proc, thread_id_req, "thread/start", thread_params)
            thread_resp = await self._read_response(proc, thread_id_req)
            thread_id = thread_resp.get("result", {}).get("thread", {}).get("id")

            if not thread_id:
                raise RuntimeError(f"Failed to start thread: {thread_resp}")

            # Read thread/started notification
            await self._read_until_method(proc, "thread/started")

            # Start turn
            turn_id_req = self._next_request_id()
            await self._send_request(proc, turn_id_req, "turn/start", {
                "threadId": thread_id,
                "input": [{"type": "text", "text": prompt}]
            })

            # Stream events until turn/completed
            async for msg in self._read_notifications(proc):
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
        proc: asyncio.subprocess.Process,
        req_id: int,
        method: str,
        params: dict[str, Any],
    ) -> None:
        msg = {"method": method, "id": req_id, "params": params}
        line = json.dumps(msg) + "\n"
        if proc.stdin:
            proc.stdin.write(line.encode())
            await proc.stdin.drain()

    async def _send_notification(
        self,
        proc: asyncio.subprocess.Process,
        method: str,
        params: dict[str, Any],
    ) -> None:
        msg = {"method": method, "params": params}
        line = json.dumps(msg) + "\n"
        if proc.stdin:
            proc.stdin.write(line.encode())
            await proc.stdin.drain()

    async def _read_response(
        self,
        proc: asyncio.subprocess.Process,
        expected_id: int,
    ) -> dict[str, Any]:
        if not proc.stdout:
            return {}
        while True:
            line = await proc.stdout.readline()
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
        proc: asyncio.subprocess.Process,
        target_method: str,
    ) -> dict[str, Any]:
        if not proc.stdout:
            return {}
        while True:
            line = await proc.stdout.readline()
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
        proc: asyncio.subprocess.Process,
    ) -> AsyncIterator[dict[str, Any]]:
        if not proc.stdout:
            return
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=300)
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
