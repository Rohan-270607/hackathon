"""Local models through Ollama's native /api/chat endpoint (stdlib HTTP, no extra dependency).

Shapes verified against the Ollama API docs (docs/api.md) and a live Ollama 0.40.4 server:
- tools: [{"type": "function", "function": {"name", "description", "parameters"}}]
- response message.tool_calls: [{"id": "call_...", "function": {"index", "name", "arguments": {...}}}]
  (`arguments` is an object; `id` is returned by 0.40.4 but not documented, so it is optional here)
- tool results go back as {"role": "tool", "content": str, "tool_name": str}
- usage: prompt_eval_count / eval_count; errors: HTTP status + {"error": "..."}

Not verified: the `done_reason` value for output truncated by num_predict. "length" is
treated as truncation; any other unknown value falls back to tool_use/end_turn.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any

from .base import (
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_TOOL_USE,
    AssistantTurn,
    LLMError,
    LLMResponse,
    ToolCall,
    ToolResultsTurn,
    ToolSpec,
    Transcript,
    Usage,
    UserTurn,
)

# Rough chars-per-token for the pre-flight context check (conservative for English + JSON).
_CHARS_PER_TOKEN = 3.0


def normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def to_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
        for t in tools
    ]


def to_messages(system: str, transcript: Transcript) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    call_names: dict[str, str] = {}
    for turn in transcript.turns:
        if isinstance(turn, UserTurn):
            messages.append({"role": "user", "content": turn.text})
        elif isinstance(turn, AssistantTurn):
            messages.append(turn.response.raw)
            for c in turn.response.tool_calls:
                call_names[c.id] = c.name
        elif isinstance(turn, ToolResultsTurn):
            for r in turn.results:
                content = f"ERROR: {r.content}" if r.is_error else r.content
                messages.append({"role": "tool", "content": content, "tool_name": call_names.get(r.call_id, "unknown")})
            if turn.note:
                messages.append({"role": "user", "content": turn.note})
        else:  # pragma: no cover - exhaustive
            raise TypeError(f"unknown turn type {type(turn)!r}")
    return messages


def parse_response(body: dict[str, Any]) -> LLMResponse:
    message = body.get("message") or {}
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls") or []
    for raw in raw_calls:
        fn = raw.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):  # tolerate a JSON string, but never guess at malformed input
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_unparseable_arguments": args}
        if not raw.get("id"):
            raw["id"] = f"ollama_{uuid.uuid4().hex[:10]}"  # keep the echoed history consistent with our ids
        calls.append(ToolCall(id=raw["id"], name=fn.get("name", ""), arguments=args if isinstance(args, dict) else {}))

    reason = body.get("done_reason")
    if reason == "length":
        stop = STOP_MAX_TOKENS
    elif calls:
        stop = STOP_TOOL_USE
    else:
        stop = STOP_END

    raw_message = {"role": "assistant", "content": message.get("content", "")}
    if raw_calls:
        raw_message["tool_calls"] = raw_calls
    return LLMResponse(
        text=(message.get("content") or "").strip(),
        tool_calls=calls,
        stop=stop,
        usage=Usage(input_tokens=int(body.get("prompt_eval_count") or 0), output_tokens=int(body.get("eval_count") or 0)),
        model=body.get("model", ""),
        raw=raw_message,
        reasoning_summary=message.get("thinking") or "",
    )


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        num_ctx: int = 16384,
        max_tokens: int = 4096,
        timeout_s: float = 600,
        temperature: float | None = None,
    ):
        self.model = model
        self.host = normalize_host(host)
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.temperature = temperature

    def build_request(self, system: str, transcript: Transcript, tools: list[ToolSpec]) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": self.num_ctx, "num_predict": self.max_tokens}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        req: dict[str, Any] = {
            "model": self.model,
            "messages": to_messages(system, transcript),
            "stream": False,
            "options": options,
        }
        if tools:
            req["tools"] = to_tools(tools)
        return req

    def complete(self, system: str, transcript: Transcript, tools: list[ToolSpec]) -> LLMResponse:
        req = self.build_request(system, transcript, tools)
        payload = json.dumps(req).encode("utf-8")
        # Ollama silently drops the oldest context when the prompt exceeds num_ctx (which can
        # drop the system prompt), so refuse instead of degrading quietly.
        estimate = len(payload) / _CHARS_PER_TOKEN
        if estimate + self.max_tokens > self.num_ctx:
            raise LLMError(
                f"conversation (~{int(estimate)} tokens) plus max output ({self.max_tokens}) may exceed "
                f"num_ctx={self.num_ctx}; raise OLLAMA_NUM_CTX or shorten the task"
            )
        http_req = urllib.request.Request(
            f"{self.host}/api/chat", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            hint = " (pull it with `ollama pull <model>`, see `ollama list`)" if e.code == 404 else ""
            raise LLMError(f"Ollama error {e.code}: {detail}{hint}", status=e.code, retryable=e.code >= 500) from e
        except urllib.error.URLError as e:
            raise LLMError(f"cannot reach Ollama at {self.host}: {e.reason} (is `ollama serve` running?)", retryable=True) from e
        except TimeoutError as e:
            raise LLMError(f"Ollama did not answer within {self.timeout_s}s", retryable=True) from e
        if "error" in body:
            raise LLMError(f"Ollama error: {body['error']}")
        return parse_response(body)
