"""Claude adapter (official `anthropic` SDK, Messages API).

Request shape choices, all verified against anthropic==1.14.0 and the bundled API docs:
- `client.beta.messages.create(...)` so the server-side refusal fallback can be enabled
  (`fallbacks="default"` with beta `server-side-fallback-2026-07-01`).
- `output_config={"effort": ...}` set explicitly (Opus 5.5 defaults to "medium").
- No `thinking` parameter: current models run adaptive thinking by default, and omitting it
  keeps the request valid if the model is swapped for one that does not support it.
- No forced `tool_choice` (400 on Opus 5.5); tool inputs are validated client-side instead.
- Assistant content is echoed back exactly as received (thinking blocks must be unchanged),
  and the transcript is append-only, so the system prompt and tool list never change mid-run.
"""

from __future__ import annotations

from typing import Any

import anthropic

from .base import (
    STOP_END,
    STOP_MAX_TOKENS,
    STOP_OTHER,
    STOP_PAUSE,
    STOP_REFUSAL,
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

FALLBACK_BETA = "server-side-fallback-2026-07-01"

_STOP_MAP = {
    "end_turn": STOP_END,
    "stop_sequence": STOP_END,
    "tool_use": STOP_TOOL_USE,
    "max_tokens": STOP_MAX_TOKENS,
    "refusal": STOP_REFUSAL,
    "pause_turn": STOP_PAUSE,
}

# Block types that must not be echoed if they precede a mid-output fallback boundary.
_PRE_FALLBACK_DROP = {"thinking", "redacted_thinking", "tool_use"}


def to_messages(transcript: Transcript) -> list[dict[str, Any]]:
    """Convert the provider-agnostic transcript to Messages API `messages`."""
    messages: list[dict[str, Any]] = []
    for turn in transcript.turns:
        if isinstance(turn, UserTurn):
            messages.append({"role": "user", "content": turn.text})
        elif isinstance(turn, AssistantTurn):
            messages.append({"role": "assistant", "content": turn.response.raw})
        elif isinstance(turn, ToolResultsTurn):
            content: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": r.call_id,
                    "content": r.content,
                    "is_error": r.is_error,
                }
                for r in turn.results
            ]
            if turn.note:
                content.append({"type": "text", "text": turn.note})
            messages.append({"role": "user", "content": content})
        else:  # pragma: no cover - exhaustive
            raise TypeError(f"unknown turn type {type(turn)!r}")
    return messages


def to_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _echo_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the documented echo rule for a mid-output fallback: drop thinking and tool_use
    blocks from the declined model that appear before the final `fallback` block."""
    last_fallback = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    if last_fallback < 0:
        return blocks
    kept = []
    for i, b in enumerate(blocks):
        if i < last_fallback and b.get("type") in _PRE_FALLBACK_DROP:
            continue
        kept.append(b)
    return kept


def parse_response(message: Any) -> LLMResponse:
    """Convert an SDK `BetaMessage` into an `LLMResponse`. Reads blocks by type, not position."""
    blocks: list[dict[str, Any]] = [b.to_dict(mode="json") for b in message.content]
    echo = _echo_blocks(blocks)
    last_fallback = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)

    texts: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for i, b in enumerate(blocks):
        btype = b.get("type")
        if btype == "text":
            texts.append(b.get("text", ""))
        elif btype == "thinking" and b.get("thinking"):
            reasoning.append(b["thinking"])
        elif btype == "tool_use" and i > last_fallback:
            args = b.get("input")
            calls.append(ToolCall(id=b["id"], name=b["name"], arguments=args if isinstance(args, dict) else {}))

    stop = _STOP_MAP.get(message.stop_reason or "", STOP_OTHER)
    refusal_detail = None
    if stop == STOP_REFUSAL and message.stop_details is not None:
        cat = message.stop_details.category or "uncategorized"
        expl = message.stop_details.explanation or ""
        refusal_detail = f"{cat}: {expl}".strip(": ")

    usage = Usage(
        input_tokens=getattr(message.usage, "input_tokens", 0) or 0,
        output_tokens=getattr(message.usage, "output_tokens", 0) or 0,
    )
    return LLMResponse(
        text="\n".join(t for t in texts if t),
        tool_calls=calls,
        stop=stop,
        usage=usage,
        model=message.model,
        raw=echo,
        refusal_detail=refusal_detail,
        reasoning_summary="\n".join(reasoning),
    )


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        max_tokens: int = 16000,
        effort: str | None = "high",
        fallbacks: str | None = "default",
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        # Zero-arg client: credentials come from the environment / `ant` profile, never code.
        self._client = client or anthropic.Anthropic()

    def build_request(self, system: str, transcript: Transcript, tools: list[ToolSpec]) -> dict[str, Any]:
        req: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": to_messages(transcript),
        }
        if tools:
            req["tools"] = to_tools(tools)
        if self.effort:
            req["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            req["fallbacks"] = self.fallbacks
            req["betas"] = [FALLBACK_BETA]
        return req

    def complete(self, system: str, transcript: Transcript, tools: list[ToolSpec]) -> LLMResponse:
        req = self.build_request(system, transcript, tools)
        try:
            message = self._client.beta.messages.create(**req)
        except anthropic.AuthenticationError as e:
            raise LLMError(f"authentication failed: {e.message}", status=401) from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"permission denied: {e.message}", status=403) from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"not found (check AGENT_MODEL): {e.message}", status=404) from e
        except anthropic.BadRequestError as e:
            raise LLMError(f"bad request: {e.message}", status=400) from e
        except anthropic.RateLimitError as e:
            raise LLMError(f"rate limited after SDK retries: {e.message}", retryable=True, status=429) from e
        except anthropic.APIStatusError as e:
            raise LLMError(
                f"API error {e.status_code}: {e.message}", retryable=e.status_code >= 500, status=e.status_code
            ) from e
        except anthropic.APITimeoutError as e:
            raise LLMError("request timed out after SDK retries", retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"connection error: {e}", retryable=True) from e
        except TypeError as e:
            # The SDK raises TypeError when it cannot find any credentials.
            if "authentication method" in str(e):
                raise LLMError(
                    "no Anthropic credentials found: set ANTHROPIC_API_KEY or run `ant auth login`"
                ) from e
            raise
        return parse_response(message)
