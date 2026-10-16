"""Provider-agnostic types for talking to an LLM.

The agent loop only sees these types. Each provider adapter converts the transcript into
its native request format and converts the native response back into `LLMResponse`.

The transcript is append-only: earlier turns are never edited. Assistant turns carry the
provider's native content in `raw` so the adapter can echo it back byte-for-byte (Claude
requires thinking blocks to be replayed unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Union


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


# Normalised stop reasons.
STOP_END = "end_turn"
STOP_TOOL_USE = "tool_use"
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
STOP_PAUSE = "pause"
STOP_OTHER = "other"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop: str
    usage: Usage
    model: str
    raw: Any = None  # provider-native assistant content, echoed back unchanged
    refusal_detail: str | None = None
    reasoning_summary: str = ""


@dataclass(frozen=True)
class UserTurn:
    text: str


@dataclass(frozen=True)
class AssistantTurn:
    response: LLMResponse


@dataclass(frozen=True)
class ToolResultsTurn:
    results: tuple[ToolResult, ...]
    note: str | None = None  # optional text appended after the results (e.g. a harness notice)


Turn = Union[UserTurn, AssistantTurn, ToolResultsTurn]


@dataclass
class Transcript:
    turns: list[Turn] = field(default_factory=list)

    def append(self, turn: Turn) -> None:
        self.turns.append(turn)


class LLMError(Exception):
    """Raised by adapters for provider failures, after the SDK's own retries are exhausted."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(self, system: str, transcript: Transcript, tools: list[ToolSpec]) -> LLMResponse:
        ...
