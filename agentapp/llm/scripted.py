"""Deterministic provider for tests and offline demos. It replays scripted responses.

Each script step is either an `LLMResponse`-like dict or a callable that receives the
transcript and returns one, so tests can make the "model" react to tool results.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Union

from .base import STOP_END, STOP_TOOL_USE, LLMResponse, ToolCall, ToolSpec, Transcript, Usage

Step = Union[dict[str, Any], Callable[[Transcript], dict[str, Any]]]

_ids = itertools.count(1)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)


class ScriptedProvider:
    name = "scripted"

    def __init__(self, steps: list[Step], model: str = "scripted-model"):
        self.model = model
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []

    def complete(self, system: str, transcript: Transcript, tools: list[ToolSpec]) -> LLMResponse:
        self.requests.append(
            {"system": system, "turns": list(transcript.turns), "tools": [t.name for t in tools]}
        )
        if not self._steps:
            return LLMResponse(text="(script exhausted)", tool_calls=[], stop=STOP_END, usage=Usage(), model=self.model)
        step = self._steps.pop(0)
        spec = step(transcript) if callable(step) else step
        calls = list(spec.get("tool_calls", []))
        stop = spec.get("stop", STOP_TOOL_USE if calls else STOP_END)
        return LLMResponse(
            text=spec.get("text", ""),
            tool_calls=calls,
            stop=stop,
            usage=Usage(input_tokens=spec.get("input_tokens", 10), output_tokens=spec.get("output_tokens", 5)),
            model=self.model,
            raw={"text": spec.get("text", ""), "tool_calls": [c.__dict__ for c in calls]},
            refusal_detail=spec.get("refusal_detail"),
        )
