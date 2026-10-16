"""Tool definitions, registry, and argument validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from ..llm.base import ToolSpec
from ..store import Store


class ToolError(Exception):
    """An expected tool failure. Its message is returned to the model as an error result."""


@dataclass
class ToolContext:
    workspace: Path
    store: Store
    run_id: str
    timeout_s: int = 30


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: str  # one of config.RISK_LEVELS
    func: Callable[[dict[str, Any], ToolContext], str] | None = None
    control: bool = False  # handled by the agent loop itself (update_plan, finish)
    _validator: Draft202012Validator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        Draft202012Validator.check_schema(self.input_schema)
        self._validator = Draft202012Validator(self.input_schema)

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def coerce_args(self, args: Any) -> tuple[Any, list[str]]:
        """Fix unambiguous type slips (common with small local models) in top-level arguments:
        "true"/"false" -> bool, "10" -> int, "1.5" -> float, and a JSON string -> array/object,
        only where the schema asks for that type. Anything ambiguous is left for validation to reject."""
        if not isinstance(args, dict):
            return args, []
        props = self.input_schema.get("properties", {})
        fixed, notes = dict(args), []
        for key, value in args.items():
            spec = props.get(key)
            if not isinstance(spec, dict) or not isinstance(value, str):
                continue
            new = _coerce_str(value, spec.get("type"))
            if new is not _UNCHANGED:
                fixed[key] = new
                notes.append(f"{key}: {value[:40]!r} -> {type(new).__name__}")
        return fixed, notes

    def validation_errors(self, args: Any) -> list[str]:
        errors = []
        for err in self._validator.iter_errors(args):
            where = "/".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(f"{where}: {err.message}")
        return errors


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        if not tool.control and tool.func is None:
            raise ValueError(f"tool {tool.name!r} needs a func")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        # Deterministic order keeps the request prefix stable (prompt caching, thinking replay).
        return [self._tools[n].spec() for n in sorted(self._tools)]


_UNCHANGED = object()
_INT = re.compile(r"-?\d+")


def _coerce_str(value: str, want: Any) -> Any:
    text = value.strip()
    if want == "boolean" and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if want == "integer" and _INT.fullmatch(text):
        return int(text)
    if want == "number":
        try:
            return float(text)
        except ValueError:
            return _UNCHANGED
    if want in ("array", "object"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _UNCHANGED
        if (want == "array" and isinstance(parsed, list)) or (want == "object" and isinstance(parsed, dict)):
            return parsed
    return _UNCHANGED


def obj_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }
