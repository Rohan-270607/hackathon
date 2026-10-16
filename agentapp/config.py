"""Settings, read from environment variables only. Secrets are never stored here.

The Anthropic SDK resolves its own credentials (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
or an `ant auth login` profile), so this module never reads or holds an API key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RISK_LEVELS = ("read", "write", "exec", "network")
POLICY_VALUES = ("allow", "ask", "deny")

DEFAULT_POLICY = {"read": "allow", "write": "allow", "exec": "ask", "network": "ask"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass
class Settings:
    provider: str = "anthropic"
    model: str = "claude-opus-5-5"
    # Opus 5.5 defaults to "medium"; agentic multi-step work does better at "high".
    # Set AGENT_EFFORT="" to omit the parameter (needed for models without effort support).
    effort: str | None = "high"
    # Server-side refusal fallback ("default" routes by refusal category). "" disables.
    fallbacks: str | None = "default"
    max_tokens: int = 16000
    # Ollama (local models)
    ollama_host: str = "http://localhost:11434"
    ollama_num_ctx: int = 16384
    ollama_num_predict: int = 4096
    ollama_timeout_s: int = 600
    max_steps: int = 25
    max_tool_calls: int = 60
    tool_timeout_s: int = 30
    tool_output_chars: int = 12000
    data_dir: Path = field(default_factory=lambda: Path("data"))
    workspace: Path = field(default_factory=lambda: Path("workspace"))
    policy: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_POLICY))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        s.provider = os.environ.get("AGENT_PROVIDER", s.provider)
        s.model = os.environ.get("AGENT_MODEL", s.model)
        if "AGENT_EFFORT" in os.environ:
            s.effort = os.environ["AGENT_EFFORT"].strip() or None
        if "AGENT_FALLBACKS" in os.environ:
            s.fallbacks = os.environ["AGENT_FALLBACKS"].strip() or None
        s.max_tokens = _env_int("AGENT_MAX_TOKENS", s.max_tokens)
        s.ollama_host = os.environ.get("OLLAMA_HOST", s.ollama_host) or s.ollama_host
        s.ollama_num_ctx = _env_int("OLLAMA_NUM_CTX", s.ollama_num_ctx)
        s.ollama_num_predict = _env_int("OLLAMA_NUM_PREDICT", s.ollama_num_predict)
        s.ollama_timeout_s = _env_int("OLLAMA_TIMEOUT", s.ollama_timeout_s)
        s.max_steps = _env_int("AGENT_MAX_STEPS", s.max_steps)
        s.max_tool_calls = _env_int("AGENT_MAX_TOOL_CALLS", s.max_tool_calls)
        s.tool_timeout_s = _env_int("AGENT_TOOL_TIMEOUT", s.tool_timeout_s)
        s.tool_output_chars = _env_int("AGENT_TOOL_OUTPUT_CHARS", s.tool_output_chars)
        s.data_dir = Path(os.environ.get("AGENT_DATA_DIR", str(s.data_dir)))
        s.workspace = Path(os.environ.get("AGENT_WORKSPACE", str(s.workspace)))
        for risk in RISK_LEVELS:
            value = os.environ.get(f"AGENT_POLICY_{risk.upper()}")
            if value:
                if value not in POLICY_VALUES:
                    raise ValueError(f"AGENT_POLICY_{risk.upper()} must be one of {POLICY_VALUES}")
                s.policy[risk] = value
        return s

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
