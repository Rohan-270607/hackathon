"""Secret redaction for anything written to logs, the audit trail, or child processes."""

from __future__ import annotations

import os
import re

SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)", re.IGNORECASE)

_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic keys
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI-style keys
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),  # Google API keys
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),  # GitHub tokens
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
]

MASK = "[REDACTED]"


def secret_env_values() -> list[str]:
    """Values of environment variables whose names look secret (min length avoids masking '1')."""
    return sorted(
        {v for k, v in os.environ.items() if SECRET_NAME.search(k) and v and len(v) >= 8},
        key=len,
        reverse=True,
    )


def redact(text: str) -> str:
    for value in secret_env_values():
        text = text.replace(value, MASK)
    for pat in _PATTERNS:
        text = pat.sub(MASK, text)
    return text


def scrubbed_env() -> dict[str, str]:
    """A copy of the environment without secret-looking variables, for child processes."""
    return {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
