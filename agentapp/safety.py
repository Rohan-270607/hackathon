"""Safety: approval policy per tool risk level, workspace path confinement, URL checks."""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.parse import urlsplit

from .llm.base import ToolCall

if TYPE_CHECKING:
    from .tools.base import Tool


@dataclass(frozen=True)
class Decision:
    approved: bool
    reason: str
    by: str  # "policy" | "user" | "auto"


Approver = Callable[[ToolCall, "Tool"], Decision]


def auto_approve(call: ToolCall, tool: "Tool") -> Decision:
    return Decision(True, "auto-approved (--yes)", "auto")


def deny_all(call: ToolCall, tool: "Tool") -> Decision:
    return Decision(False, "no approver available; denied", "auto")


def decide(policy: dict[str, str], tool: "Tool", call: ToolCall, approver: Approver) -> Decision:
    mode = policy.get(tool.risk, "ask")
    if mode == "allow":
        return Decision(True, f"policy allows '{tool.risk}' tools", "policy")
    if mode == "deny":
        return Decision(False, f"policy denies '{tool.risk}' tools", "policy")
    return approver(call, tool)


# ---- filesystem confinement -----------------------------------------------------------
def resolve_in_workspace(workspace: Path, rel: str) -> Path:
    """Resolve `rel` inside the workspace. Rejects absolute paths, `..` escapes and symlink escapes."""
    from .tools.base import ToolError

    root = workspace.resolve()
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise ToolError(f"path {rel!r} is outside the workspace; use a path relative to the workspace")
    return candidate


# ---- network checks ------------------------------------------------------------------
def _is_public_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def check_public_url(url: str) -> None:
    """Allow only http(s) URLs whose host resolves exclusively to public IPs (basic SSRF guard).

    Limitation: a DNS answer can change between this check and the connection (DNS rebinding).
    """
    from .tools.base import ToolError

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError("only http:// and https:// URLs are allowed")
    host = parts.hostname
    if not host:
        raise ToolError("URL has no host")
    try:
        infos = socket.getaddrinfo(host, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as e:
        raise ToolError(f"cannot resolve host {host!r}: {e}") from e
    for info in infos:
        ip = info[4][0]
        if not _is_public_ip(ip.split("%")[0]):
            raise ToolError(f"host {host!r} resolves to non-public address {ip}; refused")
