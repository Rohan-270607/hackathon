"""Human-in-the-loop approvals for the web UI.

The agent runs in a worker thread. When the policy says "ask", the worker registers a pending
approval and blocks until the browser answers, the run is cancelled, or the wait times out
(timeouts count as a denial).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import ToolCall
from ..safety import Approver, Decision
from ..tools.base import Tool


@dataclass
class _Pending:
    id: str
    run_id: str
    tool: str
    risk: str
    arguments: dict[str, Any]
    event: threading.Event = field(default_factory=threading.Event)
    decision: Decision | None = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "risk": self.risk, "arguments": self.arguments}


class ApprovalBroker:
    def __init__(self, timeout_s: float = 600):
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    def approver_for(self, run_id: str) -> Approver:
        def approve(call: ToolCall, tool: Tool) -> Decision:
            item = _Pending(uuid.uuid4().hex[:10], run_id, tool.name, tool.risk, call.arguments)
            with self._lock:
                self._pending[item.id] = item
            answered = item.event.wait(self.timeout_s)
            with self._lock:
                self._pending.pop(item.id, None)
            if not answered or item.decision is None:
                return Decision(False, f"no answer within {int(self.timeout_s)}s", "auto")
            return item.decision

        return approve

    def pending(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [p.public() for p in self._pending.values() if p.run_id == run_id]

    def resolve(self, run_id: str, approval_id: str, approve: bool) -> bool:
        with self._lock:
            item = self._pending.get(approval_id)
            if item is None or item.run_id != run_id:
                return False
            item.decision = Decision(approve, "approved in web UI" if approve else "declined in web UI", "user")
            item.event.set()
            return True

    def deny_all(self, run_id: str, reason: str) -> None:
        with self._lock:
            for item in self._pending.values():
                if item.run_id == run_id:
                    item.decision = Decision(False, reason, "user")
                    item.event.set()
