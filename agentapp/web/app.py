"""Local web UI: submit tasks, watch the agent's trace live, approve risky tool calls.

There is no authentication. `agent web` binds to 127.0.0.1 by default; keep it that way.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..agent import Agent
from ..config import Settings
from ..llm import make_provider
from ..llm.base import LLMError, LLMProvider
from ..safety import auto_approve
from ..store import Store
from .approvals import ApprovalBroker

STATIC = Path(__file__).parent / "static"


class RunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=20_000)
    auto_approve: bool = False


class ApprovalAnswer(BaseModel):
    approve: bool


def create_app(
    settings: Settings,
    store: Store,
    provider_factory: Callable[[Settings], LLMProvider] = make_provider,
    broker: ApprovalBroker | None = None,
) -> FastAPI:
    app = FastAPI(title="agentapp", docs_url=None, redoc_url=None)
    broker = broker or ApprovalBroker()
    agents: dict[str, Agent] = {}
    threads: dict[str, threading.Thread] = {}
    settings.ensure_dirs()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict[str, Any]:
        try:
            provider = provider_factory(settings)
        except LLMError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        run_id = store.create_run(req.task, provider.name, provider.model)
        agent = Agent(
            provider,
            store,
            settings,
            approver=auto_approve if req.auto_approve else broker.approver_for(run_id),
        )
        agents[run_id] = agent
        thread = threading.Thread(target=agent.run, args=(req.task, run_id), name=f"run-{run_id}", daemon=True)
        threads[run_id] = thread
        thread.start()
        return {"run_id": run_id}

    @app.get("/api/runs")
    def list_runs(limit: int = 30) -> list[dict[str, Any]]:
        return store.list_runs(limit)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, after: int = 0) -> dict[str, Any]:
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run": run, "events": store.events(run_id, after), "pending_approvals": broker.pending(run_id)}

    @app.post("/api/runs/{run_id}/approvals/{approval_id}")
    def answer_approval(run_id: str, approval_id: str, answer: ApprovalAnswer) -> dict[str, bool]:
        if not broker.resolve(run_id, approval_id, answer.approve):
            raise HTTPException(status_code=404, detail="no such pending approval")
        return {"ok": True}

    @app.post("/api/runs/{run_id}/cancel")
    def cancel(run_id: str) -> dict[str, bool]:
        agent = agents.get(run_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="run not active in this server")
        agent.cancel_event.set()
        broker.deny_all(run_id, "run cancelled")
        return {"ok": True}

    app.state.threads = threads  # exposed for tests
    return app
