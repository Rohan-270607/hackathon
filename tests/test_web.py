from __future__ import annotations

import time

from fastapi.testclient import TestClient

from agentapp.llm.scripted import ScriptedProvider, call
from agentapp.web.app import create_app


def wait_for(fn, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def make_client(settings, store, steps_factory):
    def factory(_settings):
        return ScriptedProvider(steps_factory())

    return TestClient(create_app(settings, store, provider_factory=factory))


def finish_with(status, summary, evidence=()):
    return {"tool_calls": [call("finish", status=status, summary=summary, evidence=list(evidence), unverified=[])]}


def test_index_served(settings, store):
    client = make_client(settings, store, lambda: [])
    r = client.get("/")
    assert r.status_code == 200 and "Agent Console" in r.text


def test_run_lifecycle_over_api(settings, store):
    def steps():
        w = call("write_file", path="web.txt", content="from web")
        return [{"text": "Writing.", "tool_calls": [w]}, finish_with("completed", "done", [{"claim": "written", "tool_call_id": w.id}])]

    client = make_client(settings, store, steps)
    run_id = client.post("/api/runs", json={"task": "write web.txt"}).json()["run_id"]
    data = wait_for(lambda: (d := client.get(f"/api/runs/{run_id}").json())["run"]["status"] != "running" and d)
    assert data["run"]["status"] == "completed"
    types = [e["type"] for e in data["events"]]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    # Incremental polling returns only newer events.
    last = data["events"][-2]["seq"]
    assert [e["type"] for e in client.get(f"/api/runs/{run_id}?after={last}").json()["events"]] == ["run_finished"]
    assert client.get("/api/runs").json()[0]["id"] == run_id
    assert (settings.workspace / "web.txt").read_text() == "from web"


def test_web_approval_approve_and_deny(settings, store):
    def steps():
        py = call("run_python", code="print('approved run')")
        return [{"tool_calls": [py]}, lambda t: finish_with("completed", t.turns[-1].results[0].content, [{"claim": "ran", "tool_call_id": py.id}])]

    client = make_client(settings, store, steps)
    for approve, expected_status, needle in [(True, "completed", "approved run"), (False, "completed_unverified", "declined in web UI")]:
        run_id = client.post("/api/runs", json={"task": "run python"}).json()["run_id"]
        pending = wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["pending_approvals"])
        assert pending[0]["tool"] == "run_python" and pending[0]["risk"] == "exec"
        assert pending[0]["arguments"]["code"] == "print('approved run')"
        r = client.post(f"/api/runs/{run_id}/approvals/{pending[0]['id']}", json={"approve": approve})
        assert r.json() == {"ok": True}
        data = wait_for(lambda: (d := client.get(f"/api/runs/{run_id}").json())["run"]["status"] != "running" and d)
        assert data["run"]["status"] == expected_status
        assert needle in data["run"]["report"]["summary"]


def test_unknown_approval_and_run(settings, store):
    client = make_client(settings, store, lambda: [])
    assert client.get("/api/runs/nope").status_code == 404
    assert client.post("/api/runs/nope/approvals/x", json={"approve": True}).status_code == 404
    assert client.post("/api/runs", json={"task": ""}).status_code == 422


def test_cancel_while_waiting_for_approval(settings, store):
    def steps():
        return [{"tool_calls": [call("http_get", url="https://example.com")]}] * 3

    client = make_client(settings, store, steps)
    run_id = client.post("/api/runs", json={"task": "fetch"}).json()["run_id"]
    wait_for(lambda: client.get(f"/api/runs/{run_id}").json()["pending_approvals"])
    assert client.post(f"/api/runs/{run_id}/cancel").json() == {"ok": True}
    data = wait_for(lambda: (d := client.get(f"/api/runs/{run_id}").json())["run"]["status"] != "running" and d)
    assert data["run"]["status"] == "cancelled"
    approvals = [e["payload"] for e in data["events"] if e["type"] == "approval"]
    assert approvals[0]["approved"] is False and approvals[0]["reason"] == "run cancelled"
