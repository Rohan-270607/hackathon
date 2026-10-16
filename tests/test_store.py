from __future__ import annotations

from agentapp.redact import redact


def test_events_are_redacted_before_storage(store, monkeypatch):
    monkeypatch.setenv("MY_SERVICE_TOKEN", "supersecretvalue42")
    run_id = store.create_run("task", "scripted", "m")
    store.log_event(
        run_id,
        "tool_result",
        {"output": "key=sk-ant-api03-abcdefghijklmnop token=supersecretvalue42 gh=ghp_" + "a" * 36},
    )
    payload = store.events(run_id)[0]["payload"]["output"]
    assert "sk-ant-api03" not in payload
    assert "supersecretvalue42" not in payload
    assert "ghp_" not in payload
    assert payload.count("[REDACTED]") == 3


def test_event_sequence_and_run_lifecycle(store):
    run_id = store.create_run("do things", "scripted", "m")
    assert store.log_event(run_id, "a", {}) == 1
    assert store.log_event(run_id, "b", {}) == 2
    assert [e["type"] for e in store.events(run_id, after_seq=1)] == ["b"]
    store.finish_run(run_id, "completed", {"summary": "ok"}, 10, 5)
    run = store.get_run(run_id)
    assert run["status"] == "completed" and run["report"] == {"summary": "ok"}
    assert run["input_tokens"] == 10
    assert store.list_runs()[0]["id"] == run_id


def test_redact_leaves_normal_text():
    assert redact("nothing secret here: 12345") == "nothing secret here: 12345"
