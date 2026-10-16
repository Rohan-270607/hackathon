from __future__ import annotations

import pytest

from agentapp import cli
from tests.test_anthropic_adapter import FakeAPI, api_message, make_provider


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_WORKSPACE", str(tmp_path / "ws"))
    return tmp_path


def use_fake_api(monkeypatch, responses) -> FakeAPI:
    fake = FakeAPI(responses)
    monkeypatch.setattr(cli, "make_provider", lambda settings: make_provider(fake))
    return fake


def test_cli_run_completes_and_reports(env, monkeypatch, capsys):
    use_fake_api(
        monkeypatch,
        [
            (200, api_message([{"type": "tool_use", "id": "toolu_w", "name": "write_file", "input": {"path": "out.txt", "content": "391\n"}}], "tool_use")),
            (200, api_message([{"type": "tool_use", "id": "toolu_f", "name": "finish", "input": {
                "status": "completed", "summary": "Wrote 391 to out.txt", "evidence": [{"claim": "file written", "tool_call_id": "toolu_w"}], "unverified": []}}], "tool_use")),
        ],
    )
    code = cli.main(["run", "write 17*23 to out.txt"])
    out = capsys.readouterr().out
    assert code == 0
    assert "STATUS: completed" in out and "-> write_file" in out
    assert (env / "ws" / "out.txt").read_text() == "391\n"

    run_id = out.strip().splitlines()[-1].split()[1]
    assert cli.main(["show", run_id, "--events"]) == 0
    shown = capsys.readouterr().out
    assert "run_started" in shown and "tool_result" in shown and "run_finished" in shown


def test_cli_non_interactive_denies_exec_without_yes(env, monkeypatch, capsys):
    fake = use_fake_api(
        monkeypatch,
        [
            (200, api_message([{"type": "tool_use", "id": "toolu_p", "name": "run_python", "input": {"code": "print(1)"}}], "tool_use")),
            (200, api_message([{"type": "tool_use", "id": "toolu_f", "name": "finish", "input": {
                "status": "failed", "summary": "Python was not allowed.", "evidence": [], "unverified": []}}], "tool_use")),
        ],
    )
    code = cli.main(["run", "run some python"])
    out = capsys.readouterr().out
    assert code == 1
    assert "run_python denied" in out
    tool_result = fake.requests[1]["body"]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True and "non-interactive" in tool_result["content"]


def test_cli_prompt_eof_is_not_recorded_as_user_decision(monkeypatch, capsys):
    from agentapp.llm.base import ToolCall
    from agentapp.tools.builtin import builtin_tools

    class TtyWithNoInput:  # what Windows reports for a stdin attached to NUL
        def isatty(self):
            return True

        def readline(self):
            return ""

    monkeypatch.setattr("sys.stdin", TtyWithNoInput())

    def eof(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    tool = next(t for t in builtin_tools() if t.name == "run_python")
    decision = cli.cli_approver(ToolCall("c1", "run_python", {"code": "print(1)"}), tool)
    assert decision.approved is False and decision.by == "auto" and "no input available" in decision.reason


def test_cli_prompt_user_answers(monkeypatch, capsys):
    from agentapp.llm.base import ToolCall
    from agentapp.tools.builtin import builtin_tools

    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Tty())
    tool = next(t for t in builtin_tools() if t.name == "run_python")
    for answer, approved in (("y", True), ("", False), ("no", False)):
        monkeypatch.setattr("builtins.input", lambda _p="", a=answer: a)
        d = cli.cli_approver(ToolCall("c1", "run_python", {"code": "print(1)"}), tool)
        assert d.approved is approved and d.by == "user"


def test_cli_yes_allows_exec(env, monkeypatch, capsys):
    use_fake_api(
        monkeypatch,
        [
            (200, api_message([{"type": "tool_use", "id": "toolu_p", "name": "run_python", "input": {"code": "print(17*23)"}}], "tool_use")),
            (200, api_message([{"type": "tool_use", "id": "toolu_f", "name": "finish", "input": {
                "status": "completed", "summary": "391", "evidence": [{"claim": "computed", "tool_call_id": "toolu_p"}], "unverified": []}}], "tool_use")),
        ],
    )
    assert cli.main(["run", "compute", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "<- ok: exit_code: 0 stdout: 391" in out


def test_cli_memory_commands(env, capsys):
    assert cli.main(["memory", "add", "likes", "tea", "--tags", "prefs"]) == 0
    assert cli.main(["memory", "search", "tea"]) == 0
    assert "#1 [prefs] likes tea" in capsys.readouterr().out
    assert cli.main(["memory", "delete", "1"]) == 0
    assert cli.main(["memory", "list"]) == 0
    assert "no memories" in capsys.readouterr().out


def test_cli_bad_config(env, monkeypatch, capsys):
    monkeypatch.setenv("AGENT_POLICY_EXEC", "sometimes")
    assert cli.main(["runs"]) == 2
    assert "config error" in capsys.readouterr().err
