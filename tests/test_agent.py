from __future__ import annotations

from agentapp.agent import NUDGE, Agent
from agentapp.llm.base import STOP_MAX_TOKENS, STOP_REFUSAL, LLMError, ToolResultsTurn, UserTurn
from agentapp.llm.scripted import ScriptedProvider, call
from agentapp.safety import Decision, auto_approve, deny_all


def last_results(transcript) -> ToolResultsTurn:
    turn = transcript.turns[-1]
    assert isinstance(turn, ToolResultsTurn)
    return turn


def test_full_plan_act_verify_report(settings, store):
    plan = call("update_plan", steps=[{"step": "write file", "status": "in_progress"}, {"step": "verify", "status": "pending"}])
    write = call("write_file", path="hello.txt", content="Hello, world!\n")
    read = call("read_file", path="hello.txt")
    provider = ScriptedProvider(
        [
            {"text": "Planning.", "tool_calls": [plan]},
            {"tool_calls": [write]},
            {"text": "Verifying.", "tool_calls": [read]},
            lambda t: {
                "tool_calls": [
                    call(
                        "finish",
                        status="completed",
                        summary="Created hello.txt and verified its contents.",
                        evidence=[
                            {"claim": "file written", "tool_call_id": write.id},
                            {"claim": "contents verified", "tool_call_id": read.id},
                        ],
                        unverified=[],
                    )
                ]
            },
        ]
    )
    result = Agent(provider, store, settings).run("Create hello.txt containing Hello, world!")

    assert result.status == "completed"
    assert (settings.workspace / "hello.txt").read_text() == "Hello, world!\n"
    assert [c["verdict"] for c in result.report["evidence"]] == ["ok", "ok"]
    assert result.report["harness_issues"] == []
    assert result.report["plan"][0]["step"] == "write file"
    assert result.usage.input_tokens == 40 and result.steps == 4

    types = [e["type"] for e in store.events(result.run_id)]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert "plan" in types and "approval" in types and "tool_call" in types and "report" in types
    run = store.get_run(result.run_id)
    assert run["status"] == "completed" and run["input_tokens"] == 40

    # The model saw the real file contents as the read_file result.
    read_turn = provider.requests[3]["turns"][-1]
    assert read_turn.results[0].content == "Hello, world!\n"


def test_fabricated_evidence_is_flagged(settings, store):
    calc = call("calculator", expression="2+2")
    provider = ScriptedProvider(
        [
            {"tool_calls": [calc]},
            {
                "tool_calls": [
                    call(
                        "finish",
                        status="completed",
                        summary="Done, also emailed the boss.",
                        evidence=[
                            {"claim": "2+2=4", "tool_call_id": calc.id},
                            {"claim": "email sent", "tool_call_id": "call_that_never_happened"},
                        ],
                        unverified=[],
                    )
                ]
            },
        ]
    )
    result = Agent(provider, store, settings).run("add numbers")
    assert result.status == "completed_unverified"
    verdicts = {c["claim"]: c["verdict"] for c in result.report["evidence"]}
    assert verdicts == {"2+2=4": "ok", "email sent": "unknown_call_id"}
    assert result.report["harness_issues"]


def test_completed_without_evidence_is_flagged(settings, store):
    provider = ScriptedProvider(
        [
            {"tool_calls": [call("calculator", expression="1+1")]},
            {"tool_calls": [call("finish", status="completed", summary="2", evidence=[], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings).run("1+1")
    assert result.status == "completed_unverified"


def test_citing_a_failed_call_is_flagged(settings, store):
    bad = call("read_file", path="missing.txt")
    provider = ScriptedProvider(
        [
            {"tool_calls": [bad]},
            {"tool_calls": [call("finish", status="completed", summary="read it", evidence=[{"claim": "read", "tool_call_id": bad.id}], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings).run("read missing.txt")
    assert result.status == "completed_unverified"
    assert result.report["evidence"][0]["verdict"] == "cites_failed_call"


def test_exec_tool_needs_approval_and_denial_is_reported(settings, store):
    py = call("run_python", code="print('should not run')")
    provider = ScriptedProvider(
        [
            {"tool_calls": [py]},
            lambda t: {
                "tool_calls": [
                    call(
                        "finish",
                        status="failed",
                        summary="Could not run code: " + last_results(t).results[0].content,
                        evidence=[],
                        unverified=[],
                    )
                ]
            },
        ]
    )
    result = Agent(provider, store, settings, approver=deny_all).run("run code")
    assert result.status == "failed"
    assert "denied" in result.report["summary"]
    approvals = [e["payload"] for e in store.events(result.run_id) if e["type"] == "approval"]
    assert approvals == [
        {"call_id": py.id, "tool": "run_python", "risk": "exec", "approved": False, "by": "auto", "reason": "no approver available; denied"}
    ]
    assert not [e for e in store.events(result.run_id) if e["type"] == "tool_call"]


def test_approver_is_consulted_with_call_details(settings, store):
    seen = []

    def approver(c, tool):
        seen.append((c.name, tool.risk, c.arguments["code"]))
        return Decision(True, "ok", "user")

    py = call("run_python", code="print(1+1)")
    provider = ScriptedProvider(
        [
            {"tool_calls": [py]},
            {"tool_calls": [call("finish", status="completed", summary="2", evidence=[{"claim": "printed 2", "tool_call_id": py.id}], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings, approver=approver).run("compute")
    assert seen == [("run_python", "exec", "print(1+1)")]
    assert result.status == "completed"


def test_policy_deny_overrides_approver(settings, store):
    settings.policy["write"] = "deny"
    w = call("write_file", path="x.txt", content="x")
    provider = ScriptedProvider(
        [{"tool_calls": [w]}, {"tool_calls": [call("finish", status="failed", summary="denied", evidence=[], unverified=[])]}]
    )
    result = Agent(provider, store, settings, approver=auto_approve).run("write")
    assert not (settings.workspace / "x.txt").exists()
    assert result.status == "failed"


def test_invalid_and_unknown_tool_calls_return_errors_to_model(settings, store):
    bad_args = call("write_file", path="a.txt")  # missing content
    unknown = call("delete_everything")
    provider = ScriptedProvider(
        [
            {"tool_calls": [bad_args, unknown]},
            lambda t: {
                "tool_calls": [
                    call("finish", status="failed", summary=" | ".join(r.content for r in last_results(t).results), evidence=[], unverified=[])
                ]
            },
        ]
    )
    result = Agent(provider, store, settings).run("x")
    assert "invalid arguments" in result.report["summary"]
    assert "unknown tool 'delete_everything'" in result.report["summary"]
    assert not (settings.workspace / "a.txt").exists()


def test_nudge_once_then_no_report(settings, store):
    provider = ScriptedProvider([{"text": "I think it's done."}, {"text": "Really done."}])
    result = Agent(provider, store, settings).run("x")
    assert result.status == "no_report"
    assert isinstance(provider.requests[1]["turns"][-1], UserTurn)
    assert provider.requests[1]["turns"][-1].text == NUDGE


def test_text_tool_calls_get_explicit_not_executed_notice(settings, store):
    fake_call = '{"name": "write_file", "parameters": {"path": "a.txt", "content": "x"}}\nThe file has been created.'
    provider = ScriptedProvider([{"text": fake_call}, {"text": fake_call}, {"text": fake_call}])
    result = Agent(provider, store, settings).run("write a.txt")
    assert result.status == "no_report"
    assert not (settings.workspace / "a.txt").exists()  # text is never executed
    notices = [r["turns"][-1].text for r in provider.requests[1:]]
    assert len(notices) == 2 and all("none of those actions happened" in n and "write_file" in n for n in notices)
    assert result.report["harness_issues"] == ["final message contains tool calls written as text (write_file); they were not executed"]


def test_argument_types_are_coerced_only_when_unambiguous(settings, store):
    w = call("write_file", path="a.txt", content="x", append="False")
    py = call("run_python", code="print(1)", timeout_s="5")
    bad = call("run_python", code="print(1)", timeout_s="five")
    json_list = '[{"claim": "w", "tool_call_id": "%s"}]' % w.id  # evidence sent as a JSON string
    fin = call("finish", status="partial", summary="s", evidence=json_list, unverified="[]")
    provider = ScriptedProvider([{"tool_calls": [w, py, bad]}, {"tool_calls": [fin]}])
    result = Agent(provider, store, settings, approver=auto_approve).run("x")
    assert (settings.workspace / "a.txt").read_text() == "x"
    assert result.status == "partial" and result.report["evidence"][0]["verdict"] == "ok"
    events = store.events(result.run_id)
    coerced = {e["payload"]["call_id"]: e["payload"]["changes"] for e in events if e["type"] == "args_coerced"}
    assert coerced[w.id] == ["append: 'False' -> bool"]
    assert coerced[py.id] == ["timeout_s: '5' -> int"]
    assert bad.id not in coerced
    bad_result = next(e["payload"] for e in events if e["type"] == "tool_result" and e["payload"]["call_id"] == bad.id)
    assert "is not of type 'integer'" in bad_result["output"]
    assert json_list.startswith("[")


def test_nudge_leads_to_report(settings, store):
    provider = ScriptedProvider(
        [
            {"text": "Paris is the capital of France."},
            {"tool_calls": [call("finish", status="completed", summary="Paris.", evidence=[], unverified=["from model knowledge, not checked with a tool"])]},
        ]
    )
    result = Agent(provider, store, settings).run("Capital of France?")
    # No tools were used, so an empty evidence list is not flagged.
    assert result.status == "completed"
    assert result.report["unverified"]


def test_refusal_stops_run(settings, store):
    provider = ScriptedProvider([{"stop": STOP_REFUSAL, "refusal_detail": "cyber: declined"}])
    result = Agent(provider, store, settings).run("something")
    assert result.status == "refused"
    assert "cyber" in result.report["summary"]


def test_max_tokens_truncated_tool_calls_are_not_executed(settings, store):
    w = call("write_file", path="partial.txt", content="half")
    provider = ScriptedProvider(
        [
            {"tool_calls": [w], "stop": STOP_MAX_TOKENS},
            {"tool_calls": [call("finish", status="failed", summary="cut off", evidence=[], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings).run("write")
    assert not (settings.workspace / "partial.txt").exists()
    turn = provider.requests[1]["turns"][-1]
    assert isinstance(turn, ToolResultsTurn) and turn.results[0].is_error and "max_tokens" in turn.note
    assert result.status == "failed"


def test_step_limit(settings, store):
    settings.max_steps = 3
    provider = ScriptedProvider([{"tool_calls": [call("calculator", expression="1+1")]} for _ in range(10)])
    result = Agent(provider, store, settings).run("loop forever")
    assert result.status == "limit_reached" and result.steps == 3


def test_tool_call_budget(settings, store):
    settings.max_tool_calls = 2
    calls = [call("calculator", expression=f"{i}+1") for i in range(3)]
    provider = ScriptedProvider(
        [
            {"tool_calls": calls},
            lambda t: {"tool_calls": [call("finish", status="partial", summary=last_results(t).results[2].content, evidence=[], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings).run("many")
    assert "budget exhausted" in result.report["summary"]


def test_llm_error_is_recorded(settings, store):
    class Failing:
        name, model = "failing", "m"

        def complete(self, *a):
            raise LLMError("no Anthropic credentials found")

    result = Agent(Failing(), store, settings).run("x")
    assert result.status == "error" and "credentials" in result.report["summary"]
    assert store.get_run(result.run_id)["status"] == "error"


def test_crashing_tool_does_not_kill_run(settings, store):
    from agentapp.agent import default_registry

    registry = default_registry()
    registry.get("calculator").func = lambda args, ctx: 1 / 0
    provider = ScriptedProvider(
        [
            {"tool_calls": [call("calculator", expression="1")]},
            lambda t: {"tool_calls": [call("finish", status="failed", summary=last_results(t).results[0].content, evidence=[], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings, registry=registry).run("x")
    assert "internal tool error: ZeroDivisionError" in result.report["summary"]


def test_tool_output_truncated(settings, store):
    settings.tool_output_chars = 50
    (settings.workspace / "big.txt").write_text("x" * 500)
    provider = ScriptedProvider(
        [
            {"tool_calls": [call("read_file", path="big.txt")]},
            lambda t: {"tool_calls": [call("finish", status="completed", summary=last_results(t).results[0].content, evidence=[], unverified=[])]},
        ]
    )
    result = Agent(provider, store, settings).run("x")
    assert "[truncated 450 characters]" in result.report["summary"]


def test_cancel(settings, store):
    provider = ScriptedProvider([{"tool_calls": [call("calculator", expression="1")]}] * 5)
    agent = Agent(provider, store, settings)
    agent.cancel_event.set()
    result = agent.run("x")
    assert result.status == "cancelled"
