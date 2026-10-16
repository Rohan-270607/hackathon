"""Exercises the real Anthropic SDK request/response path against a mocked HTTP transport.

No network and no API key are used: httpx2.MockTransport answers in place of the API.
"""

from __future__ import annotations

import json

import anthropic
import httpx2
import pytest

from agentapp.agent import Agent
from agentapp.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from agentapp.llm.base import LLMError

THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def api_message(content, stop_reason, **extra):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
        **extra,
    }


class FakeAPI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "body": json.loads(request.content)})
        status, body = self.responses.pop(0)
        return httpx2.Response(status, json=body)


def make_provider(fake: FakeAPI, **kw) -> AnthropicProvider:
    client = anthropic.Anthropic(
        api_key="test-key-not-real",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(fake)),
    )
    return AnthropicProvider("claude-opus-5-5", client=client, **kw)


def test_agent_loop_through_sdk_echoes_thinking_and_builds_valid_requests(settings, store):
    fake = FakeAPI(
        [
            (200, api_message([THINKING, {"type": "text", "text": "Computing."},
                               {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {"expression": "6*7"}}], "tool_use")),
            (200, api_message([{"type": "tool_use", "id": "toolu_2", "name": "finish", "input": {
                "status": "completed", "summary": "6*7 = 42", "evidence": [{"claim": "42", "tool_call_id": "toolu_1"}], "unverified": []}}], "tool_use")),
        ]
    )
    result = Agent(make_provider(fake), store, settings).run("What is 6*7?")
    assert result.status == "completed"
    assert result.report["evidence"][0]["verdict"] == "ok"
    assert result.usage.input_tokens == 200

    first, second = fake.requests
    assert first["url"].endswith("/v1/messages?beta=true")
    assert first["headers"]["anthropic-beta"] == FALLBACK_BETA
    body = first["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"effort": "high"}
    assert "thinking" not in body and "tool_choice" not in body
    assert [t["name"] for t in body["tools"]] == sorted(t["name"] for t in body["tools"])

    # Second request: prefix identical to the first, assistant content echoed unchanged
    # (thinking block with signature included), then the tool result.
    msgs = second["body"]["messages"]
    assert msgs[0] == body["messages"][0]
    assert second["body"]["system"] == body["system"] and second["body"]["tools"] == body["tools"]
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"][0] == THINKING
    assert msgs[1]["content"][2]["id"] == "toolu_1"
    assert msgs[2] == {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42", "is_error": False}]}


def test_options_can_be_disabled(settings):
    fake = FakeAPI([(200, api_message([{"type": "text", "text": "hi"}], "end_turn"))])
    from agentapp.llm.base import Transcript, UserTurn

    resp = make_provider(fake, effort=None, fallbacks=None).complete("sys", Transcript([UserTurn("hi")]), [])
    body = fake.requests[0]["body"]
    assert "output_config" not in body and "fallbacks" not in body and "tools" not in body
    assert "anthropic-beta" not in fake.requests[0]["headers"]
    assert resp.text == "hi" and resp.stop == "end_turn"


def test_refusal_is_mapped(settings, store):
    fake = FakeAPI([(200, api_message([], "refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "Declined."}))])
    result = Agent(make_provider(fake), store, settings).run("x")
    assert result.status == "refused" and "cyber: Declined." in result.report["summary"]


def test_mid_output_fallback_echo_rule():
    from agentapp.llm.anthropic_provider import parse_response
    from anthropic.types.beta import BetaMessage

    msg = BetaMessage.model_validate(
        api_message(
            [
                THINKING,
                {"type": "text", "text": "partial "},
                {"type": "tool_use", "id": "toolu_old", "name": "calculator", "input": {"expression": "1"}},
                {"type": "fallback", "from": {"model": "claude-opus-5-5"}, "to": {"model": "claude-opus-4-8"}, "trigger": {"type": "refusal", "category": "cyber"}},
                {"type": "tool_use", "id": "toolu_new", "name": "calculator", "input": {"expression": "2"}},
            ],
            "tool_use",
        )
    )
    resp = parse_response(msg)
    assert [c.id for c in resp.tool_calls] == ["toolu_new"]
    assert [b["type"] for b in resp.raw] == ["text", "fallback", "tool_use"]


@pytest.mark.parametrize(
    "status,needle",
    [(401, "authentication failed"), (404, "not found"), (400, "bad request"), (429, "rate limited"), (529, "API error 529")],
)
def test_http_errors_become_llm_errors(status, needle):
    from agentapp.llm.base import Transcript, UserTurn

    fake = FakeAPI([(status, {"type": "error", "error": {"type": "x", "message": "boom"}})])
    with pytest.raises(LLMError, match=needle):
        make_provider(fake).complete("sys", Transcript([UserTurn("hi")]), [])


def test_missing_credentials_message(monkeypatch):
    from agentapp.llm.base import Transcript, UserTurn

    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    fake = FakeAPI([])
    try:
        client = anthropic.Anthropic(http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(fake)), max_retries=0)
    except Exception as e:  # some SDK versions may fail at construction instead
        pytest.skip(f"client construction failed without credentials: {e!r}")
    provider = AnthropicProvider("claude-opus-5-5", client=client)
    with pytest.raises(LLMError, match="no Anthropic credentials"):
        provider.complete("sys", Transcript([UserTurn("hi")]), [])
    assert fake.requests == []
