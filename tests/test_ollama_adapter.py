"""Ollama adapter tests against a local fake server that returns the response shapes
captured from a real Ollama 0.40.4 server. Exercises the real urllib request path."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentapp.agent import Agent
from agentapp.config import Settings
from agentapp.llm import make_provider
from agentapp.llm.base import LLMError, Transcript, UserTurn
from agentapp.llm.ollama_provider import OllamaProvider, normalize_host


def chat_body(content="", tool_calls=None, done_reason="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "model": "llama3.1:8b-instruct-q4_K_M",
        "created_at": "2026-10-15T16:46:30Z",
        "message": message,
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 159,
        "eval_count": 18,
    }


def tc(name, args, id_="call_abc", index=0):
    call = {"function": {"index": index, "name": name, "arguments": args}}
    if id_ is not None:
        call["id"] = id_
    return call


@pytest.fixture
def fake_ollama():
    state = {"responses": [], "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append({"path": self.path, "body": body})
            status, payload = state["responses"].pop(0)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["host"] = f"127.0.0.1:{server.server_address[1]}"
    yield state
    server.shutdown()


def test_agent_loop_over_ollama_api(fake_ollama, settings, store):
    fake_ollama["responses"] = [
        (200, chat_body(tool_calls=[tc("write_file", {"path": "a.txt", "content": "hi"}, "call_w")])),
        (200, chat_body(tool_calls=[tc("read_file", {"path": "missing.txt"}, "call_r")])),
        (200, chat_body(tool_calls=[tc("finish", {"status": "partial", "summary": "wrote a.txt; read failed",
                                                  "evidence": [{"claim": "wrote", "tool_call_id": "call_w"}], "unverified": []}, "call_f")])),
    ]
    provider = OllamaProvider("llama3.1:8b-instruct-q4_K_M", host=fake_ollama["host"], num_ctx=32768)
    result = Agent(provider, store, settings).run("write a.txt")
    assert result.status == "partial"
    assert result.report["evidence"][0]["verdict"] == "ok"
    assert result.usage.input_tokens == 159 * 3

    first = fake_ollama["requests"][0]
    assert first["path"] == "/api/chat"
    body = first["body"]
    assert body["stream"] is False and body["model"] == "llama3.1:8b-instruct-q4_K_M"
    assert body["options"] == {"num_ctx": 32768, "num_predict": 4096}
    assert body["messages"][0]["role"] == "system" and body["messages"][1] == {"role": "user", "content": "write a.txt"}
    tool = next(t for t in body["tools"] if t["function"]["name"] == "write_file")
    assert tool["type"] == "function" and tool["function"]["parameters"]["required"] == ["path", "content"]

    third = fake_ollama["requests"][2]["body"]["messages"]
    # history: system, user, assistant(write), tool, assistant(read), tool(error)
    assert [m["role"] for m in third] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert third[2]["tool_calls"][0]["id"] == "call_w"
    assert third[3]["tool_name"] == "write_file" and "wrote 2 bytes" in third[3]["content"]
    assert third[5]["tool_name"] == "read_file" and third[5]["content"].startswith("ERROR: ")


def test_missing_call_ids_are_generated_consistently(fake_ollama, settings, store):
    fake_ollama["responses"] = [
        (200, chat_body(tool_calls=[tc("calculator", {"expression": "2+2"}, id_=None)])),
        (200, chat_body(content="4")),
        (200, chat_body(content="still 4")),
    ]
    provider = OllamaProvider("m", host=fake_ollama["host"])
    result = Agent(provider, store, settings).run("2+2")
    assert result.status == "no_report"
    second = fake_ollama["requests"][1]["body"]["messages"]
    generated = second[2]["tool_calls"][0]["id"]
    assert generated.startswith("ollama_")
    calls = [e["payload"] for e in store.events(result.run_id) if e["type"] == "tool_call"]
    assert calls[0]["call_id"] == generated


def test_truncation_is_reported(fake_ollama):
    fake_ollama["responses"] = [(200, chat_body(content="partial", done_reason="length"))]
    resp = OllamaProvider("m", host=fake_ollama["host"]).complete("s", Transcript([UserTurn("x")]), [])
    assert resp.stop == "max_tokens"


def test_missing_model_error(fake_ollama):
    fake_ollama["responses"] = [(404, {"error": "model 'nope:latest' not found"})]
    with pytest.raises(LLMError, match="Ollama error 404: model 'nope:latest' not found .*ollama pull"):
        OllamaProvider("nope:latest", host=fake_ollama["host"]).complete("s", Transcript([UserTurn("x")]), [])


def test_unreachable_server():
    with pytest.raises(LLMError, match="cannot reach Ollama"):
        OllamaProvider("m", host="127.0.0.1:1", timeout_s=5).complete("s", Transcript([UserTurn("x")]), [])


def test_context_guard_refuses_instead_of_silent_truncation():
    provider = OllamaProvider("m", host="127.0.0.1:1", num_ctx=2048, max_tokens=1024)
    with pytest.raises(LLMError, match="may exceed num_ctx=2048"):
        provider.complete("s" * 6000, Transcript([UserTurn("x")]), [])


def test_factory_and_host_normalization(monkeypatch):
    assert normalize_host("localhost:11434/") == "http://localhost:11434"
    assert normalize_host("https://box:1") == "https://box:1"
    s = Settings()
    s.provider = "ollama"
    with pytest.raises(LLMError, match="needs AGENT_MODEL"):
        make_provider(s)
    monkeypatch.setenv("AGENT_PROVIDER", "ollama")
    monkeypatch.setenv("AGENT_MODEL", "llama3.1:8b-instruct-q4_K_M")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    p = make_provider(Settings.from_env())
    assert isinstance(p, OllamaProvider) and p.num_ctx == 8192 and p.model == "llama3.1:8b-instruct-q4_K_M"
