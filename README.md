# agentapp

A general-purpose agent that carries out tasks described in natural language by **planning,
calling tools, verifying results, and reporting honestly**. It has a CLI and a local web UI,
stores runs, an audit trail and long-term memory in SQLite, and talks to the LLM through a
small provider-agnostic adapter. Claude is the provider that ships.

## Quick start

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python -m pip install -e ".[dev]"
```

Set credentials in your environment (never in code). Either export `ANTHROPIC_API_KEY`, or
run `ant auth login`; the SDK picks up either automatically.

```bash
.venv\Scripts\agent run "Create notes/todo.md with three items, then verify the file"
```

```bash
.venv\Scripts\agent web
```

Then open http://127.0.0.1:8000. The web UI has **no authentication**, so keep it on localhost.

Other commands: `agent runs`, `agent show <run_id> --events`, and
`agent memory list|search|add|delete`.

### Free: local models through Ollama

No API key needed. Set these in PowerShell, then use `agent run` / `agent web` as usual:

```bash
$env:AGENT_PROVIDER = 'ollama'; $env:AGENT_MODEL = 'llama3.1:8b-instruct-q4_K_M'
```

Optional: `OLLAMA_HOST` (default `http://localhost:11434`), `OLLAMA_NUM_CTX` (16384),
`OLLAMA_NUM_PREDICT` (4096), `OLLAMA_TIMEOUT` (600 s). If a conversation might exceed
`num_ctx`, the adapter stops with an error. Ollama would otherwise quietly drop the oldest
context, which can include the system prompt.

In live tests, llama3.1 8B rarely finished these tasks correctly. It often sends arguments
with the wrong types, writes tool calls as plain text, and files malformed reports. The
harness corrects unambiguous type slips (logged as `args_coerced`), never runs tool calls
written as text, and in those tests recorded no false `completed` status. Expect much more
reliable runs from a stronger model.

## How a run works

1. The task goes to the model with a fixed system prompt and the tool list.
2. The model calls `update_plan`, then works through the tools. Every tool argument is
   validated against its JSON schema before anything runs.
3. Each call goes through the **approval policy** for its risk level:

   | risk    | tools                                   | default |
   |---------|-----------------------------------------|---------|
   | read    | calculator, list_files, read_file, recall | allow |
   | write   | write_file (workspace only), remember   | allow   |
   | exec    | run_python                              | ask     |
   | network | http_get                                | ask     |

   `ask` prompts in the terminal or shows Approve/Deny buttons in the web UI. With no
   approver (a non-interactive shell) the call is denied. `--yes` (CLI) or the
   "auto-approve" checkbox (web) approves all `ask` calls. Override any level with
   `AGENT_POLICY_<RISK>=allow|ask|deny`.
4. The model ends with `finish`, which takes a status, a summary, **evidence** (claims, each
   citing a tool call id), and a list of what it could **not** verify.
5. **The harness checks the evidence.** Each cited call id must belong to a tool call that
   ran and succeeded in this run. If any citation fails, or "completed" cites no evidence
   after tools were used, the run is recorded as `completed_unverified`.

Run statuses: `completed`, `completed_unverified`, `partial`, `failed`, `needs_input`,
`refused`, `no_report`, `limit_reached`, `cancelled`, `error`.

## Safety notes (read these)

- **File tools** are confined to `AGENT_WORKSPACE`. Absolute paths, `..` and symlink escapes
  are rejected.
- **`run_python` is not a sandbox.** It is a separate process with a timeout, the workspace as
  its working directory, isolated mode (`-I`), and secret-looking environment variables
  removed. It still has your user's OS permissions, which is why it needs approval by default.
  For untrusted workloads, run the app inside a container or VM.
- **`http_get`** allows only http(s) and refuses hosts that resolve to private, loopback,
  link-local or reserved addresses, including after redirects. DNS rebinding is not fully
  covered.
- **Prompt injection.** Tool output is treated as untrusted data in the system prompt, and
  network and exec tools need approval. That reduces risk but does not eliminate it.
- **Secrets** are read only from the environment. The app never loads `.env` files. Audit
  events, run tasks and memories pass through redaction (secret env values and common key
  formats) before they are stored.

## Provider adapter

`agentapp/llm/base.py` defines the provider-agnostic transcript, `ToolSpec`, `ToolCall`,
`LLMResponse` and the `LLMProvider.complete()` protocol. To add a provider, implement
`complete()` in a new module and register it in `agentapp/llm/__init__.py:make_provider`.

The Claude adapter (`anthropic_provider.py`, `anthropic` 1.14):

- uses `client.beta.messages.create` with `output_config.effort` (default `high`) and the
  server-side refusal fallback (`fallbacks="default"`, beta `server-side-fallback-2026-07-01`).
  Turn either off with `AGENT_EFFORT=` / `AGENT_FALLBACKS=`.
- keeps the transcript append-only and echoes assistant content back unchanged, because
  thinking blocks must be replayed as-is.
- maps refusals, `max_tokens` truncation (truncated tool calls are not executed) and SDK
  errors to clear run statuses.

## Tests

```bash
.venv\Scripts\python -m pytest -q
```

The tests run offline. The agent loop uses a scripted provider. The Claude adapter is tested
through the real SDK against a mocked HTTP transport, which checks request bodies, headers,
thinking-block echo, refusals and error mapping. **No test calls the live API.**

## Known limits (v0.1)

- Each web/CLI submission is a separate run; there is no multi-turn chat inside one run yet.
  Use memory for continuity.
- Tool calls run one at a time, and runs are not resumable after a crash.
- Requests are non-streaming (`max_tokens` 16000 by default).
- Two adapters exist: Anthropic (Claude) and Ollama (local models).
