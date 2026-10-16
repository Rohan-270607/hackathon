"""Command-line interface.

    agent run "task" [--yes] [--max-steps N] [--model M] [--quiet]
    agent runs [--limit N]
    agent show RUN_ID [--events]
    agent memory list | search QUERY | add TEXT [--tags T] | delete ID
    agent web [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from typing import Any

from .agent import Agent
from .config import Settings
from .llm import make_provider
from .llm.base import LLMError, ToolCall
from .safety import Decision, auto_approve
from .store import Store
from .tools.base import Tool

EXIT_OK, EXIT_NOT_COMPLETED, EXIT_ERROR = 0, 1, 2


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def cli_approver(call: ToolCall, tool: Tool) -> Decision:
    if not sys.stdin or not sys.stdin.isatty():
        return Decision(False, "non-interactive session; rerun with --yes to allow", "auto")
    print(f"\n  Approval needed: {tool.name} ({tool.risk})")
    for key, val in call.arguments.items():
        body = val if isinstance(val, str) else json.dumps(val)
        print(textwrap.indent(f"{key}: {body}", "    "))
    try:
        answer = input("  Allow? [y/N] ").strip().lower()
    except EOFError:
        # No one can answer (e.g. stdin is the null device, which Windows reports as a TTY).
        # Record that honestly instead of logging it as a user decision.
        print()
        return Decision(False, "no input available to answer the prompt; rerun with --yes to allow", "auto")
    if answer in ("y", "yes"):
        return Decision(True, "approved at prompt", "user")
    return Decision(False, "declined at prompt", "user")


def print_event(type_: str, p: dict[str, Any]) -> None:
    if type_ == "run_started":
        print(f"[run] model={p['model']} tools={len(p['tools'])}")
    elif type_ == "llm_response":
        if p.get("text"):
            print(f"\n{p['text']}")
    elif type_ == "plan":
        print("\n[plan]")
        for s in p["steps"]:
            print(f"  [{s['status']}] {s['step']}")
    elif type_ == "approval" and not p["approved"]:
        print(f"  x {p['tool']} denied ({p['by']}): {p['reason']}")
    elif type_ == "tool_call":
        print(f"  -> {p['tool']} {_short(p['arguments'], 120)}")
    elif type_ == "tool_result":
        tag = "ok" if p["ok"] else "ERROR"
        print(f"  <- {tag}: {_short(p['output'])}")
    elif type_ == "harness_notice":
        print(f"  [harness] {p['notice']}")
    elif type_ == "args_coerced":
        print(f"  [harness] fixed {p['tool']} argument types: {'; '.join(p['changes'])}")
    elif type_ == "llm_error":
        print(f"[llm error] {p['error']}")


def print_report(status: str, report: dict[str, Any] | None) -> None:
    report = report or {}
    print("\n" + "=" * 60)
    print(f"STATUS: {status}")
    if report.get("summary"):
        print(f"\n{report['summary']}")
    if report.get("evidence"):
        print("\nEvidence:")
        for e in report["evidence"]:
            mark = "ok " if e["verdict"] == "ok" else "!! "
            print(f"  {mark}{e['claim']}  [{e['tool_call_id']}: {e['verdict']}]")
    if report.get("unverified"):
        print("\nNot verified:")
        for u in report["unverified"]:
            print(f"  - {u}")
    if report.get("harness_issues"):
        print("\nHarness check:")
        for i in report["harness_issues"]:
            print(f"  !! {i}")
    if report.get("text") and status != "completed":
        print(f"\nLast model text:\n{report['text']}")
    print("=" * 60)


def cmd_run(args: argparse.Namespace, settings: Settings, store: Store) -> int:
    if args.max_steps:
        settings.max_steps = args.max_steps
    if args.model:
        settings.model = args.model
    try:
        provider = make_provider(settings)
    except LLMError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_ERROR
    agent = Agent(
        provider,
        store,
        settings,
        approver=auto_approve if args.yes else cli_approver,
        on_event=None if args.quiet else print_event,
    )
    started = time.time()
    try:
        result = agent.run(args.task)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR
    print_report(result.status, result.report)
    print(
        f"run {result.run_id} | steps {result.steps} | tokens in {result.usage.input_tokens} "
        f"out {result.usage.output_tokens} | {time.time() - started:.1f}s"
    )
    if result.status == "completed":
        return EXIT_OK
    return EXIT_ERROR if result.status == "error" else EXIT_NOT_COMPLETED


def cmd_runs(args: argparse.Namespace, settings: Settings, store: Store) -> int:
    runs = store.list_runs(args.limit)
    if not runs:
        print("no runs yet")
    for r in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["created_at"]))
        print(f"{r['id']}  {when}  {r['status']:<20} {_short(r['task'], 70)}")
    return EXIT_OK


def cmd_show(args: argparse.Namespace, settings: Settings, store: Store) -> int:
    run = store.get_run(args.run_id)
    if run is None:
        print(f"no run {args.run_id!r}", file=sys.stderr)
        return EXIT_ERROR
    print(f"run {run['id']}  model={run['model']}  status={run['status']}")
    print(f"task: {run['task']}")
    if args.events:
        for e in store.events(run["id"]):
            print(f"{e['seq']:>4} {e['type']:<14} {_short(e['payload'], 200)}")
    print_report(run["status"], run["report"])
    return EXIT_OK


def cmd_memory(args: argparse.Namespace, settings: Settings, store: Store) -> int:
    if args.action == "add":
        print(f"stored memory #{store.add_memory(' '.join(args.text), args.tags or '')}")
    elif args.action == "delete":
        ok = store.delete_memory(int(args.text[0]))
        print("deleted" if ok else "not found")
        return EXIT_OK if ok else EXIT_NOT_COMPLETED
    else:
        query = " ".join(args.text) if args.action == "search" else ""
        rows = store.search_memory(query, limit=50)
        if not rows:
            print("no memories")
        for r in rows:
            print(f"#{r['id']} [{r['tags']}] {r['content']}")
    return EXIT_OK


def cmd_web(args: argparse.Namespace, settings: Settings, store: Store) -> int:
    import uvicorn

    from .web.app import create_app

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: the web UI has no authentication; exposing it beyond localhost lets others run tools on this machine.", file=sys.stderr)
    uvicorn.run(create_app(settings, store), host=args.host, port=args.port, log_level="warning")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent", description="Plan-act-verify-report agent.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run a task")
    r.add_argument("task")
    r.add_argument("--yes", action="store_true", help="approve every tool call the policy marks 'ask'")
    r.add_argument("--max-steps", type=int)
    r.add_argument("--model")
    r.add_argument("--quiet", action="store_true", help="only print the final report")
    r.set_defaults(func=cmd_run)

    ls = sub.add_parser("runs", help="list recent runs")
    ls.add_argument("--limit", type=int, default=20)
    ls.set_defaults(func=cmd_runs)

    sh = sub.add_parser("show", help="show a run's report (and audit trail with --events)")
    sh.add_argument("run_id")
    sh.add_argument("--events", action="store_true")
    sh.set_defaults(func=cmd_show)

    m = sub.add_parser("memory", help="inspect or edit long-term memory")
    m.add_argument("action", choices=["list", "search", "add", "delete"])
    m.add_argument("text", nargs="*")
    m.add_argument("--tags")
    m.set_defaults(func=cmd_memory)

    w = sub.add_parser("web", help="start the local web UI")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.set_defaults(func=cmd_web)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.command == "memory" and args.action in ("search", "add", "delete") and not args.text:
        print(f"memory {args.action} needs an argument", file=sys.stderr)
        return EXIT_ERROR
    try:
        settings = Settings.from_env()
    except ValueError as e:
        print(f"config error: {e}", file=sys.stderr)
        return EXIT_ERROR
    settings.ensure_dirs()
    store = Store(settings.db_path)
    try:
        return args.func(args, settings, store)
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
