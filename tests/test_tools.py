from __future__ import annotations

import os

import pytest

from agentapp.tools.base import ToolError
from agentapp.tools.builtin import (
    builtin_tools,
    calculator,
    http_get,
    list_files,
    read_file,
    recall,
    remember,
    run_python,
    write_file,
)


def test_calculator_exact_and_functions(ctx):
    assert calculator({"expression": "2**10 + 3*(4-1)"}, ctx) == "1033"
    assert calculator({"expression": "sqrt(16) + factorial(5)"}, ctx) == "124.0"


@pytest.mark.parametrize(
    "expr",
    ["__import__('os').system('echo hi')", "open('x')", "(1).__class__", "9**9**9", "lambda: 1"],
)
def test_calculator_rejects_unsafe_or_huge(ctx, expr):
    with pytest.raises(ToolError):
        calculator({"expression": expr}, ctx)


def test_calculator_division_by_zero(ctx):
    with pytest.raises(ToolError, match="division by zero"):
        calculator({"expression": "1/0"}, ctx)


def test_write_then_read_roundtrip_reports_disk_state(ctx):
    out = write_file({"path": "notes/a.txt", "content": "héllo\n"}, ctx)
    assert "read back from disk" in out and "notes/a.txt" in out
    assert read_file({"path": "notes/a.txt"}, ctx) == "héllo\n"
    write_file({"path": "notes/a.txt", "content": "more\n", "append": True}, ctx)
    assert read_file({"path": "notes/a.txt"}, ctx) == "héllo\nmore\n"
    listing = list_files({"recursive": True}, ctx)
    assert "notes/a.txt" in listing


@pytest.mark.parametrize("bad", ["../escape.txt", "a/../../escape.txt", os.path.abspath(os.sep) + "windows_or_root.txt"])
def test_file_tools_confined_to_workspace(ctx, bad):
    with pytest.raises(ToolError, match="outside the workspace"):
        write_file({"path": bad, "content": "x"}, ctx)
    with pytest.raises(ToolError, match="outside the workspace"):
        read_file({"path": bad}, ctx)


def test_run_python_success_and_failure(ctx):
    ok = run_python({"code": "print(6*7)"}, ctx)
    assert "exit_code: 0" in ok and "42" in ok
    with pytest.raises(ToolError, match="exit_code: 1"):
        run_python({"code": "raise SystemExit(1)"}, ctx)


def test_run_python_runs_in_workspace(ctx):
    write_file({"path": "data.txt", "content": "abc"}, ctx)
    out = run_python({"code": "print(open('data.txt').read())"}, ctx)
    assert "abc" in out


def test_run_python_timeout(ctx):
    with pytest.raises(ToolError, match="timed out"):
        run_python({"code": "import time; time.sleep(5)", "timeout_s": 1}, ctx)


def test_run_python_does_not_see_secrets(ctx, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-secret-value-123456")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    out = run_python(
        {"code": "import os; print(os.environ.get('ANTHROPIC_API_KEY')); print(os.environ.get('HARMLESS_SETTING'))"},
        ctx,
    )
    assert "sk-ant-test" not in out
    assert "None" in out and "visible" in out


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:8000/", "http://localhost/", "http://10.0.0.5/", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://example.com/"],
)
def test_http_get_blocks_non_public_targets(ctx, url):
    with pytest.raises(ToolError):
        http_get({"url": url}, ctx)


def test_memory_remember_and_recall(ctx):
    remember({"content": "User prefers metric units", "tags": "preferences"}, ctx)
    remember({"content": "Project deadline is Friday", "tags": "project"}, ctx)
    out = recall({"query": "metric units"}, ctx)
    assert "metric" in out and "deadline" not in out
    assert "deadline" in recall({"query": ""}, ctx)
    assert recall({"query": "nonexistentword"}, ctx) == "no matching memories"


def test_schemas_validate_arguments():
    tools = {t.name: t for t in builtin_tools()}
    assert tools["write_file"].validation_errors({"path": "a.txt", "content": "x"}) == []
    errs = tools["write_file"].validation_errors({"path": "a.txt"})
    assert any("content" in e for e in errs)
    errs = tools["read_file"].validation_errors({"path": "a.txt", "extra": 1})
    assert any("Additional properties" in e for e in errs)
