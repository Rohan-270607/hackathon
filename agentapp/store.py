"""SQLite storage: runs, an append-only audit trail of events, and long-term memory.

Every event payload is passed through `redact()` before it is written.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .redact import redact

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    task         TEXT NOT NULL,
    status       TEXT NOT NULL,
    provider     TEXT NOT NULL,
    model        TEXT NOT NULL,
    created_at   REAL NOT NULL,
    finished_at  REAL,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    report       TEXT
);
CREATE TABLE IF NOT EXISTS events (
    run_id   TEXT NOT NULL REFERENCES runs(id),
    seq      INTEGER NOT NULL,
    ts       REAL NOT NULL,
    type     TEXT NOT NULL,
    payload  TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    tags        TEXT NOT NULL DEFAULT '',
    run_id      TEXT,
    created_at  REAL NOT NULL
);
"""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False, sort_keys=True)


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # One shared connection guarded by a lock: simple and safe for the web server's threads.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._tx() as c:
            if self.path != ":memory:":
                c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- runs -------------------------------------------------------------------------
    def create_run(self, task: str, provider: str, model: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._tx() as c:
            c.execute(
                "INSERT INTO runs (id, task, status, provider, model, created_at) VALUES (?,?,?,?,?,?)",
                (run_id, redact(task), "running", provider, model, time.time()),
            )
        return run_id

    def finish_run(self, run_id: str, status: str, report: dict[str, Any] | None, input_tokens: int, output_tokens: int) -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE runs SET status=?, finished_at=?, report=?, input_tokens=?, output_tokens=? WHERE id=?",
                (status, time.time(), redact(_dumps(report)) if report is not None else None, input_tokens, output_tokens, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._tx() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        run["report"] = json.loads(run["report"]) if run["report"] else None
        return run

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._tx() as c:
            rows = c.execute(
                "SELECT id, task, status, model, created_at, finished_at, input_tokens, output_tokens "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- audit events -----------------------------------------------------------------
    def log_event(self, run_id: str, type_: str, payload: dict[str, Any]) -> int:
        text = redact(_dumps(payload))
        with self._tx() as c:
            row = c.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id=?", (run_id,)).fetchone()
            seq = int(row[0])
            c.execute(
                "INSERT INTO events (run_id, seq, ts, type, payload) VALUES (?,?,?,?,?)",
                (run_id, seq, time.time(), type_, text),
            )
        return seq

    def events(self, run_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        with self._tx() as c:
            rows = c.execute(
                "SELECT seq, ts, type, payload FROM events WHERE run_id=? AND seq>? ORDER BY seq",
                (run_id, after_seq),
            ).fetchall()
        return [{"seq": r["seq"], "ts": r["ts"], "type": r["type"], "payload": json.loads(r["payload"])} for r in rows]

    # ---- memory -----------------------------------------------------------------------
    def add_memory(self, content: str, tags: str = "", run_id: str | None = None) -> int:
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO memories (content, tags, run_id, created_at) VALUES (?,?,?,?)",
                (redact(content), tags, run_id, time.time()),
            )
            return int(cur.lastrowid)

    def search_memory(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        words = [w for w in query.lower().split() if w]
        sql = "SELECT id, content, tags, created_at FROM memories"
        params: list[Any] = []
        if words:
            # Match any word in content or tags; rank by number of matching words.
            score = " + ".join(["(instr(lower(content || ' ' || tags), ?) > 0)"] * len(words))
            params.extend(words)
            sql += f" WHERE ({score}) > 0 ORDER BY ({score}) DESC, created_at DESC"
            params.extend(words)
        else:
            sql += " ORDER BY created_at DESC"
        sql += " LIMIT ?"
        params.append(limit)
        with self._tx() as c:
            rows = c.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        with self._tx() as c:
            cur = c.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            return cur.rowcount > 0
