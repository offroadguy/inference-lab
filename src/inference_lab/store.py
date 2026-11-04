import contextlib
import fcntl
import json
import os
import re
from pathlib import Path
import sqlite3
import time
import uuid


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.getenv("LAB_DATA") or "data").resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(self.root / "lab.sqlite", timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                run_id TEXT, request_id TEXT, body TEXT NOT NULL, PRIMARY KEY(run_id, request_id));
            CREATE TABLE IF NOT EXISTS events (run_id TEXT, ts REAL, body TEXT);
        ''')

    @contextlib.contextmanager
    def lock(self, run_id):
        if not re.fullmatch(r"[0-9a-f]{12}", run_id):
            raise ValueError("Invalid run ID")
        with (self.root / f"{run_id}.lock").open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another CLI process owns this run") from None
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def create(self, plan):
        run = {"id": uuid.uuid4().hex[:12], "plan": plan.to_dict(), "created_at": time.time(),
               "status": "planned", "approval": None, "resource_id": None, "deploy_intent": False,
               "shutdown": "not_started", "stages": [], "segments": [], "telemetry": []}
        self.save(run)
        return run

    def save(self, run):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?)", (run["id"], json.dumps(run)))

    def get(self, run_id):
        row = self.db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise ValueError("Unknown run")
        return json.loads(row[0])

    def runs(self):
        return [json.loads(r[0]) for r in self.db.execute("SELECT body FROM runs ORDER BY rowid DESC")]

    def request(self, run_id, record):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO requests VALUES (?,?,?)",
                            (run_id, record["id"], json.dumps(record)))

    def requests(self, run_id):
        return [json.loads(r[0]) for r in self.db.execute(
            "SELECT body FROM requests WHERE run_id=? ORDER BY request_id", (run_id,))]

    def event(self, run_id, kind, **fields):
        body = {"kind": kind, **fields}
        with self.db:
            self.db.execute("INSERT INTO events VALUES (?,?,?)", (run_id, time.time(), json.dumps(body)))

    def events(self, run_id):
        return [{"ts": ts, **json.loads(body)} for ts, body in self.db.execute(
            "SELECT ts,body FROM events WHERE run_id=? ORDER BY ts", (run_id,))]

    def recover(self, run_id):
        for r in self.requests(run_id):
            if r["status"] == "running":
                r.update(status="incomplete", error="local_interruption", ended_at=time.time())
                self.request(run_id, r)
