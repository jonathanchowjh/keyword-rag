"""The SQLite file: cost ledger, response cache, catalog, questions and jobs.

The jobs table is the work queue: create_job queues, claim_next_job takes the
oldest queued job by insertion order, and the worker reports through
update_job and log_job. Cancel is a flag the running job polls.

One connection per process, opened with check_same_thread=False and in
autocommit mode. Most writes are a single statement, which SQLite's serialized
threading mode makes safe across the API's threadpool and the job threads.
log_job, request_cancel, delete_job, delete_collection_rows and the migration in
connect are several statements and are not atomic; log_job also reads then
writes, so it assumes a single writer (the worker).
"""

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ragqa_core.types import CollectionSpec, KeywordLists, Question

SCHEMA = """
create table if not exists ledger (
  at text not null, model text not null, input_tokens integer not null,
  output_tokens integer not null, usd real not null, batch integer not null,
  purpose text not null, job_id text
);
create table if not exists cache (
  key text primary key, text text not null, input_tokens integer not null,
  output_tokens integer not null
);
create table if not exists collections (
  name text primary key, qa_type text not null, summarize integer not null,
  dense_model text not null, dense_dim integer not null, keyword_version text,
  created_at text not null
);
create table if not exists keyword_lists (
  collection text not null, version text not null, lists_json text not null,
  created_at text not null, primary key (collection, version)
);
create table if not exists questions (
  collection text not null, id text not null, json text not null,
  primary key (collection, id)
);
create table if not exists jobs (
  id text primary key, kind text not null, collection text not null,
  status text not null, progress real not null default 0, setup_json text not null,
  result_json text, error text, created_at text not null, updated_at text not null,
  cancel_requested integer not null default 0, log_json text
);
create table if not exists pending_batches (
  id text primary key, model text not null, purpose text not null, created_at text not null,
  usd real not null default 0, job_id text
);
create table if not exists run_items (
  job_id text not null, method text not null, question_id text not null,
  row_json text not null, complete integer not null default 0
);
create index if not exists run_items_job on run_items (job_id, method);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("pragma table_info(pending_batches)")}
    if "usd" not in columns:  # a 0.1.0 database
        conn.execute("alter table pending_batches add column usd real not null default 0")
    columns = {row["name"] for row in conn.execute("pragma table_info(jobs)")}
    if "cancel_requested" not in columns:  # before jobs were a queue
        conn.execute("alter table jobs add column cancel_requested integer not null default 0")
    if "log_json" not in columns:
        conn.execute("alter table jobs add column log_json text")
    _migrate_run_items(conn)
    return conn


def _migrate_run_items(conn: sqlite3.Connection) -> None:
    """Make run_items resumable and let ledger rows name their job. Idempotent."""
    for table in ("ledger", "pending_batches"):
        columns = {row["name"] for row in conn.execute(f"pragma table_info({table})")}
        if "job_id" not in columns:
            conn.execute(f"alter table {table} add column job_id text")
    columns = {row["name"] for row in conn.execute("pragma table_info(run_items)")}
    if "complete" not in columns:
        conn.execute("alter table run_items add column complete integer not null default 0")
        for row in conn.execute("select id, setup_json from jobs").fetchall():
            if json.loads(row["setup_json"]).get("answer") is False:
                conn.execute("update run_items set complete = 1 where job_id = ?", (row["id"],))
        conn.execute("update run_items set complete = 1 where row_json like '%\"answer\"%'")
    conn.execute(
        "delete from run_items where rowid not in"
        " (select max(rowid) from run_items group by job_id, method, question_id)"
    )
    conn.execute(
        "create unique index if not exists run_items_key on run_items (job_id, method, question_id)"
    )
    for row in conn.execute("select id, setup_json from jobs where kind = 'eval'").fetchall():
        setup = json.loads(row["setup_json"])
        if "question_ids" in setup:
            continue
        ids = conn.execute(
            "select distinct question_id from run_items where job_id = ? order by question_id",
            (row["id"],),
        ).fetchall()
        if ids:
            set_job_setup(conn, row["id"], {**setup, "question_ids": [r[0] for r in ids]})


def save_collection(conn: sqlite3.Connection, spec: CollectionSpec) -> None:
    conn.execute(
        "insert or replace into collections values (?, ?, ?, ?, ?, ?, ?)",
        (
            spec.name,
            spec.qa_type,
            int(spec.summarize),
            spec.dense_model,
            spec.dense_dim,
            spec.keyword_version,
            now(),
        ),
    )


def _spec(row: sqlite3.Row) -> CollectionSpec:
    return CollectionSpec(
        row["name"],
        row["qa_type"],
        bool(row["summarize"]),
        row["dense_model"],
        row["dense_dim"],
        row["keyword_version"],
    )


def get_collection(conn: sqlite3.Connection, name: str) -> CollectionSpec | None:
    row = conn.execute("select * from collections where name = ?", (name,)).fetchone()
    return _spec(row) if row else None


def list_collections(conn: sqlite3.Connection) -> list[CollectionSpec]:
    return [_spec(r) for r in conn.execute("select * from collections order by name")]


def save_keyword_lists(
    conn: sqlite3.Connection, name: str, lists: KeywordLists, version: str
) -> None:
    conn.execute(
        "insert or replace into keyword_lists values (?, ?, ?, ?)",
        (name, version, json.dumps(asdict(lists)), now()),
    )


def set_keyword_version(conn: sqlite3.Connection, name: str, version: str) -> None:
    conn.execute("update collections set keyword_version = ? where name = ?", (version, name))


def get_keyword_lists(
    conn: sqlite3.Connection, name: str, version: str | None = None
) -> KeywordLists | None:
    if version is None:
        spec = get_collection(conn, name)
        version = spec.keyword_version if spec else None
        if version is None:
            return None
    row = conn.execute(
        "select lists_json from keyword_lists where collection = ? and version = ?",
        (name, version),
    ).fetchone()
    return KeywordLists(**json.loads(row["lists_json"])) if row else None


def save_questions(conn: sqlite3.Connection, name: str, questions: list[Question]) -> None:
    conn.executemany(
        "insert or replace into questions values (?, ?, ?)",
        [(name, q.id, json.dumps(asdict(q))) for q in questions],
    )


def get_questions(conn: sqlite3.Connection, name: str) -> list[Question]:
    rows = conn.execute("select json from questions where collection = ? order by id", (name,))
    return [Question(**json.loads(r["json"])) for r in rows]


def create_job(conn: sqlite3.Connection, kind: str, collection: str, setup: dict) -> str:
    job_id = uuid.uuid4().hex
    conn.execute(
        "insert into jobs (id, kind, collection, status, setup_json, created_at, updated_at)"
        " values (?, ?, ?, 'queued', ?, ?, ?)",
        (job_id, kind, collection, json.dumps(setup), now(), now()),
    )
    return job_id


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    status: str | None = None,
    progress: float | None = None,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    sets: list[str] = ["updated_at = ?"]
    args: list[str | float] = [now()]
    for column, value in (
        ("status", status),
        ("progress", progress),
        ("result_json", None if result is None else json.dumps(result)),
        ("error", error),
    ):
        if value is not None:
            sets.append(f"{column} = ?")
            args.append(value)
    conn.execute(f"update jobs set {', '.join(sets)} where id = ?", (*args, job_id))


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("select * from jobs where id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    job["setup"] = json.loads(job.pop("setup_json"))
    raw = job.pop("result_json")
    job["result"] = json.loads(raw) if raw else None
    job["log"] = json.loads(job.pop("log_json") or "[]")
    job["cancel_requested"] = bool(job["cancel_requested"])
    return job


JOBS_RESUMABLE = ("interrupted", "cancelled", "stopped_budget", "stopped_no_key", "failed")


def claim_next_job(conn: sqlite3.Connection) -> dict | None:
    """The oldest queued job by insertion, marked running in the same statement."""
    row = conn.execute(
        "update jobs set status = 'running', updated_at = ? where status = 'queued'"
        " and id = (select id from jobs where status = 'queued' order by rowid limit 1)"
        " returning id",
        (now(),),
    ).fetchone()
    return get_job(conn, row["id"]) if row else None


def list_jobs(
    conn: sqlite3.Connection,
    *,
    kind: str | None = None,
    collection: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> list[dict]:
    clauses, args = [], []
    for column, value in (("kind", kind), ("collection", collection), ("status", status)):
        if value is not None:
            clauses.append(f"{column} = ?")
            args.append(value)
    where = f" where {' and '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"select id from jobs{where} order by rowid desc limit ?", (*args, limit)
    ).fetchall()
    return [job for r in rows if (job := get_job(conn, r["id"])) is not None]


def request_cancel(conn: sqlite3.Connection, job_id: str) -> str:
    """A queued job is cancelled at once; a running one is flagged. Returns the status."""
    conn.execute(
        "update jobs set status = 'cancelled', updated_at = ? where id = ? and status = 'queued'",
        (now(), job_id),
    )
    conn.execute(
        "update jobs set cancel_requested = 1, updated_at = ? where id = ? and status = 'running'",
        (now(), job_id),
    )
    job = get_job(conn, job_id)
    return job["status"] if job else "missing"


def is_cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("select cancel_requested from jobs where id = ?", (job_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def log_job(conn: sqlite3.Connection, job_id: str, message: str) -> None:
    job = get_job(conn, job_id)
    if job is None:
        return
    log = [*job["log"], {"at": now(), "message": message}][-50:]
    conn.execute(
        "update jobs set log_json = ?, updated_at = ? where id = ?",
        (json.dumps(log), now(), job_id),
    )


def set_job_setup(conn: sqlite3.Connection, job_id: str, setup: dict) -> None:
    conn.execute(
        "update jobs set setup_json = ?, updated_at = ? where id = ?",
        (json.dumps(setup), now(), job_id),
    )


def requeue(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "update jobs set status = 'queued', cancel_requested = 0, error = null, updated_at = ?"
        " where id = ?",
        (now(), job_id),
    )


def interrupt_running(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "update jobs set status = 'interrupted', updated_at = ? where status = 'running'",
        (now(),),
    )
    return cursor.rowcount


def running_eval(conn: sqlite3.Connection, collection: str) -> bool:
    return job_running(conn, "eval", collection)


def job_running(conn: sqlite3.Connection, kind: str, collection: str) -> bool:
    row = conn.execute(
        "select 1 from jobs where kind = ? and collection = ? and status = 'running' limit 1",
        (kind, collection),
    ).fetchone()
    return row is not None


def delete_job(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute("delete from run_items where job_id = ?", (job_id,))
    conn.execute("delete from jobs where id = ?", (job_id,))


def delete_collection_rows(conn: sqlite3.Connection, name: str) -> None:
    for row in conn.execute("select id from jobs where collection = ?", (name,)).fetchall():
        delete_job(conn, row["id"])
    conn.execute("delete from questions where collection = ?", (name,))
    conn.execute("delete from keyword_lists where collection = ?", (name,))
    conn.execute("delete from collections where name = ?", (name,))


def upsert_run_item(
    conn: sqlite3.Connection,
    job_id: str,
    method: str,
    question_id: str,
    row: dict,
    complete: bool,
) -> None:
    conn.execute(
        "insert into run_items (job_id, method, question_id, row_json, complete)"
        " values (?, ?, ?, ?, ?)"
        " on conflict (job_id, method, question_id)"
        " do update set row_json = excluded.row_json, complete = excluded.complete",
        (job_id, method, question_id, json.dumps(row), int(complete)),
    )


def complete_question_ids(conn: sqlite3.Connection, job_id: str, method: str) -> set[str]:
    rows = conn.execute(
        "select question_id from run_items where job_id = ? and method = ? and complete = 1",
        (job_id, method),
    )
    return {r["question_id"] for r in rows}


def job_spend(conn: sqlite3.Connection, job_id: str) -> float:
    row = conn.execute("select sum(usd) as usd from ledger where job_id = ?", (job_id,)).fetchone()
    return row["usd"] or 0.0


def run_items(conn: sqlite3.Connection, job_id: str, method: str | None = None) -> list[dict]:
    if method is None:
        rows = conn.execute("select * from run_items where job_id = ? order by rowid", (job_id,))
    else:
        rows = conn.execute(
            "select * from run_items where job_id = ? and method = ? order by rowid",
            (job_id, method),
        )
    return [
        {
            "method": r["method"],
            "question_id": r["question_id"],
            "complete": bool(r["complete"]),
            **json.loads(r["row_json"]),
        }
        for r in rows
    ]
