"""The SQLite file: cost ledger, response cache, catalog, questions and jobs.

One connection per process, opened with check_same_thread=False and in
autocommit mode. Every write is a single statement, which SQLite's serialized
threading mode makes safe across the API's threadpool and the job threads.
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
  purpose text not null
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
  result_json text, error text, created_at text not null, updated_at text not null
);
create table if not exists pending_batches (
  id text primary key, model text not null, purpose text not null, created_at text not null,
  usd real not null default 0
);
create table if not exists run_items (
  job_id text not null, method text not null, question_id text not null,
  row_json text not null
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
    return conn


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
    return job


def active_jobs(conn: sqlite3.Connection, kind: str, collection: str) -> int:
    row = conn.execute(
        "select count(*) from jobs where kind = ? and collection = ?"
        " and status in ('queued', 'running')",
        (kind, collection),
    ).fetchone()
    return row[0]


def mark_interrupted(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "update jobs set status = 'interrupted', updated_at = ?"
        " where status in ('queued', 'running')",
        (now(),),
    )
    return cursor.rowcount


def add_run_item(
    conn: sqlite3.Connection, job_id: str, method: str, question_id: str, row: dict
) -> None:
    conn.execute(
        "insert into run_items values (?, ?, ?, ?)",
        (job_id, method, question_id, json.dumps(row)),
    )


def update_run_item(
    conn: sqlite3.Connection, job_id: str, method: str, question_id: str, row: dict
) -> None:
    conn.execute(
        "update run_items set row_json = ? where job_id = ? and method = ? and question_id = ?",
        (json.dumps(row), job_id, method, question_id),
    )


def run_items(conn: sqlite3.Connection, job_id: str, method: str | None = None) -> list[dict]:
    if method is None:
        rows = conn.execute("select * from run_items where job_id = ? order by rowid", (job_id,))
    else:
        rows = conn.execute(
            "select * from run_items where job_id = ? and method = ? order by rowid",
            (job_id, method),
        )
    return [
        {"method": r["method"], "question_id": r["question_id"], **json.loads(r["row_json"])}
        for r in rows
    ]
