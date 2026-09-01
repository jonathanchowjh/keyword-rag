"""The SQLite file: catalog, questions and jobs. The AI connections, ledger,
response cache and pending batches live in the same file, in aiconnect's ai_*
tables, through the AI class's own connection.

The jobs table is the work queue: create_job queues, claim_next_job_id takes the
oldest queued job by insertion order, and the worker reports through
update_job and log_job. Cancel is a flag the running job polls.

One connection per process (spec §5.1), shared by the API's request threads and
the worker thread: a ragqa_core.connection.Connection, in autocommit mode, which
runs each statement under one lock and returns its rows as a list.

The lock covers one statement. log_job, request_cancel, delete_job,
delete_collection_rows and the migration in connect are several statements and
are not atomic (move_ai_tables is one transaction); log_job also reads then
writes, so it assumes a single writer (the worker).
"""

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ragqa_core.connection import Connection
from ragqa_core.types import CollectionSpec, KeywordLists, Question

SCHEMA = """
create table if not exists collections (
  name text primary key, qa_type text not null, summarize integer not null,
  dense_model text not null, dense_dim integer not null, keyword_version text,
  created_at text not null, include_leaked integer not null default 0
);
create table if not exists keyword_lists (
  collection text not null, version text not null, lists_json text not null,
  created_at text not null, primary key (collection, version)
);
create table if not exists disabled_sources (
    collection text not null, source text not null, primary key (collection, source)
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
create table if not exists run_items (
  job_id text not null, method text not null, question_id text not null,
  row_json text not null, complete integer not null default 0
);
create index if not exists run_items_job on run_items (job_id, method);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def first(rows: list[sqlite3.Row]) -> sqlite3.Row | None:
    return rows[0] if rows else None


def connect(path: Path) -> Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = Connection(path)
    conn.executescript(SCHEMA)
    columns = {row["name"] for row in conn.execute("pragma table_info(jobs)")}
    if "cancel_requested" not in columns:  # before jobs were a queue
        conn.execute("alter table jobs add column cancel_requested integer not null default 0")
    if "log_json" not in columns:
        conn.execute("alter table jobs add column log_json text")
    columns = {row["name"] for row in conn.execute("pragma table_info(collections)")}
    if "include_leaked" not in columns:
        conn.execute("alter table collections add column include_leaked integer not null default 0")
    _migrate_run_items(conn)
    return conn


def move_ai_tables(conn: Connection) -> None:
    """Move the ledger, cache and pending batches from before AI connections into the
    ai_* tables AI(path) made, as the 'anthropic' connection's, in one transaction:
    this month's spend still counts and a cached prompt is never paid for twice.
    Once moved, the old tables are gone and this does nothing."""
    with conn.lock:
        conn.execute("begin immediate")
        try:
            if conn.execute("select 1 from sqlite_master where type = 'table' and name = 'ledger'"):
                # A 0.1.0 database has no batch estimate; one from before the queue, no job ids.
                for table, column, kind in (
                    ("ledger", "job_id", "text"),
                    ("pending_batches", "job_id", "text"),
                    ("pending_batches", "usd", "real not null default 0"),
                ):
                    columns = {r["name"] for r in conn.execute(f"pragma table_info({table})")}
                    if column not in columns:
                        conn.execute(f"alter table {table} add column {column} {kind}")
                conn.execute(
                    "insert into ai_ledger (at, model, input_tokens, output_tokens, usd, batch,"
                    " purpose, job_id, connection_id) select at, model, input_tokens,"
                    " output_tokens, usd, batch, purpose, job_id, 'anthropic' from ledger"
                )
                conn.execute(
                    "insert or ignore into ai_cache (key, text, input_tokens, output_tokens)"
                    " select key, text, input_tokens, output_tokens from cache"
                )
                conn.execute(
                    "insert or ignore into ai_batches (id, model, purpose, created_at, usd,"
                    " job_id, connection_id) select id, model, purpose, created_at, usd, job_id,"
                    " 'anthropic' from pending_batches"
                )
                for table in ("ledger", "cache", "pending_batches"):
                    conn.execute(f"drop table {table}")
            conn.execute("commit")
        except BaseException:
            conn.execute("rollback")
            raise


def _migrate_run_items(conn: Connection) -> None:
    """Make run_items resumable. Idempotent."""
    columns = {row["name"] for row in conn.execute("pragma table_info(run_items)")}
    if "complete" not in columns:
        conn.execute("alter table run_items add column complete integer not null default 0")
        for row in conn.execute("select id, setup_json from jobs"):
            if json.loads(row["setup_json"]).get("answer") is False:
                conn.execute("update run_items set complete = 1 where job_id = ?", (row["id"],))
        conn.execute(
            "update run_items set complete = 1 where json_extract(row_json, '$.answer') is not null"
        )
    conn.execute(
        "delete from run_items where rowid not in"
        " (select max(rowid) from run_items group by job_id, method, question_id)"
    )
    conn.execute(
        "create unique index if not exists run_items_key on run_items (job_id, method, question_id)"
    )
    # question_only retrieves nothing: older runs stored 0.0 scores, which read as measured.
    conn.execute(
        "update run_items set row_json = json_set(row_json, '$.retrieval', json('{}'))"
        " where method = 'question_only'"
        " and coalesce(json_extract(row_json, '$.retrieval'), '{}') != '{}'"
    )
    # A queued eval records its own question ids when it runs. Nothing runs at connect
    # time, so a 'running' eval is a leftover the lifespan marks interrupted: backfilled,
    # it is an untracked run that resume refuses rather than mixing versions.
    evals = conn.execute(
        "select id, setup_json from jobs where kind = 'eval' and status != 'queued'"
    )
    for row in evals:
        setup = json.loads(row["setup_json"])
        if "question_ids" in setup:
            continue
        ids = run_item_question_ids(conn, row["id"])
        if ids:
            set_job_setup(conn, row["id"], {**setup, "question_ids": ids})


def save_collection(conn: Connection, spec: CollectionSpec) -> None:
    conn.execute(
        "insert into collections"
        " (name, qa_type, summarize, dense_model, dense_dim, keyword_version, created_at)"
        " values (?, ?, ?, ?, ?, ?, ?)"
        " on conflict(name) do update set qa_type = excluded.qa_type,"
        " summarize = excluded.summarize, dense_model = excluded.dense_model,"
        " dense_dim = excluded.dense_dim, keyword_version = excluded.keyword_version,"
        " created_at = excluded.created_at",
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


def get_collection(conn: Connection, name: str) -> CollectionSpec | None:
    row = first(conn.execute("select * from collections where name = ?", (name,)))
    return _spec(row) if row else None


def get_include_leaked(conn: Connection, name: str) -> bool:
    row = first(conn.execute("select include_leaked from collections where name = ?", (name,)))
    return bool(row["include_leaked"]) if row else False


def set_include_leaked(conn: Connection, name: str, include: bool) -> None:
    conn.execute("update collections set include_leaked = ? where name = ?", (int(include), name))


def list_collections(conn: Connection) -> list[CollectionSpec]:
    return [_spec(r) for r in conn.execute("select * from collections order by name")]


def save_keyword_lists(conn: Connection, name: str, lists: KeywordLists, version: str) -> None:
    conn.execute(
        "insert or replace into keyword_lists values (?, ?, ?, ?)",
        (name, version, json.dumps(asdict(lists)), now()),
    )


def set_keyword_version(conn: Connection, name: str, version: str) -> None:
    conn.execute("update collections set keyword_version = ? where name = ?", (version, name))


def get_keyword_lists(
    conn: Connection, name: str, version: str | None = None
) -> KeywordLists | None:
    if version is None:
        spec = get_collection(conn, name)
        version = spec.keyword_version if spec else None
        if version is None:
            return None
    row = first(
        conn.execute(
            "select lists_json from keyword_lists where collection = ? and version = ?",
            (name, version),
        )
    )
    return KeywordLists(**json.loads(row["lists_json"])) if row else None


def list_keyword_versions(conn: Connection, name: str) -> list[dict]:
    """Every saved version of a collection's lists, newest first."""
    rows = conn.execute(
        "select version, created_at, lists_json from keyword_lists where collection = ?"
        " order by created_at desc, rowid desc",
        (name,),
    )
    return [_keyword_version(r) for r in rows]


def get_keyword_version(conn: Connection, name: str, version: str) -> dict | None:
    row = first(
        conn.execute(
            "select version, created_at, lists_json from keyword_lists"
            " where collection = ? and version = ?",
            (name, version),
        )
    )
    return _keyword_version(row) if row else None


def _keyword_version(row: sqlite3.Row) -> dict:
    return {
        "version": row["version"],
        "created_at": row["created_at"],
        "lists": KeywordLists(**json.loads(row["lists_json"])),
    }


def save_questions(conn: Connection, name: str, questions: list[Question]) -> None:
    conn.executemany(
        "insert or replace into questions values (?, ?, ?)",
        [(name, q.id, json.dumps(asdict(q))) for q in questions],
    )


def get_questions(conn: Connection, name: str) -> list[Question]:
    rows = conn.execute("select json from questions where collection = ? order by id", (name,))
    return [Question(**json.loads(r["json"])) for r in rows]


def create_job(conn: Connection, kind: str, collection: str, setup: dict) -> str:
    job_id = uuid.uuid4().hex
    conn.execute(
        "insert into jobs (id, kind, collection, status, setup_json, created_at, updated_at)"
        " values (?, ?, ?, 'queued', ?, ?, ?)",
        (job_id, kind, collection, json.dumps(setup), now(), now()),
    )
    return job_id


def loaded_datasets(conn: Connection, name: str) -> set[str]:
    """Every dataset a load job of the collection named, whatever its status: a failed load
    still added items."""
    rows = conn.execute(
        "select setup_json from jobs where kind = 'load' and collection = ?", (name,)
    )
    return {d for r in rows if (d := json.loads(r["setup_json"]).get("dataset"))}


def update_job(
    conn: Connection,
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


def get_job(conn: Connection, job_id: str) -> dict | None:
    row = first(conn.execute("select * from jobs where id = ?", (job_id,)))
    return None if row is None else _job(row)


def _job(row: sqlite3.Row) -> dict:
    job = dict(row)
    job["setup"] = json.loads(job.pop("setup_json"))
    raw = job.pop("result_json")
    job["result"] = json.loads(raw) if raw else None
    job["log"] = json.loads(job.pop("log_json") or "[]")
    job["cancel_requested"] = bool(job["cancel_requested"])
    return job


def newest_jobs(conn: Connection) -> list[dict]:
    """The newest job of each (collection, kind), newest first, in one query."""
    rows = conn.execute(
        "select * from jobs where rowid in"
        " (select max(rowid) from jobs group by collection, kind) order by rowid desc"
    )
    return [_job(r) for r in rows]


JOBS_RESUMABLE = ("interrupted", "cancelled", "stopped_budget", "stopped_no_key", "failed")


def requeue(conn: Connection, job_id: str) -> bool:
    """Queues a resumable job again; False when it was not resumable (already moved)."""
    marks = ", ".join("?" * len(JOBS_RESUMABLE))
    with conn.lock:  # no other statement between the update and changes()
        conn.execute(
            "update jobs set status = 'queued', cancel_requested = 0, error = null,"
            f" updated_at = ? where id = ? and status in ({marks})",
            (now(), job_id, *JOBS_RESUMABLE),
        )
        return conn.changes() > 0


def claim_next_job_id(conn: Connection) -> str | None:
    """The oldest queued job by insertion, marked running in the same statement."""
    row = first(
        conn.execute(
            "update jobs set status = 'running', updated_at = ? where status = 'queued'"
            " and id = (select id from jobs where status = 'queued' order by rowid limit 1)"
            " returning id",
            (now(),),
        )
    )
    return row["id"] if row else None


ALL = 1_000_000  # list_jobs limit meaning "every job"


def list_jobs(
    conn: Connection,
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
    rows = conn.execute(f"select * from jobs{where} order by rowid desc limit ?", (*args, limit))
    return [_job(r) for r in rows]


def request_cancel(conn: Connection, job_id: str) -> str:
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


def is_cancel_requested(conn: Connection, job_id: str) -> bool:
    row = first(conn.execute("select cancel_requested from jobs where id = ?", (job_id,)))
    return bool(row and row["cancel_requested"])


def log_job(conn: Connection, job_id: str, message: str) -> None:
    job = get_job(conn, job_id)
    if job is None:
        return
    log = [*job["log"], {"at": now(), "message": message}][-50:]
    conn.execute(
        "update jobs set log_json = ?, updated_at = ? where id = ?",
        (json.dumps(log), now(), job_id),
    )


def set_job_setup(conn: Connection, job_id: str, setup: dict) -> None:
    conn.execute(
        "update jobs set setup_json = ?, updated_at = ? where id = ?",
        (json.dumps(setup), now(), job_id),
    )


def interrupt_running(conn: Connection) -> int:
    with conn.lock:  # no other statement between the update and changes()
        conn.execute(
            "update jobs set status = 'interrupted', updated_at = ? where status = 'running'",
            (now(),),
        )
        return conn.changes()


def job_running(conn: Connection, kind: str, collection: str) -> bool:
    row = first(
        conn.execute(
            "select 1 from jobs where kind = ? and collection = ? and status = 'running' limit 1",
            (kind, collection),
        )
    )
    return row is not None


def job_waiting_or_running(conn: Connection, kind: str, collection: str) -> bool:
    row = first(
        conn.execute(
            "select 1 from jobs where kind = ? and collection = ?"
            " and status in ('queued', 'running') limit 1",
            (kind, collection),
        )
    )
    return row is not None


def delete_job(conn: Connection, job_id: str) -> None:
    conn.execute("delete from run_items where job_id = ?", (job_id,))
    conn.execute("delete from jobs where id = ?", (job_id,))


def disabled_sources(conn: Connection, name: str) -> list[str]:
    rows = conn.execute(
        "select source from disabled_sources where collection = ? order by source", (name,)
    )
    return [row["source"] for row in rows]


def set_source_enabled(conn: Connection, name: str, source: str, enabled: bool) -> None:
    if enabled:
        conn.execute(
            "delete from disabled_sources where collection = ? and source = ?", (name, source)
        )
    else:
        conn.execute(
            "insert or ignore into disabled_sources (collection, source) values (?, ?)",
            (name, source),
        )


def question_sources(conn: Connection, name: str) -> dict[str, int]:
    """How many of the collection's questions carry each meta.source."""
    rows = conn.execute(
        "select json_extract(json, '$.meta.source') as source, count(*) as n from questions"
        " where collection = ? and source is not null group by source",
        (name,),
    )
    return {row["source"]: row["n"] for row in rows}


def delete_source_questions(conn: Connection, name: str, source: str) -> None:
    conn.execute(
        "delete from questions where collection = ? and json_extract(json, '$.meta.source') = ?",
        (name, source),
    )


def delete_collection_rows(conn: Connection, name: str) -> None:
    for row in conn.execute("select id from jobs where collection = ?", (name,)):
        delete_job(conn, row["id"])
    conn.execute("delete from questions where collection = ?", (name,))
    conn.execute("delete from disabled_sources where collection = ?", (name,))
    conn.execute("delete from keyword_lists where collection = ?", (name,))
    conn.execute("delete from collections where name = ?", (name,))


def upsert_run_item(
    conn: Connection,
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


def complete_question_ids(conn: Connection, job_id: str, method: str) -> set[str]:
    rows = conn.execute(
        "select question_id from run_items where job_id = ? and method = ? and complete = 1",
        (job_id, method),
    )
    return {r["question_id"] for r in rows}


def run_item_question_ids(conn: Connection, job_id: str) -> list[str]:
    """The questions a run has rows for, sorted: an old run's question ids."""
    rows = conn.execute(
        "select distinct question_id from run_items where job_id = ? order by question_id",
        (job_id,),
    )
    return [r["question_id"] for r in rows]


def run_items(conn: Connection, job_id: str, method: str | None = None) -> list[dict]:
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
