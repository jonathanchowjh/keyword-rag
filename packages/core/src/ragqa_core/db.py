"""The SQLite file: catalog, eval sets (questions, with versioned snapshots) and jobs.
The AI connections, ledger, response cache and pending batches live in the same file,
in aiconnect's ai_* tables, through the AI class's own connection.

The jobs table is the work queue: create_job queues, claim_next_job_id takes the
oldest queued job by insertion order, and the worker reports through
update_job and log_job. Cancel is a flag the running job polls.

One connection per process (spec §5.1), shared by the API's request threads and
the worker thread: a ragqa_core.connection.Connection, in autocommit mode, which
runs each statement under one lock and returns its rows as a list.

The lock covers one statement. log_job, request_cancel, delete_job,
delete_collection_rows and the migrations in connect are several statements and
are not atomic (move_ai_tables, _drop_routing, replace_questions and each collection's
step of the eval set migration are one transaction each). The eval set writes
(save_questions, replace_questions, delete_eval_set) hold the lock for the whole write;
log_job also reads then writes, so it assumes a single writer (the worker). So do the chat
writes, which number a chat's messages and a message's attempts inside the lock.
"""

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from ragqa_core.connection import Connection
from ragqa_core.recipes import build_for, corpus_type_for, default_search
from ragqa_core.types import (
    BuildSettings,
    CollectionSpec,
    KeywordLists,
    QaType,
    Question,
    SearchSettings,
)

NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # a collection's or an eval set's name

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
create table if not exists vocabularies (
  collection text not null, version text not null, terms_json text not null,
  created_at text not null, primary key (collection, version)
);
create table if not exists disabled_sources (
    collection text not null, source text not null, primary key (collection, source)
);
create table if not exists questions (
  collection text not null, id text not null, json text not null,
  primary key (collection, id)
);
create table if not exists eval_sets (
  name text primary key, qa_type text not null, version text,
  include_leaked integer not null default 0, created_at text not null
);
create table if not exists eval_set_versions (
  eval_set text not null, version text not null, questions_json text not null,
  created_at text not null, primary key (eval_set, version)
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
create table if not exists chats (
  id text primary key, title text not null, stores_json text not null,
  settings_json text not null, created_at text not null, updated_at text not null
);
create table if not exists chat_messages (
  chat_id text not null, n integer not null, content text not null,
  created_at text not null, primary key (chat_id, n)
);
create table if not exists chat_attempts (
  chat_id text not null, n integer not null, attempt integer not null,
  search_text text not null, decisions_json text not null, evidence_json text not null,
  answer text, citations_json text not null, usd real not null, error text, model text,
  created_at text not null, search_error text, pipeline_json text,
  primary key (chat_id, n, attempt)
);
create table if not exists app_settings (key text primary key, value_json text not null);
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
    for column in NEW_COLLECTION_COLUMNS:
        if column not in columns:
            conn.execute(f"alter table collections add column {column} text")
    columns = {row["name"] for row in conn.execute("pragma table_info(chat_attempts)")}
    if "search_error" not in columns:  # before an answer kept a failed rewrite's note
        conn.execute("alter table chat_attempts add column search_error text")
    if "pipeline_json" not in columns:  # before the pool and rerank were recorded
        conn.execute("alter table chat_attempts add column pipeline_json text")
    _drop_routing(conn)  # after the columns: it clears chat_k
    _migrate_collections(conn)
    _migrate_run_items(conn)
    _migrate_eval_sets(conn)
    return conn


NEW_COLLECTION_COLUMNS = (
    "corpus_type",
    "recipe",
    "build_json",
    "search_json",
    "vocabulary_version",
    "profile_json",
    "eval_set",
    "chat_method",
    "chat_k",
)  # old databases keep the two columns, unused


ROUTING_DROPPED = "schema.routing_dropped"  # app_settings key: this database is current


def _drop_routing(conn: Connection) -> None:
    """A database from before chat searched every store: its bars go, and with them the
    chat setting that picked stores by profile ("routing"); a chat keeps every other key.
    An evidence size of 8 and rewrite off were the built-in values, so they become null,
    what is in force. Each store's chat_k is cleared, since it now means candidates per
    store. One transaction, which also writes the ROUTING_DROPPED marker; a fresh or
    migrated database with no store_bars gets the marker too. With the marker, store_bars
    made again (an older build run on this database) only drops it and any "routing" key:
    a chat's and a store's values keep their meaning."""
    has_bars = "select 1 from sqlite_master where type = 'table' and name = 'store_bars'"
    marked = "select 1 from app_settings where key = ?"
    if not conn.execute(has_bars):
        if not conn.execute(marked, (ROUTING_DROPPED,)):
            set_app_setting(conn, ROUTING_DROPPED, True)
        return
    with conn.lock:
        conn.execute("begin immediate")
        try:
            if not conn.execute(has_bars):  # another process migrated it first
                conn.execute("commit")
                return
            done = bool(conn.execute(marked, (ROUTING_DROPPED,)))
            for row in conn.execute("select id, settings_json from chats"):
                settings = json.loads(row["settings_json"])
                settings.pop("routing", None)
                if not done:
                    size, rewrite = settings.get("evidence_size"), settings.get("rewrite")
                    settings["evidence_size"] = None if size == 8 else size
                    settings["rewrite"] = None if rewrite is False else rewrite
                conn.execute(
                    "update chats set settings_json = ? where id = ?",
                    (json.dumps(settings), row["id"]),
                )
            if not done:
                conn.execute("update collections set chat_k = null")
                set_app_setting(conn, ROUTING_DROPPED, True)
            conn.execute("drop table store_bars")
            conn.execute("commit")
        except BaseException:
            conn.execute("rollback")
            raise


def _migrate_collections(conn: Connection) -> None:
    """Collections from before recipes become "paper 2024", with their old summarize flag.
    Program answers stay off: they never ran model-written code, and a migration must not
    start to."""
    for row in conn.execute("select * from collections where recipe is null"):
        qa_type = row["qa_type"]
        build = build_for("paper_2024", bool(row["summarize"]))
        search = replace(default_search(qa_type, build), program=False)
        conn.execute(
            "update collections set corpus_type = ?, recipe = ?, build_json = ?,"
            " search_json = ?, profile_json = ? where name = ?",
            (
                corpus_type_for(qa_type),
                "paper_2024",
                json.dumps(asdict(build)),
                json.dumps(asdict(search)),
                "{}",
                row["name"],
            ),
        )


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


def _migrate_eval_sets(conn: Connection) -> None:
    """Questions used to belong to a collection. Each such collection gets an eval set of its
    name, and its runs are pinned to that set's first snapshot: their questions stay as they
    were when the set is edited. Each collection's step is one transaction; runs a step left
    unpinned (a migration from before it was one) are pinned to the set's first snapshot."""
    rows = conn.execute(
        "select name, qa_type, include_leaked from collections where name in"
        " (select collection from questions) or name in (select name from eval_sets)"
    )
    for row in rows:
        name = row["name"]
        with conn.lock:
            conn.execute("begin immediate")
            try:
                if get_eval_set(conn, name) is None:
                    create_eval_set(conn, name, row["qa_type"])
                    conn.execute(
                        "update eval_sets set include_leaked = ? where name = ?",
                        (row["include_leaked"], name),
                    )
                    snapshot_eval_set(conn, name)
                _pin_old_runs(conn, name)
                conn.execute("commit")
            except BaseException:
                conn.execute("rollback")
                raise


def _pin_old_runs(conn: Connection, name: str) -> None:
    """Every eval on the collection that records no eval set gets the set of its name, at that
    set's first snapshot."""
    first_version = first(
        conn.execute(
            "select version from eval_set_versions where eval_set = ? order by rowid limit 1",
            (name,),
        )
    )
    if first_version is None:
        return
    version = first_version["version"]
    for job in conn.execute(
        "select id, setup_json from jobs where kind = 'eval' and collection = ?", (name,)
    ):
        setup = json.loads(job["setup_json"])
        if "eval_set" not in setup and "eval_set_version" not in setup:
            set_job_setup(conn, job["id"], {**setup, "eval_set": name, "eval_set_version": version})


def save_collection(conn: Connection, spec: CollectionSpec) -> None:
    build = spec.build or build_for("paper_2024", spec.summarize)
    search = spec.search or default_search(spec.qa_type, build)
    conn.execute(
        "insert into collections"
        " (name, qa_type, summarize, dense_model, dense_dim, keyword_version, created_at,"
        " corpus_type, recipe, build_json, search_json, vocabulary_version, profile_json)"
        " values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " on conflict(name) do update set qa_type = excluded.qa_type,"
        " summarize = excluded.summarize, dense_model = excluded.dense_model,"
        " dense_dim = excluded.dense_dim, keyword_version = excluded.keyword_version,"
        " created_at = excluded.created_at, corpus_type = excluded.corpus_type,"
        " recipe = excluded.recipe, build_json = excluded.build_json,"
        " search_json = excluded.search_json,"
        " vocabulary_version = excluded.vocabulary_version,"
        " profile_json = excluded.profile_json",
        (
            spec.name,
            spec.qa_type,
            int(spec.summarize),
            spec.dense_model,
            spec.dense_dim,
            spec.keyword_version,
            now(),
            spec.corpus_type,
            spec.recipe,
            json.dumps(asdict(build)),
            json.dumps(asdict(search)),
            spec.vocabulary_version,
            json.dumps(spec.profile),
        ),
    )


def set_recipe(conn: Connection, name: str, recipe: str, build: BuildSettings) -> None:
    """Switches the store's recipe and build in place. Its search stays: the Documents
    recipes share one default, and resetting it would lose a saved override."""
    with conn.lock:
        conn.execute(
            "update collections set recipe = ?, build_json = ? where name = ?",
            (recipe, json.dumps(asdict(build)), name),
        )


def set_search(conn: Connection, name: str, search: SearchSettings) -> None:
    conn.execute(
        "update collections set search_json = ? where name = ?", (json.dumps(asdict(search)), name)
    )


def _spec(row: sqlite3.Row) -> CollectionSpec:
    return CollectionSpec(
        row["name"],
        row["qa_type"],
        bool(row["summarize"]),
        row["dense_model"],
        row["dense_dim"],
        row["keyword_version"],
        row["corpus_type"],
        row["recipe"],
        BuildSettings(**json.loads(row["build_json"])),
        SearchSettings(**json.loads(row["search_json"])),
        row["vocabulary_version"],
        json.loads(row["profile_json"] or "{}"),
    )


def get_collection(conn: Connection, name: str) -> CollectionSpec | None:
    row = first(conn.execute("select * from collections where name = ?", (name,)))
    return _spec(row) if row else None


def get_include_leaked(conn: Connection, name: str) -> bool:
    eval_set = get_eval_set(conn, name)
    return eval_set["include_leaked"] if eval_set else False


def set_include_leaked(conn: Connection, name: str, include: bool) -> None:
    with conn.lock:
        _ensure_for_write(conn, name)
        conn.execute("update eval_sets set include_leaked = ? where name = ?", (int(include), name))


def list_collections(conn: Connection) -> list[CollectionSpec]:
    return [_spec(r) for r in conn.execute("select * from collections order by name")]


def save_keyword_lists(conn: Connection, name: str, lists: KeywordLists, version: str) -> None:
    conn.execute(
        "insert or replace into keyword_lists values (?, ?, ?, ?)",
        (name, version, json.dumps(asdict(lists)), now()),
    )


def set_keyword_version(conn: Connection, name: str, version: str) -> None:
    conn.execute("update collections set keyword_version = ? where name = ?", (version, name))


def set_dense_model(conn: Connection, name: str, model: str, dim: int) -> None:
    conn.execute(
        "update collections set dense_model = ?, dense_dim = ? where name = ?", (model, dim, name)
    )


def switching_model(conn: Connection, name: str) -> str | None:
    """The model the collection's newest rebuild is switching to while that switch is
    unfinished, else None."""
    row = first(
        conn.execute(
            "select status, setup_json from jobs where kind = 'reembed' and collection = ?"
            " order by rowid desc limit 1",
            (name,),
        )
    )
    if row is None or row["status"] == "done":
        return None
    setup = json.loads(row["setup_json"])
    return setup.get("model") if setup.get("switching") else None


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


def save_vocabulary(conn: Connection, name: str, terms: list[str], version: str) -> None:
    conn.execute(
        "insert or replace into vocabularies values (?, ?, ?, ?)",
        (name, version, json.dumps(terms), now()),
    )


def set_vocabulary_version(conn: Connection, name: str, version: str) -> None:
    conn.execute("update collections set vocabulary_version = ? where name = ?", (version, name))


def set_profile(conn: Connection, name: str, profile: dict[str, int]) -> None:
    conn.execute(
        "update collections set profile_json = ? where name = ?", (json.dumps(profile), name)
    )


def get_vocabulary(conn: Connection, name: str, version: str | None = None) -> list[str] | None:
    """A saved version's terms; with no version, the one in force (None when there is none)."""
    if version is None:
        spec = get_collection(conn, name)
        version = spec.vocabulary_version if spec else None
        if version is None:
            return None
    row = first(
        conn.execute(
            "select terms_json from vocabularies where collection = ? and version = ?",
            (name, version),
        )
    )
    return json.loads(row["terms_json"]) if row else None


def list_vocabularies(conn: Connection, name: str) -> list[dict]:
    """Every saved version of a collection's vocabulary, newest first."""
    spec = get_collection(conn, name)
    active = spec.vocabulary_version if spec else None
    rows = conn.execute(
        "select version, created_at, terms_json from vocabularies where collection = ?"
        " order by created_at desc, rowid desc",
        (name,),
    )
    return [
        {
            "version": r["version"],
            "created_at": r["created_at"],
            "count": len(json.loads(r["terms_json"])),
            "active": r["version"] == active,
        }
        for r in rows
    ]


def create_eval_set(conn: Connection, name: str, qa_type: QaType) -> None:
    if not NAME.match(name):
        raise ValueError(f"eval set name {name!r} must match {NAME.pattern}")
    with conn.lock:
        if get_eval_set(conn, name):
            raise ValueError(f"eval set '{name}' already exists")
        conn.execute(
            "insert into eval_sets (name, qa_type, created_at) values (?, ?, ?)",
            (name, qa_type, now()),
        )


def ensure_eval_set(conn: Connection, name: str, qa_type: QaType) -> None:
    with conn.lock:
        old = get_eval_set(conn, name)
        if old is None:
            create_eval_set(conn, name, qa_type)
        elif old["qa_type"] != qa_type:
            raise ValueError(f"eval set '{name}' is {old['qa_type']}, not {qa_type}")


def _ensure_for_write(conn: Connection, name: str) -> None:
    """A write to an eval set that does not exist yet creates it with the qa_type of the
    collection of that name, so a store's questions can be saved under the store's name."""
    if get_eval_set(conn, name):
        return
    spec = get_collection(conn, name)
    if spec is None:
        raise ValueError(f"no eval set or collection named '{name}'")
    create_eval_set(conn, name, spec.qa_type)


def _eval_set(row: sqlite3.Row, questions: int) -> dict:
    return {
        "name": row["name"],
        "qa_type": row["qa_type"],
        "version": row["version"],
        "include_leaked": bool(row["include_leaked"]),
        "created_at": row["created_at"],
        "questions": questions,
    }


EVAL_SET_SQL = (
    "select e.*, (select count(*) from questions q where q.collection = e.name) as n"
    " from eval_sets e"
)


def get_eval_set(conn: Connection, name: str) -> dict | None:
    row = first(conn.execute(f"{EVAL_SET_SQL} where e.name = ?", (name,)))
    return None if row is None else _eval_set(row, row["n"])


def list_eval_sets(conn: Connection) -> list[dict]:
    return [_eval_set(r, r["n"]) for r in conn.execute(f"{EVAL_SET_SQL} order by e.name")]


def eval_set_ids_ever(conn: Connection, name: str) -> set[str]:
    """Every question id any saved version of the set has held, read in one query."""
    rows = conn.execute(
        "select distinct json_extract(value, '$.id') as id"
        " from eval_set_versions, json_each(questions_json) where eval_set = ?",
        (name,),
    )
    return {r["id"] for r in rows}


def eval_set_versions(conn: Connection, name: str) -> list[dict]:
    rows = conn.execute(
        "select version, created_at, questions_json from eval_set_versions"
        " where eval_set = ? order by rowid desc",
        (name,),
    )
    return [
        {
            "version": r["version"],
            "created_at": r["created_at"],
            "questions": len(json.loads(r["questions_json"])),
        }
        for r in rows
    ]


def save_questions(conn: Connection, name: str, questions: list[Question]) -> None:
    with conn.lock:
        _ensure_for_write(conn, name)
        conn.executemany(
            "insert or replace into questions values (?, ?, ?)",
            [(name, q.id, json.dumps(asdict(q))) for q in questions],
        )
        snapshot_eval_set(conn, name)


def snapshot_eval_set(conn: Connection, name: str) -> str:
    """Saves the eval set's questions as a version when that content is new, makes it the
    current one and returns it. The same content always has the same version."""
    with conn.lock:
        rows = conn.execute("select json from questions where collection = ? order by id", (name,))
        text = json.dumps([json.loads(r["json"]) for r in rows], sort_keys=True)
        version = hashlib.sha256(text.encode()).hexdigest()[:12]
        conn.execute(
            "insert or ignore into eval_set_versions values (?, ?, ?, ?)",
            (name, version, text, now()),
        )
        conn.execute("update eval_sets set version = ? where name = ?", (version, name))
        return version


def replace_questions(
    conn: Connection, name: str, questions: list[Question], base_version: str | None
) -> str:
    """Makes the eval set's questions exactly `questions`; refuses when `base_version` is
    given and the set has moved on since it was read."""
    with conn.lock:
        _ensure_for_write(conn, name)
        current = (get_eval_set(conn, name) or {})["version"]
        if base_version is not None and base_version != current:
            raise ValueError(f"eval set '{name}' changed since it was read (now {current})")
        ids = [q.id for q in questions]
        if repeated := next((i for i in ids if ids.count(i) > 1), None):
            raise ValueError(f"question id '{repeated}' appears more than once")
        conn.execute("begin immediate")  # all or nothing: a failed insert keeps the old ones
        try:
            conn.execute("delete from questions where collection = ?", (name,))
            conn.executemany(
                "insert into questions values (?, ?, ?)",
                [(name, q.id, json.dumps(asdict(q))) for q in questions],
            )
            version = snapshot_eval_set(conn, name)
            conn.execute("commit")
        except BaseException:
            conn.execute("rollback")
            raise
        return version


def get_questions(conn: Connection, name: str, version: str | None = None) -> list[Question]:
    if version is None:
        rows = conn.execute("select json from questions where collection = ? order by id", (name,))
        return [Question(**json.loads(r["json"])) for r in rows]
    row = first(
        conn.execute(
            "select questions_json from eval_set_versions where eval_set = ? and version = ?",
            (name, version),
        )
    )
    if row is None:
        raise KeyError(version)
    return [Question(**q) for q in json.loads(row["questions_json"])]


def _chat_column(conn: Connection, name: str, column: str) -> str | None:
    row = first(conn.execute(f"select {column} from collections where name = ?", (name,)))
    return row[column] if row else None


def _set_chat_column(conn: Connection, name: str, column: str, value: object) -> None:
    with conn.lock:
        conn.execute(
            f"update collections set {column} = ? where name = ?",
            (None if value is None else str(value), name),
        )


def get_chat_method(conn: Connection, name: str) -> str | None:
    return _chat_column(conn, name, "chat_method")


def set_chat_method(conn: Connection, name: str, method: str | None) -> None:
    _set_chat_column(conn, name, "chat_method", method)


def get_chat_k(conn: Connection, name: str) -> int | None:
    value = _chat_column(conn, name, "chat_k")
    return None if value is None else int(value)


def set_chat_k(conn: Connection, name: str, k: int | None) -> None:
    _set_chat_column(conn, name, "chat_k", k)


def get_app_settings(conn: Connection, prefix: str) -> dict[str, object]:
    """The app-wide settings whose key starts with `prefix`, keyed without it."""
    rows = conn.execute(
        "select key, value_json from app_settings where substr(key, 1, ?) = ?",
        (len(prefix), prefix),
    )
    return {row["key"][len(prefix) :]: json.loads(row["value_json"]) for row in rows}


def set_app_setting(conn: Connection, key: str, value: object | None) -> None:
    """None deletes the setting."""
    with conn.lock:
        if value is None:
            conn.execute("delete from app_settings where key = ?", (key,))
        else:
            conn.execute(
                "insert or replace into app_settings (key, value_json) values (?, ?)",
                (key, json.dumps(value)),
            )


def default_eval_set(conn: Connection, collection: str) -> str | None:
    """The set the store names, else the set named after it, else none."""
    row = first(conn.execute("select eval_set from collections where name = ?", (collection,)))
    if row and row["eval_set"]:
        return row["eval_set"]
    return collection if get_eval_set(conn, collection) else None


def set_default_eval_set(conn: Connection, collection: str, eval_set: str | None) -> None:
    conn.execute("update collections set eval_set = ? where name = ?", (eval_set, collection))


def delete_eval_set(conn: Connection, name: str) -> int:
    """Deletes the set with its versions and the runs made on it; stores that pointed to it
    point to nothing. Returns the number of runs deleted."""
    with conn.lock:
        runs = [
            r["id"]
            for r in conn.execute("select id, setup_json from jobs where kind = 'eval'")
            if json.loads(r["setup_json"]).get("eval_set") == name
        ]
        for job_id in runs:
            delete_job(conn, job_id)
        conn.execute("delete from questions where collection = ?", (name,))
        conn.execute("delete from eval_set_versions where eval_set = ?", (name,))
        conn.execute("delete from eval_sets where name = ?", (name,))
        conn.execute("update collections set eval_set = null where eval_set = ?", (name,))
        return len(runs)


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


JOBS_RESUMABLE = (
    "interrupted",
    "cancelled",
    "stopped_budget",
    "stopped_no_key",
    "stopped_replies",
    "failed",
)


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


def retag_partly_done(conn: Connection, collection: str) -> bool:
    """A retag since the last finished one stopped after it began writing tags. Choosing
    them is the first half of its progress, and the worker records progress after its cancel
    check, so a stop just after the first batch is written still reads exactly 0.5."""
    row = first(
        conn.execute(
            "select 1 from jobs where kind = 'retag' and collection = ?1"
            " and status in ('failed', 'cancelled', 'interrupted') and progress >= 0.5"
            " and rowid > coalesce((select max(rowid) from jobs where kind = 'retag'"
            " and collection = ?1 and status = 'done'), 0) limit 1",
            (collection,),
        )
    )
    return row is not None


def tag_build_unfinished(conn: Connection, collection: str) -> bool:
    """The store's newest Add keyword tags job is not done: queued, running or stopped. Its
    vocabulary is switched to before tagging, so until then the tags may be partial."""
    row = first(
        conn.execute(
            "select status from jobs where kind = 'tag_build' and collection = ?"
            " order by rowid desc limit 1",
            (collection,),
        )
    )
    return row is not None and row["status"] != "done"


# The jobs that add, tag, re-embed or read a collection's items: its vocabulary or build
# must not change under one.
ITEM_JOBS = ("load", "import", "retag", "copy", "eval", "reembed", "tag_build")


def busy_job(conn: Connection, collection: str) -> str | None:
    """The first of ITEM_JOBS queued or running on the collection, or None."""
    return next((k for k in ITEM_JOBS if job_waiting_or_running(conn, k, collection)), None)


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


def delete_collection_rows(conn: Connection, name: str) -> None:
    for row in conn.execute("select id from jobs where collection = ?", (name,)):
        delete_job(conn, row["id"])
    conn.execute("delete from disabled_sources where collection = ?", (name,))
    conn.execute("delete from keyword_lists where collection = ?", (name,))
    conn.execute("delete from vocabularies where collection = ?", (name,))
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


# -- saved chats ---------------------------------------------------------------


def _chat(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "stores": json.loads(row["stores_json"]),
        "settings": json.loads(row["settings_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def create_chat(conn: Connection, title: str, stores: list[str], settings: dict) -> str:
    chat_id = uuid.uuid4().hex
    stamp = now()
    conn.execute(
        "insert into chats values (?, ?, ?, ?, ?, ?)",
        (chat_id, title, json.dumps(stores), json.dumps(settings), stamp, stamp),
    )
    return chat_id


def get_chat(conn: Connection, chat_id: str) -> dict | None:
    row = first(conn.execute("select * from chats where id = ?", (chat_id,)))
    return _chat(row) if row else None


def list_chats(conn: Connection) -> list[dict]:
    """Newest first. now() has one-second resolution, so rowid breaks ties."""
    rows = conn.execute(
        "select c.*, (select count(*) from chat_messages m where m.chat_id = c.id) as messages"
        " from chats c order by c.updated_at desc, c.rowid desc"
    )
    return [{**_chat(r), "messages": r["messages"]} for r in rows]


def update_chat(
    conn: Connection,
    chat_id: str,
    *,
    title: str | None = None,
    stores: list[str] | None = None,
    settings: dict | None = None,
) -> None:
    """Writes the fields given, and stamps updated_at."""
    with conn.lock:
        if title is not None:
            conn.execute("update chats set title = ? where id = ?", (title, chat_id))
        if stores is not None:
            conn.execute(
                "update chats set stores_json = ? where id = ?", (json.dumps(stores), chat_id)
            )
        if settings is not None:
            conn.execute(
                "update chats set settings_json = ? where id = ?", (json.dumps(settings), chat_id)
            )
        conn.execute("update chats set updated_at = ? where id = ?", (now(), chat_id))


def delete_chat(conn: Connection, chat_id: str) -> None:
    with conn.lock:
        conn.execute("delete from chat_attempts where chat_id = ?", (chat_id,))
        conn.execute("delete from chat_messages where chat_id = ?", (chat_id,))
        conn.execute("delete from chats where id = ?", (chat_id,))


def add_chat_message(conn: Connection, chat_id: str, content: str) -> int:
    """Saves the chat's next message and returns its number, counted from 1."""
    with conn.lock:
        n = conn.execute(
            "select coalesce(max(n), 0) + 1 as n from chat_messages where chat_id = ?",
            (chat_id,),
        )[0]["n"]
        stamp = now()
        conn.execute("insert into chat_messages values (?, ?, ?, ?)", (chat_id, n, content, stamp))
        conn.execute("update chats set updated_at = ? where id = ?", (stamp, chat_id))
        return n


def delete_chat_message(conn: Connection, chat_id: str, n: int) -> None:
    with conn.lock:
        conn.execute("delete from chat_attempts where chat_id = ? and n = ?", (chat_id, n))
        conn.execute("delete from chat_messages where chat_id = ? and n = ?", (chat_id, n))


def chat_messages(conn: Connection, chat_id: str) -> list[dict]:
    rows = conn.execute(
        "select n, content, created_at from chat_messages where chat_id = ? order by n",
        (chat_id,),
    )
    return [dict(r) for r in rows]


def _attempt(row: sqlite3.Row) -> dict:
    return {
        "n": row["n"],
        "attempt": row["attempt"],
        "search_text": row["search_text"],
        "decisions": json.loads(row["decisions_json"]),
        "evidence": json.loads(row["evidence_json"]),
        "answer": row["answer"],
        "citations": json.loads(row["citations_json"]),
        "usd": row["usd"],
        "error": row["error"],
        "model": row["model"],
        "created_at": row["created_at"],
        "pipeline": json.loads(row["pipeline_json"]) if row["pipeline_json"] else None,
    }


def add_chat_attempt(conn: Connection, chat_id: str, n: int, attempt: dict) -> int:
    """Saves message n's next attempt (search_text, decisions, evidence, answer, citations,
    usd, search_error, error, model, and pipeline when it has one) and returns its number,
    counted from 1. search_error is the part of error made before any answer, kept when it is
    answered later."""
    with conn.lock:
        number = conn.execute(
            "select coalesce(max(attempt), 0) + 1 as a from chat_attempts"
            " where chat_id = ? and n = ?",
            (chat_id, n),
        )[0]["a"]
        conn.execute(
            "insert into chat_attempts (chat_id, n, attempt, search_text, decisions_json,"
            " evidence_json, answer, citations_json, usd, error, model, created_at,"
            " search_error, pipeline_json) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chat_id,
                n,
                number,
                attempt["search_text"],
                json.dumps(attempt["decisions"]),
                json.dumps(attempt["evidence"]),
                attempt["answer"],
                json.dumps(attempt["citations"]),
                attempt["usd"],
                attempt["error"],
                attempt["model"],
                now(),
                attempt["search_error"],
                json.dumps(pipeline) if (pipeline := attempt.get("pipeline")) is not None else None,
            ),
        )
        return number


def answer_chat_attempt(
    conn: Connection, chat_id: str, n: int, attempt: int, answered: dict
) -> bool:
    """Writes an unanswered attempt's answer, citations, usd and model, and as its error its
    search_error then the answer's own error. False, writing nothing, when it has an answer."""
    where = " where chat_id = ? and n = ? and attempt = ? and answer is null"
    with conn.lock:  # no other statement between the read, the update and changes()
        row = first(
            conn.execute(f"select search_error from chat_attempts{where}", (chat_id, n, attempt))
        )
        if row is None:
            return False
        error = "; ".join(e for e in (row["search_error"], answered["error"]) if e) or None
        conn.execute(
            "update chat_attempts set answer = ?, citations_json = ?, usd = ?, error = ?,"
            f" model = ?{where}",
            (
                answered["answer"],
                json.dumps(answered["citations"]),
                answered["usd"],
                error,
                answered["model"],
                chat_id,
                n,
                attempt,
            ),
        )
        return conn.changes() > 0


def chat_attempts(conn: Connection, chat_id: str, n: int | None = None) -> list[dict]:
    """The chat's attempts (or message n's), by message, then attempt."""
    if n is None:
        rows = conn.execute(
            "select * from chat_attempts where chat_id = ? order by n, attempt", (chat_id,)
        )
    else:
        rows = conn.execute(
            "select * from chat_attempts where chat_id = ? and n = ? order by attempt",
            (chat_id, n),
        )
    return [_attempt(r) for r in rows]
