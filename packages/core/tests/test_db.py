import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

import pytest

from aiconnect import fakes
from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.db import (
    JOBS_RESUMABLE,
    claim_next_job_id,
    complete_question_ids,
    connect,
    create_job,
    delete_collection_rows,
    delete_job,
    disabled_sources,
    get_collection,
    get_job,
    get_keyword_lists,
    get_questions,
    interrupt_running,
    is_cancel_requested,
    job_running,
    list_collections,
    list_jobs,
    loaded_datasets,
    log_job,
    move_ai_tables,
    newest_jobs,
    request_cancel,
    requeue,
    run_items,
    save_collection,
    save_keyword_lists,
    save_questions,
    set_job_setup,
    set_keyword_version,
    set_source_enabled,
    update_job,
    upsert_run_item,
)
from ragqa_core.evaluator import Evaluator, changes_since
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.retriever import Retriever
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, KeywordLists, Question, SearchSettings

SPEC = CollectionSpec("arc", "mcq", True, "m", 64)


def test_collection_round_trip(tmp_path):
    conn = connect(tmp_path / "r.db")
    save_collection(conn, CollectionSpec("arc", "mcq", True, "m", 64))
    got = get_collection(conn, "arc")
    assert got is not None and got.build is not None and got.search is not None
    assert (got.name, got.qa_type, got.summarize, got.keyword_version) == ("arc", "mcq", True, None)
    assert (got.recipe, got.corpus_type, got.build.added_text) == (
        "paper_2024",
        "worked",
        "summary",
    )
    assert get_collection(conn, "nope") is None
    assert [c.name for c in list_collections(conn)] == ["arc"]


def test_keyword_lists_current_version(tmp_path):
    conn = connect(tmp_path / "r.db")
    save_collection(conn, SPEC)
    save_keyword_lists(conn, "arc", KeywordLists(["a"], ["b"], ["c"]), "v1")
    save_keyword_lists(conn, "arc", KeywordLists(["x"], ["y"], ["z"]), "v2")
    set_keyword_version(conn, "arc", "v2")
    current = get_keyword_lists(conn, "arc")
    first = get_keyword_lists(conn, "arc", "v1")
    assert current is not None and current.topic == ["x"]
    assert first is not None and first.topic == ["a"]
    spec = get_collection(conn, "arc")
    assert spec is not None and spec.keyword_version == "v2"


def test_keyword_lists_absent(tmp_path):
    conn = connect(tmp_path / "r.db")
    save_collection(conn, SPEC)
    assert get_keyword_lists(conn, "arc") is None


def test_questions_round_trip_sorted_by_id(tmp_path):
    conn = connect(tmp_path / "r.db")
    qs = [
        Question("b", "Q2?", None, ["x"], ["i1"], {}),
        Question("a", "Q1?", ["p", "q"], ["A"], [], {"skill": "s"}),
    ]
    save_collection(conn, SPEC)
    save_questions(conn, "arc", qs)
    assert get_questions(conn, "arc") == [qs[1], qs[0]]


def test_job_round_trip(tmp_path):
    conn = connect(tmp_path / "r.db")
    job = create_job(conn, "load", "arc", {"dataset": "scienceqa"})
    update_job(conn, job, progress=0.5, result={"added": 3})
    row = get_job(conn, job)
    assert row is not None
    assert row["status"] == "queued" and row["progress"] == 0.5
    assert row["setup"] == {"dataset": "scienceqa"} and row["result"] == {"added": 3}
    assert get_job(conn, "missing") is None


def test_loaded_datasets_names_every_load_once(tmp_path):
    conn = connect(tmp_path / "r.db")
    for dataset, status in (("gpqa", "done"), ("scienceqa", "failed"), ("gpqa", "done")):
        update_job(conn, create_job(conn, "load", "arc", {"dataset": dataset}), status=status)
    create_job(conn, "load", "other", {"dataset": "tatqa"})
    # A failed load still added items.
    assert loaded_datasets(conn, "arc") == {"gpqa", "scienceqa"}
    assert loaded_datasets(conn, "empty") == set()


def test_run_items_filter_by_method(tmp_path):
    conn = connect(tmp_path / "r.db")
    job = create_job(conn, "eval", "arc", {})
    upsert_run_item(conn, job, "hybrid", "q1", {"hit_ids": ["a"]}, complete=False)
    upsert_run_item(conn, job, "vector", "q1", {"hit_ids": ["b"]}, complete=False)
    assert [r["hit_ids"] for r in run_items(conn, job, "hybrid")] == [["a"]]
    assert len(run_items(conn, job)) == 2


def test_old_collections_migrate_to_paper_2024(tmp_path):
    path = tmp_path / "r.db"
    raw = sqlite3.connect(path)
    raw.execute(
        "create table collections (name text primary key, qa_type text not null,"
        " summarize integer not null, dense_model text not null, dense_dim integer not null,"
        " keyword_version text, created_at text not null,"
        " include_leaked integer not null default 0)"
    )
    raw.execute("insert into collections values ('old', 'multihop', 1, 'm', 8, null, 'now', 0)")
    raw.execute("insert into collections values ('sci', 'science', 0, 'm', 8, null, 'now', 0)")
    raw.commit()
    raw.close()
    conn = connect(path)
    science = get_collection(conn, "sci")
    # It never ran model-written code, so migrating does not start to.
    assert science is not None and science.search is not None
    assert science.search.program is False
    spec = get_collection(conn, "old")
    assert spec is not None and spec.build is not None and spec.search is not None
    assert (spec.corpus_type, spec.recipe, spec.build.added_text) == (
        "reference",
        "paper_2024",
        "summary",
    )
    assert spec.search.query_tags == "similarity" and spec.search.program is False
    connect(path)  # a second connect leaves it alone
    assert get_collection(conn, "old") == spec


def test_set_search_round_trips(tmp_path):
    conn = connect(tmp_path / "r.db")
    save_collection(conn, CollectionSpec("arc", "mcq", False, "m", 64))
    db.set_search(conn, "arc", SearchSettings({"meaning": 0.0, "words": 1.0, "keywords": 2.0}))
    got = get_collection(conn, "arc")
    assert got is not None and got.search is not None
    assert got.search.weights["keywords"] == 2.0


def test_connect_is_idempotent(tmp_path):
    connect(tmp_path / "r.db")
    connect(tmp_path / "r.db")


# The ledger, cache and pending_batches DDL as it was before AI connections (main d80f1b9).
PRE_PHASE1_AI_TABLES = """
create table ledger (
  at text not null, model text not null, input_tokens integer not null,
  output_tokens integer not null, usd real not null, batch integer not null,
  purpose text not null, job_id text
);
create table cache (
  key text primary key, text text not null, input_tokens integer not null,
  output_tokens integer not null
);
create table pending_batches (
  id text primary key, model text not null, purpose text not null, created_at text not null,
  usd real not null default 0, job_id text
);
"""


def old_cache_key(model: str, system: str, prompt: str, max_tokens: int) -> str:
    """The cache key the old Claude class wrote, spelled out so a change to it shows here."""
    return hashlib.sha256(json.dumps([model, system, prompt, max_tokens]).encode()).hexdigest()


def make_pre_phase1_db(path):
    """A database written before AI connections: this month's $1.50 ledger row, a cached
    answer to "hello" and one batch still pending ($0.20)."""
    old = sqlite3.connect(path)
    old.executescript(PRE_PHASE1_AI_TABLES)
    at = datetime.now(UTC).isoformat(timespec="seconds")
    old.execute(
        "insert into ledger values (?, 'claude-haiku-5-5', 10, 10, 1.5, 0, 'answer', 'j1')",
        (at,),
    )
    old.execute(
        "insert into cache values (?, 'cached hello', 3, 2)",
        (old_cache_key("claude-haiku-5-5", "", "hello", 1024),),
    )
    old.execute(
        "insert into pending_batches values ('b1', 'claude-haiku-5-5', 'answer', ?, 0.2, 'j1')",
        (at,),
    )
    old.commit()
    old.close()


def table_exists(conn, name: str) -> bool:
    return bool(
        conn.execute("select 1 from sqlite_master where type = 'table' and name = ?", (name,))
    )


def test_upgrade_moves_spend_cache_and_batches(tmp_path):
    path = tmp_path / "ragqa.db"
    make_pre_phase1_db(path)
    conn = connect(path)
    ai = fake_ai(conn, lambda p: pytest.fail("cached prompt was re-sent"))
    move_ai_tables(conn)
    assert ai.month_spend() == pytest.approx(1.5)
    assert ai.job_spend("j1") == pytest.approx(1.5)
    done = ai.complete("hello", purpose="t")
    assert done.cached is True and done.text == "cached hello" and done.usd == 0
    assert [b["connection_id"] for b in ai.pending_batches()] == ["anthropic"]
    assert ai.pending_batches()[0]["usd"] == pytest.approx(0.2)
    assert not table_exists(conn, "ledger")
    assert not table_exists(conn, "cache") and not table_exists(conn, "pending_batches")
    move_ai_tables(conn)  # second run: no-op
    assert ai.month_spend() == pytest.approx(1.5)
    assert len(ai.pending_batches()) == 1


def test_upgrade_batch_is_collected_into_the_ledger(tmp_path, monkeypatch):
    path = tmp_path / "ragqa.db"
    make_pre_phase1_db(path)
    conn = connect(path)
    ai = fake_ai(conn, lambda p: "x")
    move_ai_tables(conn)
    # The provider still holds the old batch's answer.
    monkeypatch.setitem(fakes.BATCHES, "b1", {"k-b1": ("batched answer", 40, 60)})
    ai.collect_pending()
    rows = conn.execute("select batch, job_id, connection_id, model from ai_ledger where batch = 1")
    assert [tuple(r) for r in rows] == [(1, "j1", "anthropic", "claude-haiku-5-5")]
    assert ai.pending_batches() == []
    assert ai.month_spend() > 1.5


def test_upgrade_of_a_new_database_does_nothing(tmp_path):
    conn = connect(tmp_path / "ragqa.db")
    ai = fake_ai(conn, lambda p: "x")
    move_ai_tables(conn)
    assert ai.month_spend() == 0 and ai.pending_batches() == []
    assert not table_exists(conn, "ledger")


def test_upgrade_carries_rows_from_before_job_ids(tmp_path):
    """A 0.1.0 database: no job_id on the ledger or batches, and no batch estimate."""
    path = tmp_path / "ragqa.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        create table ledger (at text not null, model text not null,
          input_tokens integer not null, output_tokens integer not null,
          usd real not null, batch integer not null, purpose text not null);
        create table cache (key text primary key, text text not null,
          input_tokens integer not null, output_tokens integer not null);
        create table pending_batches (id text primary key, model text not null,
          purpose text not null, created_at text not null);
        insert into ledger values ('t', 'm', 1, 1, 0.5, 0, 'p');
        insert into pending_batches values ('b', 'm', 'p', 'now');
        """
    )
    old.commit()
    old.close()
    conn = connect(path)
    ai = fake_ai(conn, lambda p: "x")
    move_ai_tables(conn)
    row = conn.execute("select job_id, connection_id from ai_ledger")[0]
    assert tuple(row) == (None, "anthropic")
    assert ai.pending_batches()[0]["usd"] == 0 and ai.pending_batches()[0]["job_id"] is None


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path / "q.db")


OLD_STORE_BARS = """
create table if not exists store_bars (
  collection text not null, kind text not null default 'gate', key text not null,
  value real, n integer not null, computed_at text not null,
  primary key (collection, kind)
);
"""

OLD_COLLECTION_COLUMNS = (
    "corpus_type",
    "recipe",
    "build_json",
    "search_json",
    "vocabulary_version",
    "profile_json",
    "eval_set",
    "chat_method",
    "chat_bar",
    "chat_k",
    "route_profile",
)


def test_an_old_database_drops_store_bars_and_migrates_chat_settings(tmp_path):
    path = tmp_path / "r.db"
    old = sqlite3.connect(path)
    old.executescript(db.SCHEMA + OLD_STORE_BARS)
    for column in OLD_COLLECTION_COLUMNS:
        old.execute(f"alter table collections add column {column} text")
    old.execute(
        "insert into collections (name, qa_type, summarize, dense_model, dense_dim,"
        " created_at, recipe, chat_k) values ('c', 'multihop', 0, 'm', 8, 'now', 'paper_2024', '5')"
    )
    old.execute("insert into store_bars values ('c', 'gate', 'k', 0.5, 20, 'now')")
    for chat_id, settings in (
        ("A", {"evidence_size": 8, "rewrite": False, "routing": "profiles"}),
        ("B", {"evidence_size": 5, "rewrite": True, "routing": "all"}),
    ):
        old.execute(
            "insert into chats values (?, 'chat', '[\"c\"]', ?, 'now', 'now')",
            (chat_id, json.dumps(settings)),
        )
    old.commit()
    old.close()

    conn = connect(path)
    assert not conn.execute("select 1 from sqlite_master where name = 'store_bars'")
    chat_a, chat_b = db.get_chat(conn, "A"), db.get_chat(conn, "B")
    assert chat_a is not None and chat_b is not None
    assert chat_a["settings"] == {"evidence_size": None, "rewrite": None}
    assert chat_b["settings"] == {"evidence_size": 5, "rewrite": True}
    assert db.get_chat_k(conn, "c") is None  # it now means candidates per store

    db.set_chat_k(conn, "c", 5)
    connect(path)  # a second connect leaves it alone: it runs once
    assert db.get_chat(conn, "B") == chat_b
    assert db.get_chat_k(conn, "c") == 5


def test_a_rerun_of_the_routing_migration_keeps_newer_chat_settings(tmp_path):
    settings = {
        "evidence_size": 8,
        "rewrite": False,
        "rerank": True,
        "stores": {"c": {"method": "vector", "candidates": None}},
    }
    # A fresh database, and one migrated from before chat searched every store.
    fresh, old = tmp_path / "fresh.db", tmp_path / "old.db"
    connect(fresh)
    legacy = sqlite3.connect(old)
    legacy.executescript(db.SCHEMA + OLD_STORE_BARS)
    legacy.commit()
    legacy.close()
    connect(old)
    for path in (fresh, old):
        conn = connect(path)
        assert db.get_app_settings(conn, "schema.") == {"routing_dropped": True}
        conn.execute(
            "insert into collections (name, qa_type, summarize, dense_model, dense_dim,"
            " created_at) values ('c', 'multihop', 0, 'm', 8, 'now')"
        )
        db.set_chat_k(conn, "c", 12)
        conn.execute(
            "insert into chats values ('A', 'chat', '[\"c\"]', ?, 'now', 'now')",
            (json.dumps({**settings, "routing": "all"}),),
        )
        conn.execute(OLD_STORE_BARS)  # as an older build makes it again on this database
        connect(path)
        chat = db.get_chat(conn, "A")
        assert chat is not None and chat["settings"] == settings, path.name
        assert db.get_chat_k(conn, "c") == 12, path.name
        assert not conn.execute("select 1 from sqlite_master where name = 'store_bars'")


def test_claim_takes_the_oldest_queued_job_by_insertion(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    assert claim_next_job_id(conn) == a
    assert get_job(conn, a)["status"] == "running"
    assert get_job(conn, b)["status"] == "queued"
    assert claim_next_job_id(conn) == b and claim_next_job_id(conn) is None


def test_interrupt_running_keeps_the_queue(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    claim_next_job_id(conn)
    assert interrupt_running(conn) == 1
    assert get_job(conn, a)["status"] == "interrupted"
    assert get_job(conn, b)["status"] == "queued"


def test_cancel_queued_is_immediate_running_is_requested(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    claim_next_job_id(conn)
    assert request_cancel(conn, b) == "cancelled"
    assert request_cancel(conn, a) == "running" and is_cancel_requested(conn, a)
    assert not is_cancel_requested(conn, b)


def test_job_running_ignores_queued(conn):
    create_job(conn, "eval", "c", {})
    assert job_running(conn, "eval", "c") is False
    claim_next_job_id(conn)
    assert job_running(conn, "eval", "c") is True
    assert job_running(conn, "eval", "other") is False


def test_job_running_matches_kind_and_collection(conn):
    create_job(conn, "load", "c", {})
    assert job_running(conn, "load", "c") is False
    claim_next_job_id(conn)
    assert job_running(conn, "load", "c") is True
    assert job_running(conn, "eval", "c") is False


def test_log_keeps_last_50(conn):
    j = create_job(conn, "load", "c", {})
    for n in range(60):
        log_job(conn, j, f"m{n}")
    log = get_job(conn, j)["log"]
    assert len(log) == 50 and log[-1]["message"] == "m59" and log[0]["message"] == "m10"
    assert set(log[0]) == {"at", "message"}


def test_new_job_has_empty_log_and_no_cancel(conn):
    j = create_job(conn, "load", "c", {})
    job = get_job(conn, j)
    assert job["log"] == [] and job["cancel_requested"] is False


def test_lists_saved_without_switching_version(conn):
    save_collection(conn, CollectionSpec("c", "mcq", False, "d", 4))
    save_keyword_lists(conn, "c", KeywordLists(["a"], ["b"], ["c"]), "v1")
    assert get_collection(conn, "c").keyword_version is None
    set_keyword_version(conn, "c", "v1")
    assert get_collection(conn, "c").keyword_version == "v1"


def test_vocabulary_versions_saved_without_switching(conn):
    save_collection(conn, CollectionSpec("c", "science", False, "d", 4))
    assert db.get_vocabulary(conn, "c") is None  # no version in force
    db.save_vocabulary(conn, "c", ["ohm's law", "ideal gas law"], "v1")
    db.save_vocabulary(conn, "c", ["ohm's law"], "v2")
    assert get_collection(conn, "c").vocabulary_version is None
    assert db.get_vocabulary(conn, "c") is None
    assert db.get_vocabulary(conn, "c", "v1") == ["ohm's law", "ideal gas law"]
    assert db.get_vocabulary(conn, "c", "missing") is None
    db.set_vocabulary_version(conn, "c", "v1")
    assert db.get_vocabulary(conn, "c") == ["ohm's law", "ideal gas law"]
    rows = db.list_vocabularies(conn, "c")
    assert [(r["version"], r["count"], r["active"]) for r in rows] == [
        ("v2", 1, False),
        ("v1", 2, True),
    ]
    assert all(r["created_at"] for r in rows)
    delete_collection_rows(conn, "c")
    assert db.get_vocabulary(conn, "c", "v1") is None and db.list_vocabularies(conn, "c") == []


def test_requeue_clears_cancel_and_error(conn):
    j = create_job(conn, "eval", "c", {})
    claim_next_job_id(conn)
    request_cancel(conn, j)
    update_job(conn, j, status="failed", error="boom")
    requeue(conn, j)
    job = get_job(conn, j)
    assert job["status"] == "queued" and job["cancel_requested"] is False
    assert job["error"] is None


def test_requeue_moves_only_a_resumable_job(conn):
    j = create_job(conn, "eval", "c", {})
    assert requeue(conn, j) is False  # already queued
    claim_next_job_id(conn)
    assert requeue(conn, j) is False and get_job(conn, j)["status"] == "running"
    for status in JOBS_RESUMABLE:
        update_job(conn, j, status=status)
        assert requeue(conn, j) is True and get_job(conn, j)["status"] == "queued"
    update_job(conn, j, status="done")
    assert requeue(conn, j) is False and get_job(conn, j)["status"] == "done"
    assert requeue(conn, "missing") is False


def test_set_job_setup_and_list_jobs(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    create_job(conn, "eval", "d", {})
    set_job_setup(conn, a, {"limit": 5})
    assert get_job(conn, a)["setup"] == {"limit": 5}
    assert [j["id"] for j in list_jobs(conn, collection="c")] == [b, a]
    assert [j["id"] for j in list_jobs(conn, kind="load")] == [a]
    claim_next_job_id(conn)
    assert [j["id"] for j in list_jobs(conn, status="running")] == [a]
    assert len(list_jobs(conn, limit=2)) == 2


def test_jobs_resumable_statuses():
    assert JOBS_RESUMABLE == (
        "interrupted",
        "cancelled",
        "stopped_budget",
        "stopped_no_key",
        "stopped_replies",
        "failed",
    )


def test_delete_job_removes_its_run_items(conn):
    a = create_job(conn, "eval", "c", {})
    b = create_job(conn, "eval", "c", {})
    upsert_run_item(conn, a, "hybrid", "q1", {}, complete=False)
    upsert_run_item(conn, b, "hybrid", "q1", {}, complete=False)
    delete_job(conn, a)
    assert get_job(conn, a) is None and run_items(conn, a) == []
    assert len(run_items(conn, b)) == 1


def test_delete_collection_rows_removes_its_jobs_and_rows(conn):
    for name in ("c", "keep"):
        save_collection(conn, CollectionSpec(name, "mcq", False, "d", 4))
        save_keyword_lists(conn, name, KeywordLists(["a"], ["b"], ["c"]), "v1")
        save_questions(conn, name, [Question("q1", "t", None, ["a"], [])])
    gone = create_job(conn, "eval", "c", {})
    kept = create_job(conn, "eval", "keep", {})
    upsert_run_item(conn, gone, "hybrid", "q1", {}, complete=False)
    upsert_run_item(conn, kept, "hybrid", "q1", {}, complete=False)
    delete_collection_rows(conn, "c")
    assert get_collection(conn, "c") is None
    assert get_keyword_lists(conn, "c", "v1") is None
    assert get_job(conn, gone) is None and run_items(conn, gone) == []
    assert get_collection(conn, "keep") is not None
    assert get_keyword_lists(conn, "keep", "v1") is not None
    assert len(get_questions(conn, "keep")) == 1
    assert get_job(conn, kept) is not None and len(run_items(conn, kept)) == 1


def test_connect_adds_queue_columns_to_an_older_jobs_table(tmp_path):
    old = sqlite3.connect(tmp_path / "r.db")
    old.execute(
        "create table jobs (id text primary key, kind text not null, collection text not null,"
        " status text not null, progress real not null default 0, setup_json text not null,"
        " result_json text, error text, created_at text not null, updated_at text not null)"
    )
    old.execute(
        "insert into jobs (id, kind, collection, status, setup_json, created_at, updated_at)"
        " values ('j', 'load', 'c', 'queued', '{}', 'x', 'x')"
    )
    old.commit()
    old.close()
    job = get_job(connect(tmp_path / "r.db"), "j")
    assert job["cancel_requested"] is False and job["log"] == []


def test_upsert_run_item_never_duplicates(conn):
    j = create_job(conn, "eval", "c", {"answer": True})
    upsert_run_item(conn, j, "hybrid", "q1", {"retrieval": {}}, complete=False)
    assert complete_question_ids(conn, j, "hybrid") == set()
    upsert_run_item(
        conn, j, "hybrid", "q1", {"retrieval": {}, "answer": {"accuracy": 1.0}}, complete=True
    )
    rows = run_items(conn, j)
    assert len(rows) == 1 and rows[0]["complete"] is True
    assert complete_question_ids(conn, j, "hybrid") == {"q1"}
    assert complete_question_ids(conn, j, "vector") == set()


def test_migration_dedupes_and_backfills(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        create table jobs (id text primary key, kind text not null, collection text not null,
          status text not null, progress real not null default 0, setup_json text not null,
          result_json text, error text, created_at text not null, updated_at text not null);
        create table run_items (job_id text not null, method text not null,
          question_id text not null, row_json text not null);
        insert into jobs values ('j1', 'eval', 'c', 'done', 1, '{"answer": false}',
          null, null, 't', 't');
        insert into jobs values ('j2', 'eval', 'c', 'done', 1, '{"answer": true}',
          null, null, 't', 't');
        insert into run_items values ('j1', 'hybrid', 'q1', '{"v": 1}');
        insert into run_items values ('j1', 'hybrid', 'q1', '{"v": 2}');
        insert into run_items values ('j2', 'hybrid', 'q2', '{"answer": {"accuracy": 1}}');
        insert into run_items values ('j2', 'hybrid', 'q3', '{"hit_ids": []}');
        """
    )
    old.commit()
    old.close()

    conn = connect(path)
    rows = {(r["job_id"], r["question_id"]): r for r in conn.execute("select * from run_items")}
    assert len(rows) == 3
    assert json.loads(rows[("j1", "q1")]["row_json"]) == {"v": 2}
    assert rows[("j1", "q1")]["complete"] == 1
    assert rows[("j2", "q2")]["complete"] == 1
    assert rows[("j2", "q3")]["complete"] == 0
    assert get_job(conn, "j1")["setup"] == {"answer": False, "question_ids": ["q1"]}
    assert get_job(conn, "j2")["setup"]["question_ids"] == ["q2", "q3"]

    before = [tuple(r) for r in conn.execute("select * from run_items order by rowid")]
    setups = [get_job(conn, j)["setup"] for j in ("j1", "j2")]
    conn.close()
    again = connect(path)
    assert [tuple(r) for r in again.execute("select * from run_items order by rowid")] == before
    assert [get_job(again, j)["setup"] for j in ("j1", "j2")] == setups


def test_migration_marks_complete_by_the_answer_key_not_the_word(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        create table jobs (id text primary key, kind text not null, collection text not null,
          status text not null, progress real not null default 0, setup_json text not null,
          result_json text, error text, created_at text not null, updated_at text not null);
        create table run_items (job_id text not null, method text not null,
          question_id text not null, row_json text not null);
        insert into jobs values ('j', 'eval', 'c', 'done', 1, '{"answer": true}',
          null, null, 't', 't');
        insert into run_items values ('j', 'hybrid', 'q1', '{"hit_ids": ["answer"]}');
        insert into run_items values ('j', 'hybrid', 'q2', '{"answer": {"accuracy": 0}}');
        """
    )
    old.commit()
    old.close()

    conn = connect(path)
    complete = {r["question_id"]: r["complete"] for r in conn.execute("select * from run_items")}
    assert complete == {"q1": 0, "q2": 1}


def test_question_ids_backfill_skips_only_queued_evals(tmp_path):
    path = tmp_path / "r.db"
    conn = connect(path)
    jobs = {
        status: create_job(conn, "eval", "c", {"answer": False})
        for status in ("queued", "running", "interrupted")
    }
    for status, job in jobs.items():
        update_job(conn, job, status=status)
        upsert_run_item(conn, job, "hybrid", "q1", {}, complete=True)
    conn.close()

    # Nothing runs at connect time: a running eval is a leftover, interrupted by the lifespan.
    again = connect(path)
    assert "question_ids" not in get_job(again, jobs["queued"])["setup"]
    assert get_job(again, jobs["running"])["setup"]["question_ids"] == ["q1"]
    assert get_job(again, jobs["interrupted"])["setup"]["question_ids"] == ["q1"]


def test_one_connection_is_safe_from_many_threads(conn):
    # The API's request threads and the worker share one connection (spec §5.1).
    import threading

    errors: list[BaseException] = []
    threads, ops = 8, 300

    def hammer(n: int) -> None:
        try:
            for i in range(ops):
                job = create_job(conn, "eval", f"c{n}", {"i": i})
                row = get_job(conn, job)
                assert row is not None and row["setup"] == {"i": i}
                if i % 3 == 0:
                    log_job(conn, job, "tick")
                    assert len(list_jobs(conn, collection=f"c{n}", limit=5)) <= 5
                if i % 5 == 0:
                    update = "update jobs set progress = 0.5 where id = ?"
                    with conn.lock:
                        conn.execute(update, (job,))
                        assert conn.changes() == 1
        except BaseException as e:  # every failure is the finding
            errors.append(e)

    workers = [threading.Thread(target=hammer, args=(n,)) for n in range(threads)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    assert errors == []
    rows = conn.execute("select collection, count(*) as n from jobs group by collection")
    assert {r["collection"]: r["n"] for r in rows} == {f"c{n}": ops for n in range(threads)}
    assert conn.execute("select count(*) as n from jobs where progress = 0.5")[0]["n"] == (
        threads * len(range(0, ops, 5))
    )


def test_connection_returns_rows_already_fetched(conn):
    for i in range(3):
        create_job(conn, "eval", "c", {"i": i})
    rows = conn.execute("select id, kind from jobs order by rowid")
    conn.execute("delete from jobs")  # a later statement cannot touch rows already returned
    conn.execute("select * from collections")
    assert isinstance(rows, list) and len(rows) == 3
    assert [r["kind"] for r in rows] == ["eval"] * 3
    assert conn.execute("select * from jobs") == []


def test_newest_jobs_is_one_per_collection_and_kind(conn):
    old = create_job(conn, "eval", "a", {})
    new = create_job(conn, "eval", "a", {"n": 2})
    load = create_job(conn, "load", "a", {})
    other = create_job(conn, "eval", "b", {})
    newest = newest_jobs(conn)
    assert [j["id"] for j in newest] == [other, load, new]
    assert newest[2]["setup"] == {"n": 2} and old not in {j["id"] for j in newest}


def test_interrupt_running_counts_the_jobs_it_changed(conn):
    jobs = [create_job(conn, "eval", "c", {}) for _ in range(3)]
    for job in jobs[:2]:
        update_job(conn, job, status="running")
    assert interrupt_running(conn) == 2
    assert interrupt_running(conn) == 0


def test_disabled_sources_round_trip_and_cleared_with_collection(conn):
    save_collection(conn, SPEC)
    assert disabled_sources(conn, "arc") == []
    set_source_enabled(conn, "arc", "b", False)
    set_source_enabled(conn, "arc", "a", False)
    set_source_enabled(conn, "arc", "a", False)  # twice is the same as once
    set_source_enabled(conn, "other", "z", False)
    assert disabled_sources(conn, "arc") == ["a", "b"]
    set_source_enabled(conn, "arc", "a", True)
    assert disabled_sources(conn, "arc") == ["b"]
    delete_collection_rows(conn, "arc")
    assert disabled_sources(conn, "arc") == []
    assert disabled_sources(conn, "other") == ["z"]


def test_include_leaked_defaults_false_and_survives_migration(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        create table collections (
          name text primary key, qa_type text not null, summarize integer not null,
          dense_model text not null, dense_dim integer not null, keyword_version text,
          created_at text not null
        );
        insert into collections values ('old', 'mcq', 0, 'm', 64, null, 't');
        """
    )
    old.commit()
    old.close()
    conn = connect(path)
    assert db.get_include_leaked(conn, "old") is False
    assert get_collection(conn, "old").name == "old"
    db.set_include_leaked(conn, "old", True)
    save_collection(conn, CollectionSpec("new", "mcq", False, "m", 64))
    assert db.get_include_leaked(conn, "new") is False
    save_collection(conn, CollectionSpec("old", "mcq", True, "m", 64))
    assert db.get_include_leaked(conn, "old") is True  # a re-save keeps the switch
    assert [c.name for c in list_collections(conn)] == ["new", "old"]


def questions_of(*texts: str) -> list[Question]:
    return [Question(f"q{i}", t, None, ["a"], []) for i, t in enumerate(texts)]


def test_saving_questions_snapshots_a_version_only_when_they_change(conn):
    db.create_eval_set(conn, "e", "mcq")
    save_questions(conn, "e", questions_of("one", "two"))
    v1 = db.get_eval_set(conn, "e")["version"]
    assert len(v1) == 12 and int(v1, 16) >= 0
    save_questions(conn, "e", questions_of("one", "two"))
    assert db.get_eval_set(conn, "e")["version"] == v1
    assert len(conn.execute("select * from eval_set_versions")) == 1
    save_questions(conn, "e", questions_of("one", "changed"))
    v2 = db.get_eval_set(conn, "e")["version"]
    assert v2 != v1
    assert [q.text for q in get_questions(conn, "e", v1)] == ["one", "two"]
    assert [v["version"] for v in db.eval_set_versions(conn, "e")] == [v2, v1]
    with pytest.raises(KeyError):
        get_questions(conn, "e", "nope")


def test_eval_set_ids_ever_is_the_union_of_ids_across_versions(conn):
    db.create_eval_set(conn, "e", "mcq")
    assert db.eval_set_ids_ever(conn, "e") == set()
    save_questions(conn, "e", questions_of("one", "two"))
    save_questions(conn, "e", questions_of("one"))
    save_questions(conn, "e", [Question("z", "new", None, ["a"], [])])
    assert db.eval_set_ids_ever(conn, "e") == {"q0", "q1", "z"}


def test_save_questions_on_a_store_name_creates_its_eval_set_and_refuses_an_unknown_name(conn):
    save_collection(conn, CollectionSpec("c", "table", False, "d", 4))
    save_questions(conn, "c", questions_of("one"))
    found = db.get_eval_set(conn, "c")
    assert found["qa_type"] == "table" and found["questions"] == 1
    assert [e["name"] for e in db.list_eval_sets(conn)] == ["c"]
    with pytest.raises(ValueError):
        save_questions(conn, "nowhere", questions_of("one"))


def test_create_eval_set_refuses_a_bad_or_taken_name(conn):
    with pytest.raises(ValueError):
        db.create_eval_set(conn, "bad name", "mcq")
    db.create_eval_set(conn, "e", "mcq")
    with pytest.raises(ValueError, match="eval set 'e' already exists"):
        db.create_eval_set(conn, "e", "mcq")


def test_replace_questions_refuses_a_stale_base_and_writes_nothing(conn):
    db.create_eval_set(conn, "e", "mcq")
    save_questions(conn, "e", questions_of("one", "two"))
    v1 = db.get_eval_set(conn, "e")["version"]
    v2 = db.replace_questions(conn, "e", questions_of("three"), v1)
    assert v2 != v1 and [q.text for q in get_questions(conn, "e")] == ["three"]
    with pytest.raises(ValueError, match=f"changed since it was read \\(now {v2}\\)"):
        db.replace_questions(conn, "e", questions_of("four"), v1)
    assert [q.text for q in get_questions(conn, "e")] == ["three"]
    assert db.get_eval_set(conn, "e")["version"] == v2
    assert db.replace_questions(conn, "e", questions_of("four"), None) != v2


def test_ensure_eval_set_refuses_another_type(conn):
    db.ensure_eval_set(conn, "e", "mcq")
    db.ensure_eval_set(conn, "e", "mcq")
    with pytest.raises(ValueError, match="eval set 'e' is mcq, not table"):
        db.ensure_eval_set(conn, "e", "table")


def test_default_eval_set_falls_back_to_the_set_named_after_the_store(conn):
    save_collection(conn, CollectionSpec("c", "mcq", False, "d", 4))
    assert db.default_eval_set(conn, "c") is None
    db.create_eval_set(conn, "c", "mcq")
    db.create_eval_set(conn, "other", "mcq")
    assert db.default_eval_set(conn, "c") == "c"
    db.set_default_eval_set(conn, "c", "other")
    assert db.default_eval_set(conn, "c") == "other"
    db.set_default_eval_set(conn, "c", None)
    assert db.default_eval_set(conn, "c") == "c"


def test_deleting_a_collection_keeps_its_eval_set(conn):
    save_collection(conn, CollectionSpec("c", "mcq", False, "d", 4))
    save_questions(conn, "c", questions_of("one"))
    delete_collection_rows(conn, "c")
    assert len(get_questions(conn, "c")) == 1 and db.get_eval_set(conn, "c") is not None


def test_delete_eval_set_returns_the_runs_it_deleted_and_clears_pointers(conn):
    save_collection(conn, CollectionSpec("c", "mcq", False, "d", 4))
    db.create_eval_set(conn, "e", "mcq")
    db.create_eval_set(conn, "f", "mcq")
    save_questions(conn, "e", questions_of("one"))
    db.set_default_eval_set(conn, "c", "e")
    a = create_job(conn, "eval", "c", {"eval_set": "e"})
    create_job(conn, "eval", "c", {"eval_set": "e"})
    kept = create_job(conn, "eval", "c", {"eval_set": "f"})
    upsert_run_item(conn, a, "hybrid", "q0", {}, complete=False)
    assert db.delete_eval_set(conn, "e") == 2
    assert db.get_eval_set(conn, "e") is None and get_questions(conn, "e") == []
    assert run_items(conn, a) == [] and get_job(conn, kept) is not None
    assert conn.execute("select * from eval_set_versions where eval_set = 'e'") == []
    assert conn.execute("select eval_set from collections")[0]["eval_set"] is None
    assert db.default_eval_set(conn, "c") is None


def evaluator_for(conn):
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "Answer: A")
    ix = Indexer(conn, store, models, ai)
    return Evaluator(
        conn,
        Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix)),
        models,
        ai,
    )


def test_migration_pins_old_runs_to_the_migrated_snapshot(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    setup = json.dumps(
        {
            "collection": "c",
            "split": "full",
            "methods": ["hybrid"],
            "answer": False,
            "k": 5,
            "rrf_k": 60,
            "search": {},
            "keyword_version": "kv",
            "vocabulary_version": None,
            "dense_model": "d",
            "disabled_sources": [],
            "question_ids": ["q0", "q1"],
        }
    )
    old.executescript(
        f"""
        create table collections (
          name text primary key, qa_type text not null, summarize integer not null,
          dense_model text not null, dense_dim integer not null, keyword_version text,
          created_at text not null, include_leaked integer not null default 1
        );
        insert into collections values ('c', 'mcq', 0, 'd', 4, 'kv', 't', 1);
        create table questions (collection text not null, id text not null, json text not null,
          primary key (collection, id));
        create table jobs (id text primary key, kind text not null, collection text not null,
          status text not null, progress real not null default 0, setup_json text not null,
          result_json text, error text, created_at text not null, updated_at text not null,
          cancel_requested integer not null default 0, log_json text);
        create table run_items (job_id text not null, method text not null,
          question_id text not null, row_json text not null, complete integer not null default 0);
        insert into jobs (id, kind, collection, status, setup_json, created_at, updated_at)
          values ('done', 'eval', 'c', 'done', '{setup}', 't', 't'),
                 ('stopped', 'eval', 'c', 'stopped_budget', '{setup}', 't', 't');
        """
    )
    for q in questions_of("one", "two"):
        old.execute("insert into questions values ('c', ?, ?)", (q.id, json.dumps(asdict(q))))
    for job in ("done", "stopped"):
        for q in ("q0", "q1"):
            row = json.dumps(
                {"hit_ids": ["i1"], "retrieval": {"solution_sim@1": 0.5, "same_skill@1": 1.0}}
            )
            old.execute("insert into run_items values (?, 'hybrid', ?, ?, 1)", (job, q, row))
    old.commit()
    old.close()

    def current(conn):
        return db.get_eval_set(conn, "c")["version"]

    conn = connect(path)  # migrates
    ev = evaluator_for(conn)
    done_table = ev.table("done")
    spec = get_collection(conn, "c")
    assert spec is not None
    for j in ("done", "stopped"):
        table = ev.table(j)
        assert table["question_count"] == 2
        hybrid = table["methods"]["hybrid"]
        assert hybrid["n"] == 2
        assert hybrid["solution_sim@1"]["mean"] == 0.5 and hybrid["same_skill@1"]["mean"] == 1.0
        assert [(r["question"], r["gold_answers"]) for r in ev.items(j, "hybrid", False)] == [
            ("one", ["a"]),
            ("two", ["a"]),
        ]
        assert sorted(q["question"] for q in ev.questions(j, False)) == ["one", "two"]
        got = changes_since(db.get_job(conn, j)["setup"], spec, [], current(conn))
        assert got == {"untracked": False, "changes": []}
    conn.close()
    conn = connect(path)  # a second connect changes nothing
    ev = evaluator_for(conn)
    assert ev.table("done") == done_table
    version = db.get_eval_set(conn, "c")["version"]
    assert db.get_eval_set(conn, "c")["include_leaked"] is True
    for j in ("done", "stopped"):
        got = db.get_job(conn, j)["setup"]
        assert got["eval_set"] == "c" and got["eval_set_version"] == version
    db.replace_questions(conn, "c", questions_of("one edited", "two"), version)
    for j in ("done", "stopped"):
        assert [r["question"] for r in ev.items(j, "hybrid", False)] == ["one", "two"]
        assert "one" in json.dumps(ev.questions(j, False))
        assert "one edited" not in json.dumps(ev.questions(j, False))
        got = changes_since(
            db.get_job(conn, j)["setup"],
            get_collection(conn, "c"),
            [],
            current(conn),
        )
        assert got["changes"] == ["eval set"]
    before = [tuple(r) for r in conn.execute("select * from jobs order by id")]
    conn.close()
    again = connect(path)
    assert [tuple(r) for r in again.execute("select * from jobs order by id")] == before


def test_replace_questions_with_a_repeated_id_changes_nothing(conn):
    db.create_eval_set(conn, "e", "mcq")
    save_questions(conn, "e", questions_of("one", "two"))
    v1 = db.get_eval_set(conn, "e")["version"]
    twice = [Question("x", "a", None, ["a"], []), Question("x", "b", None, ["a"], [])]
    with pytest.raises(ValueError, match="'x'"):
        db.replace_questions(conn, "e", twice, v1)
    assert [q.text for q in get_questions(conn, "e")] == ["one", "two"]
    assert db.get_eval_set(conn, "e")["version"] == v1


def test_replace_questions_that_fail_midway_keep_the_old_ones(conn):
    db.create_eval_set(conn, "e", "mcq")
    save_questions(conn, "e", questions_of("one", "two"))
    v1 = db.get_eval_set(conn, "e")["version"]

    def boom(sql, rows):
        raise sqlite3.OperationalError("disk full")

    conn.executemany = boom  # the insert fails after the delete ran
    with pytest.raises(sqlite3.OperationalError):
        db.replace_questions(conn, "e", questions_of("three"), v1)
    del conn.executemany
    assert [q.text for q in get_questions(conn, "e")] == ["one", "two"]
    assert db.get_eval_set(conn, "e")["version"] == v1
    assert db.replace_questions(conn, "e", questions_of("three"), v1) != v1


def test_migration_pins_runs_a_step_left_unpinned(tmp_path):
    path = tmp_path / "half.db"
    conn = connect(path)
    save_collection(conn, CollectionSpec("c", "mcq", False, "d", 4))
    save_questions(conn, "c", questions_of("one"))  # the set of its name and a first snapshot
    first_version = db.get_eval_set(conn, "c")["version"]
    save_questions(conn, "c", questions_of("one", "two"))
    old = create_job(conn, "eval", "c", {"split": "full"})  # an old run, never pinned
    pinned = create_job(conn, "eval", "c", {"eval_set": "c", "eval_set_version": "v"})
    conn.close()
    conn = connect(path)
    assert get_job(conn, old)["setup"] == {
        "split": "full",
        "eval_set": "c",
        "eval_set_version": first_version,
    }
    assert get_job(conn, pinned)["setup"] == {"eval_set": "c", "eval_set_version": "v"}


def test_chat_attempts_from_before_search_error_migrate_and_answer(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        create table chat_attempts (
          chat_id text not null, n integer not null, attempt integer not null,
          search_text text not null, decisions_json text not null, evidence_json text not null,
          answer text, citations_json text not null, usd real not null, error text,
          model text, created_at text not null, primary key (chat_id, n, attempt)
        );
        insert into chat_attempts
          values ('c', 1, 1, 's', '[]', '[]', null, '[]', 0, 'x', null, 't');
        """
    )
    old.commit()
    old.close()
    conn = connect(path)
    row = {"search_text": "s", "decisions": [], "evidence": [], "answer": None, "citations": []}
    row.update(usd=0.0, search_error="the note", error="the note; failed", model=None)
    assert db.add_chat_attempt(conn, "c", 1, row) == 2
    answered = {"answer": "yes", "citations": [], "usd": 0.5, "error": None, "model": "m"}
    assert db.answer_chat_attempt(conn, "c", 1, 1, answered)  # an old row has no note
    assert db.answer_chat_attempt(conn, "c", 1, 2, answered)
    assert not db.answer_chat_attempt(conn, "c", 1, 2, answered)  # answered: nothing written
    got = db.chat_attempts(conn, "c", 1)
    assert [(a["answer"], a["error"]) for a in got] == [("yes", None), ("yes", "the note")]


def test_chat_attempts_from_before_the_pipeline_read_with_no_pipeline(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        create table chat_attempts (
          chat_id text not null, n integer not null, attempt integer not null,
          search_text text not null, decisions_json text not null, evidence_json text not null,
          answer text, citations_json text not null, usd real not null, error text,
          model text, created_at text not null, search_error text,
          primary key (chat_id, n, attempt)
        );
        insert into chat_attempts
          values ('c', 1, 1, 's', '[]', '[]', 'old', '[]', 0, null, 'm', 't', null);
        """
    )
    old.commit()
    old.close()
    conn = connect(path)
    row = {"search_text": "s", "decisions": [], "evidence": [], "answer": None, "citations": []}
    row.update(usd=0.0, search_error=None, error=None, model=None)
    assert db.add_chat_attempt(conn, "c", 1, row) == 2  # an old-shape row: no pipeline
    pipeline = {"merged_by": "rerank", "pool": 3, "notes": []}
    assert db.add_chat_attempt(conn, "c", 1, {**row, "pipeline": pipeline}) == 3
    got = db.chat_attempts(conn, "c", 1)
    assert [a["pipeline"] for a in got] == [None, None, pipeline]


def test_switching_model_reads_the_newest_unfinished_switch(conn):
    assert db.switching_model(conn, "c") is None
    job = create_job(conn, "reembed", "c", {"model": "m", "switching": True})
    update_job(conn, job, status="interrupted")
    assert db.switching_model(conn, "c") == "m"
    assert db.switching_model(conn, "other") is None
    update_job(conn, job, status="done")
    assert db.switching_model(conn, "c") is None
    update_job(conn, create_job(conn, "reembed", "c", {"model": "n"}), status="interrupted")
    assert db.switching_model(conn, "c") is None


def test_set_dense_model_changes_only_the_model_and_dimension(conn):
    save_collection(conn, SPEC)
    db.set_dense_model(conn, SPEC.name, "fake/small", 64)
    spec = get_collection(conn, SPEC.name)
    assert spec is not None and (spec.dense_model, spec.dense_dim) == ("fake/small", 64)
    assert spec.keyword_version == SPEC.keyword_version and spec.qa_type == SPEC.qa_type


def test_a_rebuild_is_an_item_job():
    assert "reembed" in db.ITEM_JOBS


# -- judged_ids ----------------------------------------------------------------


def stored_json(conn, name, id_):
    return conn.execute("select json from questions where collection = ? and id = ?", (name, id_))[
        0
    ]["json"]


def test_a_write_keeps_judged_a_superset_of_gold(conn):
    db.create_eval_set(conn, "e", "documents")
    q = Question("q1", "t", None, [], ["a", "b"], {}, ["a", "c"])
    save_questions(conn, "e", [q])
    assert get_questions(conn, "e")[0].judged_ids == ["a", "c", "b"]
    unknown = Question("q2", "t", None, [], ["a"], {}, None)
    save_questions(conn, "e", [unknown])
    assert get_questions(conn, "e")[1].judged_ids is None
    assert "judged_ids" not in json.loads(stored_json(conn, "e", "q2"))
    db.replace_questions(conn, "e", [q, unknown], None)
    got = {x.id: x.judged_ids for x in get_questions(conn, "e")}
    assert got == {"q1": ["a", "c", "b"], "q2": None}
    assert "judged_ids" not in json.loads(stored_json(conn, "e", "q2"))


OLD_QUESTION = {
    "id": "q1",
    "text": "t",
    "choices": None,
    "gold_answers": [],
    "gold_ids": ["a"],
    "meta": {},
}


def old_set(conn):
    db.create_eval_set(conn, "e", "documents")
    conn.execute("insert into questions values (?, ?, ?)", ("e", "q1", json.dumps(OLD_QUESTION)))
    return db.snapshot_eval_set(conn, "e")


def test_old_question_json_reads_with_judged_unknown(conn):
    version = old_set(conn)
    assert get_questions(conn, "e")[0].judged_ids is None
    assert get_questions(conn, "e", version)[0].judged_ids is None


def test_an_unchanged_old_set_keeps_its_version(conn):
    version = old_set(conn)
    save_questions(conn, "e", get_questions(conn, "e"))
    assert db.get_eval_set(conn, "e")["version"] == version
    assert len(conn.execute("select * from eval_set_versions")) == 1


def test_a_snapshot_keeps_judged_ids(conn):
    db.create_eval_set(conn, "e", "documents")
    save_questions(conn, "e", [Question("q1", "t", None, [], ["a"], {}, ["a", "b"])])
    version = db.get_eval_set(conn, "e")["version"]
    save_questions(conn, "e", [Question("q1", "t2", None, [], ["a"], {}, ["a"])])
    assert get_questions(conn, "e", version)[0].judged_ids == ["a", "b"]


def test_connect_adds_the_eval_set_store_column(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(db.SCHEMA)
    old.executescript(
        """
        drop table eval_sets;
        create table eval_sets (
          name text primary key, qa_type text not null, version text,
          include_leaked integer not null default 0, created_at text not null
        );
        """
    )
    old.execute(
        "insert into collections (name, qa_type, summarize, dense_model, dense_dim, created_at)"
        " values ('c', 'mcq', 0, 'm', 8, 'now')"
    )
    old.execute("insert into eval_sets (name, qa_type, created_at) values ('c', 'mcq', 'now')")
    old.execute("insert into questions values (?, ?, ?)", ("c", "q1", json.dumps(OLD_QUESTION)))
    old.commit()
    old.close()
    conn = connect(path)  # _migrate_eval_sets reads the set through _eval_set
    assert db.get_eval_set(conn, "c")["store"] is None
    db.set_eval_set_store(conn, "c", "c")
    assert db.get_eval_set(conn, "c")["store"] == "c"


def test_upsert_question_refuses_a_stale_base(conn):
    db.create_eval_set(conn, "e", "mcq")
    q1, q2 = questions_of("one", "two")
    save_questions(conn, "e", [q1, q2])
    q1a = Question(q1.id, "one a", None, ["a"], [])
    q1b = Question(q1.id, "one b", None, ["a"], [])
    q2a = Question(q2.id, "two a", None, ["a"], [])
    db.upsert_question(conn, "e", q1a, q1)
    with pytest.raises(ValueError, match="changed by someone else since you opened it"):
        db.upsert_question(conn, "e", q1b, q1)
    db.upsert_question(conn, "e", q2a, q2)
    assert [q.text for q in get_questions(conn, "e")] == ["one a", "two a"]
    db.delete_question(conn, "e", q2.id)
    with pytest.raises(KeyError):
        db.upsert_question(conn, "e", q2a, q2a)


def test_add_question_picks_a_free_id(conn):
    db.create_eval_set(conn, "e", "mcq")
    q = Question("q-x", "one", None, ["a"], [])
    first, _ = db.add_question(conn, "e", q)
    second, _ = db.add_question(conn, "e", q)
    third, _ = db.add_question(conn, "e", q)
    assert (first, second, third) == ("q-x", "q-x-2", "q-x-3")


def test_upsert_and_delete_make_a_version_each(conn):
    db.create_eval_set(conn, "e", "mcq")
    q = Question("q0", "t0", None, ["a"], [], judged_ids=["j"])
    _, v = db.add_question(conn, "e", q)
    seen = {v}
    for i in (1, 2):
        new = Question("q0", f"t{i}", None, ["a"], [], judged_ids=["j"])
        v = db.upsert_question(conn, "e", new, q)
        q = new
        assert v not in seen
        seen.add(v)
        assert len(db.eval_set_versions(conn, "e")) == len(seen)
    assert get_questions(conn, "e")[0].judged_ids == ["j"]
    v = db.delete_question(conn, "e", "q0")
    assert v not in seen and get_questions(conn, "e") == []
    with pytest.raises(KeyError):
        db.delete_question(conn, "e", "q0")
