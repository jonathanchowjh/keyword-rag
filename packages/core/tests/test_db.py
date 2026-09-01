import hashlib
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from ragqa_core import db
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
from ragqa_core.fakes import fake_ai
from ragqa_core.types import CollectionSpec, KeywordLists, Question

SPEC = CollectionSpec("arc", "mcq", True, "m", 64)


def test_collection_round_trip(tmp_path):
    conn = connect(tmp_path / "r.db")
    save_collection(conn, CollectionSpec("arc", "mcq", True, "m", 64))
    assert get_collection(conn, "arc") == CollectionSpec("arc", "mcq", True, "m", 64, None)
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
    assert get_questions(conn, "c") == []
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
