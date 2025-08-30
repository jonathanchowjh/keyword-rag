import pytest

from ragqa_core.db import (
    JOBS_RESUMABLE,
    add_run_item,
    claim_next_job,
    connect,
    create_job,
    delete_collection_rows,
    delete_job,
    get_collection,
    get_job,
    get_keyword_lists,
    get_questions,
    interrupt_running,
    is_cancel_requested,
    job_running,
    list_collections,
    list_jobs,
    log_job,
    request_cancel,
    requeue,
    run_items,
    running_eval,
    save_collection,
    save_keyword_lists,
    save_questions,
    set_job_setup,
    set_keyword_version,
    update_job,
)
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


def test_run_items_filter_by_method(tmp_path):
    conn = connect(tmp_path / "r.db")
    job = create_job(conn, "eval", "arc", {})
    add_run_item(conn, job, "hybrid", "q1", {"hit_ids": ["a"]})
    add_run_item(conn, job, "vector", "q1", {"hit_ids": ["b"]})
    assert [r["hit_ids"] for r in run_items(conn, job, "hybrid")] == [["a"]]
    assert len(run_items(conn, job)) == 2


def test_connect_is_idempotent(tmp_path):
    connect(tmp_path / "r.db")
    connect(tmp_path / "r.db")


def test_connect_adds_the_batch_estimate_to_a_0_1_0_database(tmp_path):
    import sqlite3

    old = sqlite3.connect(tmp_path / "r.db")
    old.execute(
        "create table pending_batches (id text primary key, model text not null,"
        " purpose text not null, created_at text not null)"
    )
    old.execute("insert into pending_batches values ('b', 'm', 'p', 'now')")
    old.commit()
    old.close()
    row = connect(tmp_path / "r.db").execute("select usd from pending_batches").fetchone()
    assert row["usd"] == 0


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path / "q.db")


def test_claim_takes_the_oldest_queued_job_by_insertion(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    first = claim_next_job(conn)
    assert first is not None and first["id"] == a and first["status"] == "running"
    assert get_job(conn, a)["status"] == "running"
    second = claim_next_job(conn)
    assert second is not None and second["id"] == b and claim_next_job(conn) is None


def test_interrupt_running_keeps_the_queue(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    claim_next_job(conn)
    assert interrupt_running(conn) == 1
    assert get_job(conn, a)["status"] == "interrupted"
    assert get_job(conn, b)["status"] == "queued"


def test_cancel_queued_is_immediate_running_is_requested(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    claim_next_job(conn)
    assert request_cancel(conn, b) == "cancelled"
    assert request_cancel(conn, a) == "running" and is_cancel_requested(conn, a)
    assert not is_cancel_requested(conn, b)


def test_running_eval_ignores_queued(conn):
    create_job(conn, "eval", "c", {})
    assert running_eval(conn, "c") is False
    claim_next_job(conn)
    assert running_eval(conn, "c") is True
    assert running_eval(conn, "other") is False


def test_job_running_matches_kind_and_collection(conn):
    create_job(conn, "load", "c", {})
    assert job_running(conn, "load", "c") is False
    claim_next_job(conn)
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
    claim_next_job(conn)
    request_cancel(conn, j)
    update_job(conn, j, status="failed", error="boom")
    requeue(conn, j)
    job = get_job(conn, j)
    assert job["status"] == "queued" and job["cancel_requested"] is False
    assert job["error"] is None


def test_set_job_setup_and_list_jobs(conn):
    a = create_job(conn, "load", "c", {})
    b = create_job(conn, "eval", "c", {})
    create_job(conn, "eval", "d", {})
    set_job_setup(conn, a, {"limit": 5})
    assert get_job(conn, a)["setup"] == {"limit": 5}
    assert [j["id"] for j in list_jobs(conn, collection="c")] == [b, a]
    assert [j["id"] for j in list_jobs(conn, kind="load")] == [a]
    claim_next_job(conn)
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
    add_run_item(conn, a, "hybrid", "q1", {})
    add_run_item(conn, b, "hybrid", "q1", {})
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
    add_run_item(conn, gone, "hybrid", "q1", {})
    add_run_item(conn, kept, "hybrid", "q1", {})
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
    import sqlite3

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
