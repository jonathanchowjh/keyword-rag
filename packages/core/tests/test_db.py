from ragqa_core.db import (
    add_run_item,
    connect,
    create_job,
    get_collection,
    get_job,
    get_keyword_lists,
    get_questions,
    list_collections,
    mark_interrupted,
    run_items,
    save_collection,
    save_keyword_lists,
    save_questions,
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


def test_mark_interrupted(tmp_path):
    conn = connect(tmp_path / "r.db")
    a = create_job(conn, "eval", "arc", {})
    update_job(conn, a, status="running")
    b = create_job(conn, "eval", "arc", {})
    update_job(conn, b, status="done")
    assert mark_interrupted(conn) == 1
    job_a, job_b = get_job(conn, a), get_job(conn, b)
    assert job_a is not None and job_a["status"] == "interrupted"
    assert job_b is not None and job_b["status"] == "done"


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
