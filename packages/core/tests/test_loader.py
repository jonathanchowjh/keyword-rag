import pytest

from ragqa_core import loader
from ragqa_core.db import connect, create_job, get_job
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.llm import Cancelled
from ragqa_core.loader import Loader
from ragqa_core.store import Store
from ragqa_core.types import Item, Question


def build(tmp_path, monkeypatch, reply=lambda p: "I cannot help with that."):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_claude(conn, reply))
    ix.create("m", "multihop", False)
    items = [
        Item("a", "enzyme rate rose shared words", "A"),
        Item("b", "market price fell shared words", "B"),
        Item("c", "volcano lava flowed shared words", "C"),
    ]
    questions = [Question("q1", "enzyme rate", None, ["x"], ["a"], {})]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, questions))
    job_id = create_job(conn, "load", "m", {"dataset": "musique", "limit": None})
    return conn, ix, Loader(conn, ix, ix.claude, tmp_path), get_job(conn, job_id)


def test_loader_runs_a_load_job(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    seen: list[float] = []
    result = ld.run(job, lambda: seen.append(0.0))
    assert result["added"] == 3 and result["questions"] == 1
    assert result["lists_source"] == "tfidf"
    assert result["keyword_version"] == ix.spec("m").keyword_version is not None
    assert seen  # check() ran through the add progress callback
    assert get_job(conn, job["id"])["progress"] == pytest.approx(0.9)


def test_loader_falls_back_to_tfidf_when_claude_proposes_nothing(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    assert ld.run(job, lambda: None)["lists_source"] == "tfidf (claude proposal unusable)"


def test_loader_propagates_cancelled_from_check(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)

    def check():
        raise Cancelled()

    with pytest.raises(Cancelled):
        ld.run(job, check)


def test_load_limit_keeps_gold_passages_and_only_answerable_questions():
    from ragqa_core.loader import limited
    from ragqa_core.types import Item, Question

    items = [Item(f"i{n}", "t") for n in range(6)]
    questions = [
        Question("q1", "x", None, ["a"], ["i4", "i5"]),
        Question("q2", "x", None, ["a"], ["i0", "i3"]),  # would pass the limit
        Question("q3", "x", None, ["a"], ["i1"]),
        Question("mcq", "x", ["a"], ["A"], []),  # retrieves examples, not gold passages
    ]
    kept, answerable = limited(items, questions, 3)
    assert [i.id for i in kept] == ["i1", "i4", "i5"]
    assert [q.id for q in answerable] == ["q1", "q3", "mcq"]
    assert limited(items, questions, None) == (items, questions)
