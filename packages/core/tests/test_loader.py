import pytest

from ragqa_core import loader
from ragqa_core.db import connect, create_job, get_job, get_questions
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.llm import Cancelled
from ragqa_core.loader import Loader
from ragqa_core.store import Store
from ragqa_core.types import Item, KeywordLists, Question


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
    result = ld.run(job, seen.append)
    assert result["added"] == 3 and result["questions"] == 1
    assert result["lists_source"] == "tfidf"
    assert result["keyword_version"] == ix.spec("m").keyword_version is not None
    assert seen[-1] == pytest.approx(1.0) and seen == sorted(seen)


def test_a_load_that_makes_lists_reports_and_cancels_inside_the_rekey(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    real = ix.store.scroll
    ix.store.scroll = lambda name: real(name, batch=1)  # one item per re-keyed step
    seen: list[float] = []
    ld.run(job, seen.append)
    assert 0.5 in seen  # adding is the first half
    assert any(0.5 < p < 1.0 for p in seen)  # the re-key reports its own steps

    conn, ix, ld, job = build(tmp_path / "again", monkeypatch)
    ld.claude.api_key = None
    ix.store.scroll = lambda name: real(name, batch=1)

    def progress(p):
        if 0.5 < p < 1.0:
            raise Cancelled()

    with pytest.raises(Cancelled):
        ld.run(job, progress)
    assert ix.spec("m").keyword_version is None  # the lists never took effect


def test_a_load_into_a_collection_with_lists_reports_adding_as_the_whole(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    ld.run(job, lambda p: None)
    version = ix.spec("m").keyword_version
    seen: list[float] = []
    ld.run(job, seen.append)
    assert seen == [1.0] and ix.spec("m").keyword_version == version


def test_loader_falls_back_to_tfidf_when_claude_proposes_nothing(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    assert ld.run(job, lambda p: None)["lists_source"] == "tfidf (claude proposal unusable)"


def test_loader_propagates_cancelled_from_check(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)

    def progress(p):
        raise Cancelled()

    with pytest.raises(Cancelled):
        ld.run(job, progress)


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


def test_load_twice_reports_unchanged(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    first = ld.run(job, lambda p: None)
    assert first["new"] == 3 and first["added"] == 3
    second = ld.run(job, lambda p: None)
    assert second["unchanged"] == 3 and second["added"] == 0


def test_sample_items_skips_disabled(tmp_path):
    conn = connect(tmp_path / "r.db")
    store = Store(None, None)
    ix = Indexer(conn, store, fake_models(), fake_claude(conn, lambda p: ""))
    spec = ix.create("m", "multihop", False)
    ix.add(spec, [Item("a", "enzyme rate", "A")], "one")
    ix.add(spec, [Item("b", "market price", "B")], "two")
    assert {i.id for i in loader.sample_items(store, "m")} == {"a", "b"}
    assert [i.id for i in loader.sample_items(store, "m", skip=("two",))] == ["a"]


def upload(tmp_path, name, rows):
    import json

    from ragqa_core import imports

    data = "\n".join(json.dumps(r) for r in rows).encode()
    return imports.save_upload(tmp_path, "m", name, [data])


def import_job(conn, tmp_path, kind, rows, mapping, source="upload:notes", name="m"):
    upload_id = upload(tmp_path, "rows.jsonl", rows)
    setup = {"upload_id": upload_id, "kind": kind, "mapping": mapping, "source": source}
    return get_job(conn, create_job(conn, "import", name, setup))


ITEM_MAP = {"id": "id", "text": "text"}
ITEM_ROWS = [{"id": f"n{i}", "text": f"note number {i} about topic {i}"} for i in range(4)]


def test_import_items_then_reimport_is_free(tmp_path, monkeypatch):
    calls: list[str] = []
    conn = connect(tmp_path / "r.db")
    claude = fake_claude(conn, lambda p: calls.append(p) or "A summary.")
    ix = Indexer(conn, Store(None, None), fake_models(), claude)
    spec = ix.create("m", "multihop", True)
    ix.set_lists(spec, KeywordLists(topic=["note"], method=["topic"], common=["number"]))
    ld = Loader(conn, ix, claude, tmp_path)
    first = ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert len(calls) == 4 and first["new"] == 4
    calls.clear()
    again = ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert calls == [] and again["unchanged"] == 4 and again["new"] == 0
    assert ix.store.get("m", ["n0"])[0].meta["source"] == "upload:notes"


def test_import_items_without_lists_makes_tfidf_lists(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    assert ix.spec("m").keyword_version is None
    ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert ix.spec("m").keyword_version is not None


LONG = "the enzyme rate rose sharply after the temperature was raised"


def test_import_questions_marks_leaked(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    ld.run_import(
        import_job(conn, tmp_path, "items", [{"id": "i1", "text": LONG + " in the lab"}], ITEM_MAP),
        lambda p: None,
    )
    rows = [
        {"id": "q1", "text": LONG, "answer": "x"},
        {"id": "q2", "text": "an unrelated question that appears nowhere", "answer": "y"},
    ]
    mapping = {"id": "id", "text": "text", "answer": "answer"}
    job = import_job(conn, tmp_path, "questions", rows, mapping, source="upload:qs")
    result = ld.run_import(job, lambda p: None)
    assert result["leaked"] == 1 and result["new"] == 2
    got = {q.id: q for q in get_questions(conn, "m")}
    assert got["q1"].meta["leaked"] is True and "leaked" not in got["q2"].meta
    assert got["q1"].meta["source"] == "upload:qs"
    again = ld.run_import(job, lambda p: None)
    assert again["unchanged"] == 2 and again["leaked"] == 1


def test_dataset_load_never_marks_builtin_questions(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.claude.api_key = None
    items = [Item("a", LONG + " in the lab", "A")]
    questions = [Question("q1", LONG, None, ["x"], ["a"], {})]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, questions))
    result = ld.run(job, lambda p: None)
    assert result["leaked"] == 0
    assert "leaked" not in get_questions(conn, "m")[0].meta


def test_estimates_count_only_changed_items(tmp_path, monkeypatch):
    conn = connect(tmp_path / "r.db")
    claude = fake_claude(conn, lambda p: "A summary.")
    ix = Indexer(conn, Store(None, None), fake_models(), claude)
    spec = ix.create("m", "multihop", True)
    ld = Loader(conn, ix, claude, tmp_path)
    items = [Item("a", "enzyme rate", "A"), Item("b", "market price", "B")]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    assert ld.estimate_load(spec, "musique", None)["calls"] == 3  # two summaries and a proposal
    ix.add(spec, items[:1], "musique")
    assert ld.estimate_load(spec, "musique", None)["calls"] == 2  # one summary and a proposal
    job = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP)
    est = ld.estimate_import(spec, job["setup"])
    assert est["new"] == 4 and est["skipped"] == 0 and est["calls"] == 5 and est["est_usd"] > 0
