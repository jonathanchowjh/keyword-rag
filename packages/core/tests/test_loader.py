import json
from dataclasses import replace
from pathlib import Path

import pytest

from aiconnect.ai import Cancelled
from ragqa_core import db, loader
from ragqa_core.builder import Builder
from ragqa_core.datasets import load_scibench
from ragqa_core.db import connect, create_job, get_job, get_keyword_lists, get_questions
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.loader import Loader
from ragqa_core.store import Store
from ragqa_core.types import Item, KeywordLists, Question

PROPOSAL = (
    '{"topic": ["enzyme", "market"], "method": ["assay"], "common": ["shared", "words", "rate"]}'
)


def proposing(prompt: str) -> str:
    return PROPOSAL


def purposes(conn) -> list[str]:
    return [r["purpose"] for r in conn.execute("select purpose from ai_ledger")]


def loader_for(ix: Indexer, home) -> Loader:
    return Loader(ix.conn, ix, Builder(ix.conn, ix.store, ix.models, ix.ai, ix), ix.ai, home)


def build(tmp_path, monkeypatch, reply=lambda p: "I cannot help with that."):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, reply))
    ix.create("m", "multihop", False)
    items = [
        Item("a", "enzyme rate rose shared words", "A"),
        Item("b", "market price fell shared words", "B"),
        Item("c", "volcano lava flowed shared words", "C"),
    ]
    questions = [Question("q1", "enzyme rate", None, ["x"], ["a"], {})]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, questions))
    job_id = create_job(conn, "load", "m", {"dataset": "musique", "limit": None})
    return conn, ix, loader_for(ix, tmp_path), get_job(conn, job_id)


def test_loader_runs_a_load_job(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    seen: list[float] = []
    result = ld.run(job, seen.append)
    assert result["added"] == 3 and result["questions"] == 1
    assert result["lists_source"] == "ai"
    assert result["keyword_version"] == ix.spec("m").keyword_version is not None
    assert seen[-1] == pytest.approx(1.0) and seen == sorted(seen)


def test_load_without_ai_saves_no_lists(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    ld.ai.update_connection("anthropic", api_key=None)
    seen: list[float] = []
    result = ld.run(job, seen.append)
    assert result["added"] == 3 and result["lists_source"] is None
    assert result["keyword_version"] is None and get_keyword_lists(conn, "m") is None
    assert seen[-1] == pytest.approx(1.0)


def test_proposed_lists_include_common(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    ld.run(job, lambda p: None)
    lists = get_keyword_lists(conn, "m")
    assert lists is not None and lists.common == ["shared", "words", "rate"]


def test_a_load_that_makes_lists_reports_and_cancels_inside_the_rekey(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    real = ix.store.scroll
    ix.store.scroll = lambda name: real(name, batch=1)  # one item per re-keyed step
    seen: list[float] = []
    ld.run(job, seen.append)
    assert 0.5 in seen  # adding is the first half
    assert any(0.5 < p < 1.0 for p in seen)  # the re-key reports its own steps

    conn, ix, ld, job = build(tmp_path / "again", monkeypatch, proposing)
    ix.store.scroll = lambda name: real(name, batch=1)

    def progress(p):
        if 0.5 < p < 1.0:
            raise Cancelled()

    with pytest.raises(Cancelled):
        ld.run(job, progress)
    assert ix.spec("m").keyword_version is None  # the lists never took effect


def test_a_load_into_a_collection_with_lists_reports_adding_as_the_whole(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    ld.run(job, lambda p: None)
    version = ix.spec("m").keyword_version
    seen: list[float] = []
    ld.run(job, seen.append)
    assert seen == [1.0] and ix.spec("m").keyword_version == version


def test_an_unusable_proposal_saves_no_lists(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    seen: list[float] = []
    result = ld.run(job, seen.append)
    # Named, unlike "no AI model" (None): the proposal was paid for and is why there are none.
    assert result["lists_source"] == loader.UNUSABLE_PROPOSAL == "none (ai proposal unusable)"
    assert result["keyword_version"] is None and ix.spec("m").keyword_version is None
    assert [r["purpose"] for r in conn.execute("select purpose from ai_ledger")] == ["propose"]
    assert seen[-1] == pytest.approx(1.0)


def test_an_induced_recipe_load_never_proposes_lists(tmp_path, monkeypatch):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, proposing))
    spec = ix.create("w", "science", False, recipe="worked_examples")
    ld = loader_for(ix, tmp_path)
    items = [Item("a", "a ball falls 5 m", "10 m/s", "use conservation of energy")]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    assert ld.ai.ready("keywords") and not ld.makes_lists(spec)
    estimate = ld.estimate_load(spec, "scibench", None)
    assert estimate["calls"] == 3  # induce, tag and abstract: no propose priced
    job = get_job(conn, create_job(conn, "load", "w", {"dataset": "scibench", "limit": None}))
    seen: list[float] = []
    result = ld.run(job, seen.append)
    assert result["added"] == 1 and result["lists_source"] is None
    assert get_keyword_lists(conn, "w") is None
    assert "propose" not in purposes(conn)
    assert seen[-1] == pytest.approx(1.0)
    ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP, name="w"), seen.append)
    assert get_keyword_lists(conn, "w") is None
    assert "propose" not in purposes(conn)


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
    ld.ai.update_connection("anthropic", api_key=None)
    first = ld.run(job, lambda p: None)
    assert first["new"] == 3 and first["added"] == 3
    second = ld.run(job, lambda p: None)
    assert second["unchanged"] == 3 and second["added"] == 0


def test_a_load_keeps_only_the_excluded_ids_the_collection_stores(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch)
    ld.ai.update_connection("anthropic", api_key=None)
    items = [Item(f"i{n}", f"text {n} shared words", "A") for n in range(6)]
    question = Question("q", "x", None, ["a"], ["i1"], {"excluded_ids": ["i0", "i4", "z9"]})
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, [question]))
    small = create_job(conn, "load", "m", {"dataset": "musique", "limit": 2})
    ld.run(get_job(conn, small), lambda p: None)  # stores i1 (the gold) and i0
    (saved,) = get_questions(conn, "m")
    assert saved.meta["excluded_ids"] == ["i0"]
    full = create_job(conn, "load", "m", {"dataset": "musique", "limit": None})
    ld.run(get_job(conn, full), lambda p: None)  # now i4 is stored: back in force
    (saved,) = get_questions(conn, "m")
    assert saved.meta["excluded_ids"] == ["i0", "i4"]


def test_sample_items_skips_disabled(tmp_path):
    conn = connect(tmp_path / "r.db")
    store = Store(None, None)
    ix = Indexer(conn, store, fake_models(), fake_ai(conn, lambda p: ""))
    spec = ix.create("m", "multihop", False)
    ix.add(spec, [Item("a", "enzyme rate", "A")], "one")
    ix.add(spec, [Item("b", "market price", "B")], "two")
    assert {i.id for i in loader.sample_items(store, "m")} == {"a", "b"}
    assert [i.id for i in loader.sample_items(store, "m", skip=("two",))] == ["a"]


def upload(tmp_path, name, rows, collection="m"):
    import json

    from ragqa_core import imports

    data = "\n".join(json.dumps(r) for r in rows).encode()
    return imports.save_upload(tmp_path, collection, name, [data])


def import_job(conn, tmp_path, kind, rows, mapping, source="upload:notes", name="m"):
    upload_id = upload(tmp_path, "rows.jsonl", rows, name)
    setup = {"upload_id": upload_id, "kind": kind, "mapping": mapping, "source": source}
    return get_job(conn, create_job(conn, "import", name, setup))


ITEM_MAP = {"id": "id", "text": "text"}
ITEM_ROWS = [{"id": f"n{i}", "text": f"note number {i} about topic {i}"} for i in range(4)]


def test_import_items_then_reimport_is_free(tmp_path, monkeypatch):
    calls: list[str] = []
    conn = connect(tmp_path / "r.db")
    ai = fake_ai(conn, lambda p: calls.append(p) or "A summary.")
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("m", "multihop", True)
    ix.set_lists(spec, KeywordLists(topic=["note"], method=["topic"], common=["number"]))
    ld = loader_for(ix, tmp_path)
    first = ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert len(calls) == 4 and first["new"] == 4
    calls.clear()
    again = ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert calls == [] and again["unchanged"] == 4 and again["new"] == 0
    assert ix.store.get("m", ["n0"])[0].meta["source"] == "upload:notes"


def test_an_import_estimate_on_cached_work_counts_no_calls(tmp_path, monkeypatch):
    conn = connect(tmp_path / "r.db")
    ai = fake_ai(conn, lambda p: "A summary.")
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("m", "multihop", True)
    ix.set_lists(spec, KeywordLists(topic=["note"], method=["topic"], common=["number"]))
    ld = loader_for(ix, tmp_path)
    ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    ix.store.delete_items("m", [r["id"] for r in ITEM_ROWS])  # new again, summaries cached
    setup = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP)["setup"]
    est = ld.estimate_import(spec, setup)
    assert est["new"] == 4 and est["calls"] == 0 and est["est_usd"] == 0


def test_a_build_estimate_on_cached_work_counts_no_calls(tmp_path, monkeypatch):
    conn, ix, ld, _ = worked_load(tmp_path, monkeypatch)
    ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP, name="w"), lambda p: 0)
    ix.store.delete_items("w", [r["id"] for r in ITEM_ROWS])  # untagged again, tags cached
    setup = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP, name="w")["setup"]
    spec = ix.spec("w")
    est = ld.estimate_import(spec, setup)
    assert est["new"] == 4 and est["calls"] == 0 and est["est_usd"] == 0
    items = [Item(r["id"], r["text"], None) for r in ITEM_ROWS]
    assert ld.builder.build_calls(spec, items) == 4  # four tags to make, all cached


def test_import_items_without_lists_makes_ai_lists(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch, proposing)
    assert ix.spec("m").keyword_version is None
    ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert ix.spec("m").keyword_version is not None


def test_import_items_without_ai_makes_no_lists(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch, proposing)
    ld.ai.update_connection("anthropic", api_key=None)
    ld.run_import(import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP), lambda p: None)
    assert ix.spec("m").keyword_version is None


LONG = "the enzyme rate rose sharply after the temperature was raised"


def test_import_questions_keeps_their_source_and_marks_nothing(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch)
    ld.ai.update_connection("anthropic", api_key=None)
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
    assert result["new"] == 2 and "leaked" not in result
    got = {q.id: q for q in get_questions(conn, "m")}
    assert all("leaked" not in q.meta for q in got.values())  # worked out per run instead
    assert got["q1"].meta["source"] == "upload:qs"
    again = ld.run_import(job, lambda p: None)
    assert again["unchanged"] == 2


def test_a_load_refuses_a_default_eval_set_of_another_type(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ix.create("x", "multihop", False)
    db.create_eval_set(conn, "x", "mcq")
    job = get_job(conn, create_job(conn, "load", "x", {"dataset": "musique", "limit": None}))
    with pytest.raises(ValueError, match="eval set 'x' is mcq, not multihop"):
        ld.run(job, lambda p: None)
    assert ix.store.count("x") == 0 and get_questions(conn, "x") == []


def test_a_load_makes_the_default_eval_set(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.run(job, lambda p: None)
    assert db.get_eval_set(conn, "m")["qa_type"] == "multihop"
    assert db.default_eval_set(conn, "m") == "m" and len(get_questions(conn, "m")) == 1


def test_dataset_load_never_marks_builtin_questions(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch)
    ld.ai.update_connection("anthropic", api_key=None)
    items = [Item("a", LONG + " in the lab", "A")]
    questions = [Question("q1", LONG, None, ["x"], ["a"], {})]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, questions))
    result = ld.run(job, lambda p: None)
    assert "leaked" not in result
    assert "leaked" not in get_questions(conn, "m")[0].meta


def test_estimates_count_only_changed_items(tmp_path, monkeypatch):
    conn = connect(tmp_path / "r.db")
    ai = fake_ai(conn, lambda p: "A summary.")
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("m", "multihop", True)
    ld = loader_for(ix, tmp_path)
    items = [Item("a", "enzyme rate", "A"), Item("b", "market price", "B")]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    assert ld.estimate_load(spec, "musique", None)["calls"] == 3  # two summaries and a proposal
    ix.add(spec, items[:1], "musique")
    assert ld.estimate_load(spec, "musique", None)["calls"] == 2  # one summary and a proposal
    job = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP)
    est = ld.estimate_import(spec, job["setup"])
    assert est["new"] == 4 and est["skipped"] == 0 and est["calls"] == 5 and est["est_usd"] > 0


def test_import_records_an_unusable_proposal(tmp_path, monkeypatch):
    conn, ix, ld, _ = build(tmp_path, monkeypatch)  # the AI replies with no lists
    job = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP)
    result = ld.run_import(job, lambda p: None)
    assert result["lists_source"] == loader.UNUSABLE_PROPOSAL and purposes(conn) == ["propose"]


SCIBENCH = Path(__file__).parent / "data" / "scibench_mini"
ABSTRACTED = json.dumps(
    {"principle": "Energy is conserved.", "method": "Equate the energies.", "answer": "3 m/s"}
)


def by_purpose(prompt: str) -> str:
    if "skills or principles" in prompt:
        return '{"terms": ["energy conservation", "kinematics"]}'
    if "Pick 1 to" in prompt:
        return '{"terms": ["kinematics"]}'
    if '"principle"' in prompt:
        return ABSTRACTED
    return "Answer: A"


def scibench_items(cache) -> list[Item]:
    items, _ = load_scibench(cache, {p.name: p for p in SCIBENCH.glob("*.json")})
    return items


def worked_load(tmp_path, monkeypatch):
    """A fresh worked-examples collection w and a load job of the scibench mini files."""
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, by_purpose))
    ix.create("w", "science", False, recipe="worked_examples")
    files = {p.name: p for p in SCIBENCH.glob("*.json")}
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: load_scibench(cache, files))
    job = get_job(conn, create_job(conn, "load", "w", {"dataset": "scibench", "limit": None}))
    return conn, ix, loader_for(ix, tmp_path), job


def test_worked_load_builds_vocabulary_tags_and_abstracts(tmp_path, monkeypatch):
    conn, ix, ld, job = worked_load(tmp_path, monkeypatch)
    items = scibench_items(tmp_path)
    seen: list[float] = []
    result = ld.run(job, seen.append)
    spec = ix.spec("w")
    # Induced, saved and switched to before tagging: tag() reads the version in force.
    assert result["vocabulary_version"] == spec.vocabulary_version is not None
    assert db.get_vocabulary(conn, "w") == ["energy conservation", "kinematics"]
    assert result["tagged"] == len(items) and result["abstracted"] == 2
    hits = [h for batch in ix.store.scroll("w") for h in batch]
    assert len(hits) == len(items) and all(h.tags == ["kinematics"] for h in hits)
    explained = {i.id for i in items if i.explanation}
    assert {h.id for h in hits if h.abstracted} == explained and len(explained) == 2
    assert all(h.abstracted.startswith("Principle: ") for h in hits if h.abstracted)
    assert spec.profile == {"kinematics": len(items)}
    assert set(purposes(conn)) == {"induce", "tag", "abstract"}
    assert 0.5 in seen and seen == sorted(seen) and seen[-1] == pytest.approx(1.0)
    # The same load again adds nothing, so it builds and spends nothing.
    spent = len(purposes(conn))
    again = ld.run(job, lambda p: None)
    assert again["tagged"] == 0 and again["abstracted"] == 0 and len(purposes(conn)) == spent


def test_paper_2024_load_unchanged(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    seen: list[float] = []
    result = ld.run(job, seen.append)
    assert not {"vocabulary_version", "tagged", "abstracted"} & set(result)
    assert purposes(conn) == ["propose"] and result["lists_source"] == "ai"
    assert ix.spec("m").vocabulary_version is None
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_load_estimate_includes_build_cost(tmp_path, monkeypatch):
    conn, ix, ld, job = worked_load(tmp_path, monkeypatch)
    spec = ix.spec("w")
    items = ix.sourced(scibench_items(tmp_path), "scibench")
    estimate = ld.estimate_load(spec, "scibench", None)
    build_usd = ld.builder.estimate_build_usd(spec, items)
    assert build_usd and estimate["est_usd"] == pytest.approx(build_usd)
    # One induction batch, a tag per item, an abstract per explained item.
    assert estimate["calls"] == ld.builder.build_calls(spec, items) == 1 + len(items) + 2
    setup = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP, name="w")["setup"]
    imported = ld.estimate_import(spec, setup)
    assert imported["calls"] == 1 + 4 and imported["est_usd"] > 0  # no explanations
    assert purposes(conn) == []
    ld.run(job, lambda p: None)
    assert ld.estimate_load(spec, "scibench", None) == {
        "est_usd": 0,
        "items": len(items),
        "calls": 0,
        "hub": {},  # nothing left to build, so the Hub is not read
    }


def test_load_estimate_is_none_without_a_keywords_model(tmp_path, monkeypatch):
    conn, ix, ld, job = worked_load(tmp_path, monkeypatch)
    ld.ai.set_role("default", None)
    assert ld.estimate_load(ix.spec("w"), "scibench", None)["est_usd"] is None


def test_worked_import_builds_its_items(tmp_path, monkeypatch):
    conn, ix, ld, _ = worked_load(tmp_path, monkeypatch)
    job = import_job(conn, tmp_path, "items", ITEM_ROWS, ITEM_MAP, name="w")
    seen: list[float] = []
    result = ld.run_import(job, seen.append)
    assert result["tagged"] == 4 and result["abstracted"] == 0
    assert result["lists_source"] is None and result["vocabulary_version"] is not None
    assert purposes(conn) == ["induce"] + ["tag"] * 4
    assert 0.5 in seen and seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def stop_between(low: float, high: float):
    def progress(p: float) -> None:
        if low < p < high:
            raise Cancelled()

    return progress


def test_a_stopped_build_is_finished_by_loading_again(tmp_path, monkeypatch):
    conn, ix, ld, job = worked_load(tmp_path, monkeypatch)
    items = scibench_items(tmp_path)
    # Induce, tag, abstract: the build's thirds of the job's second half.
    with pytest.raises(Cancelled):
        ld.run(job, stop_between(0.6, 0.7))  # as the induction ends, before it is saved
    assert ix.spec("w").vocabulary_version is None
    # The items are stored, so this load adds none, yet induces over all of them.
    with pytest.raises(Cancelled):
        ld.run(job, stop_between(0.7, 0.8))  # inside the tagging, before any tag is written
    assert ix.spec("w").vocabulary_version is not None
    assert all(not h.tags for batch in ix.store.scroll("w") for h in batch)
    # What is left is priced: a tag per item and an abstract per explained one, no induction.
    # The stopped tagging's replies are cached, so only the abstracts would be sent.
    spec = ix.spec("w")
    assert ld.builder.build_calls(spec, []) == len(items) + 2
    assert ld.estimate_load(spec, "scibench", None)["calls"] == 2
    result = ld.run(job, lambda p: None)
    assert result["added"] == 0 and result["tagged"] == len(items) and result["abstracted"] == 2
    hits = [h for batch in ix.store.scroll("w") for h in batch]
    assert all(h.tags == ["kinematics"] for h in hits)
    assert sum(1 for h in hits if h.abstracted) == 2
    assert ix.spec("w").profile == {"kinematics": len(items)}
    # Built in full: loading again spends nothing.
    spent = len(purposes(conn))
    again = ld.run(job, lambda p: None)
    assert again["tagged"] == 0 and again["abstracted"] == 0 and len(purposes(conn)) == spent
    assert ld.estimate_load(spec, "scibench", None)["calls"] == 0


def scibench_paper(tmp_path):
    """Collection p (paper 2024, science) holding the scibench mini items, and one question."""
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, by_purpose))
    spec = ix.create("p", "science", False)
    items = scibench_items(tmp_path)
    ix.add(spec, items, "scibench")
    question = Question("q1", "x", None, ["3"], [items[0].id], {"excluded_ids": [items[1].id]})
    db.save_questions(conn, "p", [question])
    return conn, ix, loader_for(ix, tmp_path), items


def test_copy_rebuilds_the_items_under_another_recipe(tmp_path):
    conn, ix, ld, items = scibench_paper(tmp_path)
    target = ld.copy_spec(ix.spec("p"), "w", "worked_examples", None)
    estimate = ld.estimate_copy("p", target)
    # One induction batch, a tag per item, an abstract per explained item.
    assert estimate["items"] == len(items) and estimate["calls"] == 1 + len(items) + 2
    assert estimate["est_usd"] > 0 and purposes(conn) == [] and ix.spec("w") is None
    job = get_job(conn, ld.start_copy("p", target))
    assert job is not None and (job["kind"], job["collection"]) == ("copy", "w")
    assert db.default_eval_set(conn, "w") == "p"  # points to the source's set, copies nothing
    assert db.get_eval_set(conn, "w") is None
    seen: list[float] = []
    result = ld.run_copy(job, seen.append)
    spec = ix.spec("w")
    assert spec is not None and spec.recipe == "worked_examples" and spec.qa_type == "science"
    assert spec.search is not None and spec.search.example == "abstracted"
    assert result["added"] == len(items) and result["sources"] == ["scibench"]
    assert result["vocabulary_version"] == spec.vocabulary_version is not None
    assert result["tagged"] == len(items) and result["abstracted"] == 2
    source = {h.id: h for batch in ix.store.scroll("p") for h in batch}
    copied = {h.id: h for batch in ix.store.scroll("w") for h in batch}
    assert {i: (h.text, h.answer, h.explanation, h.meta) for i, h in copied.items()} == {
        i: (h.text, h.answer, h.explanation, h.meta) for i, h in source.items()
    }
    assert all(h.tags == ["kinematics"] for h in copied.values())
    assert set(purposes(conn)) == {"induce", "tag", "abstract"}
    assert 0.5 in seen and seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_copy_points_to_the_source_eval_set(tmp_path):
    conn, ix, ld, items = scibench_paper(tmp_path)
    ld.start_copy("p", ld.copy_spec(ix.spec("p"), "w", "worked_examples", None))
    assert db.default_eval_set(conn, "w") == "p"
    assert get_questions(conn, "w") == []


def test_copy_into_paper_2024_takes_the_source_lists(tmp_path, monkeypatch):
    conn, ix, ld, job = build(tmp_path, monkeypatch, proposing)
    ld.run(job, lambda p: None)
    source = ix.spec("m")
    assert source is not None and source.keyword_version and purposes(conn) == ["propose"]
    target = ld.copy_spec(source, "m2", "paper_2024", {"chunk_overlap": 10})
    assert ld.estimate_copy("m", target) == {"est_usd": 0.0, "items": 3, "calls": 0}
    result = ld.run_copy(get_job(conn, ld.start_copy("m", target)), lambda p: None)
    assert result == {"added": 3, "sources": ["musique"]}
    spec = ix.spec("m2")
    assert spec is not None and spec.build is not None and spec.build.chunk_overlap == 10
    assert spec.keyword_version == source.keyword_version
    assert get_keyword_lists(conn, "m2") == get_keyword_lists(conn, "m")
    assert all(h.keywords["all"] for batch in ix.store.scroll("m2") for h in batch)  # keyed
    assert purposes(conn) == ["propose"]  # the lists were copied, not proposed again


def test_copy_prices_summaries_for_a_summarising_target(tmp_path):
    conn, ix, ld, items = scibench_paper(tmp_path)
    target = ld.copy_spec(ix.spec("p"), "s", "paper_2024", {"added_text": "summary"})
    assert target.summarize
    estimate = ld.estimate_copy("p", target)
    assert estimate["calls"] == len(items) and estimate["est_usd"] > 0
    ld.ai.set_role("default", None)
    assert ld.estimate_copy("p", target)["est_usd"] is None  # no model to price it with
    worked = ld.copy_spec(ix.spec("p"), "w", "worked_examples", None)
    assert ld.estimate_copy("p", worked)["est_usd"] is None


def test_a_copy_keeps_its_sources_model(tmp_path):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(
        conn, Store(None, None), fake_models(dims={"fake/small": 64}), fake_ai(conn, by_purpose)
    )
    source = ix.create("s", "multihop", False, dense_model="fake/small")
    ld = loader_for(ix, tmp_path)
    assert db.get_app_settings(conn, "search.") == {}  # Settings names no model
    target = ld.copy_spec(source, "cp", "paper_2024", None)
    assert target.dense_model == "fake/small"
    ld.start_copy("s", target)
    created = ix.spec("cp")
    assert created is not None and (created.dense_model, created.dense_dim) == ("fake/small", 64)
    ix.add(created, [Item("a", "enzyme rate rose", "A")], "x")
    assert ix.store.vectors("cp", "text_dense", 3).shape[1] == 64


def test_copy_spec_refuses_a_bad_name_recipe_or_override(tmp_path):
    conn, ix, ld, items = scibench_paper(tmp_path)
    source = ix.spec("p")
    for name, recipe, override in (
        ("bad name!", "reference", None),
        ("r", "nope", None),
        ("r", "reference", {"tagging": "magic"}),
        ("r", "reference", {"vocabulary": "lists"}),
    ):
        with pytest.raises(ValueError):
            ld.copy_spec(source, name, recipe, override)
    assert ix.spec("r") is None


def test_chunk_of_reads_old_and_new_ids():
    assert loader.chunk_of("a.md#3") == "a.md"
    assert loader.chunk_of("a.md#0123456789ab") == "a.md"
    assert loader.chunk_of("a.md#0123456789ab-2") == "a.md"
    assert loader.chunk_of("a.md#b.md#0123456789ab") == "a.md#b.md"
    assert loader.chunk_of("a.md#notes") is None
    assert loader.chunk_of("a.md#0123456789AB") is None
    assert loader.chunk_of("a.md#0123456789ab-") is None
    assert loader.chunk_of("a.md#٣") is None  # an Arabic-Indic digit is not a part number
    assert loader.chunk_of("a.md") is None


def stored_labelled(tmp_path, qa_type):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, lambda p: ""))
    db.create_eval_set(conn, "s", qa_type)
    old = Question("q1", "old text", None, ["Ten days"], ["a"], {}, ["a", "b"])
    db.save_questions(
        conn, "s", [old] if qa_type == "documents" else [replace(old, judged_ids=None)]
    )
    return loader_for(ix, tmp_path)


def test_reimporting_a_documents_set_keeps_its_labels(tmp_path):
    ld = stored_labelled(tmp_path, "documents")
    counts, todo = ld.question_counts("s", [Question("q1", "new text", None, [], ["c"], {}, None)])
    assert counts["updated"] == 1
    assert todo[0].gold_ids == ["a", "c"] and todo[0].judged_ids == ["a", "b", "c"]
    assert todo[0].gold_answers == ["Ten days"] and todo[0].text == "new text"
    (tmp_path / "m").mkdir()
    ld = stored_labelled(tmp_path / "m", "multihop")
    _, todo = ld.question_counts("s", [Question("q1", "new text", None, [], ["c"], {}, None)])
    assert todo[0].gold_ids == ["c"] and todo[0].judged_ids is None


def test_reimporting_labelled_questions_replaces_them(tmp_path):
    ld = stored_labelled(tmp_path, "documents")
    new = Question("q1", "new text", None, [], ["c"], {}, ["c"])
    _, todo = ld.question_counts("s", [new])
    assert todo[0].gold_ids == ["c"] and todo[0].judged_ids == ["c"]
    assert todo[0].gold_answers == []


def test_a_label_saved_while_a_question_import_job_merges_is_never_overwritten(tmp_path):
    import threading

    from ragqa_core import imports

    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, lambda p: ""))
    ix.create("d", "documents", False, "documents")
    read = Question("q1", "one?", None, [], ["p1"], {}, ["p1"])
    db.save_questions(conn, "d", [read])
    ld = loader_for(ix, tmp_path)
    counts, saved, threads = ld.question_counts, [], []

    def label() -> None:  # a labeller's PUT of q1, sent while the job is merging
        try:
            db.upsert_question(conn, "d", replace(read, gold_ids=["p9"], judged_ids=["p9"]), read)
            saved.append("saved")
        except ValueError:
            saved.append("refused")

    def merge_slowly(name, questions):
        out = counts(name, questions)
        threads.append(threading.Thread(target=label))
        threads[0].start()
        threads[0].join(0.3)  # time for the PUT to land between the merge and its write
        return out

    ld.question_counts = merge_slowly
    rows = [{"id": "q1", "text": "one?", "gold_ids": ["p2"]}]
    mapping = imports.guess_mapping(imports.columns(rows), "questions")
    ld.run_import(import_job(conn, tmp_path, "questions", rows, mapping, name="d"), lambda p: None)
    threads[0].join(5)
    assert saved == ["refused"]  # it waited for the job, then found q1 had changed
    assert get_questions(conn, "d")[0].gold_ids == ["p1", "p2"]


def test_question_counts_on_a_set_not_made_yet(tmp_path):
    ld = stored_labelled(tmp_path, "documents")
    counts, todo = ld.question_counts("nope", [Question("q", "t", None, [], [], {}, None)])
    assert counts["new"] == 1 and len(todo) == 1
