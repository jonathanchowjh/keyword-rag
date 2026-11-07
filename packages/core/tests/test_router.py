from dataclasses import replace

import numpy as np
import pytest

from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.datasets import DEV_SLICE, dev_slice
from ragqa_core.db import connect, replace_questions, set_source_enabled
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.prompts import QUERY_TASK
from ragqa_core.retriever import Retriever
from ragqa_core.router import (
    CENTROID_SAMPLE,
    PROFILE_SAMPLE_CHARS,
    PROFILE_SAMPLES,
    ROUTE_SAMPLE,
    Router,
)
from ragqa_core.store import Store
from ragqa_core.types import Bar, Item, Question, Skip

WORDS = {
    "enzyme": "enzyme kinetics catalyst assay substrate",
    "market": "market economics trade survey demand",
    "volcano": "volcano magma tectonics seismograph lava",
    "optics": "optics lens refraction imaging focus",
}
ITEMS = [Item(f"{n}-item", w, "A", f"{n} expl") for n, w in WORDS.items()]


def questions(n: int, text=lambda i: f"enzyme kinetics catalyst q{i}", meta=None) -> list[Question]:
    return [Question(f"q{i:02d}", text(i), ["A", "B"], ["A"], [], meta or {}) for i in range(n)]


def build(tmp_path, n_questions=20, items=ITEMS, **kw):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "A")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("c", "mcq", False)
    if items:
        ix.add(spec, items[:2], "x")
        ix.add(spec, items[2:], "y")
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    evaluator = Evaluator(conn, retriever, models, ai)
    if n_questions:
        replace_questions(conn, "c", questions(n_questions, **kw), None)
    return Router(conn, retriever, evaluator), spec


def top(router: Router, spec, text: str, skip: Skip | None = None) -> float:
    skip = skip or Skip()
    vec = router.retriever.models.embed_query([text], task=QUERY_TASK[spec.qa_type])[0]
    hits = router.retriever.store.search(spec.name, "text_dense", vec, 1, skip.sources, skip.items)
    return hits[0].score


def counting(router: Router) -> list[int]:
    calls: list[int] = []
    real = router.retriever.store.search

    def search(*a, **k):
        calls.append(1)
        return real(*a, **k)

    router.retriever.store.search = search  # type: ignore[method-assign]
    return calls


def test_best_match_is_the_top_dense_cosine_over_the_texts_and_skips_disabled_sources(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    texts = ["nothing alike zzz", WORDS["volcano"]]
    assert router.best_match(spec, texts) == pytest.approx(max(top(router, spec, t) for t in texts))
    assert router.best_match(spec, texts) == pytest.approx(1.0, abs=1e-3)
    set_source_enabled(router.conn, "c", "y", False)
    off = router.best_match(spec, texts)
    assert off == pytest.approx(max(top(router, spec, t, Skip(("y",))) for t in texts))
    assert off < 0.9


def test_best_match_refuses_a_store_built_with_another_model(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    router.retriever.models.dense_model = "other/model"
    with pytest.raises(ModelMismatch):
        router.best_match(spec, ["enzyme"])


def test_bar_is_the_5th_percentile_of_the_sets_own_best_matches(tmp_path):
    parts = ["enzyme kinetics", "market trade", "lens focus"]
    texts = lambda i: " ".join(parts[i % 3 :]) + f" q{i}"  # noqa: E731
    router, spec = build(tmp_path, text=texts)
    scores = [top(router, spec, q.text) for q in questions(20, texts)]
    bar = router.bar(spec)
    assert bar.source == "set"
    assert bar.n == 20
    assert bar.eval_set == "c"
    assert bar.value == pytest.approx(float(np.quantile(scores, 0.05)))
    assert len(set(round(s, 4) for s in scores)) > 1


def test_bar_leaves_out_each_questions_excluded_ids(tmp_path):
    same = lambda i: WORDS["enzyme"]  # noqa: E731
    plain, spec = build(tmp_path, text=same)
    assert plain.bar(spec).value == pytest.approx(1.0, abs=1e-3)
    router, spec = build(tmp_path / "b", text=same, meta={"excluded_ids": ["enzyme-item"]})
    value = router.bar(spec).value
    assert value is not None and value < 0.9


def test_bar_drops_leaked_questions(tmp_path):
    router, spec = build(tmp_path, n_questions=22)
    router.evaluator.leaked_ids = lambda spec, eval_set: {"q00", "q01"}  # type: ignore[method-assign]
    assert router.bar(spec).n == 20
    router.evaluator.leaked_ids = lambda spec, eval_set: {"q00", "q01", "q02"}  # type: ignore[method-assign]
    router.conn.execute("delete from store_bars")
    bar = router.bar(spec)
    assert (bar.source, bar.value, bar.n) == ("few", None, 19)


def test_too_few_questions_give_no_bar(tmp_path):
    router, spec = build(tmp_path, n_questions=19)
    bar = router.bar(spec)
    assert (bar.source, bar.value, bar.n) == ("few", None, 19)
    assert router.kept(0.0, bar)


def test_a_store_without_an_eval_set_has_no_bar_and_always_keeps(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    bar = router.bar(spec)
    assert bar == Bar(None, "none", None, 0)
    assert router.kept(0.0, bar)
    assert router.kept(0.5, Bar(0.5, "set", "c", 20))
    assert not router.kept(0.49, Bar(0.5, "set", "c", 20))


def test_bar_recomputes_when_the_set_or_store_changes_and_manual_wins(tmp_path):
    router, spec = build(tmp_path)
    calls = counting(router)
    first = router.bar(spec)
    assert len(calls) == 20
    calls.clear()
    assert router.bar(spec) == first  # unchanged: from store_bars, no search
    fresh = Router(router.conn, router.retriever, router.evaluator)  # a restart keeps the row
    assert fresh.bar(spec) == first
    assert calls == []

    replace_questions(router.conn, "c", questions(21), None)
    router.bar(spec)
    assert len(calls) == 21
    calls.clear()

    router.retriever.indexer.add(spec, [Item("new-item", "fresh words", "A", "e")], "z")
    router.bar(spec)
    assert len(calls) == 21
    calls.clear()

    set_source_enabled(router.conn, "c", "z", False)
    router.bar(spec)
    assert len(calls) == 21
    calls.clear()
    router.bar(spec)
    assert calls == []

    db.set_chat_bar(router.conn, "c", 0.9)
    assert router.bar(spec) == Bar(0.9, "manual", None, 0)
    assert calls == []

    assert db.get_store_bar(router.conn, "c", "gate") is not None
    db.delete_collection_rows(router.conn, "c")
    assert db.get_store_bar(router.conn, "c", "gate") is None


def test_question_vectors_are_embedded_once_per_set_version(tmp_path):
    router, spec = build(tmp_path)
    embeds: list[int] = []
    real = router.retriever.models.embed_query

    def embed(texts, task=""):
        embeds.append(len(texts))
        return real(texts, task=task)

    router.retriever.models.embed_query = embed  # type: ignore[method-assign]
    router.bar(spec)
    router.retriever.indexer.add(spec, [Item("new-item", "fresh words", "A", "e")], "z")
    router.bar(spec)
    assert embeds == [20]


def test_chat_settings_round_trip_as_numbers(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    conn = router.conn
    assert db.get_chat_bar(conn, "c") is None
    db.set_chat_bar(conn, "c", 0.25)
    db.set_chat_k(conn, "c", 6)
    db.set_chat_method(conn, "c", "hybrid")
    assert (db.get_chat_bar(conn, "c"), db.get_chat_k(conn, "c")) == (0.25, 6)
    assert db.get_chat_method(conn, "c") == "hybrid"
    db.set_chat_bar(conn, "c", None)
    assert db.get_chat_bar(conn, "c") is None


def test_bar_refuses_a_store_built_with_another_model(tmp_path):
    router, spec = build(tmp_path)
    router.retriever.models.dense_model = "other/model"
    with pytest.raises(ModelMismatch):
        router.bar(spec)
    db.set_chat_bar(router.conn, "c", 0.9)  # checked before a saved bar is read
    with pytest.raises(ModelMismatch):
        router.bar(spec)


def test_bar_recomputes_when_the_stores_model_changes(tmp_path):
    router, spec = build(tmp_path)
    calls = counting(router)
    router.bar(spec)
    calls.clear()
    router.retriever.models.dense_model = "other/model"  # rebuilt under another model
    router.bar(replace(spec, dense_model="other/model"))
    assert len(calls) == 20


def test_bar_unsaved_is_the_eval_set_bar_whatever_is_saved(tmp_path):
    router, spec = build(tmp_path)
    own = router.bar(spec)
    db.set_chat_bar(router.conn, "c", 0.9)
    assert router.bar(spec) == Bar(0.9, "manual", None, 0)
    assert router.bar(spec, saved=False) == own


def test_forget_drops_the_stores_bar(tmp_path):
    router, spec = build(tmp_path)
    router.bar(spec)
    calls = counting(router)
    router.forget("c")
    assert db.get_store_bar(router.conn, "c", "gate") is None
    router.bar(spec)
    assert len(calls) == 20


# -- routing ---------------------------------------------------------------------------


def stored_vectors(router: Router, name: str) -> np.ndarray:
    records, _ = router.retriever.store.client.scroll(name, limit=100, with_vectors=["text_dense"])
    return np.array([r.vector["text_dense"] for r in records])  # type: ignore[index]


def test_draft_route_profile_lists_terms_sources_and_samples(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    long = [Item(f"long-{i}", f"lens {'focus ' * 60}{i}", "A", "e") for i in range(3)]
    router.retriever.indexer.add(spec, long, "z")
    texts = [i.text for i in ITEMS + long]
    tagged = replace(spec, profile={f"t{i:02d}": 40 - i for i in range(25)})
    real = router.retriever.store.scroll

    def scroll(name, batch=256, skip=()):
        assert batch <= PROFILE_SAMPLES
        pages = real(name, batch, skip)
        yield next(pages)
        raise AssertionError("the draft scrolled past its samples")

    router.retriever.store.scroll = scroll  # type: ignore[method-assign]
    lines = router.draft_route_profile(tagged).split("\n")
    assert lines[0] == f"Store c (mcq, {spec.corpus_type})."
    assert lines[1] == "Topics: " + ", ".join(f"t{i:02d}" for i in range(20))
    assert lines[2] == "Sources: x, y, z"
    samples = lines[3:]
    assert len(samples) == PROFILE_SAMPLES
    assert all(
        s.startswith("- ") and s[2:] in [t[:PROFILE_SAMPLE_CHARS] for t in texts] for s in samples
    )

    set_source_enabled(router.conn, "c", "z", False)  # a switched-off source is left out
    lines = router.draft_route_profile(tagged).split("\n")
    assert lines[2] == "Sources: x, y"
    assert sorted(s[2:] for s in lines[3:]) == sorted(i.text for i in ITEMS)

    terms = [f"v{i:02d}" for i in range(30)]  # no tag counts: the vocabulary's terms
    db.save_vocabulary(router.conn, "c", terms, "v1")
    db.set_vocabulary_version(router.conn, "c", "v1")
    lines = router.draft_route_profile(router.retriever.spec("c")).split("\n")
    assert lines[1] == "Topics: " + ", ".join(terms[:20])
    db.set_vocabulary_version(router.conn, "c", None)  # type: ignore[arg-type]
    assert router.draft_route_profile(router.retriever.spec("c")).split("\n")[1] == "Sources: x, y"


def test_route_profile_is_the_stored_text_else_the_draft(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    assert router.route_profile(spec) == (router.draft_route_profile(spec), True)
    db.set_route_profile(router.conn, "c", "Enzymes, by hand.")
    assert db.get_route_profile(router.conn, "c") == "Enzymes, by hand."
    assert router.route_profile(spec) == ("Enzymes, by hand.", False)
    db.set_route_profile(router.conn, "c", None)
    assert router.route_profile(spec)[1] is True


def test_route_score_is_the_max_of_profile_and_normalised_centroid_cosines(tmp_path):
    same = [Item(f"s{i}", "enzyme kinetics", "A", None) for i in range(4)]
    router, spec = build(tmp_path, n_questions=0, items=same)
    vector = stored_vectors(router, "c")[0]
    assert np.allclose(stored_vectors(router, "c"), vector)
    assert router.route_score(spec, ["nothing alike zzz", "enzyme kinetics"]) == pytest.approx(1.0)

    router, spec = build(tmp_path / "b", n_questions=0)
    models = router.retriever.models
    mean = stored_vectors(router, "c").mean(axis=0)
    assert np.linalg.norm(mean) < 0.9  # the mean of unit vectors is shorter
    centroid = mean / np.linalg.norm(mean)
    profile = models.embed([router.draft_route_profile(spec)])[0]
    texts = ["volcano magma", "market survey demand"]
    queries = models.embed_query(texts, task=QUERY_TASK[spec.qa_type])
    want = max(max(float(q @ profile), float(q @ centroid)) for q in queries)
    assert router.route_score(spec, texts) == pytest.approx(want)
    db.set_route_profile(router.conn, "c", "volcano magma")  # the profile's cosine wins
    assert router.route_score(spec, texts) == pytest.approx(1.0)


def test_a_store_without_profile_or_items_is_always_routed(tmp_path):
    router, spec = build(tmp_path, n_questions=0, items=[])
    assert router.route_score(spec, ["enzyme kinetics"]) is None
    db.set_route_profile(router.conn, "c", "enzyme kinetics")  # a profile alone routes
    assert router.route_score(spec, ["enzyme kinetics"]) == pytest.approx(1.0)


def test_route_score_and_route_bar_refuse_a_store_built_with_another_model(tmp_path):
    router, spec = build(tmp_path)
    router.retriever.models.dense_model = "other/model"
    with pytest.raises(ModelMismatch):
        router.route_score(spec, ["enzyme"])
    with pytest.raises(ModelMismatch):
        router.route_bar(spec)


def test_editing_the_route_profile_refreshes_its_vector_and_route_bar(tmp_path):
    router, spec = build(tmp_path)
    conn = router.conn
    db.set_route_profile(conn, "c", "enzyme kinetics catalyst q")
    asked = ["enzyme kinetics catalyst"]
    before = router.route_score(spec, asked)
    assert before is not None and before > 0.8
    bar = router.route_bar(spec)
    scores = [router.route_score(spec, [q.text]) for q in questions(20)]
    assert (bar.source, bar.eval_set, bar.n) == ("set", "c", 20)
    assert bar.value == pytest.approx(float(np.quantile(scores, 0.05)))  # type: ignore[arg-type]
    assert db.get_store_bar(conn, "c", "gate") is None  # a route bar is a row of its own
    kept = db.get_store_bar(conn, "c", "route")
    assert kept is not None and router.route_bar(spec) == bar

    db.set_route_profile(conn, "c", "volcano magma tectonics")
    after = router.route_score(spec, asked)
    fresh = Router(conn, router.retriever, router.evaluator)  # nothing held in memory
    assert after is not None and after < 0.6
    assert after == pytest.approx(fresh.route_score(spec, asked))
    moved = router.route_bar(spec)
    assert moved.value is not None and moved.value < bar.value  # type: ignore[operator]
    assert db.get_store_bar(conn, "c", "route")["key"] != kept["key"]  # type: ignore[index]


def test_centroid_is_sampled_and_cached(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    conn = router.conn
    calls: list[tuple] = []
    real = router.retriever.store.vectors

    def vectors(name, using, limit, skip=()):
        calls.append((using, limit, skip))
        return real(name, using, limit, skip)

    router.retriever.store.vectors = vectors  # type: ignore[method-assign]
    router.route_score(spec, ["enzyme"])
    router.route_score(spec, ["market"])
    assert CENTROID_SAMPLE == 2000
    assert calls == [("text_dense", 2000, ())]

    set_source_enabled(conn, "c", "y", False)
    router.route_score(spec, ["enzyme"])
    assert calls[-1] == ("text_dense", 2000, ("y",)) and len(calls) == 2

    router.retriever.indexer.add(spec, [Item("new-item", "fresh words", "A", "e")], "z")
    router.route_score(spec, ["enzyme"])
    assert len(calls) == 3

    # A job the worker finishes in its own process: same count, but the items may differ.
    job = db.create_job(conn, "import", "c", {"kind": "items"})
    router.route_score(spec, ["enzyme"])
    assert len(calls) == 3  # queued: nothing written yet
    db.update_job(conn, job, status="done")
    router.route_score(spec, ["enzyme"])
    assert len(calls) == 4
    db.create_job(conn, "eval", "c", {})  # a job that adds no items
    router.route_score(spec, ["enzyme"])
    assert len(calls) == 4

    router.forget("c")
    router.route_score(spec, ["enzyme"])
    assert len(calls) == 5


def test_forget_drops_the_route_bar_and_the_profile_vector(tmp_path):
    router, spec = build(tmp_path)
    db.set_route_profile(router.conn, "c", "enzyme kinetics")
    router.route_bar(spec)
    embeds: list[list[str]] = []
    real = router.retriever.models.embed

    def embed(texts):
        embeds.append(texts)
        return real(texts)

    router.retriever.models.embed = embed  # type: ignore[method-assign]
    router.route_score(spec, ["enzyme"])
    assert embeds == []  # held from the bar
    router.forget("c")
    assert db.get_store_bar(router.conn, "c", "route") is None
    router.route_score(spec, ["enzyme"])
    assert embeds == [["enzyme kinetics"]]


def test_a_kept_bar_reads_no_questions_and_scans_for_no_leaks(tmp_path, monkeypatch):
    router, spec = build(tmp_path, meta={"source": "hand"})  # hand-written: a leak scan scrolls
    db.set_route_profile(router.conn, "c", "enzyme kinetics")
    gate, route = router.bar(spec), router.route_bar(spec)
    assert gate.source == "set" and route.source == "set"

    def refuse(*a, **k):
        raise AssertionError("a kept bar read the questions or scanned for leaks")

    monkeypatch.setattr(db, "get_questions", refuse)
    router.evaluator.leaked_ids = refuse  # type: ignore[method-assign]
    assert router.bar(spec) == gate and router.route_bar(spec) == route

    # A restart: a new Router and Evaluator on the same file, nothing held in memory.
    evaluator = Evaluator(
        router.conn, router.retriever, router.retriever.models, router.evaluator.ai
    )
    evaluator.leaked_ids = refuse  # type: ignore[method-assign]
    fresh = Router(router.conn, router.retriever, evaluator)
    assert fresh.bar(spec) == gate and fresh.route_bar(spec) == route


def test_a_job_finished_while_a_bar_is_set_leaves_no_bar_under_the_old_key(tmp_path):
    router, spec = build(tmp_path)
    conn = router.conn
    db.set_route_profile(conn, "c", "enzyme kinetics")
    real = router._questions

    def worker_finishes_meanwhile(*a, **k):
        # An import in the worker replaces items at the same count and drops the bars.
        db.update_job(conn, db.create_job(conn, "import", "c", {}), status="done")
        db.delete_store_bars(conn, "c")
        return real(*a, **k)

    read: list[int] = []

    def questions_(*a, **k):
        read.append(1)
        return real(*a, **k)

    for set_bar in (router.bar, router.route_bar):
        router._questions = worker_finishes_meanwhile  # type: ignore[method-assign]
        set_bar(spec)
        router._questions = questions_  # type: ignore[method-assign]
        set_bar(spec)
        set_bar(spec)
        assert len(read) == 1  # saved under the key read before the job: set again, once
        read.clear()


def test_best_match_and_route_score_take_the_texts_embedded_once(tmp_path):
    router, spec = build(tmp_path, n_questions=0)
    texts = ["nothing alike zzz", WORDS["volcano"]]
    vectors = router.embed(spec, texts)
    assert np.allclose(vectors, router.retriever.models.embed_query(texts, task=QUERY_TASK["mcq"]))
    router.retriever.models.embed_query = None  # type: ignore[assignment]  # never embedded again
    assert router.best_match(spec, texts, vectors) == pytest.approx(1.0, abs=1e-3)
    assert router.route_score(spec, texts, vectors) is not None


# -- the routing eval ------------------------------------------------------------------

TOPICS = {"a": WORDS["enzyme"], "b": WORDS["volcano"], "n": WORDS["market"]}
HELD_OUT = 30  # each set's questions past its dev slice


def topic_questions(store: str) -> list[Question]:
    """DEV_SLICE + HELD_OUT questions on the store's topic. The dev ones, which set its bars,
    carry two of its words and the held-out ones four, so a held-out question clears its
    home's bars by a margin, not by a tie; another topic shares no word with it."""
    words = TOPICS[store].split()
    qs = [Question(f"q{i:03d}", "", None, ["A"], [], {}) for i in range(DEV_SLICE + HELD_OUT)]
    dev = {q.id for q in dev_slice(qs)}
    return [replace(q, text=" ".join(words[: 2 if q.id in dev else 4] + [q.id])) for q in qs]


def routed(tmp_path, names=("a", "b"), sets=("a", "b")) -> Router:
    """A store per name, ten items on its topic each, and an eval set of topic_questions
    for each name in `sets`, the default of the store named after it."""
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "A")
    ix = Indexer(conn, store, models, ai)
    for name in names:
        topic = TOPICS[name[0]]  # a copy, "a2", holds what "a" holds
        spec = ix.create(name, "mcq", False)
        ix.add(spec, [Item(f"{name}-{i}", f"{topic} note{i}", "A", "e") for i in range(10)], "x")
    for name in sets:
        replace_questions(conn, name, topic_questions(name), None)
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    return Router(conn, retriever, Evaluator(conn, retriever, models, ai))


def test_routing_eval_scores_gate_and_profiles_on_home_stores(tmp_path):
    router = routed(tmp_path)
    spec = {s: router.retriever.spec(s) for s in "ab"}
    bars = {s: (router.bar(spec[s]).value, router.route_bar(spec[s]).value) for s in "ab"}
    held = {s: [q for q in topic_questions(s) if q.text.count(" ") == 4] for s in "ab"}
    for home, other in ("ab", "ba"):  # clear margins, by construction
        for q in held[home]:
            mine = router.best_match(spec[home], [q.text])
            theirs = router.best_match(spec[other], [q.text])
            route_mine = router.route_score(spec[home], [q.text])
            route_theirs = router.route_score(spec[other], [q.text])
            assert mine > bars[home][0] + 0.05 and theirs < bars[other][0] - 0.1  # type: ignore[operator]
            assert route_mine > bars[home][1] + 0.05 and route_theirs < bars[other][1] - 0.1  # type: ignore[operator]

    got = router.evaluate(["a", "b"])
    assert got["split"] == "held_out" and got["questions"] == 2 * HELD_OUT
    assert got["modes"] == {
        "gate": {"recall": 1.0, "precision": 1.0, "none_rate": 0.0, "mean_searched": 2.0},
        "profiles": {"recall": 1.0, "precision": 1.0, "none_rate": 0.0, "mean_searched": 1.0},
    }
    one = {"questions": HELD_OUT, "recall_gate": 1.0, "recall_profiles": 1.0}
    assert got["per_store"] == {"a": {"eval_set": "a", **one}, "b": {"eval_set": "b", **one}}
    row = next(r for r in got["rows"] if r["eval_set"] == "b")
    assert row == {
        "eval_set": "b",
        "id": row["id"],
        "homes": ["b"],
        "gate_kept": ["b"],
        "routed": ["b"],
        "profiles_kept": ["b"],
    }


def test_routing_eval_defaults_to_held_out_questions(tmp_path):
    router = routed(tmp_path)
    dev = {q.id for q in dev_slice(topic_questions("a"))}
    rows = router.evaluate(["a", "b"])["rows"]
    assert len(rows) == 2 * HELD_OUT and not {r["id"] for r in rows} & dev
    assert {r["id"] for r in router.evaluate(["a", "b"], "dev")["rows"]} == dev
    full = router.evaluate(["a", "b"], "full")
    every = sorted(q.id for q in topic_questions("a"))
    assert full["questions"] == 2 * ROUTE_SAMPLE  # the first ROUTE_SAMPLE of each set, by id
    assert [r["id"] for r in full["rows"] if r["eval_set"] == "a"] == every[:ROUTE_SAMPLE]
    with pytest.raises(ValueError):
        router.evaluate(["a", "b"], "train")  # type: ignore[arg-type]


def test_routing_eval_counts_every_store_sharing_a_set_as_home(tmp_path, monkeypatch):
    router = routed(tmp_path, names=("a", "a2", "b"))
    db.set_default_eval_set(router.conn, "a2", "a")
    first = router.evaluate(["a", "a2", "b"])
    reads: list[str] = []
    real = db.get_questions

    def get_questions(conn, name, version=None):
        reads.append(name)
        return real(conn, name, version)

    monkeypatch.setattr(db, "get_questions", get_questions)
    got = router.evaluate(["a", "a2", "b"])
    assert got == first
    assert sorted(reads) == ["a", "b"]  # the bars are kept; each set is read once
    assert got["questions"] == 2 * HELD_OUT
    shared = [r for r in got["rows"] if r["eval_set"] == "a"]
    assert len(shared) == HELD_OUT and all(r["homes"] == ["a", "a2"] for r in shared)
    assert all(r["gate_kept"] == ["a", "a2"] for r in shared)
    one = {"questions": HELD_OUT, "recall_gate": 1.0, "recall_profiles": 1.0}
    assert got["per_store"]["a2"] == {"eval_set": "a", **one}  # a's set, a2's default
    assert got["modes"]["gate"]["recall"] == 1.0 and got["modes"]["gate"]["precision"] == 1.0


def test_routing_eval_handles_a_store_without_an_eval_set(tmp_path):
    router = routed(tmp_path, names=("a", "b", "n"))
    got = router.evaluate(["a", "b", "n"])
    assert got["questions"] == 2 * HELD_OUT  # n has no questions of its own
    no_set = {"eval_set": None, "questions": 0, "recall_gate": None, "recall_profiles": None}
    assert got["per_store"]["n"] == no_set
    assert got["per_store"]["a"]["eval_set"] == "a"  # so the page can say why one adds none
    # With no bar and no route bar, n is kept for every question: a wrong destination.
    assert all(r["gate_kept"] == [r["eval_set"], "n"] for r in got["rows"])
    assert all(r["profiles_kept"] == [r["eval_set"], "n"] for r in got["rows"])
    gate, profiles = got["modes"]["gate"], got["modes"]["profiles"]
    assert (gate["recall"], gate["precision"], gate["mean_searched"]) == (1.0, 0.5, 3.0)
    assert (profiles["recall"], profiles["precision"], profiles["mean_searched"]) == (1.0, 0.5, 2.0)
    with pytest.raises(ValueError, match="eval set"):
        router.evaluate(["n"])


def test_routing_eval_leaves_out_each_questions_excluded_ids(tmp_path):
    router = routed(tmp_path)
    qs = topic_questions("a")
    dev = {q.id for q in dev_slice(qs)}
    blind = {"excluded_ids": [f"a-{i}" for i in range(10)]}  # every item a holds
    qs = [q if q.id in dev else replace(q, meta=blind) for q in qs]
    replace_questions(router.conn, "a", qs, None)
    got = router.evaluate(["a", "b"])
    assert got["per_store"]["a"]["recall_gate"] == 0.0
    assert got["per_store"]["b"]["recall_gate"] == 1.0
    assert all(r["routed"] == ["a"] for r in got["rows"] if r["eval_set"] == "a")
