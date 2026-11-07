from dataclasses import replace

import numpy as np
import pytest

from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.db import connect, replace_questions, set_source_enabled
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.prompts import QUERY_TASK
from ragqa_core.retriever import Retriever
from ragqa_core.router import Router
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


def build(tmp_path, n_questions=20, **kw):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "A")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS[:2], "x")
    ix.add(spec, ITEMS[2:], "y")
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
