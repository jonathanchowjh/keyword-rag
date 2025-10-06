import pytest

from ragqa_core.db import claim_next_job, connect, create_job
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.llm import NoApiKey
from ragqa_core.prompts import QUERY_TASK
from ragqa_core.retriever import (
    CONTROL_METHODS,
    METHOD_NAMES,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
    rrf,
)
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Hit, Item, KeywordLists

DOMAINS = {
    "enzyme": (
        ["enzyme", "kinetics", "catalyst"],
        ["experiment", "assay", "titration"],
        ["rate", "substrate", "temperature"],
    ),
    "market": (
        ["market", "economics", "trade"],
        ["survey", "regression", "audit"],
        ["price", "demand", "supply"],
    ),
    "volcano": (
        ["volcano", "magma", "tectonics"],
        ["sampling", "seismograph", "mapping"],
        ["lava", "pressure", "ash"],
    ),
    "optics": (
        ["optics", "lens", "refraction"],
        ["spectroscopy", "imaging", "calibration"],
        ["light", "angle", "focus"],
    ),
}
LISTS = KeywordLists(
    topic=[w for d in DOMAINS.values() for w in d[0]],
    method=[w for d in DOMAINS.values() for w in d[1]],
    common=[w for d in DOMAINS.values() for w in d[2]],
)
ITEMS = [
    Item(f"{name}-item", " ".join(t + m + c) + f" notes about {name}", "A", f"{name} expl")
    for name, (t, m, c) in DOMAINS.items()
]
FREE = [m for m in METHOD_NAMES if m not in ("question_only", "iterative", *CONTROL_METHODS)]


def hit(i: str) -> Hit:
    return Hit(i, i, 0.0, {}, [], None, None, {})


def build(tmp_path, summarize=False, lists=True, reply=lambda p: "A summary."):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    claude = fake_claude(conn, reply)
    ix = Indexer(conn, store, models, claude)
    spec = ix.create("c", "mcq", summarize)
    ix.add(spec, ITEMS)
    if lists:
        ix.set_lists(spec, LISTS)
    return Retriever(conn, store, models, claude, ix), conn


def test_keyword_methods_refuse_while_rekeying(tmp_path):
    r, conn = build(tmp_path)
    create_job(conn, "rekey", "c", {"version": "v"})
    claim_next_job(conn)
    with pytest.raises(MethodUnavailable, match="re-keyed"):
        r.retrieve("c", "enzyme kinetics", "kw_individual")
    assert r.retrieve("c", "enzyme kinetics", "hybrid")  # plain methods still answer


def test_rrf_sums_reciprocal_ranks():
    a, b, c = hit("a"), hit("b"), hit("c")
    fused = rrf([[a, b], [b, c]], k=1)
    assert [h.id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert fused[1].score == pytest.approx(1 / 2)


def test_rrf_ties_keep_first_appearance():
    fused = rrf([[hit("x")], [hit("y")]], k=60)
    assert [h.id for h in fused] == ["x", "y"]


@pytest.mark.parametrize("method", FREE)
def test_each_method_finds_the_obvious_item(tmp_path, method):
    r, _ = build(tmp_path)
    assert r.retrieve("c", "enzyme kinetics rate experiment", method, k=1)[0].id == "enzyme-item"


@pytest.mark.parametrize("method", FREE)
def test_each_method_respects_k(tmp_path, method):
    r, _ = build(tmp_path)
    hits = r.retrieve("c", "enzyme kinetics rate market price demand", method, k=2)
    assert len(hits) == 2 and len({h.id for h in hits}) == 2


def test_question_only_returns_nothing(tmp_path):
    r, _ = build(tmp_path)
    assert r.retrieve("c", "x", "question_only") == []


def test_random_example_is_the_same_for_the_same_question(tmp_path):
    r, _ = build(tmp_path)
    first = [h.id for h in r.retrieve("c", "enzyme", "random_example", k=2)]
    assert first == [h.id for h in r.retrieve("c", "enzyme", "random_example", k=2)]
    assert len(set(first)) == 2


def test_random_example_varies_between_questions(tmp_path):
    r, _ = build(tmp_path)
    picks = {r.retrieve("c", f"question {n}", "random_example", k=1)[0].id for n in range(12)}
    assert len(picks) > 1


def test_fixed_example_is_the_same_for_every_question(tmp_path):
    r, _ = build(tmp_path)
    a = [h.id for h in r.retrieve("c", "enzyme kinetics", "fixed_example", k=2)]
    b = [h.id for h in r.retrieve("c", "market price demand", "fixed_example", k=2)]
    assert a == b


@pytest.mark.parametrize("method", CONTROL_METHODS)
def test_controls_keep_their_first_pick_whatever_k(tmp_path, method):
    # mcq answers use only the first hit, so it must not depend on k.
    r, _ = build(tmp_path)
    one = r.retrieve("c", "enzyme", method, k=1)[0].id
    assert r.retrieve("c", "enzyme", method, k=4)[0].id == one


@pytest.mark.parametrize("method", CONTROL_METHODS)
def test_controls_see_items_added_later(tmp_path, method):
    r, _ = build(tmp_path)
    r.retrieve("c", "enzyme", method, k=1)
    r.indexer.add(r.spec("c"), [Item("late-item", "added after the first query", "B", "late")])
    ids = {h.id for h in r.retrieve("c", "enzyme", method, k=10)}
    assert ids == {i.id for i in ITEMS} | {"late-item"}


def test_dense_query_carries_the_collection_task(tmp_path):
    r, _ = build(tmp_path)
    tasks: list[str] = []
    real = r.models.embed_query
    r.models.embed_query = lambda texts, task="": tasks.append(task) or real(texts)
    r.retrieve("c", "enzyme", "hybrid", k=1)
    assert tasks == [QUERY_TASK["mcq"]]


def test_kw_hits_carry_matched_keywords(tmp_path):
    r, _ = build(tmp_path)
    [h] = r.retrieve("c", "enzyme kinetics rate experiment", "kw_individual", k=1)
    assert "enzyme" in h.matched
    assert set(h.matched) <= {w for ws in h.keywords.values() for w in ws}


def test_non_kw_hits_have_no_matched(tmp_path):
    r, _ = build(tmp_path)
    assert r.retrieve("c", "enzyme", "hybrid", k=1)[0].matched == []


def test_kw_method_without_lists_is_unavailable(tmp_path):
    r, _ = build(tmp_path, lists=False)
    with pytest.raises(MethodUnavailable, match="keyword lists not set for collection 'c'"):
        r.retrieve("c", "x", "kw_rrf")


def test_unknown_method_and_collection(tmp_path):
    r, _ = build(tmp_path)
    with pytest.raises(ValueError):
        r.retrieve("c", "x", "nope")
    with pytest.raises(UnknownCollection):
        r.retrieve("missing", "x", "hybrid")


def test_summarize_collection_summarises_the_query(tmp_path):
    seen: list[str] = []
    r, _ = build(tmp_path, summarize=True, reply=lambda p: seen.append(p) or "A summary.")
    before = len(seen)
    r.retrieve("c", "enzyme kinetics rate experiment", "kw_rrf", k=1)
    assert len(seen) == before + 1 and "enzyme kinetics rate experiment" in seen[-1]


def test_summarize_query_without_key_raises(tmp_path):
    r, _ = build(tmp_path, summarize=True)
    r.claude.api_key = None
    with pytest.raises(NoApiKey):
        r.retrieve("c", "a brand new question", "kw_individual")
    assert r.retrieve("c", "enzyme", "keyword", k=1)[0].id == "enzyme-item"


def test_prepare_summarises_all_queries_in_one_call(tmp_path):
    seen: list[str] = []
    r, _ = build(tmp_path, summarize=True, reply=lambda p: seen.append(p) or "S.")
    spec = r.spec("c")
    before = len(seen)
    r.prepare(spec, ["q one", "q two"])
    assert len(seen) == before + 2
    r.retrieve("c", "q one", "kw_all", k=1)
    assert len(seen) == before + 2


def test_iterative_stops_when_done(tmp_path):
    replies = iter(['{"done": false, "next_query": "market price demand"}', '{"done": true}'])
    r, conn = build(tmp_path)
    r.claude = fake_claude(conn, lambda p: next(replies))
    ids = [h.id for h in r.retrieve("c", "enzyme kinetics rate", "iterative", k=4)]
    assert "enzyme-item" in ids and "market-item" in ids


def test_iterative_makes_at_most_three_calls(tmp_path):
    r, conn = build(tmp_path)
    claude = fake_claude(conn, lambda p: '{"done": false, "next_query": "market"}')
    calls: list[str] = []
    real = claude.complete
    claude.complete = lambda prompt, **kw: calls.append(prompt) or real(prompt, **kw)
    r.claude = claude
    r.retrieve("c", "enzyme", "iterative", k=4)
    assert len(calls) == 3


def test_is_paid(tmp_path):
    r, _ = build(tmp_path)
    plain = CollectionSpec("c", "mcq", False, "fake/dense", 256)
    summarizing = CollectionSpec("s", "mcq", True, "fake/dense", 256)
    assert r.is_paid(summarizing, "kw_all") and r.is_paid(summarizing, "kw_hybrid")
    assert not r.is_paid(plain, "kw_all")
    assert r.is_paid(plain, "iterative") and not r.is_paid(summarizing, "hybrid")


def test_retrieve_refuses_model_mismatch(tmp_path):
    from ragqa_core.indexer import ModelMismatch

    r, _ = build(tmp_path)
    r.models.dense_model = "other/model"
    with pytest.raises(
        ModelMismatch, match="built with fake/dense; service is running other/model"
    ):
        r.retrieve("c", "enzyme", "hybrid")


def test_kw_hybrid_computes_query_keywords_once(tmp_path):
    r, _ = build(tmp_path)
    calls: list[str] = []
    real = r.indexer.keywords_for
    r.indexer.keywords_for = lambda *a: calls.append("x") or real(*a)
    hits = r.retrieve("c", "enzyme kinetics rate experiment", "kw_hybrid", k=1)
    assert hits[0].id == "enzyme-item" and hits[0].matched and len(calls) == 1


def test_iterative_keeps_the_best_passage_of_every_hop(tmp_path):
    replies = iter(
        [
            '{"done": false, "next_query": "market price demand"}',
            '{"done": false, "next_query": "volcano magma lava"}',
            '{"done": true}',
        ]
    )
    r, conn = build(tmp_path)
    r.claude = fake_claude(conn, lambda p: next(replies))
    ids = [h.id for h in r.retrieve("c", "enzyme kinetics rate", "iterative", k=3)]
    assert ids == ["enzyme-item", "market-item", "volcano-item"]
