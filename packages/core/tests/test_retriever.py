from dataclasses import replace

import pytest

from aiconnect.ai import NoConnection
from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.catalog import CONTROLS
from ragqa_core.db import claim_next_job_id, connect, create_job, set_source_enabled
from ragqa_core.fakes import fake_ai, fake_models, fake_replies
from ragqa_core.indexer import Indexer
from ragqa_core.prompts import QUERY_TASK
from ragqa_core.retriever import (
    METHOD_NAMES,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
    rrf,
)
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Hit, Item, KeywordLists, SearchSettings

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
# The vocabulary: each domain's topic words, so an item's tags are its own domain's.
TERMS = [w for d in DOMAINS.values() for w in d[0]]
FREE = [m for m in METHOD_NAMES if m not in ("question_only", "iterative", *CONTROLS)]


def hit(i: str) -> Hit:
    return Hit(i, i, 0.0, {}, [], None, None, {})


def hits(*ids: str) -> list[Hit]:
    return [hit(i) for i in ids]


def build(
    tmp_path,
    summarize=False,
    lists=True,
    reply=lambda p: "A summary.",
    vocabulary=True,
    qa_type="mcq",
    recipe="paper_2024",
):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, reply)
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("c", qa_type, summarize, recipe)
    ix.add(spec, ITEMS, "x")
    if lists:
        ix.set_lists(spec, LISTS)
    r = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    if vocabulary:
        tag_all(r, TERMS)
    return r, conn


def tag_all(r: Retriever, terms: list[str]) -> None:
    """Put a vocabulary in force and tag every stored item with it by similarity."""
    spec = r.spec("c")
    db.set_vocabulary_version(r.conn, "c", r.builder.save_vocabulary(spec, terms))
    r.builder.tag(r.spec("c"), r.builder.stored_items("c"))


def tagged(r: Retriever, tags: dict[str, list[str]]) -> None:
    """Set the given items' tags (and kw_tags vectors) by hand."""
    ids = list(tags)
    vectors = r.models.embed(["; ".join(tags[i]) for i in ids])
    r.store.set_tags("c", ids, [tags[i] for i in ids], vectors)


def test_keyword_methods_refuse_while_rekeying(tmp_path):
    r, conn = build(tmp_path)
    create_job(conn, "rekey", "c", {"version": "v"})
    claim_next_job_id(conn)
    with pytest.raises(MethodUnavailable, match="re-keyed"):
        r.retrieve("c", "enzyme kinetics", "kw_individual")
    assert r.retrieve("c", "enzyme kinetics", "hybrid")  # plain methods still answer


def test_rrf_sums_reciprocal_ranks():
    a, b, c = hit("a"), hit("b"), hit("c")
    fused = rrf([[a, b], [b, c]], k=1)
    assert [h.id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 3 + 1 / 2)
    assert fused[1].score == pytest.approx(1 / 2)


def test_weighted_rrf():
    a, b = hits("x", "y"), hits("y", "x")
    assert [h.id for h in rrf([a, b], 60, [2.0, 1.0])] == ["x", "y"]
    assert [h.id for h in rrf([a, b], 60, [1.0, 2.0])] == ["y", "x"]
    assert rrf([a, b], 1, [2.0, 1.0])[0].score == pytest.approx(2 / 2 + 1 / 3)


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


@pytest.mark.parametrize("method", CONTROLS)
def test_controls_keep_their_first_pick_whatever_k(tmp_path, method):
    # mcq answers use only the first hit, so it must not depend on k.
    r, _ = build(tmp_path)
    one = r.retrieve("c", "enzyme", method, k=1)[0].id
    assert r.retrieve("c", "enzyme", method, k=4)[0].id == one


@pytest.mark.parametrize("method", CONTROLS)
def test_controls_see_items_added_later(tmp_path, method):
    r, _ = build(tmp_path)
    r.retrieve("c", "enzyme", method, k=1)
    r.indexer.add(r.spec("c"), [Item("late-item", "added after the first query", "B", "late")], "x")
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
    r.ai.update_connection("anthropic", api_key=None)
    with pytest.raises(NoConnection):
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
    r.ai = fake_ai(conn, lambda p: next(replies))
    ids = [h.id for h in r.retrieve("c", "enzyme kinetics rate", "iterative", k=4)]
    assert "enzyme-item" in ids and "market-item" in ids


def test_iterative_makes_at_most_three_calls(tmp_path):
    r, conn = build(tmp_path)
    ai = fake_ai(conn, lambda p: '{"done": false, "next_query": "market"}')
    calls: list[str] = []
    real = ai.complete
    ai.complete = lambda prompt, **kw: calls.append(prompt) or real(prompt, **kw)
    r.ai = ai
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
    r.ai = fake_ai(conn, lambda p: next(replies))
    ids = [h.id for h in r.retrieve("c", "enzyme kinetics rate", "iterative", k=3)]
    assert ids == ["enzyme-item", "market-item", "volcano-item"]


SKIP_METHODS = [m for m in METHOD_NAMES if m != "question_only"]


def build_sources(tmp_path):
    r, conn = build(tmp_path, lists=False, vocabulary=False)
    spec = r.spec("c")
    r.indexer.add(spec, [Item("a1", "enzyme rate shared words", "A", "e")], "a")
    r.indexer.add(spec, [Item("a2", "market price shared words", "A", "e")], "a")
    r.indexer.add(spec, [Item("b1", "enzyme market shared words", "A", "e")], "b")
    r.indexer.set_lists(spec, LISTS)
    tag_all(r, TERMS)
    r.ai = fake_ai(conn, lambda p: '{"done": true}')
    return r, conn


@pytest.mark.parametrize("method", SKIP_METHODS)
def test_disabled_source_is_never_retrieved(tmp_path, method):
    r, conn = build_sources(tmp_path)
    query = "enzyme market rate price shared words"
    assert method in CONTROLS or "b1" in [h.id for h in r.retrieve("c", query, method, k=10)]
    set_source_enabled(conn, "c", "b", False)
    hits = r.retrieve("c", query, method, k=10)
    assert hits and "b1" not in [h.id for h in hits]


@pytest.mark.parametrize("method", SKIP_METHODS)
def test_excluded_items_are_never_retrieved(tmp_path, method):
    r, _ = build_sources(tmp_path)
    query = "enzyme market rate price shared words"
    assert method in CONTROLS or "b1" in [h.id for h in r.retrieve("c", query, method, k=10)]
    hits = r.retrieve("c", query, method, k=10, exclude=["b1", "a2"])
    assert hits and not {"b1", "a2"} & {h.id for h in hits}


@pytest.mark.parametrize("method", ["hybrid", "keyword_rag"])
def test_fused_methods_fill_k_past_an_excluded_pool(tmp_path, monkeypatch, method):
    monkeypatch.setattr("ragqa_core.retriever.POOL", 2)
    r, conn = build(tmp_path, lists=False, vocabulary=False)
    spec = r.spec("c")
    fillers = [Item(f"f{n}", f"enzyme market shared words filler {n}", "A", "e") for n in range(8)]
    r.indexer.add(spec, fillers, "x")
    tag_all(r, TERMS)
    query = "enzyme market rate price shared words"
    # Every channel's whole pool: at most POOL from each of three channels.
    pooled = [h.id for h in r.retrieve("c", query, method, k=6)]
    hits = r.retrieve("c", query, method, k=2, exclude=pooled)
    assert len(hits) == 2 and not set(pooled) & {h.id for h in hits}


def test_controls_refresh_after_a_switch(tmp_path):
    r, conn = build_sources(tmp_path)
    assert "b1" in [h.id for h in r.retrieve("c", "enzyme", "random_example", k=3)]
    set_source_enabled(conn, "c", "b", False)
    assert "b1" not in [h.id for h in r.retrieve("c", "enzyme", "random_example", k=3)]


def test_all_ids_returns_the_list_it_fetched_even_if_another_thread_replaces_the_entry(tmp_path):
    r, _ = build_sources(tmp_path)
    real = r.store.ids

    def racing(name, *args, **kwargs):
        ids = real(name, *args, **kwargs)
        r._ids[name] = ((0, ("other",)), ["someone-elses"])
        return ids

    r.store.ids = racing
    assert "someone-elses" not in r._all_ids("c", ("b",))


# -- Keyword-RAG -------------------------------------------------------------------

OHM = ["ohm-1", "ohm-2", "ohm-3"]


def build_physics(tmp_path):
    """Three items on Ohm's law, one on kinematics nearest the query alone."""
    r, conn = build(tmp_path, lists=False, vocabulary=False)
    spec = r.spec("c")
    physics = [Item(i, "current voltage resistor", "A", "V = IR") for i in OHM]
    r.indexer.add(spec, [*physics, Item("kin", "car velocity", "B", "v = d/t")], "x")
    db.set_vocabulary_version(
        conn, "c", r.builder.save_vocabulary(spec, ["ohm's law", "kinematics", "optics"])
    )
    tagged(r, {**{i: ["ohm's law"] for i in OHM}, "kin": ["kinematics"]})
    return r, conn


def test_classifier_votes_neighbours_tags(tmp_path):
    r, _ = build_physics(tmp_path)
    spec = r.spec("c")
    classify = SearchSettings(query_tags="classifier")
    # "kin" is the single nearest unit, but three Ohm's-law neighbours outvote it.
    query = r.models.embed(["car velocity current"])[0]
    assert r.store.search("c", "text_dense", query, 1)[0].id == "kin"
    assert r.query_tags(spec, "car velocity current", classify, ()) == ["ohm's law", "kinematics"]
    one = replace(spec, build=replace(spec.build, tag_k=1))
    assert r.query_tags(one, "car velocity current", classify, ()) == ["ohm's law"]
    # Disabled sources cast no vote.
    assert r.query_tags(spec, "car velocity current", classify, ("x",)) == []
    # Nor do a question's excluded items.
    tags = r.query_tags(spec, "car velocity current", classify, (), exclude=OHM)
    assert tags == ["kinematics"]


def test_similarity_query_tags_are_the_nearest_terms(tmp_path):
    r, _ = build(tmp_path)
    tags = r.query_tags(r.spec("c"), "volcano magma", SearchSettings(query_tags="similarity"), ())
    assert tags[:2] == ["volcano", "magma"] and len(tags) == 3


def test_model_query_tags_are_the_ais_pick_of_the_vocabulary(tmp_path):
    r, conn = build_physics(tmp_path)
    sent: list[str] = []
    reply = '{"terms": ["optics", "made-up", "kinematics"]}'
    fake_replies(r.ai, lambda p: sent.append(p) or reply)
    asked = SearchSettings(query_tags="model")
    spec = r.spec("c")
    # Terms outside the vocabulary are dropped; the rest keep the AI's order.
    assert r.query_tags(spec, "a lens bends light", asked, ()) == ["optics", "kinematics"]
    [prompt] = sent
    assert "1. ohm's law" in prompt and prompt.endswith("Question:\na lens bends light")
    assert "a solution to the question below would use" in prompt  # a worked-examples corpus
    [row] = conn.execute("select purpose from ai_ledger")
    assert row["purpose"] == "query_tags"
    # A reply naming no term falls back to similarity: a question never goes untagged.
    fake_replies(r.ai, lambda p: "I am not sure.")
    similar = r.query_tags(spec, "car velocity", SearchSettings(query_tags="similarity"), ())
    assert r.query_tags(spec, "car velocity", asked, ()) == similar


def test_prepared_model_tags_cost_one_call_each(tmp_path):
    r, conn = build_physics(tmp_path)
    fake_replies(r.ai, lambda p: '{"terms": ["optics"]}')
    questions = ["a lens bends light", "a mirror reflects it"]
    r.prepare_tags(r.spec("c"), questions)
    asked = SearchSettings(query_tags="model")
    for q in questions:
        assert r.query_tags(r.spec("c"), q, asked, ()) == ["optics"]
    assert conn.execute("select count(*) from ai_ledger")[0][0] == len(questions)


def test_oracle_query_tags_need_gold(tmp_path):
    r, _ = build(tmp_path)
    with pytest.raises(ValueError, match="oracle tags need a question's gold"):
        r.query_tags(r.spec("c"), "x", SearchSettings(query_tags="oracle"), ())


def test_keyword_rag_fuses_three_channels(tmp_path):
    r, _ = build(tmp_path)
    r.indexer.add(r.spec("c"), [Item("odd-item", "nothing in common here", "D", "e")], "x")
    tagged(r, {"odd-item": ["enzyme", "kinetics"]})
    no_meaning = SearchSettings(weights={"meaning": 0.0, "words": 1.0, "keywords": 1.0})
    found = r.retrieve("c", "enzyme kinetics", "keyword_rag", k=10, search=no_meaning)
    by_id = {h.id: h for h in found}
    assert by_id["odd-item"].channels == ["keywords"]  # no shared word: only its tags found it
    assert by_id["enzyme-item"].channels == ["words", "keywords"]
    [top] = r.retrieve("c", "enzyme kinetics", "keyword_rag", k=1)
    assert top.id == "enzyme-item" and top.channels == ["meaning", "words", "keywords"]
    spec = r.spec("c")
    tags = r.query_tags(spec, "enzyme kinetics", spec.search or SearchSettings(), ())
    assert top.matched == [t for t in tags if t in top.tags]  # the query's tags the hit carries
    assert "enzyme" in top.matched and "kinetics" in top.matched


def test_keyword_rag_uses_given_tags_and_weights(tmp_path):
    r, _ = build(tmp_path)
    only_keywords = SearchSettings(weights={"meaning": 0.0, "words": 0.0, "keywords": 1.0})
    [top] = r.retrieve(
        "c", "enzyme kinetics", "keyword_rag", k=1, search=only_keywords, tags=["lava", "magma"]
    )
    assert top.id == "volcano-item" and top.channels == ["keywords"]
    assert top.matched == ["magma"]  # "lava" is not a tag of any item


def test_zero_weight_channel_not_searched(tmp_path):
    r, _ = build(tmp_path)
    calls: list[str] = []
    real = r.store.search_sparse
    r.store.search_sparse = lambda *a, **kw: calls.append("words") or real(*a, **kw)
    no_words = SearchSettings(weights={"meaning": 1.0, "words": 0.0, "keywords": 1.0})
    found = r.retrieve("c", "enzyme kinetics", "keyword_rag", k=4, search=no_words)
    assert calls == [] and found and all("words" not in h.channels for h in found)
    r.retrieve("c", "enzyme kinetics", "keyword_rag", k=4)
    assert calls == ["words"]


def test_keyword_rag_defaults_to_the_collections_search(tmp_path):
    r, conn = build(tmp_path)
    db.set_search(
        conn, "c", SearchSettings(weights={"meaning": 0.0, "words": 1.0, "keywords": 0.0})
    )
    found = r.retrieve("c", "enzyme kinetics", "keyword_rag", k=4)
    assert [h.id for h in found] == ["enzyme-item"] and found[0].channels == ["words"]


def test_reference_meaning_channel_uses_chunks(tmp_path):
    r, _ = build(tmp_path, lists=False, qa_type="multihop", recipe="reference")
    used: list[str] = []
    real_search, real_chunks = r.store.search, r.store.search_chunks
    r.store.search = lambda name, using, *a, **kw: (
        used.append(using) or real_search(name, using, *a, **kw)
    )
    r.store.search_chunks = lambda *a, **kw: used.append("chunks") or real_chunks(*a, **kw)
    similarity = SearchSettings(query_tags="similarity")
    [top] = r.retrieve("c", "enzyme kinetics", "keyword_rag", k=1, search=similarity)
    assert top.id == "enzyme-item" and "meaning" in top.channels
    assert sorted(used) == ["chunks", "kw_tags"]  # never the item's text_dense


def test_worked_meaning_channel_uses_item_vectors(tmp_path):
    r, _ = build(tmp_path)
    used: list[str] = []
    real_search = r.store.search
    r.store.search = lambda name, using, *a, **kw: (
        used.append(using) or real_search(name, using, *a, **kw)
    )
    r.store.search_chunks = lambda *a, **kw: pytest.fail("a worked collection searched chunks")
    r.retrieve("c", "enzyme kinetics", "keyword_rag", k=1, tags=["enzyme"])
    assert sorted(used) == ["kw_tags", "text_dense"]


def test_keyword_rag_without_vocabulary_is_unavailable(tmp_path):
    r, _ = build(tmp_path, vocabulary=False)
    message = (
        "no vocabulary for collection 'c'; build it with the worked examples or reference recipe"
    )
    with pytest.raises(MethodUnavailable, match=message):
        r.retrieve("c", "enzyme", "keyword_rag")
    with pytest.raises(MethodUnavailable, match=message):
        r.query_tags(r.spec("c"), "enzyme", SearchSettings(), ())


def test_keyword_rag_refuses_while_retagging(tmp_path):
    r, conn = build(tmp_path)
    create_job(conn, "retag", "c", {"version": "v"})
    claim_next_job_id(conn)
    with pytest.raises(MethodUnavailable, match="re-tagged"):
        r.retrieve("c", "enzyme kinetics", "keyword_rag")
    assert r.retrieve("c", "enzyme kinetics", "kw_individual")  # the lists still answer


def stopped_retag(conn, status: str, progress: float) -> None:
    job = create_job(conn, "retag", "c", {"version": "v"})
    claim_next_job_id(conn)
    db.update_job(conn, job, status=status, progress=progress)


def test_keyword_rag_refuses_after_a_retag_stopped_writing(tmp_path):
    r, conn = build(tmp_path)
    stopped_retag(conn, "cancelled", 0.5)  # the worker checks before it records progress
    create_job(conn, "retag", "c", {"version": "v2"})  # saved again, still waiting
    with pytest.raises(MethodUnavailable, match="partly re-tagged"):
        r.retrieve("c", "enzyme kinetics", "keyword_rag")
    with pytest.raises(MethodUnavailable, match="partly re-tagged"):
        r.query_tags(r.spec("c"), "enzyme", SearchSettings(), ())
    claim_next_job_id(conn)
    db.update_job(conn, db.newest_jobs(conn)[0]["id"], status="done", progress=1.0)
    assert r.retrieve("c", "enzyme kinetics", "keyword_rag")


def test_keyword_rag_answers_after_a_retag_stopped_before_writing(tmp_path):
    r, conn = build(tmp_path)
    stopped_retag(conn, "cancelled", 0.3)  # still choosing tags: nothing was written
    assert r.retrieve("c", "enzyme kinetics", "keyword_rag")


def test_keyword_rag_is_not_paid(tmp_path):
    r, _ = build(tmp_path)
    summarizing = CollectionSpec("s", "mcq", True, "fake/dense", 256)
    assert not r.is_paid(summarizing, "keyword_rag")
