import json
from dataclasses import replace

import pytest

from aiconnect.ai import Cancelled
from ragqa_core import db, search_defaults
from ragqa_core.builder import Builder
from ragqa_core.db import connect, get_keyword_lists
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer, ModelMismatch, chunk, embed_text
from ragqa_core.keywords import keyword_string, list_version
from ragqa_core.retriever import Retriever
from ragqa_core.store import CHUNK, Store, chunks_of, item_hash, rebuild_of
from ragqa_core.types import CollectionSpec, Item, KeywordLists

LISTS = KeywordLists(
    topic=["enzyme kinetics", "genetics", "market economics", "plate tectonics", "optics"],
    method=["titration", "tabular scrutiny", "field survey", "spectroscopy"],
    common=["rate", "mass", "price", "temperature", "volume", "voltage"],
)
OTHER_LISTS = KeywordLists(
    topic=["history", "chemistry", "astronomy"],
    method=["interview", "simulation", "meta-analysis"],
    common=["year", "count", "speed", "length"],
)
ITEMS = [
    Item("1", "enzyme kinetics rate rose with temperature", "A", "Rates rise."),
    Item("2", "stock market price fell sharply", "B", "Prices fall."),
    Item("3", "genetics of pea plants mass", "C", "Mendel."),
]


@pytest.fixture
def env(tmp_path):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    calls: list[str] = []
    ai = fake_ai(conn, lambda p: calls.append(p) or "A summary.")
    ix = Indexer(conn, store, models, ai)
    return conn, store, models, ix, calls


def test_create_saves_spec_with_model(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    assert (spec.name, spec.qa_type, spec.summarize, spec.dense_model) == (
        "c",
        "mcq",
        False,
        "fake/dense",
    )
    assert (spec.recipe, spec.corpus_type) == ("paper_2024", "worked")
    assert spec == ix.spec("c")
    assert store.count("c") == 0


def test_create_with_recipe_records_settings(env):
    _, _, _, ix, _ = env
    spec = ix.create("w", "science", True, recipe="worked_examples")
    assert (spec.corpus_type, spec.recipe, spec.summarize) == ("worked", "worked_examples", False)
    assert spec.search is not None and spec.search.example == "abstracted"
    assert spec.build is not None and spec.build.units == "whole"
    assert spec == ix.spec("w")
    paper = ix.create("p", "mcq", True)
    assert paper.summarize and paper.build is not None and paper.build.added_text == "summary"
    with pytest.raises(ValueError):
        ix.create("x", "mcq", False, recipe="nope")


def test_create_rejects_bad_name(env):
    *_, ix, _ = env
    with pytest.raises(ValueError):
        ix.create("bad name!", "mcq", False)


def test_add_without_lists_indexes_text_only(env):
    conn, store, models, ix, calls = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    assert store.count("c") == 3 and calls == []
    assert store.get("c", ["1"])[0].keywords == {}
    assert store.search_chunks("c", models.embed(["pea plants"])[0], 1)[0][0] == "3"


def test_a_text_that_is_its_own_only_chunk_is_embedded_once(env):
    # Whole units, and any item short enough to be one chunk, used to embed every text
    # twice: once as the item, once as its chunk. On BRIGHT that was half the load.
    _, store, models, ix, _ = env
    embedded: list[str] = []
    real = models.embed
    models.embed = lambda texts, model=None: embedded.extend(texts) or real(texts, model)  # type: ignore[method-assign]
    long = Item("4", " ".join(f"w{n}" for n in range(150)), "D", "Long.")
    spec = ix.create("c", "mcq", False)
    ix.add(spec, [*ITEMS, long], "x")
    assert sorted(embedded).count(ITEMS[0].text) == 1
    assert len(embedded) == 4 + 2  # four items, and the long one's two windows
    assert store.search_chunks("c", real(["pea plants"])[0], 1)[0][0] == "3"
    whole = ix.create("w", "science", False, recipe="worked_examples")
    embedded.clear()
    ix.add(whole, [*ITEMS, long], "x")
    assert len(embedded) == 4
    assert store.search_chunks("w", real([long.text])[0], 1)[0][0] == "4"


def test_add_reports_progress(env):
    *_, ix, _ = env
    spec = ix.create("c", "mcq", False)
    seen: list[float] = []
    ix.add(spec, ITEMS, "x", progress=seen.append)
    assert seen and seen[-1] == 1.0


def test_set_lists_assigns_keywords_and_returns_version(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    version = ix.set_lists(spec, LISTS)
    assert version == list_version(LISTS)
    assert get_keyword_lists(conn, "c") == LISTS
    hit = store.get("c", ["1"])[0]
    assert set(hit.keywords) == {"topic", "method", "common", "all"}
    assert len(hit.keywords["topic"]) == 3 and len(hit.keywords["method"]) == 3
    assert hit.keywords["topic"][0] == "enzyme kinetics"
    own = keyword_string(
        hit.keywords["topic"] + hit.keywords["method"] + hit.keywords["common"], None
    )
    found = store.search("c", "kw_individual", models.embed([own])[0], 1)
    assert found[0].id == "1" and found[0].score == pytest.approx(1.0, abs=1e-5)


def test_add_after_lists_assigns_keywords(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.set_lists(spec, LISTS)
    spec = ix.spec("c")
    ix.add(spec, ITEMS, "x")
    assert store.get("c", ["2"])[0].keywords["topic"][0] == "market economics"


def test_add_with_summarize_calls_claude_once_per_item(env):
    conn, store, models, ix, calls = env
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS, "x")
    assert len(calls) == len(ITEMS)
    assert store.get("c", ["1"])[0].summary == "A summary."


def test_summary_vector_only_when_summarize(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS, "x")
    ix.set_lists(spec, LISTS)
    assert store.search("c", "kw_summary", models.embed(["A summary."])[0], 3)
    plain = ix.create("p", "mcq", False)
    ix.add(plain, ITEMS, "x")
    ix.set_lists(plain, LISTS)
    assert store.search("p", "kw_summary", models.embed(["A summary."])[0], 3) == []


def test_estimate_is_zero_without_summarize(env):
    *_, ix, _ = env
    assert ix.estimate_add_usd(ix.create("c", "mcq", False), ITEMS, "x") == 0.0
    assert ix.estimate_add_usd(ix.create("s", "mcq", True), ITEMS, "x") > 0.0


def small_env(tmp_path):
    """An Indexer whose fake models give fake/small 64 dimensions, and every model an
    embed call named."""
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models(dims={"fake/small": 64})
    ix = Indexer(conn, store, models, fake_ai(conn, lambda p: "A summary."))
    named: list[str | None] = []
    real = models.embed

    def embed(texts, model=None):
        named.append(model)
        return real(texts, model)

    models.embed = embed  # type: ignore[method-assign]
    return store, models, ix, named


def test_a_store_is_embedded_with_its_own_model(tmp_path):
    store, _, ix, named = small_env(tmp_path)
    spec = ix.create("s", "mcq", False, dense_model="fake/small")
    assert spec.dense_dim == 64
    ix.add(spec, ITEMS, "x")
    assert store.vectors("s", "text_dense", 10).shape[1] == 64
    assert named and all(m == "fake/small" for m in named)


def test_list_and_term_vector_caches_are_per_model(tmp_path):
    store, _, ix, _ = small_env(tmp_path)
    for name, model in (("a", "fake/dense"), ("b", "fake/small")):
        spec = ix.create(name, "mcq", False, dense_model=model)
        ix.add(spec, ITEMS, "x")
        ix.set_lists(ix.spec(name), LISTS)  # the same lists version: no cached vectors reused
    assert store.vectors("a", "kw_all", 3).shape[1] == 256
    assert store.vectors("b", "kw_all", 3).shape[1] == 64


def test_check_model_refuses_a_model_this_service_cannot_load(env):
    *_, models, ix, _ = env
    real = models.dense_dim

    def dense_dim(model=None):
        if model == "gone/model":
            raise ValueError("gone/model is not in fastembed's supported models")
        return real(model)

    models.dense_dim = dense_dim  # type: ignore[method-assign]
    gone = CollectionSpec("c", "mcq", False, "gone/model", 256)
    with pytest.raises(ModelMismatch, match="built with gone/model, which this service can't load"):
        ix.add(gone, ITEMS, "x")
    narrow = CollectionSpec("c", "mcq", False, "fake/dense", 64)
    with pytest.raises(ModelMismatch, match="at 64 dimensions; fake/dense gives 256"):
        ix.add(narrow, ITEMS, "x")


def test_an_existing_store_keeps_its_model_after_the_service_default_changes(tmp_path):
    store, models, ix, _ = small_env(tmp_path)
    ix.add(ix.create("c", "mcq", False), ITEMS, "x")  # fake/dense, the service's model then
    models.dense_model = "fake/small"
    ix.add(ix.spec("c"), [Item("4", "enzyme assay of a new substrate", "D", "New.")], "x")
    assert store.vectors("c", "text_dense", 10).shape == (4, 256)
    asked: list[str | None] = []
    real = models.embed_query

    def embed_query(texts, task="", model=None):
        asked.append(model)
        return real(texts, task, model)

    models.embed_query = embed_query  # type: ignore[method-assign]
    builder = Builder(ix.conn, store, models, ix.ai, ix)
    retriever = Retriever(ix.conn, store, models, ix.ai, ix, builder)
    assert "1" in {h.id for h in retriever.retrieve("c", "enzyme", "hybrid")}
    assert asked == ["fake/dense"]
    assert ix.create("n", "mcq", False).dense_model == "fake/small"


def test_a_new_store_takes_the_settings_model_and_keeps_it(tmp_path, monkeypatch):
    _, models, ix, _ = small_env(tmp_path)
    monkeypatch.setattr(
        search_defaults, "supported_embedders", lambda: {"fake/small": {"dim": 64, "size_gb": 0.1}}
    )
    search_defaults.save(ix.conn, {"dense_model": "fake/small"})
    spec = ix.create("n", "mcq", False)
    assert (spec.dense_model, spec.dense_dim) == ("fake/small", 64)
    search_defaults.save(ix.conn, {"dense_model": None})
    assert ix.spec("n") == spec
    assert ix.create("m", "mcq", False).dense_model == models.dense_model


def test_save_lists_does_not_switch_until_rekey_finishes(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    v1 = ix.set_lists(spec, LISTS)
    v2 = ix.save_lists(spec, OTHER_LISTS)
    assert v2 != v1 and ix.spec("c").keyword_version == v1
    assert get_keyword_lists(conn, "c", v2) == OTHER_LISTS
    ix.rekey("c", v2)
    assert ix.spec("c").keyword_version == v2


def test_rekey_that_fails_leaves_the_old_version(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    v1 = ix.set_lists(spec, LISTS)
    v2 = ix.save_lists(spec, OTHER_LISTS)

    def boom(p):
        raise Cancelled()

    with pytest.raises(Cancelled):
        ix.rekey("c", v2, progress=boom)
    assert ix.spec("c").keyword_version == v1


def test_rekey_reports_progress_per_batch_and_once_for_an_empty_collection(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    seen: list[float] = []
    ix.rekey("c", ix.save_lists(spec, LISTS), seen.append)
    assert seen == [1.0]
    ix.add(spec, ITEMS, "x")
    seen.clear()
    ix.rekey("c", ix.save_lists(spec, OTHER_LISTS), seen.append)
    assert seen == [1.0]


def test_rekey_rejects_an_unknown_version(env):
    *_, ix, _ = env
    ix.create("c", "mcq", False)
    with pytest.raises(ValueError, match="no keyword lists"):
        ix.rekey("c", "missing")


def test_propose_lists_returns_all_three_lists(tmp_path):
    conn = connect(tmp_path / "r.db")
    prompts: list[str] = []
    reply = '{"topic": ["genetics", "genetics"], "method": ["titration"], "common": ["rate"]}'
    ai = fake_ai(conn, lambda p: prompts.append(p) or reply)
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("c", "mcq", False)
    lists = ix.propose_lists(spec, ITEMS + [Item("4", "enzyme rate genetics")])
    assert (lists.topic, lists.method, lists.common) == (["genetics"], ["titration"], ["rate"])
    assert '"common"' in prompts[0]


@pytest.mark.parametrize("missing", ["topic", "method", "common"])
def test_propose_lists_refuses_a_missing_list(tmp_path, missing):
    conn = connect(tmp_path / "r.db")
    lists = {"topic": ["genetics"], "method": ["titration"], "common": ["rate"], missing: []}
    ai = fake_ai(conn, lambda p: json.dumps(lists))
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("c", "mcq", False)
    with pytest.raises(ValueError, match="no usable"):
        ix.propose_lists(spec, ITEMS)


def test_keyword_list_embeddings_are_reused_across_calls(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    embedded: list[str] = []
    real = models.embed
    models.embed = lambda texts, model=None: embedded.extend(texts) or real(texts, model)
    ix.set_lists(spec, LISTS)
    assert "enzyme kinetics" in embedded
    before = len(embedded)
    ix.keywords_for(spec, LISTS, ["first query"], [None])
    ix.keywords_for(spec, LISTS, ["second query"], [None])
    assert "enzyme kinetics" not in embedded[before:]  # the lists are not embedded again


def test_keywords_for_embeds_in_two_passes(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", True)
    ix.keywords_for(spec, LISTS, ["warm the list cache"], ["s"])
    calls: list[int] = []
    real = models.embed
    models.embed = lambda texts, model=None: calls.append(len(texts)) or real(texts, model)
    keywords, vectors = ix.keywords_for(spec, LISTS, ["one query"], ["a summary"])
    assert len(calls) == 2
    assert set(vectors) == {
        "kw_all",
        "kw_individual",
        "kw_topic",
        "kw_method",
        "kw_common",
        "kw_summary",
    }
    assert all(v.shape[0] == 1 for v in vectors.values())


def test_set_lists_rejects_an_empty_facet_and_keeps_the_old_lists(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    version = ix.set_lists(spec, LISTS)
    with pytest.raises(ValueError, match="'topic' is empty"):
        ix.set_lists(spec, KeywordLists(topic=[], method=LISTS.method, common=LISTS.common))
    assert get_keyword_lists(conn, "c") == LISTS and ix.spec("c").keyword_version == version


def test_set_lists_saves_nothing_if_the_lists_cannot_be_embedded(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)

    def broken(texts, model=None):
        raise RuntimeError("onnx failed")

    models.embed = broken
    with pytest.raises(RuntimeError, match="onnx failed"):
        ix.set_lists(spec, LISTS)
    assert get_keyword_lists(conn, "c") is None and ix.spec("c").keyword_version is None


def test_propose_lists_rejects_a_reply_without_lists(tmp_path):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_ai(conn, lambda p: "Sorry,"))
    spec = ix.create("c", "mcq", False)
    with pytest.raises(ValueError, match="no usable topic or method"):
        ix.propose_lists(spec, ITEMS)


def counting_embed(models):
    n = [0]
    real = models.embed

    def embed(texts, model=None):
        n[0] += 1
        return real(texts, model)

    models.embed = embed
    return n


def test_add_twice_embeds_nothing_the_second_time(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    n = counting_embed(models)
    assert ix.add(spec, ITEMS, "x") == {"new": 0, "updated": 0, "unchanged": 3}
    assert n[0] == 0


def test_add_counts_new_updated_unchanged(env):
    *_, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    again = [Item("1", "edited text", "A", "Rates rise."), *ITEMS[1:], Item("4", "brand new", "D")]
    assert ix.add(spec, again, "x") == {"new": 1, "updated": 1, "unchanged": 2}


def chunk_texts(store, name):
    points, _ = store.client.scroll(f"{name}__chunks", limit=100, with_payload=True)
    return [p.payload["text"] for p in points if p.payload["parent_id"] == "long"]


def test_shorter_text_leaves_no_stale_chunks(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    words = [f"w{i}" for i in range(260)]
    ix.add(spec, [Item("long", " ".join(words), "A")], "x")
    assert len(chunk_texts(store, "c")) > 1
    ix.add(spec, [Item("long", " ".join(words[:20]), "A")], "x")
    left = chunk_texts(store, "c")
    assert len(left) == 1 and "w250" not in left[0]


def test_add_summarises_only_changed_items(env):
    conn, store, models, ix, calls = env
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS, "x")
    calls.clear()
    ix.add(spec, [Item("1", "edited", "A"), *ITEMS[1:]], "x")
    assert len(calls) == 1


def test_estimate_add_prices_only_changed(env):
    *_, ix, _ = env
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS, "x")
    assert ix.estimate_add_usd(spec, ITEMS, "x") == 0.0
    assert ix.estimate_add_usd(spec, [Item("1", "edited", "A"), *ITEMS[1:]], "x") > 0.0


def test_add_sets_source_on_items_and_chunks(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    items = [Item("1", "plain text", "A")]
    ix.add(spec, items, "notes")
    assert items[0].meta == {}
    assert store.get("c", ["1"])[0].meta["source"] == "notes"
    assert ix.add(spec, items, "notes")["unchanged"] == 1
    assert ix.add(spec, items, "other")["updated"] == 1


def boom(*args, **kwargs):
    raise RuntimeError("died")


def test_an_add_that_dies_after_the_items_is_redone(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    store.upsert_chunks = boom
    with pytest.raises(RuntimeError):
        ix.add(spec, ITEMS, "x")
    del store.upsert_chunks
    assert ix.add(spec, ITEMS, "x") == {"new": 0, "updated": 3, "unchanged": 0}
    assert store.search_chunks("c", models.embed(["pea plants"])[0], 1)[0][0] == "3"
    assert ix.add(spec, ITEMS, "x")["unchanged"] == 3


def test_an_add_that_dies_before_the_keywords_is_redone(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.set_lists(spec, LISTS)
    store.set_keywords = boom
    with pytest.raises(RuntimeError):
        ix.add(spec, ITEMS, "x")
    del store.set_keywords
    assert ix.add(spec, ITEMS, "x")["unchanged"] == 0
    assert store.get("c", ["1"])[0].keywords != {}


def write_legacy(store, models, items):
    texts = [i.text for i in items]
    store.upsert_items("c", items, models.embed(texts), models.sparse(texts), [None] * len(items))
    store.upsert_chunks("c", [i.id for i in items], texts, models.embed(texts), [None] * len(items))


def test_backfill_labels_items_and_chunks(env):
    conn, store, models, ix, _ = env
    ix.create("c", "mcq", False)
    legacy = [
        Item("sqa-train-1", "enzyme kinetics rate", "A", "Rates.", {"skill": "x"}),
        Item("note-1", "stock market price", "B", None, {}),
    ]
    write_legacy(store, models, legacy)
    conn.execute(
        "insert into jobs (id, kind, collection, status, setup_json, created_at, updated_at)"
        " values ('j', 'load', 'c', 'failed', '{\"dataset\": \"scienceqa\"}', 'n', 'n')"
    )
    assert store.unsourced("c") == 4
    assert ix.backfill("c", lambda p: None) == {"labelled": 2}
    assert store.unsourced("c") == 0
    want = {"sqa-train-1": "scienceqa", "note-1": "upload:unknown"}
    hits = {h.id: h for h in store.get("c", list(want))}
    assert {i: h.meta["source"] for i, h in hits.items()} == want
    assert hits["sqa-train-1"].meta["skill"] == "x"
    labelled = [
        Item(i.id, i.text, i.answer, i.explanation, {**i.meta, "source": want[i.id]})
        for i in legacy
    ]
    assert store.hashes("c", list(want)) == {i.id: item_hash(i) for i in labelled}
    assert store.source_counts("c") == {"scienceqa": 1, "upload:unknown": 1}
    chunk_sources = {
        r.payload["parent_id"]: r.payload["meta"]["source"]
        for r in store.client.scroll("c__chunks", limit=100, with_payload=True)[0]
    }
    assert chunk_sources == want
    assert ix.backfill("c", lambda p: None) == {"labelled": 0}


def test_backfill_labels_chunks_of_already_labelled_items(env):
    conn, store, models, ix, _ = env
    ix.create("c", "mcq", False)
    write_legacy(store, models, ITEMS)
    store.set_payloads(
        "c",
        {
            i.id: {
                "meta": {"source": "hand"},
                "content_hash": item_hash(
                    Item(i.id, i.text, i.answer, i.explanation, {"source": "hand"})
                ),
            }
            for i in ITEMS
        },
    )
    assert store.unsourced("c") > 0
    assert ix.backfill("c", lambda p: None) == {"labelled": 0}
    assert store.unsourced("c") == 0
    chunks = store.client.scroll("c__chunks", limit=100, with_payload=True)[0]
    assert {r.payload["meta"]["source"] for r in chunks} == {"hand"}


def test_chunk_respects_words_and_overlap():
    text = " ".join(str(n) for n in range(250))
    assert len(chunk(text, 100, 10)) == 3 and chunk(text, 100, 10)[1].split()[0] == "90"
    assert chunk(text, 0) == [text]
    assert len(chunk(text)) == 3 and chunk(text)[1].split()[0] == "75"


def long_item():
    return Item("long", " ".join(f"w{n}" for n in range(250)), "A", "E")


def test_paper_2024_chunks_unchanged(env):
    _, store, _, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, [long_item()], "x")
    assert sorted(chunk_texts(store, "c")) == sorted(chunk(long_item().text, 100, 25))


def test_whole_units_store_one_chunk_per_item(env):
    _, store, _, ix, _ = env
    spec = ix.create("w", "science", False, recipe="worked_examples")
    ix.add(spec, [long_item()], "x")
    assert chunk_texts(store, "w") == [long_item().text]


def test_reference_recipe_uses_low_overlap_chunks(env):
    _, store, _, ix, _ = env
    spec = ix.create("r", "multihop", False, recipe="reference")
    ix.add(spec, [long_item()], "x")
    assert sorted(chunk_texts(store, "r")) == sorted(chunk(long_item().text, 100, 10))


def test_title_prefix_changes_embedding_not_text(env):
    _, store, models, ix, _ = env
    spec = ix.create("t", "mcq", False)
    spec = replace(spec, build=replace(spec.build, added_text="title_prefix"))
    item = Item("1", "volts and amps", "A", "E", {"title": "Ohm"})
    assert embed_text(spec, item) == "Ohm: volts and amps"
    assert embed_text(spec, Item("2", "plain", "A", "E", {"title": " "})) == "plain"
    seen: list[str] = []
    real = models.embed
    models.embed = lambda texts, model=None: seen.extend(texts) or real(texts, model)
    ix.add(spec, [item], "x")
    assert "Ohm: volts and amps" in seen and "volts and amps" not in seen
    assert store.get("t", ["1"])[0].text == "volts and amps"
    assert [p.payload["text"] for p in store.client.scroll("t__chunks", limit=10)[0]] == [
        "Ohm: volts and amps"
    ]


def test_induced_recipe_skips_keyword_lists(env):
    conn, store, _, ix, _ = env
    spec = ix.create("w", "science", False, recipe="worked_examples")
    db.save_keyword_lists(conn, "w", LISTS, list_version(LISTS))
    db.set_keyword_version(conn, "w", list_version(LISTS))
    ix.add(spec, ITEMS, "x")
    assert store.get("w", ["1"])[0].keywords == {}


def test_spec_without_build_acts_as_paper_2024(env):
    _, store, _, ix, _ = env
    spec = replace(ix.create("c", "mcq", False), build=None)
    ix.add(spec, [long_item()], "x")
    assert len(chunk_texts(store, "c")) == 3


def retriever_for(ix):
    builder = Builder(ix.conn, ix.store, ix.models, ix.ai, ix)
    return Retriever(ix.conn, ix.store, ix.models, ix.ai, ix, builder)


def dense_width(store, collection, vector="text_dense"):
    vectors = store.client.get_collection(collection).config.params.vectors
    return vectors[vector].size


TAGS = [["enzymes", "rates"], ["markets"], ["heredity", "peas"]]


def test_reembed_switches_the_model_and_keeps_items_payloads_lists_and_tags(tmp_path):
    store, models, ix, _ = small_env(tmp_path)
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS, "x")
    ix.set_lists(spec, LISTS)
    ids = [i.id for i in ITEMS]
    store.set_tags("c", ids, TAGS, models.embed([keyword_string(t, None) for t in TAGS]))
    before = ix.spec("c")
    assert before is not None
    kept = {
        h.id: (h.text, h.answer, h.explanation, h.meta, h.summary, h.tags)
        for h in store.get("c", ids)
    }
    hashes = store.hashes("c", ids)
    chunks = store.count(chunks_of("c"))
    result = ix.reembed("c", "fake/small")
    after = ix.spec("c")
    assert after is not None and (after.dense_model, after.dense_dim) == ("fake/small", 64)
    assert store.count("c") == 3 and after.keyword_version == before.keyword_version
    hits = store.get("c", ids)
    assert {h.id: (h.text, h.answer, h.explanation, h.meta, h.summary, h.tags) for h in hits} == (
        kept
    )
    assert all(set(h.keywords) == {"topic", "method", "common", "all"} for h in hits)
    # The picks and kw_* vectors are the new model's, recomputed, not carried over.
    target = replace(before, dense_model="fake/small", dense_dim=64)
    texts, summaries = [i.text for i in ITEMS], [kept[i][4] for i in ids]
    keywords, kw_vectors = ix.keywords_for(target, LISTS, texts, summaries)
    assert [h.keywords for h in hits] == keywords
    held = {p["item_id"]: v for batch in store.points("c") for _, v, p in batch}
    for row, item_id in enumerate(ids):
        assert held[item_id]["kw_all"] == pytest.approx(
            kw_vectors["kw_all"][row].tolist(), abs=1e-6
        )
    assert store.hashes("c", ids) == hashes
    for vector in ("text_dense", "kw_all", "kw_tags"):
        assert store.vectors("c", vector, 10).shape == (3, 64)
    assert store.vectors(chunks_of("c"), CHUNK, 100).shape == (chunks, 64)
    assert store.search_sparse("c", models.sparse_query(["enzyme"])[0], 1)[0].id == "1"
    assert store.search("c", "kw_tags", models.embed(["markets"], "fake/small")[0], 1)[0].id == "2"
    assert not store.exists(rebuild_of("c"))
    assert result == {"model": "fake/small", "dim": 64, "items": 3, "chunks": chunks}


def test_reembed_recomputes_hand_set_lists_whatever_the_vocabulary(tmp_path):
    # PUT keyword-lists rekeys any store, and search offers the kw_* methods whenever
    # lists are in force, so a rebuild must not drop them on an induced-vocabulary store.
    store, models, ix, _ = small_env(tmp_path)
    spec = ix.create("w", "science", False, recipe="worked_examples")
    assert spec.build is not None and spec.build.vocabulary != "lists"
    ix.add(spec, ITEMS, "x")
    ix.set_lists(spec, LISTS)
    assert store.vectors("w", "kw_all", 10).shape == (3, 256)
    ix.reembed("w", "fake/small")
    assert store.vectors("w", "kw_all", 10).shape == (3, 64)
    assert store.vectors("w", "kw_individual", 10).shape == (3, 64)
    kw = store.get("w", ["1"])[0].keywords
    own = keyword_string(kw["topic"] + kw["method"] + kw["common"], None)
    found = store.search("w", "kw_individual", models.embed([own], "fake/small")[0], 3)
    assert found and found[0].id == "1"


def test_a_resume_after_the_switch_drops_a_left_over_chunks_copy(tmp_path):
    store, _, ix, _ = small_env(tmp_path)
    ix.add(ix.create("c", "mcq", False), ITEMS, "x")
    real_drop = store.drop_rebuild

    def drop_half(name):  # dies after the copy is deleted, before its chunks are
        store.client.delete_collection(rebuild_of(name))
        raise RuntimeError("the store went away")

    def half_drop_from_now():  # phase 1's own drops run as they should
        store.drop_rebuild = drop_half  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        ix.reembed("c", "fake/small", on_switch=half_drop_from_now)
    spec = ix.spec("c")
    assert spec is not None and spec.dense_model == "fake/small"  # the switch itself finished
    assert not store.exists(rebuild_of("c")) and store.exists(chunks_of(rebuild_of("c")))
    store.drop_rebuild = real_drop  # type: ignore[method-assign]
    assert ix.reembed("c", "fake/small", switching=True)["items"] == 3
    assert not store.exists(chunks_of(rebuild_of("c")))


def test_reembed_reuses_an_items_vector_for_its_only_chunk(tmp_path):
    store, models, ix, _ = small_env(tmp_path)
    ix.add(ix.create("w", "science", False, recipe="worked_examples"), ITEMS, "x")
    embedded: list[str] = []
    real = models.embed
    models.embed = lambda texts, model=None: embedded.extend(texts) or real(texts, model)  # type: ignore[method-assign]
    ix.reembed("w", "fake/small")
    assert len(embedded) == len(ITEMS)
    assert store.search_chunks("w", real(["pea plants"], "fake/small")[0], 1)[0][0] == "3"


def test_a_cancelled_rebuild_leaves_the_store_as_it_was(tmp_path):
    store, _, ix, _ = small_env(tmp_path)
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    switched: list[int] = []

    def stop(p):
        raise Cancelled()

    with pytest.raises(Cancelled):
        ix.reembed("c", "fake/small", progress=stop, on_switch=lambda: switched.append(1))
    assert ix.spec("c") == spec and switched == []
    assert store.vectors("c", "text_dense", 10).shape == (3, 256)
    assert "1" in {h.id for h in retriever_for(ix).retrieve("c", "enzyme", "hybrid")}
    assert not store.exists(rebuild_of("c")) and not store.exists(chunks_of(rebuild_of("c")))


def test_a_switch_that_stops_resumes_from_the_rebuilt_copy(tmp_path):
    store, _, ix, _ = small_env(tmp_path)
    ix.add(ix.create("c", "mcq", False), ITEMS, "x")
    real_copy = store.copy
    failed: list[str] = []

    def copy_once(src, dst):
        if not failed:
            failed.append(src)
            raise RuntimeError("the store went away")
        return real_copy(src, dst)

    store.copy = copy_once  # type: ignore[method-assign]
    switched: list[int] = []
    with pytest.raises(RuntimeError):
        ix.reembed("c", "fake/small", on_switch=lambda: switched.append(1))
    spec = ix.spec("c")
    assert switched == [1] and spec is not None and spec.dense_model == "fake/dense"
    assert store.exists(rebuild_of("c"))
    job = db.create_job(ix.conn, "reembed", "c", {"model": "fake/small", "switching": True})
    db.update_job(ix.conn, job, status="failed")
    with pytest.raises(
        ModelMismatch,
        match="store 'c' is switching to fake/small; try again when its rebuild finishes,"
        " or resume it on the store page",
    ):
        ix.check_model(spec)
    del store.copy  # type: ignore[attr-defined]
    result = ix.reembed("c", "fake/small", switching=True)
    spec = ix.spec("c")
    assert store.count("c") == 3 and spec is not None and spec.dense_model == "fake/small"
    assert result["items"] == 3 and store.vectors("c", "text_dense", 10).shape == (3, 64)
    assert not store.exists(rebuild_of("c"))


def test_a_resumed_switch_after_the_copy_was_dropped_changes_nothing(tmp_path):
    store, _, ix, _ = small_env(tmp_path)
    ix.add(ix.create("c", "mcq", False), ITEMS, "x")
    done = ix.reembed("c", "fake/small")
    assert ix.reembed("c", "fake/small", switching=True) == done
    assert store.count("c") == 3
    old = ix.create("d", "mcq", False)
    ix.add(old, ITEMS, "x")
    gone = "the copy made for the switch to fake/small is gone; delete the store and load it again"
    with pytest.raises(ValueError, match=gone):
        ix.reembed("d", "fake/small", switching=True)
    assert ix.spec("d") == old and store.count("d") == 3


def test_reembed_of_an_empty_store_switches_its_model(tmp_path):
    store, _, ix, _ = small_env(tmp_path)
    ix.create("c", "mcq", False)
    assert ix.reembed("c", "fake/small")["items"] == 0
    spec = ix.spec("c")
    assert spec is not None and (spec.dense_model, spec.dense_dim) == ("fake/small", 64)
    assert dense_width(store, "c") == 64 and dense_width(store, chunks_of("c"), CHUNK) == 64
    ix.add(spec, ITEMS, "x")
    assert store.vectors("c", "text_dense", 10).shape == (3, 64)


def test_a_store_whose_model_cannot_load_can_still_be_rebuilt(tmp_path):
    store, models, ix, _ = small_env(tmp_path)
    ix.add(ix.create("c", "mcq", False), ITEMS, "x")
    db.set_dense_model(ix.conn, "c", "gone/model", 256)
    real = models.dense_dim

    def dense_dim(model=None):
        if model == "gone/model":
            raise ValueError("gone/model is not in fastembed's supported models")
        return real(model)

    models.dense_dim = dense_dim  # type: ignore[method-assign]
    assert ix.reembed("c", "fake/small")["items"] == 3
    spec = ix.spec("c")
    assert spec is not None and spec.dense_model == "fake/small"
    ix.check_model(spec)


def test_check_model_refuses_while_a_switch_is_unfinished(env):
    conn, _, _, ix, _ = env
    spec = ix.create("c", "mcq", False)
    job = db.create_job(conn, "reembed", "c", {"model": "fake/small", "switching": True})
    db.update_job(conn, job, status="interrupted")
    with pytest.raises(ModelMismatch, match="is switching to fake/small"):
        ix.check_model(spec)
    db.update_job(conn, job, status="done")
    ix.check_model(spec)
