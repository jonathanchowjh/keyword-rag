import pytest

from ragqa_core.db import connect, get_keyword_lists
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.keywords import keyword_string, list_version
from ragqa_core.llm import Cancelled
from ragqa_core.store import Store
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
    claude = fake_claude(conn, lambda p: calls.append(p) or "A summary.")
    ix = Indexer(conn, store, models, claude)
    return conn, store, models, ix, calls


def test_create_saves_spec_with_model(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    assert spec == CollectionSpec("c", "mcq", False, "fake/dense", 256, None)
    assert store.count("c") == 0


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


def test_add_refuses_model_mismatch(env):
    *_, ix, _ = env
    spec = CollectionSpec("c", "mcq", False, "other/model", 64)
    with pytest.raises(
        ModelMismatch, match="built with other/model; service is running fake/dense"
    ):
        ix.add(spec, ITEMS, "x")


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


def test_propose_lists_uses_claude_json_and_tfidf(tmp_path):
    conn = connect(tmp_path / "r.db")
    claude = fake_claude(
        conn, lambda p: '{"topic": ["genetics", "genetics"], "method": ["titration"]}'
    )
    ix = Indexer(conn, Store(None, None), fake_models(), claude)
    spec = ix.create("c", "mcq", False)
    lists = ix.propose_lists(spec, ITEMS + [Item("4", "enzyme rate genetics")])
    assert lists.topic == ["genetics"] and lists.method == ["titration"]
    assert "rate" in lists.common


def test_keyword_list_embeddings_are_reused_across_calls(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    embedded: list[str] = []
    real = models.embed
    models.embed = lambda texts: embedded.extend(texts) or real(texts)
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
    models.embed = lambda texts: calls.append(len(texts)) or real(texts)
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

    def broken(texts):
        raise RuntimeError("onnx failed")

    models.embed = broken
    with pytest.raises(RuntimeError, match="onnx failed"):
        ix.set_lists(spec, LISTS)
    assert get_keyword_lists(conn, "c") is None and ix.spec("c").keyword_version is None


def test_propose_lists_rejects_a_reply_without_lists(tmp_path):
    conn = connect(tmp_path / "r.db")
    ix = Indexer(conn, Store(None, None), fake_models(), fake_claude(conn, lambda p: "Sorry,"))
    spec = ix.create("c", "mcq", False)
    with pytest.raises(ValueError, match="no usable topic or method"):
        ix.propose_lists(spec, ITEMS)


def counting_embed(models):
    n = [0]
    real = models.embed

    def embed(texts):
        n[0] += 1
        return real(texts)

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
