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
    ix.add(spec, ITEMS)
    assert store.count("c") == 3 and calls == []
    assert store.get("c", ["1"])[0].keywords == {}
    assert store.search_chunks("c", models.embed(["pea plants"])[0], 1)[0][0] == "3"


def test_add_reports_progress(env):
    *_, ix, _ = env
    spec = ix.create("c", "mcq", False)
    seen: list[float] = []
    ix.add(spec, ITEMS, progress=seen.append)
    assert seen and seen[-1] == 1.0


def test_set_lists_assigns_keywords_and_returns_version(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS)
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
    ix.add(spec, ITEMS)
    assert store.get("c", ["2"])[0].keywords["topic"][0] == "market economics"


def test_add_with_summarize_calls_claude_once_per_item(env):
    conn, store, models, ix, calls = env
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS)
    assert len(calls) == len(ITEMS)
    assert store.get("c", ["1"])[0].summary == "A summary."


def test_summary_vector_only_when_summarize(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", True)
    ix.add(spec, ITEMS)
    ix.set_lists(spec, LISTS)
    assert store.search("c", "kw_summary", models.embed(["A summary."])[0], 3)
    plain = ix.create("p", "mcq", False)
    ix.add(plain, ITEMS)
    ix.set_lists(plain, LISTS)
    assert store.search("p", "kw_summary", models.embed(["A summary."])[0], 3) == []


def test_estimate_is_zero_without_summarize(env):
    *_, ix, _ = env
    assert ix.estimate_add_usd(ix.create("c", "mcq", False), ITEMS) == 0.0
    assert ix.estimate_add_usd(ix.create("s", "mcq", True), ITEMS) > 0.0


def test_add_refuses_model_mismatch(env):
    *_, ix, _ = env
    spec = CollectionSpec("c", "mcq", False, "other/model", 64)
    with pytest.raises(
        ModelMismatch, match="built with other/model; service is running fake/dense"
    ):
        ix.add(spec, ITEMS)


def test_save_lists_does_not_switch_until_rekey_finishes(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS)
    v1 = ix.set_lists(spec, LISTS)
    v2 = ix.save_lists(spec, OTHER_LISTS)
    assert v2 != v1 and ix.spec("c").keyword_version == v1
    assert get_keyword_lists(conn, "c", v2) == OTHER_LISTS
    ix.rekey("c", v2)
    assert ix.spec("c").keyword_version == v2


def test_rekey_that_fails_leaves_the_old_version(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS)
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
    ix.add(spec, ITEMS)
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
    ix.add(spec, ITEMS)
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
