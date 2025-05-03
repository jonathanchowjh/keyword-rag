import pytest

from ragqa_core.db import connect, create_job, get_keyword_lists, update_job
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer, ListsBusy, ModelMismatch
from ragqa_core.keywords import keyword_string, list_version
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item, KeywordLists

LISTS = KeywordLists(
    topic=["enzyme kinetics", "genetics", "market economics", "plate tectonics", "optics"],
    method=["titration", "tabular scrutiny", "field survey", "spectroscopy"],
    common=["rate", "mass", "price", "temperature", "volume", "voltage"],
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


def test_set_lists_refused_while_eval_runs(env):
    conn, store, models, ix, _ = env
    spec = ix.create("c", "mcq", False)
    job = create_job(conn, "eval", "c", {})
    update_job(conn, job, status="running")
    with pytest.raises(ListsBusy, match="an eval is running on 'c'"):
        ix.set_lists(spec, LISTS)
    update_job(conn, job, status="done")
    assert ix.set_lists(spec, LISTS)


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
