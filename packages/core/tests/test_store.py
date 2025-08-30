import pytest

from ragqa_core.fakes import fake_models
from ragqa_core.store import Store
from ragqa_core.types import Item

ITEMS = [
    Item("1", "enzyme kinetics rate", answer="A", explanation="Because rate.", meta={"skill": "s"}),
    Item("2", "stock market prices"),
]


@pytest.fixture
def setup():
    s, m = Store(None, None), fake_models()
    s.create("c", m.dense_dim())
    texts = [i.text for i in ITEMS]
    s.upsert_items("c", ITEMS, m.embed(texts), m.sparse(texts), ["sum one", None])
    return s, m


def test_dense_and_sparse_search_find_the_matching_item(setup):
    s, m = setup
    assert s.search("c", "text_dense", m.embed_query(["enzyme rate"])[0], 1)[0].id == "1"
    assert s.search_sparse("c", m.sparse_query(["market"])[0], 1)[0].id == "2"
    assert s.count("c") == 2


def test_hit_carries_payload(setup):
    s, m = setup
    hit = s.get("c", ["1"])[0]
    assert (hit.text, hit.answer, hit.explanation, hit.meta, hit.summary) == (
        "enzyme kinetics rate",
        "A",
        "Because rate.",
        {"skill": "s"},
        "sum one",
    )
    assert hit.keywords == {} and hit.matched == []


def test_get_keeps_input_order(setup):
    s, _ = setup
    assert [h.id for h in s.get("c", ["2", "1"])] == ["2", "1"]


def test_upsert_is_idempotent_by_item_id(setup):
    s, m = setup
    new = Item("1", "enzyme kinetics rewritten")
    s.upsert_items("c", [new], m.embed([new.text]), m.sparse([new.text]), [None])
    assert s.count("c") == 2 and s.get("c", ["1"])[0].text == "enzyme kinetics rewritten"


def test_set_keywords_writes_vectors_and_payload(setup):
    s, m = setup
    kw = [{"topic": ["biology"], "all": ["biology"]}, {"topic": ["finance"], "all": ["finance"]}]
    s.set_keywords("c", ["1", "2"], kw, {"kw_individual": m.embed(["biology", "finance"])})
    hit = s.search("c", "kw_individual", m.embed(["finance"])[0], 1)[0]
    assert hit.id == "2" and hit.keywords == kw[1]
    assert s.get("c", ["1"])[0].text == "enzyme kinetics rate"


def test_set_keywords_drops_vectors_not_given(setup):
    s, m = setup
    vec = m.embed(["biology", "finance"])
    s.set_keywords("c", ["1", "2"], [{}, {}], {"kw_individual": vec, "kw_summary": vec})
    s.set_keywords("c", ["1", "2"], [{}, {}], {"kw_individual": vec})
    assert s.search("c", "kw_summary", vec[0], 5) == []


def test_chunks_map_to_parent(setup):
    s, m = setup
    s.upsert_chunks(
        "c", ["1", "1"], ["enzyme kinetics", "rate"], m.embed(["enzyme kinetics", "rate"])
    )
    parent, score = s.search_chunks("c", m.embed(["enzyme"])[0], 1)[0]
    assert parent == "1" and score > 0


def test_scroll_yields_every_item(setup):
    s, _ = setup
    assert sorted(h.id for batch in s.scroll("c", batch=1) for h in batch) == ["1", "2"]


def test_create_twice_is_a_no_op(setup):
    s, m = setup
    s.create("c", m.dense_dim())
    assert s.count("c") == 2


def test_drop(setup):
    s, m = setup
    s.drop("c")
    s.create("c", m.dense_dim())
    assert s.count("c") == 0


def test_ids_lists_every_item_sorted(setup):
    s, _ = setup
    assert s.ids("c") == ["1", "2"]
