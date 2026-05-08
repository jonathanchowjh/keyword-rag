import pytest

from ragqa_core.fakes import fake_models
from ragqa_core.store import Store, chunks_of, item_hash, point_id, skip_filter
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
        "c",
        ["1", "1"],
        ["enzyme kinetics", "rate"],
        m.embed(["enzyme kinetics", "rate"]),
        [None, None],
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


def sourced():
    """Items a1, a2 (source a) and b1 (source b), each with one chunk, in collection c."""
    s, m = Store(None, None), fake_models()
    s.create("c", m.dense_dim())
    items = [
        Item(
            "a1", "enzyme kinetics rate", explanation="Because.", meta={"source": "a", "skill": "k"}
        ),
        Item("a2", "photosynthesis light", meta={"source": "a"}),
        Item("b1", "stock market prices", meta={"source": "b"}),
    ]
    texts = [i.text for i in items]
    s.upsert_items("c", items, m.embed(texts), m.sparse(texts), [None] * 3)
    s.upsert_chunks("c", [i.id for i in items], texts, m.embed(texts), ["a", "a", "b"])
    return s, m, items


def test_skip_filter_is_none_when_nothing_is_skipped():
    assert skip_filter(()) is None


def test_skip_filters_every_search():
    s, m, _ = sourced()
    q = m.embed_query(["enzyme market light"])[0]
    sq = m.sparse_query(["enzyme market light"])[0]
    everywhere = {"a1", "a2", "b1"}
    assert {h.id for h in s.search("c", "text_dense", q, 10)} == everywhere
    assert {h.id for h in s.search_sparse("c", sq, 10)} <= everywhere
    assert {p for p, _ in s.search_chunks("c", q, 10)} == everywhere
    assert s.ids("c") == ["a1", "a2", "b1"]
    assert {h.id for b in s.scroll("c") for h in b} == everywhere
    skip = ("b",)
    assert "b1" not in {h.id for h in s.search("c", "text_dense", q, 10, skip=skip)}
    assert "b1" not in {h.id for h in s.search_sparse("c", sq, 10, skip=skip)}
    assert {p for p, _ in s.search_chunks("c", q, 10, skip=skip)} == {"a1", "a2"}
    assert s.ids("c", skip=skip) == ["a1", "a2"]
    assert {h.id for b in s.scroll("c", skip=skip) for h in b} == {"a1", "a2"}


def test_points_without_a_source_are_never_skipped(setup):
    s, m = setup
    q = m.embed_query(["enzyme"])[0]
    assert {h.id for h in s.search("c", "text_dense", q, 10, skip=("b",))} == {"1", "2"}


def test_item_hash_covers_content_and_source():
    base = Item("x", "t", answer="A", explanation="e", meta={"skill": "s", "source": "a"})
    same = Item("x", "t", answer="A", explanation="e", meta={"source": "a", "skill": "s"})
    assert item_hash(base) == item_hash(same) and len(item_hash(base)) == 16
    other_explanation = Item("x", "t", answer="A", explanation="f", meta=base.meta)
    other_skill = Item("x", "t", answer="A", explanation="e", meta={"skill": "z", "source": "a"})
    other_source = Item("x", "t", answer="A", explanation="e", meta={"skill": "s", "source": "b"})
    hashes = {item_hash(i) for i in (base, other_explanation, other_skill, other_source)}
    assert len(hashes) == 4


def test_hashes_reads_back_content_hash():
    s, _, items = sourced()
    assert s.hashes("c", ["a1"]) == {"a1": ""}  # upsert_items alone writes no hash
    s.set_payloads("c", {"a1": {"content_hash": item_hash(items[0])}})
    assert s.hashes("c", ["a1", "zz"]) == {"a1": item_hash(items[0])}


def test_hashes_maps_a_point_without_one_to_empty():
    s, _, _ = sourced()
    s.client.delete_payload("c", keys=["content_hash"], points=[point_id("c", "a2")])
    assert s.hashes("c", ["a2"]) == {"a2": ""}


def test_delete_chunks_removes_only_named_parents():
    s, _, _ = sourced()
    s.delete_chunks("c", ["a1", "b1"])
    left, _ = s.client.scroll(chunks_of("c"), limit=10)
    assert [r.payload["parent_id"] for r in left if r.payload] == ["a2"]


def test_chunks_carry_source():
    s, _, _ = sourced()
    left, _ = s.client.scroll(chunks_of("c"), limit=10)
    got = {r.payload["parent_id"]: r.payload["meta"]["source"] for r in left if r.payload}
    assert got == {"a1": "a", "a2": "a", "b1": "b"}


def test_source_counts_unsourced_field_counts_delete_source():
    s, m, items = sourced()
    assert s.source_counts("c") == {"a": 2, "b": 1}
    assert s.unsourced("c") == 0
    assert s.field_counts("c") == {"explanation": 1, "skill": 1}
    loose = Item("n1", "no source here")
    s.upsert_items("c", [loose], m.embed([loose.text]), m.sparse([loose.text]), [None])
    assert s.unsourced("c") == 1
    s.delete_source("c", "a")
    s.delete_chunks("c", ["n1"])
    assert s.source_counts("c") == {"b": 1}
    left, _ = s.client.scroll(chunks_of("c"), limit=10)
    assert {r.payload["parent_id"] for r in left if r.payload} == {"b1"}
    assert s.ids("c") == ["b1", "n1"]


def test_set_payloads_and_set_chunk_sources():
    s, _, _ = sourced()
    s.set_payloads("c", {"a2": {"meta": {"source": "z"}}, "b1": {"explanation": "new"}})
    assert s.get("c", ["a2"])[0].meta == {"source": "z"}
    assert s.get("c", ["b1"])[0].explanation == "new"
    s.set_chunk_sources("c", {"a1": "q", "b1": "r"})
    left, _ = s.client.scroll(chunks_of("c"), limit=10)
    got = {r.payload["parent_id"]: r.payload["meta"]["source"] for r in left if r.payload}
    assert got == {"a1": "q", "a2": "a", "b1": "r"}


def test_unsourced_counts_chunks_too():
    s, m, items = sourced()
    loose = Item("n1", "no source here", meta={"source": "a"})
    s.upsert_items("c", [loose], m.embed([loose.text]), m.sparse([loose.text]), [None])
    s.upsert_chunks("c", ["n1"], [loose.text], m.embed([loose.text]), [None])
    assert s.unsourced("c") == 1
    s.set_chunk_sources("c", {"n1": "a"})
    assert s.unsourced("c") == 0
