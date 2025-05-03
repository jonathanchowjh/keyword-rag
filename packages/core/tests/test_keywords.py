from ragqa_core.fakes import fake_models
from ragqa_core.keywords import (
    assign,
    keyword_string,
    list_version,
    matched,
    pick_all,
    tfidf_terms,
)
from ragqa_core.types import FACETS, KeywordLists

LISTS = KeywordLists(
    topic=["genetics", "kinetic energy"],
    method=["tabular scrutiny", "titration"],
    common=["mass", "temperature", "concentration"],
)


def test_assign_picks_closest_per_facet():
    m = fake_models()
    vecs = {f: m.embed(getattr(LISTS, f)) for f in FACETS}
    [kw] = assign(m.embed(["the titration changed the concentration and mass"]), vecs, LISTS, k=1)
    assert set(kw) == set(FACETS)
    assert kw["method"] == ["titration"]
    assert kw["common"][0] in {"concentration", "mass"}
    assert len(kw["topic"]) == 1


def test_assign_caps_k_at_list_length():
    m = fake_models()
    vecs = {f: m.embed(getattr(LISTS, f)) for f in FACETS}
    [kw] = assign(m.embed(["mass"]), vecs, LISTS, k=3)
    assert len(kw["topic"]) == 2 and len(kw["common"]) == 3 and kw["common"][0] == "mass"


def test_pick_all_returns_k_unique():
    m = fake_models()
    words = sorted(set(LISTS.topic + LISTS.method + LISTS.common))
    [picked] = pick_all(m.embed(["mass temperature"]), m.embed(words), words, k=3)
    assert len(picked) == 3 == len(set(picked)) and "mass" in picked


def test_keyword_string():
    assert keyword_string(["a", "b"], None) == "a; b"
    assert keyword_string(["a"], "Sum.") == "a. Sum."


def test_tfidf_requires_two_docs_and_drops_stopwords():
    terms = tfidf_terms(["the enzyme rate rose", "enzyme rate fell", "a solitary word"], n=10)
    assert "enzyme" in terms and "rate" in terms
    assert "the" not in terms and "solitary" not in terms


def test_tfidf_respects_n():
    texts = [f"alpha beta gamma delta {i}" for i in range(5)]
    assert len(tfidf_terms(texts, n=2)) == 2


def test_list_version_is_order_sensitive_and_stable():
    assert list_version(LISTS) == list_version(KeywordLists(**vars(LISTS)))
    assert len(list_version(LISTS)) == 12
    reordered = KeywordLists(LISTS.topic[::-1], LISTS.method, LISTS.common)
    assert list_version(LISTS) != list_version(reordered)


def test_matched():
    query = {"topic": ["a", "b"], "method": ["c"]}
    item = {"topic": ["b"], "common": ["c"]}
    assert matched(query, item) == ["b", "c"]
    assert matched({"topic": ["b"], "all": ["b"]}, {"topic": ["b"]}) == ["b"]
