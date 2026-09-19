import pytest

from ragqa_core.recipes import (
    RECIPES,
    build_for,
    corpus_type_for,
    default_search,
    search_settings,
)
from ragqa_core.types import SearchSettings


def test_recipes_match_spec_table():
    w, r, p = RECIPES["worked_examples"], RECIPES["reference"], RECIPES["paper_2024"]
    assert (w.units, w.added_text, w.vocabulary, w.describes, w.tagging) == (
        "whole",
        "none",
        "induced",
        "solution",
        "ai",
    )
    assert (w.abstracted, w.classifier, w.profile) == (True, True, True)
    assert (r.units, r.chunk_words, r.chunk_overlap, r.added_text, r.describes, r.tagging) == (
        "chunks",
        100,
        10,
        "title_prefix",
        "text",
        "similarity",
    )
    assert (r.abstracted, r.classifier, r.profile) == (False, True, True)
    assert (p.units, p.chunk_words, p.chunk_overlap, p.vocabulary, p.describes, p.tagging) == (
        "chunks",
        100,
        25,
        "lists",
        "text",
        "similarity",
    )
    assert (p.abstracted, p.classifier, p.profile) == (False, False, True)


def test_corpus_type_by_qa_type():
    types = ("science", "mcq", "multihop", "table")
    assert [corpus_type_for(t) for t in types] == ["worked", "worked", "reference", "reference"]


def test_default_search_follows_build():
    s = default_search("science", RECIPES["worked_examples"])
    assert (s.example, s.query_tags, s.program) == ("abstracted", "classifier", True)
    assert default_search("mcq", RECIPES["paper_2024"]).example == "full"
    assert default_search("table", RECIPES["reference"]).query_tags == "classifier"


def test_build_for_copies_and_sets_paper_summary():
    assert build_for("paper_2024", True).added_text == "summary"
    assert build_for("paper_2024").added_text == "none"
    build_for("paper_2024", True).chunk_words = 1
    assert RECIPES["paper_2024"].chunk_words == 100
    with pytest.raises(ValueError):
        build_for("nope")


def test_search_override_merges_weights_and_rejects_bad_input():
    s = search_settings(SearchSettings(), {"weights": {"keywords": 2.0}})
    assert s.weights == {"meaning": 1.0, "words": 1.0, "keywords": 2.0}
    bad_inputs = (
        {"nope": 1},
        {"weights": {"x": 1}},
        {"weights": {"meaning": -1}},
        {"weights": {"meaning": float("nan")}},
        {"weights": {"meaning": 0, "words": 0, "keywords": 0}},
        {"example": "half"},
        {"query_tags": "x"},
        {"program": "yes"},
    )
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            search_settings(SearchSettings(), bad)


def test_search_override_does_not_change_the_base():
    base = SearchSettings()
    search_settings(base, {"weights": {"words": 0}})
    assert base.weights["words"] == 1.0
