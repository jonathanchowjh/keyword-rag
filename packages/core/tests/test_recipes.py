import pytest

from ragqa_core.recipes import (
    RECIPES,
    build_for,
    build_settings,
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


def test_build_override_merges_and_rejects_bad_input():
    base = RECIPES["reference"]
    b = build_settings(base, {"chunk_words": 50, "chunk_overlap": 5, "tagging": "ai"})
    assert (b.chunk_words, b.chunk_overlap, b.tagging, b.units) == (50, 5, "ai", "chunks")
    assert base.chunk_words == 100 and build_settings(base, None) == base
    bad_inputs = (
        {"tagging": "magic"},
        {"nope": 1},
        {"units": "pages"},
        {"describes": "title"},
        {"abstracted": "yes"},
        {"profile": 1},
        {"chunk_words": 0},
        {"chunk_words": -5},
        {"chunk_words": True},
        {"chunk_words": 2.5},
        {"chunk_overlap": -1},
        {"chunk_overlap": 100},  # a stride of 0: the windows would never move
        {"chunk_words": 10},  # under the recipe's overlap of 10
        {"tag_k": 0},
        {"tag_cutoff": float("nan")},
        {"tag_cutoff": 2},
    )
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            build_settings(base, bad)
    # Whole units ignore the overlap; the window size must still be a size.
    assert build_settings(base, {"units": "whole", "chunk_overlap": 100}).units == "whole"


def test_build_override_keeps_one_vocabulary_path():
    """A collection makes the three lists or runs the build, never both: the recipe says which."""
    worked, paper = RECIPES["worked_examples"], build_for("paper_2024")
    with pytest.raises(ValueError, match="recipe"):
        build_settings(worked, {"vocabulary": "lists"})
    with pytest.raises(ValueError, match="recipe"):
        build_settings(paper, {"vocabulary": "induced"})
    assert build_settings(worked, {"vocabulary": "induced"}).vocabulary == "induced"
    # Summaries feed the lists' keyword vectors only.
    with pytest.raises(ValueError, match="summary"):
        build_settings(worked, {"added_text": "summary"})
    assert build_settings(paper, {"added_text": "summary"}).added_text == "summary"
