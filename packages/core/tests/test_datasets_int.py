"""Real downloads: checks the loaders against the published files and prints
the counts the README reports."""

import pytest

from ragqa_core.datasets import load_musique, load_scienceqa, load_tatqa

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


def test_musique_matches_hipporag2_protocol(cache):
    items, questions = load_musique(cache)
    ids = {i.id for i in items}
    print(f"\nmusique: {len(items)} passages, {len(questions)} questions")
    assert len(questions) == 1000
    assert 11_000 <= len(items) <= 11_656
    assert all(len(q.gold_ids) >= 2 and set(q.gold_ids) <= ids for q in questions)


def test_tatqa_dev_questions_link_to_items(cache):
    items, questions = load_tatqa(cache)
    ids = {i.id for i in items}
    print(f"\ntatqa: {len(items)} contexts, {len(questions)} dev questions")
    assert len(questions) == 1668 and len(items) >= 2700
    assert all(set(q.gold_ids) <= ids for q in questions)


def test_scienceqa_filter_leaves_a_dev_slice(cache):
    items, questions = load_scienceqa(cache)
    print(f"\nscienceqa: {len(items)} train examples, {len(questions)} test questions")
    assert len(questions) >= 200 and len(items) >= 200
