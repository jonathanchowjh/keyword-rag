"""Real downloads: checks the loaders against the published files and prints
the counts the README reports."""

import pytest
from huggingface_hub import get_token

from ragqa_core.datasets import load_gpqa, load_musique, load_scibench, load_scienceqa, load_tatqa
from ragqa_core.metrics import NUMBER

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


def test_scibench_gold_is_a_number(cache):
    items, questions = load_scibench(cache)
    solved = sum(i.explanation is not None for i in items)
    print(f"\nscibench: {len(items)} items ({solved} solved), {len(questions)} questions")
    # 493 rows; thermo_sol.json lists one problem twice, which is kept once.
    assert len(items) + len(questions) == 492
    assert solved == 76
    assert all(NUMBER.fullmatch(q.gold_answers[0]) for q in questions)


@pytest.mark.skipif(
    get_token() is None, reason="GPQA is gated: needs HF_TOKEN after accepting its terms"
)
def test_gpqa_diamond_is_held_out_of_the_store(cache):
    items, questions = load_gpqa(cache)
    print(f"\ngpqa: {len(items)} examples, {len(questions)} diamond questions")
    assert 150 <= len(questions) <= 198 and len(items) >= 300
    assert not {i.id for i in items} & {q.id for q in questions}
    assert all(q.meta["skill"] and q.meta["solution"] for q in questions)
