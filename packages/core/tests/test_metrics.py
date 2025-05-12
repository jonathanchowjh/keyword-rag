import math

import numpy as np
import pytest

from ragqa_core.metrics import (
    bootstrap_ci,
    cosine,
    paired_diff_ci,
    recall_at,
    separated,
    spearman,
    squad_em_f1,
    tatqa_em_f1,
)


def test_recall():
    assert recall_at(["a", "x", "b"], ["a", "b"], 2) == 0.5
    assert recall_at(["a", "x", "b"], ["a", "b"], 3) == 1.0
    assert recall_at(["a"], [], 5) == 0.0


def test_squad_exact():
    assert squad_em_f1("The June 1982", ["june 1982"]) == (1.0, 1.0)


def test_squad_partial():
    em, f1 = squad_em_f1("June", ["June 1982"])
    assert em == 0.0 and f1 == pytest.approx(2 / 3)


def test_squad_takes_best_gold():
    assert squad_em_f1("Paris", ["London", "paris."]) == (1.0, 1.0)


def test_squad_empty_prediction():
    assert squad_em_f1("", ["x"]) == (0.0, 0.0)


def test_tatqa_numbers():
    assert tatqa_em_f1("$1,452.40", ["1452.4"], "") == (1.0, 1.0)


def test_tatqa_percent():
    assert tatqa_em_f1("12.5%", ["12.5"], "percent") == (1.0, 1.0)


def test_tatqa_scale_word_in_prediction():
    assert tatqa_em_f1("1,496.5 million", ["$1,496.5"], "million") == (1.0, 1.0)


def test_tatqa_wrong_number():
    assert tatqa_em_f1("1452.5", ["1452.4"], "") == (0.0, 0.0)


def test_tatqa_text_falls_back_to_squad():
    assert tatqa_em_f1("fixed price", ["Fixed Price"], "") == (1.0, 1.0)


def test_cosine():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 2.0])) == pytest.approx(0.0)


def test_bootstrap_constant():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0, 1.0)


def test_bootstrap_brackets_the_mean():
    mean, lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < mean == 0.5 < hi


def test_bootstrap_empty():
    mean, lo, hi = bootstrap_ci([])
    assert math.isnan(mean) and math.isnan(lo) and math.isnan(hi)


def test_paired_identical_not_separated():
    m, lo, hi = paired_diff_ci([1, 0, 1, 0] * 25, [1, 0, 1, 0] * 25)
    assert m == 0 and not separated(lo, hi)


def test_paired_clear_gap_separated():
    m, lo, hi = paired_diff_ci([1.0] * 100, [0.0] * 100)
    assert m == 1.0 and separated(lo, hi)


def test_paired_rejects_unequal_lengths():
    with pytest.raises(ValueError):
        paired_diff_ci([1.0], [1.0, 0.0])


def test_spearman():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 1, 2, 3]) == pytest.approx(1.0)
    assert math.isnan(spearman([1, 2], [1, 2]))
