import math

import numpy as np
import pytest

from ragqa_core.metrics import (
    bootstrap_ci,
    cosine,
    paired_diff_ci,
    recall_at,
    science_correct,
    science_number,
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


def test_science_number_reads_the_first_number_in_common_forms():
    assert science_number("3.2 J") == 3.2
    assert science_number("-4.1 kJ mol^-1") == -4.1
    assert science_number("1.2e-3") == 0.0012
    assert science_number("1.2 × 10^-3 J") == pytest.approx(0.0012)
    assert science_number("1.2 x 10^{-3}") == pytest.approx(0.0012)
    assert science_number("≈ 0.25") == 0.25
    assert science_number("2,500 m") == 2500.0
    assert science_number("$1.6 \\times 10^{-19}$ C") == pytest.approx(1.6e-19)
    assert science_number("1.6 × 10⁻¹⁹ C") == pytest.approx(1.6e-19)
    assert science_number("9.13 \\cdot 10^{-35} J s") == pytest.approx(9.13e-35)
    assert science_number("−0.5") == -0.5  # Unicode minus
    assert science_number(".5 mol") == 0.5
    assert science_number("E_1 = -13.6 eV") == -13.6
    assert science_number("t_{1/2} = 5.2 s") == 5.2
    assert science_number("about two") is None
    assert science_number("CO2") is None
    assert science_number("1e400") is None  # too large to represent
    assert science_number("1 × 10^500") is None


def test_science_correct_by_letter_number_or_text():
    assert science_correct("B", ["B"], has_choices=True)
    assert not science_correct("C", ["B"], has_choices=True)
    assert science_correct("3.1", ["3.0"], has_choices=False)  # within 5%
    assert not science_correct("3.2", ["3.0"], has_choices=False)  # 6.7% off
    assert science_correct("9.0e-35 J s", ["9.13e-35"], has_choices=False)
    assert not science_correct("5e-35", ["9.13e-35"], has_choices=False)  # tiny but 45% off
    assert not science_correct("0", ["9.13e-35"], has_choices=False)
    assert science_correct("0", ["0"], has_choices=False)
    assert not science_correct("0.001", ["0"], has_choices=False)  # absolute only when gold is 0
    assert science_correct("The Mitochondria.", ["mitochondria"], has_choices=False)
    assert not science_correct("H2O", ["CO2"], has_choices=False)  # text gold, not the number 2
    assert not science_correct("", ["4.2"], has_choices=False)
    assert not science_correct("1e400", ["3.0"], has_choices=False)
