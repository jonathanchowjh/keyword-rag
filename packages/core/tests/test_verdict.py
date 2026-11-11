import pytest

from ragqa_core.verdict import (
    answer,
    best_of_each,
    gaps,
    headline_metric,
    proxy,
    scale,
    score,
)

SAME = {f"q{n}": 0.5 for n in range(40)}


def test_clear_gap_is_yes():
    ref = {f"q{n}": 0.0 if n < 30 else 1.0 for n in range(40)}
    cand = {f"q{n}": 1.0 for n in range(40)}
    a = answer({"kw_individual": cand, "hybrid": ref}, "accuracy", 100)
    assert a["verdict"] == "yes" and a["lo"] > 0
    assert a["candidate"] == "kw_individual" and a["reference"] == "hybrid"


def test_clear_loss_is_no():
    ref = {f"q{n}": 1.0 for n in range(40)}
    cand = {f"q{n}": 0.0 if n < 30 else 1.0 for n in range(40)}
    assert answer({"kw_individual": cand, "hybrid": ref}, "accuracy", 100)["verdict"] == "no"


def test_no_answer_without_the_pair():
    assert answer({"vector": SAME, "hybrid": SAME}, "accuracy", 100) is None


def test_tie_is_cant_tell():
    a = answer({"kw_individual": SAME, "hybrid": SAME}, "accuracy", 100)
    assert a["verdict"] == "cant_tell"


def test_reasoning_match_rescales_against_random():
    assert scale("solution_sim@1", 0.6) == {"unit": "match", "factor": pytest.approx(250.0)}
    assert scale("solution_sim@1", None) == {"unit": "raw", "factor": 100}
    assert scale("accuracy", 0.6) == {"unit": "pts", "factor": 100}


def test_gaps_are_paired_sorted_and_skip_the_reference():
    ref = {f"q{n}": 0.0 if n < 30 else 1.0 for n in range(40)}
    up = {f"q{n}": 1.0 for n in range(40)}
    down = {f"q{n}": 0.0 for n in range(40)}
    out = gaps({"hybrid": ref, "up": up, "down": down, "empty": {}}, "hybrid", 100)
    assert [g["method"] for g in out] == ["up", "down"]
    assert out[0]["state"] == "ahead" and out[0]["diff"] == pytest.approx(75.0)
    assert out[1]["state"] == "behind"
    assert gaps({"up": up}, "hybrid", 100) == []


def test_gaps_use_only_shared_questions():
    ref = {"q0": 0.0, "q1": 0.0}
    other = {"q1": 1.0, "q2": 0.0}
    (g,) = gaps({"hybrid": ref, "x": other}, "hybrid", 100)
    assert g["diff"] == pytest.approx(100.0)


def test_best_of_each_compares_keywords_with_plain_or_modern():
    hi = {f"q{n}": 1.0 for n in range(40)}
    lo = {f"q{n}": 0.0 for n in range(40)}
    mid = {f"q{n}": float(n % 2) for n in range(40)}
    per = {
        "question_only": hi,  # baseline, excluded
        "random_example": hi,  # control, excluded
        "kw_all": mid,
        "kw_rrf": hi,
        "vector": lo,
        "hybrid_rerank": mid,
    }
    b = best_of_each(per, "accuracy", 100)
    assert b["candidate"] == "kw_rrf" and b["reference"] == "hybrid_rerank"
    assert b["after_the_fact"] is True and b["verdict"] == "yes"
    assert best_of_each({"vector": lo, "hybrid": lo}, "accuracy", 100) is None


def test_proxy_excludes_baseline_and_controls_and_needs_four():
    means = {
        m: {"solution_sim@1": s, "accuracy": a}
        for m, s, a in [
            ("question_only", 0.0, 0.5),
            ("random_example", 0.4, 0.4),
            ("hybrid", 0.6, 0.7),
            ("vector", 0.55, 0.6),
            ("kw_individual", 0.7, 0.8),
        ]
    }
    assert proxy(means) is None  # only 3 eligible
    means["keyword"] = {"solution_sim@1": 0.5, "accuracy": 0.55}
    p = proxy(means)
    assert p["methods"] == 4 and p["pair_agrees"] is True and p["trusted"] is True


def test_proxy_is_untrusted_when_the_pair_points_the_other_way():
    means = {
        "hybrid": {"solution_sim@1": 0.6, "accuracy": 0.6},
        "kw_individual": {"solution_sim@1": 0.5, "accuracy": 0.7},
        "vector": {"solution_sim@1": 0.4, "accuracy": 0.5},
        "keyword": {"solution_sim@1": 0.7, "accuracy": 0.8},
    }
    p = proxy(means)
    assert p["pair_agrees"] is False and p["trusted"] is False


def test_proxy_pair_agrees_is_none_when_a_pair_method_is_missing():
    means = {
        m: {"solution_sim@1": s, "accuracy": s}
        for m, s in [("vector", 0.1), ("keyword", 0.2), ("hybrid", 0.3), ("kw_all", 0.4)]
    }
    p = proxy(means)
    assert p["pair_agrees"] is None and p["trusted"] is False


def test_headline_per_type():
    full = {"solution_sim@1", "same_skill@1"}
    assert headline_metric("mcq", True, None, full) == "accuracy"
    assert headline_metric("multihop", True, None, set()) == "f1"
    assert headline_metric("table", True, None, set()) == "f1"
    assert headline_metric("mcq", False, None, full) == "solution_sim@1"
    assert headline_metric("mcq", False, True, full) == "solution_sim@1"
    assert headline_metric("mcq", False, None, {"same_skill@1"}) == "same_skill@1"
    assert headline_metric("multihop", False, None, set()) == "recall@2"
    assert headline_metric("table", False, None, set()) == "recall@1"


def test_headline_for_documents_is_ndcg_at_10():
    measured = {"ndcg@10", "recall@10", "mrr@10"}
    assert headline_metric("documents", False, None, measured) == "ndcg@10"
    assert headline_metric("documents", False, None, set()) == "ndcg@10"
    assert headline_metric("documents", True, None, measured | {"accuracy"}) == "ndcg@10"


def test_headline_falls_back_to_same_skill_when_proxy_failed():
    assert headline_metric("mcq", False, False, {"solution_sim@1", "same_skill@1"}) == (
        "same_skill@1"
    )


def test_headline_is_ndcg_when_only_gold_ids_measure_the_example():
    assert headline_metric("science", False, None, {"ndcg@10", "recall@1"}) == "ndcg@10"
    assert headline_metric("science", False, None, {"same_skill@1", "ndcg@10"}) == "same_skill@1"
    both = {"solution_sim@1", "same_skill@1", "ndcg@10"}
    assert headline_metric("mcq", False, None, both) == "solution_sim@1"


def test_headline_for_science_follows_mcq():
    assert headline_metric("science", True, None, set()) == "accuracy"
    assert headline_metric("science", False, None, {"same_skill@1"}) == "same_skill@1"


def test_score_is_absolute_with_switch_only_when_ahead():
    # Hybrid and the control are further ahead and come first: only the filter picks vector.
    pm = {
        "hybrid_rerank": {"q1": 0.5, "q2": 0.5, "q3": 0.5},
        "hybrid": {"q1": 1.0, "q2": 1.0, "q3": 1.0},
        "random_example": {"q1": 1.0, "q2": 1.0, "q3": 1.0},
        "vector": {"q1": 0.9, "q2": 0.9, "q3": 0.9},
    }
    s = score(pm, "hybrid_rerank", 100, False, True)
    assert s is not None
    assert s["kind"] == "score" and s["mean"] == 50 and s["n"] == 3
    assert s["metric"] == "ndcg@10" and s["method"] == "hybrid_rerank"
    assert s["before_rerank"] is False
    assert s["switch"]["method"] == "vector"  # hybrid isn't offered with rerank on; controls never
    assert set(s["switch"]) == {"method", "diff", "lo", "hi"}
    assert score(pm, "keyword", 100, False, True) is None


def test_score_offers_no_switch_unless_one_is_ahead():
    pm = {"hybrid": {"q1": 1.0, "q2": 1.0}, "vector": {"q1": 0.5, "q2": 0.5}}
    s = score(pm, "hybrid", 100, False, False)
    assert s is not None and s["switch"] is None and s["lo"] <= s["mean"] <= s["hi"]
    assert score({"hybrid": {}}, "hybrid", 100, False, False) is None


def test_pair_answers_are_tagged_pair():
    ref = {f"q{n}": 0.0 for n in range(5)}
    a = answer({"kw_individual": SAME, "hybrid": ref}, "accuracy", 100)
    assert a is not None and a["kind"] == "pair"
