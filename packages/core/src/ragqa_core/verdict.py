"""What a run says: the headline metric, paired gaps, the pre-declared answer and
the proxy check. Pure functions over per-question values."""

import math

from ragqa_core.catalog import ANSWER_PAIR, CONTROLS, METHODS
from ragqa_core.metrics import paired_diff_ci, spearman
from ragqa_core.types import QaType

PROXY_TRUST = 0.5
PROXY_MIN_METHODS = 4

PerMethod = dict[str, dict[str, float]]


def headline_metric(
    qa_type: QaType, has_answers: bool, proxy_trusted: bool | None, measurable: set[str]
) -> str:
    if has_answers:
        return "accuracy" if qa_type == "mcq" else "f1"
    if qa_type == "mcq":
        if proxy_trusted is False or "solution_sim@1" not in measurable:
            return "same_skill@1"
        return "solution_sim@1"
    return "recall@2" if qa_type == "multihop" else "recall@1"


def scale(metric: str, random_mean: float | None) -> dict:
    if metric == "solution_sim@1":
        if random_mean is not None and random_mean < 1:
            return {"unit": "match", "factor": 100 / (1 - random_mean)}
        return {"unit": "raw", "factor": 100}
    return {"unit": "pts", "factor": 100}


def _state(lo: float, hi: float) -> str:
    return "ahead" if lo > 0 else "behind" if hi < 0 else "cant_tell"


def _paired(
    candidate: dict[str, float], reference: dict[str, float], factor: float
) -> tuple[float, float, float] | None:
    shared = sorted(set(candidate) & set(reference))
    if not shared:
        return None
    diff, lo, hi = paired_diff_ci([candidate[q] for q in shared], [reference[q] for q in shared])
    return diff * factor, lo * factor, hi * factor


def gaps(per_method: PerMethod, reference: str, metric: str, factor: float) -> list[dict]:
    """Each method against the reference, over the questions both have."""
    ref = per_method.get(reference)
    if not ref:
        return []
    out = []
    for method, values in per_method.items():
        paired = None if method == reference or not values else _paired(values, ref, factor)
        if paired is not None:
            diff, lo, hi = paired
            out.append(
                {"method": method, "diff": diff, "lo": lo, "hi": hi, "state": _state(lo, hi)}
            )
    return sorted(out, key=lambda g: g["diff"], reverse=True)


def _compare(
    per_method: PerMethod, candidate: str, reference: str, metric: str, factor: float
) -> dict | None:
    cand, ref = per_method.get(candidate), per_method.get(reference)
    paired = _paired(cand, ref, factor) if cand and ref else None
    if paired is None:
        return None
    diff, lo, hi = paired
    verdict = {"ahead": "yes", "behind": "no", "cant_tell": "cant_tell"}[_state(lo, hi)]
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "verdict": verdict,
    }


def answer(
    per_method: PerMethod, metric: str, factor: float, pair: tuple[str, str] = ANSWER_PAIR
) -> dict | None:
    """The one pair declared in advance, never the winners of a run."""
    return _compare(per_method, pair[0], pair[1], metric, factor)


def best_of_each(per_method: PerMethod, metric: str, factor: float) -> dict | None:
    """Picked after the fact, so the gap is likely overstated."""

    def best(groups: tuple[str, ...]) -> str | None:
        names = [
            m
            for m, values in per_method.items()
            if values and m in METHODS and METHODS[m]["group"] in groups
        ]
        return max(
            names, key=lambda m: sum(per_method[m].values()) / len(per_method[m]), default=None
        )

    keywords, other = best(("keywords",)), best(("plain", "modern"))
    result = _compare(per_method, keywords, other, metric, factor) if keywords and other else None
    return {**result, "after_the_fact": True} if result else None


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def proxy(
    methods_mean: dict[str, dict[str, float]], pair: tuple[str, str] = ANSWER_PAIR
) -> dict | None:
    """Can the free score be trusted? Only retrieval methods, no baseline, no controls."""
    eligible = {
        m: v
        for m, v in methods_mean.items()
        if m in METHODS
        and METHODS[m]["retrieves"]
        and m not in CONTROLS
        and "solution_sim@1" in v
        and "accuracy" in v
    }
    if len(eligible) < PROXY_MIN_METHODS:
        return None
    rho = spearman(
        [v["solution_sim@1"] for v in eligible.values()], [v["accuracy"] for v in eligible.values()]
    )
    agrees = None
    if pair[0] in eligible and pair[1] in eligible:
        c, r = eligible[pair[0]], eligible[pair[1]]
        agrees = _sign(c["solution_sim@1"] - r["solution_sim@1"]) == _sign(
            c["accuracy"] - r["accuracy"]
        )
    return {
        "spearman": None if math.isnan(rho) else rho,
        "methods": len(eligible),
        "pair_agrees": agrees,
        "trusted": bool(not math.isnan(rho) and rho >= PROXY_TRUST and agrees is True),
    }
