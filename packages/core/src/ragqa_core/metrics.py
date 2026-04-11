"""Retrieval and answer metrics, and the intervals that say whether a gap is real.

At 200 questions a few points of difference is noise; the paired bootstrap
resamples questions, so two methods are compared on the same draws.
"""

import math
import re
import string
from collections import Counter

import numpy as np

ARTICLES = re.compile(r"\b(a|an|the)\b")
SCALE_WORDS = re.compile(r"\b(thousand|million|billion|percent)\b", re.I)
BOOTSTRAP_DRAWS = 1000
# The first number in a science answer: "3.2 J", "1.2e-3", "1.2 × 10^-3", "1.6 × 10⁻¹⁹".
# The lookbehind keeps "E_1" and "t_{1/2}" from being read as the answer.
WRAPPER = re.compile(r"\\(?:boxed|text|mathrm)\s*\{")
NUMBER = re.compile(
    r"(?<![\w.^{}/+-])"
    r"([-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+))"
    r"(?:[eE]([-+]?\d+))?"
    r"(?:\s*(?:×|\\times|\\cdot|·|\*|x|X)\s*10\s*"
    r"(?:\^\s*\{?\s*([-+]?\d+)\s*\}?|([⁺⁻]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)))?"
)
SUPERSCRIPT = str.maketrans("⁺⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "+-0123456789")


def recall_at(hit_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(hit_ids[:k]) & set(gold_ids)) / len(set(gold_ids))


def normalise_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _em_f1(pred: str, gold: str) -> tuple[float, float]:
    p, g = normalise_answer(pred).split(), normalise_answer(gold).split()
    em = float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or common == 0:
        return em, em
    precision, recall = common / len(p), common / len(g)
    return em, 2 * precision * recall / (precision + recall)


def squad_em_f1(pred: str, golds: list[str]) -> tuple[float, float]:
    scores = [_em_f1(pred, g) for g in golds] or [(0.0, 0.0)]
    return max(s[0] for s in scores), max(s[1] for s in scores)


def _number(text: str) -> float | None:
    cleaned = SCALE_WORDS.sub("", text).replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def science_number(text: str) -> float | None:
    match = NUMBER.search(WRAPPER.sub(" ", text.replace("−", "-")))
    if not match:
        return None
    mantissa, e_exp, ten_exp, sup_exp = match.groups()
    try:
        exponent = int(e_exp or 0) + int(ten_exp or (sup_exp or "0").translate(SUPERSCRIPT))
        value = float(mantissa.replace(",", "")) * 10.0**exponent
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def science_correct(pred: str, golds: list[str], has_choices: bool) -> bool:
    """With choices, the letter must match. Otherwise a numeric gold is matched
    within 5% (relative to the larger magnitude, like SciBench; absolute only when
    the gold is 0) and any other gold as SQuAD-normalised text."""
    if has_choices:
        return pred.strip().upper() in {g.strip().upper() for g in golds}
    p = science_number(pred)
    for gold in golds:
        gold = gold.strip().replace("−", "-")
        if NUMBER.fullmatch(gold):
            g = science_number(gold)
            if p is None or g is None:
                continue
            if g == 0:
                if abs(p) <= 1e-9:
                    return True
            elif math.isclose(p, g, rel_tol=0.05):
                return True
        elif normalise_answer(pred) == normalise_answer(gold) != "":
            return True
    return False


def tatqa_em_f1(pred: str, golds: list[str], scale: str) -> tuple[float, float]:
    """Simplified TAT-QA scoring: numbers compared at 2 dp after stripping $ , %
    and scale words; anything else is scored like SQuAD. `scale` is accepted so
    callers pass it, but is not applied — both sides are stated in the same scale."""
    p = _number(pred)
    numeric = [g for g in (_number(g) for g in golds) if g is not None]
    if p is not None and numeric:
        hit = any(round(p, 2) == round(g, 2) for g in numeric)
        return (1.0, 1.0) if hit else (0.0, 0.0)
    return squad_em_f1(pred, golds)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bootstrap_ci(
    values: list[float], n: int = BOOTSTRAP_DRAWS, seed: int = 0
) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    data = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).integers(0, len(data), size=(n, len(data)))
    means = data[draws].mean(axis=1)
    return float(data.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(
    a: list[float], b: list[float], n: int = BOOTSTRAP_DRAWS, seed: int = 0
) -> tuple[float, float, float]:
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} vs {len(b)}")
    return bootstrap_ci([x - y for x, y in zip(a, b, strict=True)], n, seed)


def separated(lo: float, hi: float) -> bool:
    return lo > 0 or hi < 0


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return math.nan
    rx, ry = np.asarray(_ranks(xs)), np.asarray(_ranks(ys))
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])
