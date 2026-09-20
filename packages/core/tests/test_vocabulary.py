import numpy as np

from ragqa_core.types import Item
from ragqa_core.vocabulary import clean_terms, describe_text, merge_terms, terms_version


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_merge_terms_clusters_near_duplicates():
    # 25 inputs, so the count >= 2 rule applies
    terms = (
        ["conservation of energy"] * 10
        + ["energy conservation"] * 4
        + ["ideal gas law"] * 10
        + ["one-off"]
    )
    by_term = {
        "conservation of energy": unit(1, 0, 0, 0),
        "energy conservation": unit(0.95, np.sqrt(1 - 0.95**2), 0, 0),  # cos 0.95
        "ideal gas law": unit(0, 0, 1, 0),
        "one-off": unit(0, 0, 0, 1),
    }
    vecs = np.stack([by_term[t] for t in terms])
    # 14 vs 10; "one-off" (count 1) dropped
    assert merge_terms(terms, vecs) == ["conservation of energy", "ideal gas law"]
    # Under 20 inputs every cluster is kept.
    few = merge_terms(terms[:5] + ["one-off"], vecs[:5].tolist() + [vecs[-1]])
    assert few == ["conservation of energy", "one-off"]


def test_merge_terms_keeps_the_most_frequent_surface_form_and_caps():
    terms = ["energy conservation"] + ["conservation of energy"] * 3
    vecs = np.stack([unit(0.95, np.sqrt(1 - 0.95**2)), *[unit(1, 0)] * 3])
    assert merge_terms(terms, vecs) == ["conservation of energy"]
    many = [f"t{n}" for n in range(30) for _ in range(2)]
    eye = np.eye(30, dtype=np.float32)
    vecs = np.stack([eye[n] for n in range(30) for _ in range(2)])
    assert merge_terms(many, vecs, cap=5) == ["t0", "t1", "t2", "t3", "t4"]
    assert merge_terms([], np.zeros((0, 4))) == []


def test_clean_terms():
    assert clean_terms(["  Ohm's  Law ", "", 3, "x" * 61, "ohm's law"]) == ["ohm's law"]


def test_terms_version_ignores_order():
    a = terms_version(["ohm's law", "ideal gas law"])
    assert a == terms_version(["ideal gas law", "ohm's law"]) and len(a) == 12
    assert a != terms_version(["ohm's law"])


def test_describe_text_falls_back():
    worked = Item("1", "A ball falls 5 m.", "10 m/s", "Use conservation of energy.")
    bare = Item("2", "A ball falls 5 m.", "10 m/s", "  ")
    assert describe_text(worked, "solution") == "Use conservation of energy."
    assert describe_text(bare, "solution") == "A ball falls 5 m."
    assert describe_text(Item("3", "text only"), "solution") == "text only"
    assert describe_text(worked, "text") == "A ball falls 5 m."
