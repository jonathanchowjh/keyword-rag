"""An induced vocabulary (spec §15, D2): clean the AI's terms, merge near-duplicates, version.

The terms are proposed batch by batch, so the same idea comes back under several
surface forms; merging keeps the most frequent form of each and drops one-offs.
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ragqa_core.types import Item

MAX_TERM_CHARS = 60
MAX_TERMS = 80  # a merged vocabulary keeps at most this many
SPACES = re.compile(r"\s+")


def read_vocabulary(path: Path) -> tuple[list[str], dict[str, str]]:
    """An authored {"terms": [{term, definition}]} file: (its terms as written, each
    definition by the term as a collection stores it, cleaned)."""
    listed = json.loads(path.read_text())["terms"]
    terms = [t["term"] for t in listed]
    definitions: dict[str, str] = {}
    for t in listed:
        cleaned = clean_terms([t.get("term")])
        if cleaned and t.get("definition"):
            definitions.setdefault(cleaned[0], t["definition"])
    return terms, definitions


def terms_version(terms: list[str]) -> str:
    canonical = json.dumps(sorted(terms), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def clean_terms(raw: list) -> list[str]:
    """Strings only, whitespace collapsed, lower case, 1-60 characters, first of each kept."""
    cleaned = (SPACES.sub(" ", t).strip().lower() for t in raw if isinstance(t, str))
    return list(dict.fromkeys(t for t in cleaned if 0 < len(t) <= MAX_TERM_CHARS))


def merge_terms(
    terms: list[str], vectors: np.ndarray, threshold: float = 0.88, cap: int = MAX_TERMS
) -> list[str]:
    """One representative per cluster of near-duplicate terms. `vectors[n]` embeds
    `terms[n]`; repeats count, and the most frequent surface form names its cluster."""
    if not terms:
        return []
    rows = np.asarray(vectors, dtype=np.float32)
    rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    counts = Counter(terms)
    first_seen: dict[str, int] = {}
    for n, term in enumerate(terms):
        first_seen.setdefault(term, n)
    # Most frequent first, so a cluster's first member is its most frequent form.
    forms = sorted(counts, key=lambda t: (-counts[t], first_seen[t]))
    reps: list[str] = []
    rep_vecs: list[np.ndarray] = []
    sizes: list[int] = []
    firsts: list[int] = []
    for form in forms:
        vec = rows[first_seen[form]]
        home = next((c for c, r in enumerate(rep_vecs) if float(r @ vec) >= threshold), None)
        if home is None:
            reps.append(form)
            rep_vecs.append(vec)
            sizes.append(counts[form])
            firsts.append(first_seen[form])
        else:
            sizes[home] += counts[form]
            firsts[home] = min(firsts[home], first_seen[form])
    # A small corpus has too few repeats for the one-off rule: keep every cluster.
    keep = [c for c in range(len(reps)) if sizes[c] >= 2 or len(terms) < 20]
    keep.sort(key=lambda c: (-sizes[c], firsts[c]))
    return [reps[c] for c in keep[:cap]]


def describe_text(item: Item, describes: str) -> str:
    """What the vocabulary describes for an item: its solution when the build says so
    and it has one, else its text. Never its meta (spec §12: no gold skill leaks in)."""
    if describes == "solution" and item.explanation and item.explanation.strip():
        return item.explanation
    return item.text
