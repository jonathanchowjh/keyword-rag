"""The paper's core idea: describe each item by keywords picked from curated lists.

Assignment is by embedding similarity between the item and every keyword, as
in the 2024 paper (§4.5): top-3 per list, or top-9 from the pooled lists.
"""

import hashlib
import json
from dataclasses import asdict

import numpy as np

from ragqa_core.types import FACETS, KeywordLists


def top_k(item_vecs: np.ndarray, word_vecs: np.ndarray, k: int) -> list[list[int]]:
    scores = item_vecs @ word_vecs.T
    k = min(k, word_vecs.shape[0])
    return [list(np.argsort(-row, kind="stable")[:k]) for row in scores]


def assign(
    item_vecs: np.ndarray,
    facet_vecs: dict[str, np.ndarray],
    lists: KeywordLists,
    k: int = 3,
) -> list[dict[str, list[str]]]:
    out: list[dict[str, list[str]]] = [{} for _ in range(item_vecs.shape[0])]
    for facet in FACETS:
        words = getattr(lists, facet)
        for row, picks in enumerate(top_k(item_vecs, facet_vecs[facet], k)):
            out[row][facet] = [words[i] for i in picks]
    return out


def pick_all(
    item_vecs: np.ndarray, pooled_vecs: np.ndarray, pooled_words: list[str], k: int = 9
) -> list[list[str]]:
    return [[pooled_words[i] for i in picks] for picks in top_k(item_vecs, pooled_vecs, k)]


def keyword_string(words: list[str], summary: str | None) -> str:
    text = "; ".join(words)
    return f"{text}. {summary}" if summary else text


def list_version(lists: KeywordLists) -> str:
    canonical = json.dumps(asdict(lists), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def matched(query_kw: dict[str, list[str]], item_kw: dict[str, list[str]]) -> list[str]:
    item_words = {w for words in item_kw.values() for w in words}
    seen: dict[str, None] = {}
    for words in query_kw.values():
        for w in words:
            if w in item_words:
                seen.setdefault(w)
    return list(seen)
