"""Test doubles: real objects with their outward-facing methods overridden.

Only tests import this module (tests/test_boundaries.py enforces it). Nothing
here calls an AI provider or downloads a model.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qdrant_client.models import SparseVector

from aiconnect import fakes
from aiconnect.ai import AI
from ragqa_core.connection import Connection
from ragqa_core.models import Models, normalise

WORD = re.compile(r"[a-z0-9]+")


def fake_models(dim: int = 256) -> Models:
    return fake_embeddings(
        Models("fake/dense", "fake/judge", "fake/rerank", Path("/nonexistent")), dim
    )


def fake_embeddings(models: Models, dim: int = 256) -> Models:
    """Bag-of-words hashing: texts that share words get similar vectors, so
    retrieval behaves sensibly in tests without any model download. Patches
    `models` in place, so everything already holding it sees the fake."""

    def bucket(token: str) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim

    def vectors(texts: list[str]) -> np.ndarray:
        rows = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in WORD.findall(text.lower()):
                rows[row, bucket(token)] += 1
        return normalise(rows)

    def sparse(texts: list[str]) -> list[SparseVector]:
        out = []
        for text in texts:
            counts = Counter(bucket(t) for t in WORD.findall(text.lower()))
            indices = sorted(counts)
            out.append(SparseVector(indices=indices, values=[float(counts[i]) for i in indices]))
        return out

    def rerank(query: str, texts: list[str]) -> list[float]:
        q = set(WORD.findall(query.lower()))
        return [float(len(q & set(WORD.findall(t.lower())))) for t in texts]

    models.embed = vectors  # type: ignore[method-assign]
    models.embed_query = lambda texts, task="": vectors(texts)  # type: ignore[method-assign]
    models.judge = vectors  # type: ignore[method-assign]
    models.sparse = sparse  # type: ignore[method-assign]
    models.sparse_query = sparse  # type: ignore[method-assign]
    models.rerank = rerank  # type: ignore[method-assign]
    models.dense_dim = lambda: dim  # type: ignore[method-assign]
    return models


def fake_ai(
    conn: Connection,
    reply: Callable[[str], str],
    api_key: str | None = "test",
    cap_usd: float = 20.0,
) -> AI:
    """An AI on conn's file, with a priced connection and a default model, that answers
    every prompt with reply(prompt). A second one on the same file shares its tables."""
    return fakes.fake_ai(conn.path, reply, api_key=api_key, cap_usd=cap_usd)


def fake_replies(ai: AI, reply: Callable[[str], str]) -> AI:
    """Patches `ai` in place to answer every prompt with reply(prompt)."""
    return fakes.fake_replies(ai, reply)
