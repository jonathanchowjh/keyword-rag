"""Test doubles: real objects with their outward-facing methods overridden.

Only tests import this module (tests/test_boundaries.py enforces it). Nothing
here calls Claude or downloads a model.
"""

import hashlib
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qdrant_client.models import SparseVector

from ragqa_core.llm import Claude
from ragqa_core.models import Models, normalise

WORD = re.compile(r"[a-z0-9]+")


def fake_models(dim: int = 256) -> Models:
    """Bag-of-words hashing: texts that share words get similar vectors, so
    retrieval behaves sensibly in tests without any model download."""
    models = Models("fake/dense", "fake/judge", "fake/rerank", Path("/nonexistent"))

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
    models.embed_query = vectors  # type: ignore[method-assign]
    models.judge = vectors  # type: ignore[method-assign]
    models.sparse = sparse  # type: ignore[method-assign]
    models.sparse_query = sparse  # type: ignore[method-assign]
    models.rerank = rerank  # type: ignore[method-assign]
    models.dense_dim = lambda: dim  # type: ignore[method-assign]
    return models


def fake_claude(
    conn: sqlite3.Connection,
    reply: Callable[[str], str],
    api_key: str | None = "test",
    cap_usd: float = 20.0,
) -> Claude:
    claude = Claude(conn, api_key, cap_usd, "claude-haiku-5-5")

    def send(model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        text = reply(prompt)
        return text, math.ceil(len(prompt) / 4), math.ceil(len(text) / 4)

    def send_batch(requests: list[dict]) -> dict[str, tuple[str, int, int]]:
        return {
            r["custom_id"]: send(r["model"], r["system"], r["prompt"], r["max_tokens"])
            for r in requests
        }

    claude._send = send  # type: ignore[method-assign]
    claude._send_batch = send_batch  # type: ignore[method-assign]
    return claude
