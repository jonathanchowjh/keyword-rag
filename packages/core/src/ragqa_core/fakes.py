"""Test doubles: real objects with their outward-facing methods overridden.

Only tests import this module (tests/test_boundaries.py enforces it). Nothing
here calls Claude or downloads a model.
"""

import hashlib
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qdrant_client.models import SparseVector

from ragqa_core.connection import Connection
from ragqa_core.llm import Claude
from ragqa_core.models import Models, normalise

WORD = re.compile(r"[a-z0-9]+")
# The fake Batch API's server side: batches outlive the fake that submitted
# them, as real ones outlive a restarted process.
BATCHES: dict[str, dict[str, tuple[str, int, int]]] = {}
CANCELLED: set[str] = set()


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


def fake_claude(
    conn: Connection,
    reply: Callable[[str], str],
    api_key: str | None = "test",
    cap_usd: float = 20.0,
) -> Claude:
    return fake_replies(Claude(conn, api_key, cap_usd, "claude-haiku-5-5"), reply)


def fake_replies(claude: Claude, reply: Callable[[str], str]) -> Claude:
    """Patches `claude` in place to answer every prompt with reply(prompt)."""

    def send(model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        text = reply(prompt)
        return text, math.ceil(len(prompt) / 4), math.ceil(len(text) / 4)

    def submit_batch(requests: list[dict]) -> str:
        batch_id = f"fake-batch-{len(BATCHES)}"
        BATCHES[batch_id] = {
            r["custom_id"]: send(r["model"], r["system"], r["prompt"], 0) for r in requests
        }
        return batch_id

    def batch_results(batch_id: str) -> dict[str, tuple[str, int, int] | None]:
        # A reply of "FAIL" stands for an errored or expired batch entry.
        return {k: None if v[0] == "FAIL" else v for k, v in BATCHES[batch_id].items()}

    claude._cancel_batch = CANCELLED.add  # type: ignore[method-assign]
    claude._send = send  # type: ignore[method-assign]
    claude._submit_batch = submit_batch  # type: ignore[method-assign]
    claude._batch_done = lambda batch_id: True  # type: ignore[method-assign]
    claude._batch_results = batch_results  # type: ignore[method-assign]
    return claude
