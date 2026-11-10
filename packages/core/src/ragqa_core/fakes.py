"""Test doubles: real objects with their outward-facing methods overridden.

Only tests import this module (tests/test_boundaries.py enforces it). Nothing
here calls an AI provider or downloads a model.
"""

import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from qdrant_client.models import SparseVector

from aiconnect import fakes
from aiconnect.ai import AI
from ragqa_core.connection import Connection
from ragqa_core.hub import Hub
from ragqa_core.models import Models, normalise

WORD = re.compile(r"[a-z0-9]+")

ITEMS = [
    {
        "id": "enzyme-item",
        "text": "enzyme kinetics rate experiment",
        "answer": "A",
        "explanation": "Rates rise.",
        "meta": {"skill": "enzymes"},
    },
    {
        "id": "market-item",
        "text": "stock market price demand",
        "answer": "B",
        "explanation": "Prices fall.",
        "meta": {"skill": "markets"},
    },
    {
        "id": "volcano-item",
        "text": "volcano magma lava pressure",
        "answer": "C",
        "explanation": "Magma rises.",
        "meta": {"skill": "volcanoes"},
    },
]
LISTS = {
    "topic": ["enzyme", "market", "volcano", "optics"],
    "method": ["experiment", "survey", "sampling"],
    "common": ["rate", "price", "lava", "light"],
}


def until[T](
    check: Callable[[], T | None], seconds: float = 10, step: float = 0.02, what: str = "condition"
) -> T:
    """Calls `check` until it returns something truthy, and returns that. Waits with
    time.sleep, so `check` must touch no Playwright object: sync Playwright delivers route
    and event callbacks only during a Playwright call."""
    deadline = time.monotonic() + seconds
    while True:
        got = check()
        if got:
            return got
        if time.monotonic() >= deadline:
            raise AssertionError(f"{what} did not happen in {seconds}s")
        time.sleep(step)


def fake_models(dim: int = 256, dims: dict[str, int] | None = None) -> Models:
    return fake_embeddings(
        Models("fake/dense", "fake/judge", "fake/rerank", Path("/nonexistent")), dim, dims
    )


def fake_embeddings(models: Models, dim: int = 256, dims: dict[str, int] | None = None) -> Models:
    """Bag-of-words hashing: texts that share words get similar vectors, so
    retrieval behaves sensibly in tests without any model download. Patches
    `models` in place, so everything already holding it sees the fake.

    Every dense model has `dims.get(model, dim)` dimensions (None is models.dense_model,
    read at each call), so any name loads; the judge keeps `dim`."""
    widths = dims or {}

    def width(model: str | None = None) -> int:
        return widths.get(model or models.dense_model, dim)

    def bucket(token: str, size: int = dim) -> int:
        return int(hashlib.md5(token.encode()).hexdigest(), 16) % size

    def vectors(texts: list[str], size: int = dim) -> np.ndarray:
        rows = np.zeros((len(texts), size), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in WORD.findall(text.lower()):
                rows[row, bucket(token, size)] += 1
        return normalise(rows)

    def embed(texts: list[str], model: str | None = None) -> np.ndarray:
        return vectors(texts, width(model))

    def embed_query(texts: list[str], task: str = "", model: str | None = None) -> np.ndarray:
        return vectors(texts, width(model))

    def sparse(texts: list[str]) -> list[SparseVector]:
        out = []
        for text in texts:
            counts = Counter(bucket(t) for t in WORD.findall(text.lower()))
            indices = sorted(counts)
            out.append(SparseVector(indices=indices, values=[float(counts[i]) for i in indices]))
        return out

    def rerank(query: str, texts: list[str], model: str | None = None) -> list[float]:
        q = set(WORD.findall(query.lower()))
        return [float(len(q & set(WORD.findall(t.lower())))) for t in texts]

    models.embed = embed  # type: ignore[method-assign]
    models.embed_query = embed_query  # type: ignore[method-assign]
    models.judge = lambda texts: vectors(texts)  # type: ignore[method-assign]
    models.sparse = sparse  # type: ignore[method-assign]
    models.sparse_query = sparse  # type: ignore[method-assign]
    models.rerank = rerank  # type: ignore[method-assign]
    models.warm_rerank = lambda model=None: None  # type: ignore[method-assign]
    models.warm_rerank_in_background = lambda model=None: None  # type: ignore[method-assign]
    models.rerank_status = lambda model=None: "ready"  # type: ignore[method-assign]
    models.dense_dim = width  # type: ignore[method-assign]
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


def fake_hub(hub: Hub, root: Path, newest: str | None) -> list[tuple[str, str]]:
    """Serves `hub` from root/<commit>/<path>, with `newest` the commit on main; None makes
    the Hub unreachable. Returns the (path, revision) of every download asked for."""
    asked: list[tuple[str, str]] = []

    def commit() -> str:
        if newest is None:
            raise ConnectionError("no route to host")
        return newest

    def download(path: str, revision: str) -> Path:
        asked.append((path, revision))
        local = root / revision / path
        if not local.exists():
            raise FileNotFoundError(path)
        return local

    hub._commit = commit  # type: ignore[method-assign]
    hub._download = download  # type: ignore[method-assign]
    return asked
