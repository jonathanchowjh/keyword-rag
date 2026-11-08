"""The local models, through fastembed (ONNX Runtime): free, offline, no PyTorch.

Each model loads on first use and downloads into cache_dir, which in the
container is the ragqa volume, so a restart does not download again.
"""

import functools
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client.models import SparseVector

SPARSE_MODEL = "Qdrant/bm25"
# TEXTS PER ONNX RUN. fastembed's default is 256, so a caller's batch of 64 runs as one,
# padded to its longest text: memory grows with batch x length squared. 64 BRIGHT
# documents of ~3,000 characters took 6 GB and 54 min that way; in runs of 8, 64 typical
# ones take 18 s with Qwen3-Embedding-0.6B.
EMBED_BATCH = 8
# PAIRS PER RERANKER RUN, chosen by scripts/time_rerank.py on the service (experiments.md,
# "Reranker timing"): 16 beat 8 at every pool size; fastembed's default is 64.
RERANK_BATCH = 16
# A failed reranker warm-up (say, the Hub is down) is retried at most this often.
RERANK_RETRY_SECONDS = 300
# Qwen3-Embedding was trained with an instruction on queries and none on
# documents; fastembed's query_embed does not add it, so it is added here.
INSTRUCTED_PREFIX = "Qwen/Qwen3-Embedding"


def normalise(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms == 0, 1, norms)).astype(np.float32)


@functools.cache
def supported_dims() -> dict[str, int]:
    """Each model fastembed lists, with its dimension: static metadata, read once."""
    return {e["model"]: int(e["dim"]) for e in TextEmbedding.list_supported_models()}


class Models:
    def __init__(self, dense: str, judge: str, rerank: str, cache_dir: Path) -> None:
        self.dense_model = dense
        self.judge_model = judge
        self.rerank_model = rerank
        self.cache_dir = cache_dir
        self._loaded: dict[str, Any] = {}
        # One lock per model, so the reranker's download never holds up the dense model;
        # _loading only guards handing them out. The Lab and the worker share one Models.
        self._locks: dict[str, threading.Lock] = {}
        self._loading = threading.Lock()
        self._job = threading.local()  # this thread's job log
        # Reranker warm-ups, by model name: Settings can pick one other than rerank_model.
        self._warm_threads: dict[str, threading.Thread] = {}
        self.rerank_errors: dict[str, str] = {}
        self.rerank_failed_at: dict[str, float] = {}

    @contextmanager
    def for_job(self, log: Callable[[str], None]) -> Iterator[None]:
        """A model first loaded on this thread says so in the job's log."""
        self._job.log = log
        try:
            yield
        finally:
            self._job.log = None

    def dense_dim(self, model: str | None = None) -> int:
        """A dense model's dimension (None is dense_model), from fastembed's static list."""
        name = model or self.dense_model
        dim = supported_dims().get(name)
        if dim is None:
            raise ValueError(f"{name} is not in fastembed's supported models")
        return dim

    def _load(self, kind: str, name: str) -> Any:
        key = f"{kind}:{name}"
        if key in self._loaded:
            return self._loaded[key]
        with self._loading:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:  # checked again inside: another thread may have just built it
            if key not in self._loaded:
                log = getattr(self._job, "log", None) or (lambda message: None)
                log(f"loading model {name}")
                self._loaded[key] = self._build(kind, name)
                log(f"model {name} ready")
        return self._loaded[key]

    def _build(self, kind: str, name: str) -> Any:
        cls = {"text": TextEmbedding, "sparse": SparseTextEmbedding, "rerank": TextCrossEncoder}
        return cls[kind](model_name=name, cache_dir=str(self.cache_dir))

    def warm_rerank(self, model: str | None = None) -> None:
        """Loads a reranker (None is rerank_model); a failure is kept for rerank_status,
        not raised."""
        name = model or self.rerank_model
        try:
            self._load("rerank", name)
        except Exception as e:
            self.rerank_errors[name] = str(e) or type(e).__name__  # never an empty "failed"
            self.rerank_failed_at[name] = time.monotonic()
        else:
            self.rerank_errors.pop(name, None)
            self.rerank_failed_at.pop(name, None)

    def warm_rerank_in_background(self, model: str | None = None) -> None:
        """Starts warm_rerank on a thread, unless that model is loaded, is loading, or last
        failed under RERANK_RETRY_SECONDS ago: a Hub outage costs one retry per 5 minutes
        per model, not one per message."""
        name = model or self.rerank_model
        with self._loading:
            if f"rerank:{name}" in self._loaded:
                return
            thread = self._warm_threads.get(name)
            if thread is not None and thread.is_alive():
                return
            failed = self.rerank_failed_at.get(name)
            if failed is not None and time.monotonic() - failed < RERANK_RETRY_SECONDS:
                return
            thread = threading.Thread(
                target=self.warm_rerank, args=(name,), name=f"warm-rerank {name}", daemon=True
            )
            self._warm_threads[name] = thread
            thread.start()

    def retry_rerank(self, model: str | None = None) -> None:
        """warm_rerank_in_background without waiting out a recent failure: someone just
        picked this model, say after fixing the network."""
        self.rerank_failed_at.pop(model or self.rerank_model, None)
        self.warm_rerank_in_background(model)

    def rerank_status(
        self, model: str | None = None
    ) -> Literal["idle", "loading", "ready", "failed"]:
        """An idle reranker loads on first use."""
        name = model or self.rerank_model
        if f"rerank:{name}" in self._loaded:
            return "ready"
        thread = self._warm_threads.get(name)
        if thread is not None and thread.is_alive():
            return "loading"
        return "failed" if name in self.rerank_errors else "idle"

    # Each store is embedded with its own model; None is the service's, dense_model.
    def embed(self, texts: list[str], model: str | None = None) -> np.ndarray:
        encoder = self._load("text", model or self.dense_model)
        return normalise(np.array(list(encoder.embed(texts, batch_size=EMBED_BATCH))))

    def query_instruction(self, task: str, model: str | None = None) -> str:
        """The prefix a dense model expects on a query; "" for models trained without one."""
        if not task or not (model or self.dense_model).startswith(INSTRUCTED_PREFIX):
            return ""
        return f"Instruct: {task}\nQuery:"

    def embed_query(self, texts: list[str], task: str = "", model: str | None = None) -> np.ndarray:
        prefix = self.query_instruction(task, model)
        encoder = self._load("text", model or self.dense_model)
        queries = [prefix + t for t in texts]
        return normalise(np.array(list(encoder.query_embed(queries, batch_size=EMBED_BATCH))))

    def judge(self, texts: list[str]) -> np.ndarray:
        model = self._load("text", self.judge_model)
        return normalise(np.array(list(model.embed(texts, batch_size=EMBED_BATCH))))

    def sparse(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._load("sparse", SPARSE_MODEL).embed(texts)
        ]

    def sparse_query(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._load("sparse", SPARSE_MODEL).query_embed(texts)
        ]

    def rerank(self, query: str, texts: list[str], model: str | None = None) -> list[float]:
        """Scores each text against the query; model None is rerank_model."""
        encoder = self._load("rerank", model or self.rerank_model)
        return [float(s) for s in encoder.rerank(query, texts, batch_size=RERANK_BATCH)]
