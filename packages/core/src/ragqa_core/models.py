"""The local models, through fastembed (ONNX Runtime): free, offline, no PyTorch.

Each model loads on first use and downloads into cache_dir, which in the
container is the ragqa volume, so a restart does not download again.
"""

from pathlib import Path
from typing import Any

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client.models import SparseVector

SPARSE_MODEL = "Qdrant/bm25"
# Qwen3-Embedding was trained with an instruction on queries and none on
# documents; fastembed's query_embed does not add it, so it is added here.
INSTRUCTED_PREFIX = "Qwen/Qwen3-Embedding"


def normalise(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms == 0, 1, norms)).astype(np.float32)


class Models:
    def __init__(self, dense: str, judge: str, rerank: str, cache_dir: Path) -> None:
        self.dense_model = dense
        self.judge_model = judge
        self.rerank_model = rerank
        self.cache_dir = cache_dir
        self._loaded: dict[str, Any] = {}

    def dense_dim(self) -> int:
        for entry in TextEmbedding.list_supported_models():
            if entry["model"] == self.dense_model:
                return int(entry["dim"])
        raise ValueError(f"{self.dense_model} is not in fastembed's supported models")

    def _load(self, kind: str, name: str) -> Any:
        key = f"{kind}:{name}"
        if key not in self._loaded:
            cls = {"text": TextEmbedding, "sparse": SparseTextEmbedding, "rerank": TextCrossEncoder}
            self._loaded[key] = cls[kind](model_name=name, cache_dir=str(self.cache_dir))
        return self._loaded[key]

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalise(np.array(list(self._load("text", self.dense_model).embed(texts))))

    def query_instruction(self, task: str) -> str:
        """The prefix the dense model expects on a query; "" for models trained without one."""
        if not task or not self.dense_model.startswith(INSTRUCTED_PREFIX):
            return ""
        return f"Instruct: {task}\nQuery:"

    def embed_query(self, texts: list[str], task: str = "") -> np.ndarray:
        prefix = self.query_instruction(task)
        model = self._load("text", self.dense_model)
        return normalise(np.array(list(model.query_embed([prefix + t for t in texts]))))

    def judge(self, texts: list[str]) -> np.ndarray:
        return normalise(np.array(list(self._load("text", self.judge_model).embed(texts))))

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

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._load("rerank", self.rerank_model).rerank(query, texts)]
