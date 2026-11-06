"""The chat gate: how well a message matches a store, and the bar a store sets."""

import json

import numpy as np

from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.datasets import dev_slice
from ragqa_core.evaluator import Evaluator
from ragqa_core.prompts import QUERY_TASK
from ragqa_core.retriever import Retriever
from ragqa_core.types import Bar, CollectionSpec, Skip

BAR_QUANTILE = 0.05
BAR_SAMPLE = 200
BAR_MIN = 20


class Router:
    def __init__(self, conn: Connection, retriever: Retriever, evaluator: Evaluator):
        self.conn = conn
        self.retriever = retriever
        self.evaluator = evaluator
        self._vectors: dict[tuple[str, str | None, str], tuple[list[str], np.ndarray]] = {}

    def best_match(self, spec: CollectionSpec, texts: list[str]) -> float:
        """The highest top-1 meaning cosine over the texts; 0.0 for an empty store."""
        self.retriever.indexer.check_model(spec)
        vectors = self.retriever.models.embed_query(texts, task=QUERY_TASK[spec.qa_type])
        skip = Skip(tuple(db.disabled_sources(self.conn, spec.name)))
        return max((self._top(spec.name, v, skip) for v in vectors), default=0.0)

    def _top(self, name: str, vector: np.ndarray, skip: Skip) -> float:
        found = self.retriever.store.search(name, "text_dense", vector, 1, skip.sources, skip.items)
        return found[0].score if found else 0.0

    def bar(self, spec: CollectionSpec) -> Bar:
        manual = db.get_chat_bar(self.conn, spec.name)
        if manual is not None:
            return Bar(manual, "manual", None, 0)
        set_name = db.default_eval_set(self.conn, spec.name)
        if set_name is None:
            return Bar(None, "none", None, 0)
        found = db.get_eval_set(self.conn, set_name)
        version = found["version"] if found else None
        disabled = db.disabled_sources(self.conn, spec.name)
        count = self.retriever.store.count(spec.name)  # before scoring, see the cache key
        key = json.dumps([set_name, version, count, sorted(disabled)])
        kept = db.get_store_bar(self.conn, spec.name, "gate")
        if kept and kept["key"] == key:
            return self._made(kept["value"], kept["n"], set_name)
        questions = db.get_questions(self.conn, set_name, version)
        leaked = self.evaluator.leaked_ids(spec, set_name)
        usable = [q for q in dev_slice(questions) if q.id not in leaked]
        usable = sorted(usable, key=lambda q: q.id)[:BAR_SAMPLE]
        value = None
        if len(usable) >= BAR_MIN:
            vectors = self._question_vectors(spec, set_name, version, usable)
            skip = tuple(disabled)
            scores = []
            for q, vector in zip(usable, vectors, strict=True):
                excluded = frozenset(q.meta.get("excluded_ids", []))
                scores.append(self._top(spec.name, vector, Skip(skip, excluded)))
            value = float(np.quantile(scores, BAR_QUANTILE))
        db.save_store_bar(
            self.conn, spec.name, "gate", {"key": key, "value": value, "n": len(usable)}
        )
        return self._made(value, len(usable), set_name)

    def _made(self, value: float | None, n: int, set_name: str) -> Bar:
        return Bar(value, "set" if value is not None else "few", set_name, n)

    def _question_vectors(self, spec, set_name, version, usable) -> np.ndarray:
        """Question vectors are embedded once per (set, version, qa type)."""
        key = (set_name, version, spec.qa_type)
        ids = [q.id for q in usable]
        have = self._vectors.get(key)
        if have is None or have[0] != ids:
            texts = [q.text for q in usable]
            vectors = self.retriever.models.embed_query(texts, task=QUERY_TASK[spec.qa_type])
            have = self._vectors[key] = (ids, vectors)
        return have[1]

    def kept(self, best: float, bar: Bar) -> bool:
        return bar.value is None or best >= bar.value
