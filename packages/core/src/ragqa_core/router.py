"""The chat gate: how well a message matches a store, and the bar a store sets. And the
router before it: how well a message matches a store's routing profile and average item."""

import hashlib
import json
from collections.abc import Callable

import numpy as np

from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.datasets import dev_slice
from ragqa_core.evaluator import Evaluator
from ragqa_core.prompts import QUERY_TASK
from ragqa_core.retriever import Retriever
from ragqa_core.types import Bar, CollectionSpec, Question, Skip

BAR_QUANTILE = 0.05
BAR_SAMPLE = 200
BAR_MIN = 20
PROFILE_TERMS = 20
PROFILE_SAMPLES = 5
PROFILE_SAMPLE_CHARS = 200
CENTROID_SAMPLE = 2000
ROUTE_SAMPLE = 200  # the most questions the routing eval takes from each eval set
SPLITS = ("held_out", "dev", "full")  # the routing eval's: held_out is all but the dev slice
MODE_SCORES = ("recall", "precision", "none_rate", "mean_searched")


class Router:
    def __init__(self, conn: Connection, retriever: Retriever, evaluator: Evaluator):
        self.conn = conn
        self.retriever = retriever
        self.evaluator = evaluator
        self._vectors: dict[tuple[str, str | None, str, str], tuple[list[str], np.ndarray]] = {}
        self._profiles: dict[str, tuple[str, np.ndarray]] = {}  # store: (text sha256, vector)
        self._centroids: dict[str, tuple[tuple, np.ndarray | None]] = {}  # store: (key, centroid)

    def embed(self, spec: CollectionSpec, texts: list[str]) -> np.ndarray:
        """The texts embedded as queries to a store of spec's type: what best_match and
        route_score take as `vectors`, so a chat embeds a message once per type."""
        return self.retriever.models.embed_query(texts, task=QUERY_TASK[spec.qa_type])

    def best_match(
        self, spec: CollectionSpec, texts: list[str], vectors: np.ndarray | None = None
    ) -> float:
        """The highest top-1 meaning cosine over the texts (or their `vectors`, from embed);
        0.0 for an empty store."""
        self.retriever.indexer.check_model(spec)
        vectors = self.embed(spec, texts) if vectors is None else vectors
        skip = Skip(tuple(db.disabled_sources(self.conn, spec.name)))
        return max((self._top(spec.name, v, skip) for v in vectors), default=0.0)

    def _top(self, name: str, vector: np.ndarray, skip: Skip) -> float:
        found = self.retriever.store.search(name, "text_dense", vector, 1, skip.sources, skip.items)
        return found[0].score if found else 0.0

    def bar(self, spec: CollectionSpec, saved: bool = True) -> Bar:
        """The store's bar: its saved chat_bar, else the one its eval set sets. With
        saved=False, the eval set's bar as if no chat_bar were saved."""
        self.retriever.indexer.check_model(spec)
        manual = db.get_chat_bar(self.conn, spec.name) if saved else None
        if manual is not None:
            return Bar(manual, "manual", None, 0)

        def score_one(q: Question, vector: np.ndarray, skip: tuple[str, ...]) -> float:
            excluded = frozenset(q.meta.get("excluded_ids", []))
            return self._top(spec.name, vector, Skip(skip, excluded))

        return self._calibrate(spec, "gate", [], score_one)

    def _set(self, spec: CollectionSpec) -> tuple[str, str | None] | None:
        """The store's eval set and its version in force; None when it has none."""
        set_name = db.default_eval_set(self.conn, spec.name)
        if set_name is None:
            return None
        found = db.get_eval_set(self.conn, set_name)
        return set_name, found["version"] if found else None

    def _questions(
        self,
        homes: list[CollectionSpec],
        set_name: str,
        version: str | None,
        split: str = "dev",
        limit: int = BAR_SAMPLE,
    ) -> list[Question]:
        """The set version's questions in `split`, less those leaked into any of the homes'
        stores, the first `limit` by id. With the defaults, the questions a bar is set from:
        the dev slice. "held_out" is the rest of the set and "full" all of it. Reads the
        whole version, and may scan the stores for leaks, so a kept bar never calls it."""
        questions = db.get_questions(self.conn, set_name, version)
        leaked = set().union(*(self.evaluator.leaked_ids(h, set_name) for h in homes))
        if split != "full":
            dev = dev_slice(questions)
            in_dev = {q.id for q in dev}
            questions = dev if split == "dev" else [q for q in questions if q.id not in in_dev]
        usable = [q for q in questions if q.id not in leaked]
        return sorted(usable, key=lambda q: q.id)[:limit]

    def _sample(self, spec: CollectionSpec) -> tuple[str, str | None, list[Question]] | None:
        """The store's eval set, its version and the questions a bar is set from."""
        found = self._set(spec)
        return None if found is None else (*found, self._questions([spec], *found))

    def _calibrate(
        self,
        spec: CollectionSpec,
        kind: str,
        key_extra: list,
        score_one: Callable[[Question, np.ndarray, tuple[str, ...]], float | None],
    ) -> Bar:
        """The BAR_QUANTILE of score_one over the sample's questions, kept in store_bars
        under `kind` until its key changes. The key is read before scoring, so a store that
        changes meanwhile is scored again next time, and is checked before the questions are
        read: the leaked ones depend only on the count and the version, both in the key.
        A None score is left out."""
        count = self.retriever.store.count(spec.name)
        disabled = tuple(db.disabled_sources(self.conn, spec.name))
        found = self._set(spec)
        if found is None:
            return Bar(None, "none", None, 0)
        set_name, version = found
        key = json.dumps([set_name, version, count, sorted(disabled), spec.dense_model, *key_extra])
        kept = db.get_store_bar(self.conn, spec.name, kind)
        if kept and kept["key"] == key:
            return self._made(kept["value"], kept["n"], set_name)
        usable = self._questions([spec], set_name, version)
        value = None
        if len(usable) >= BAR_MIN:
            vectors = self._question_vectors(spec, set_name, version, usable)
            scores = [
                s
                for q, vector in zip(usable, vectors, strict=True)
                if (s := score_one(q, vector, disabled)) is not None
            ]
            if len(scores) >= BAR_MIN:
                value = float(np.quantile(scores, BAR_QUANTILE))
        row = {"key": key, "value": value, "n": len(usable)}
        db.save_store_bar(self.conn, spec.name, kind, row)
        return self._made(value, len(usable), set_name)

    def _made(self, value: float | None, n: int, set_name: str) -> Bar:
        return Bar(value, "set" if value is not None else "few", set_name, n)

    def _question_vectors(self, spec, set_name, version, usable) -> np.ndarray:
        """Question vectors are embedded once per (set, version, qa type, model)."""
        key = (set_name, version, spec.qa_type, spec.dense_model)
        ids = [q.id for q in usable]
        have = self._vectors.get(key)
        if have is None or have[0] != ids:
            vectors = self.embed(spec, [q.text for q in usable])
            have = self._vectors[key] = (ids, vectors)
        return have[1]

    def forget(self, name: str) -> None:
        """Drops the store's kept bars, centroid and profile vector: what it holds changed in
        a way the keys may miss, as when a source is replaced by as many items. Question
        vectors are kept per eval set, not per store, so none are dropped."""
        db.delete_store_bars(self.conn, name)
        self._centroids.pop(name, None)
        self._profiles.pop(name, None)

    def kept(self, best: float, bar: Bar) -> bool:
        return bar.value is None or best >= bar.value

    # -- routing -----------------------------------------------------------------------

    def draft_route_profile(self, spec: CollectionSpec) -> str:
        """A routing profile made from what the store already knows of itself: its most
        carried tags (else its vocabulary), its sources and the opening of a few items."""
        return self._draft(spec)[0]

    def _draft(self, spec: CollectionSpec) -> tuple[str, bool]:
        """The draft, and whether it says what the store holds: a Topics line or a sample.
        The tag counts are read as saved, never counted again: that would scroll the store."""
        store = self.retriever.store
        disabled = tuple(db.disabled_sources(self.conn, spec.name))
        lines = [f"Store {spec.name} ({spec.qa_type}, {spec.corpus_type})."]
        vocabulary = db.get_vocabulary(self.conn, spec.name, spec.vocabulary_version) or []
        terms = list(spec.profile)[:PROFILE_TERMS] or vocabulary[:PROFILE_TERMS]
        if terms:
            lines.append("Topics: " + ", ".join(terms))
        sources = [s for s in sorted(store.source_counts(spec.name)) if s not in disabled]
        if sources:
            lines.append("Sources: " + ", ".join(sources))
        first = next(store.scroll(spec.name, PROFILE_SAMPLES, disabled), [])
        samples = [" ".join(h.text.split())[:PROFILE_SAMPLE_CHARS] for h in first]
        samples = [t for t in samples[:PROFILE_SAMPLES] if t]
        lines += [f"- {t}" for t in samples]
        return "\n".join(lines), bool(terms or samples)

    def route_profile(self, spec: CollectionSpec) -> tuple[str, bool]:
        """The routing profile and whether it is drafted (no text was saved by hand)."""
        stored = db.get_route_profile(self.conn, spec.name)
        if stored is not None:
            return stored, False
        return self.draft_route_profile(spec), True

    def route_score(
        self, spec: CollectionSpec, texts: list[str], vectors: np.ndarray | None = None
    ) -> float | None:
        """The highest cosine of the texts (or their `vectors`, from embed) with the store's
        routing profile or its centroid; None when it has neither, so a store that says
        nothing of itself is never left out."""
        self.retriever.indexer.check_model(spec)
        _, targets = self._targets(spec)
        if not targets:
            return None
        vectors = self.embed(spec, texts) if vectors is None else vectors
        return max((self._closest(targets, v) for v in vectors), default=None)

    def route_bar(self, spec: CollectionSpec) -> Bar:
        """The bar the store's eval set sets for its route score, the way `bar` sets the
        gate's: its key also holds the profile's sha256, so an edit sets it again."""
        self.retriever.indexer.check_model(spec)
        sha, targets = self._targets(spec)

        def score_one(q: Question, vector: np.ndarray, skip: tuple[str, ...]) -> float | None:
            return self._closest(targets, vector) if targets else None

        return self._calibrate(spec, "route", [sha], score_one)

    def _closest(self, targets: list[np.ndarray], vector: np.ndarray) -> float:
        return max(float(t @ vector) for t in targets)

    def _targets(self, spec: CollectionSpec) -> tuple[str | None, list[np.ndarray]]:
        """The profile text's sha256 (None when the profile is missing) and the vectors a
        message is compared with: the profile's and the centroid, those the store has."""
        sha, profile = self._profile(spec)
        centroid = self._centroid(spec)
        return sha, [v for v in (profile, centroid) if v is not None]

    def _profile(self, spec: CollectionSpec) -> tuple[str | None, np.ndarray | None]:
        """The profile's sha256 and embedding, kept until its text changes. Missing when no
        text was saved and the draft names no topic and holds no sample."""
        stored = db.get_route_profile(self.conn, spec.name)
        text, says = (stored, True) if stored is not None else self._draft(spec)
        if not says:
            return None, None
        sha = hashlib.sha256(text.encode()).hexdigest()
        have = self._profiles.get(spec.name)
        if have is None or have[0] != sha:
            have = self._profiles[spec.name] = (sha, self.retriever.models.embed([text])[0])
        return sha, have[1]

    def _centroid(self, spec: CollectionSpec) -> np.ndarray | None:
        """The mean text_dense vector of up to CENTROID_SAMPLE items, at unit length: the
        mean of unit vectors is shorter, and would lose to the profile. Kept until the count,
        the switched-off sources or the last finished load, import or copy changes, which
        a job run by the worker in another process also moves."""
        store = self.retriever.store
        disabled = tuple(db.disabled_sources(self.conn, spec.name))
        key = (store.count(spec.name), disabled, db.last_item_job(self.conn, spec.name))
        have = self._centroids.get(spec.name)
        if have is not None and have[0] == key:
            return have[1]
        vectors = store.vectors(spec.name, "text_dense", CENTROID_SAMPLE, disabled)
        centroid = None
        if len(vectors):
            mean = vectors.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            centroid = mean / norm if norm > 0 else None
        self._centroids[spec.name] = (key, centroid)
        return centroid

    # -- the routing eval ----------------------------------------------------------------

    def evaluate(self, stores: list[str], split: str = "held_out") -> dict:
        """How well the gate, and the router before it, send the listed stores' own eval set
        questions to their homes: the listed stores whose default set the question is from.
        Each store's set is read once at its version in force, in `split`, leaked questions
        dropped and at most ROUTE_SAMPLE taken by id. A store with no set adds no questions,
        but may still be kept for others'. Mode "gate" is a chat routing "all", "profiles" one
        routing by profile, neither with a sticky store or one searched anyway. Embeddings
        only: it spends nothing."""
        if split not in SPLITS:
            raise ValueError(f"unknown split '{split}'; known: {', '.join(SPLITS)}")
        specs = [self.retriever.spec(name) for name in dict.fromkeys(stores)]
        sets: dict[str, tuple[str | None, list[CollectionSpec]]] = {}
        for spec in specs:
            if (found := self._set(spec)) is not None:
                sets.setdefault(found[0], (found[1], []))[1].append(spec)
        if not sets:
            raise ValueError("none of these stores has an eval set to check routing on")
        asked = [
            (q, {"eval_set": name, "id": q.id, "homes": [h.name for h in homes]})
            for name, (version, homes) in sets.items()
            for q in self._questions(homes, name, version, split, ROUTE_SAMPLE)
        ]
        rows = [row | {"gate_kept": [], "routed": [], "profiles_kept": []} for _, row in asked]
        questions = [q for q, _ in asked]
        embedded: dict[str, np.ndarray] = {}  # the questions as queries, once per qa type
        for spec in specs:
            if not questions:
                break
            if spec.qa_type not in embedded:
                embedded[spec.qa_type] = self.embed(spec, [q.text for q in questions])
            self._judge(spec, questions, embedded[spec.qa_type], rows)
        per_store = {}
        for spec in specs:
            mine = [r for r in rows if spec.name in r["homes"]]
            per_store[spec.name] = {
                "questions": len(mine),
                "recall_gate": share([spec.name in r["gate_kept"] for r in mine]),
                "recall_profiles": share([spec.name in r["profiles_kept"] for r in mine]),
            }
        modes = {
            "gate": mode_scores(rows, "gate_kept", [len(specs)] * len(rows)),
            "profiles": mode_scores(rows, "profiles_kept", [len(r["routed"]) for r in rows]),
        }
        return {
            "split": split,
            "questions": len(rows),
            "per_store": per_store,
            "modes": modes,
            "rows": rows,
        }

    def _judge(
        self,
        spec: CollectionSpec,
        questions: list[Question],
        vectors: np.ndarray,
        rows: list[dict],
    ) -> None:
        """Adds the store to each question's row where the gate keeps it, the router routes
        to it, and both: the rules a chat follows. Its best match leaves out the question's
        excluded_ids, as its bar does; a store with no profile or no route bar is routed."""
        bar = self.bar(spec)
        _, targets = self._targets(spec)
        route_bar = self.route_bar(spec).value if targets else None
        disabled = tuple(db.disabled_sources(self.conn, spec.name))
        for q, vector, row in zip(questions, vectors, rows, strict=True):
            excluded = frozenset(q.meta.get("excluded_ids", []))
            kept = self.kept(self._top(spec.name, vector, Skip(disabled, excluded)), bar)
            routed = route_bar is None or self._closest(targets, vector) >= route_bar
            if kept:
                row["gate_kept"].append(spec.name)
            if routed:
                row["routed"].append(spec.name)
            if kept and routed:
                row["profiles_kept"].append(spec.name)


def share(flags: list[bool]) -> float | None:
    """The share of the flags that are true; None for none."""
    return sum(flags) / len(flags) if flags else None


def mode_scores(rows: list[dict], kept: str, searched: list[int]) -> dict:
    """A routing mode's scores over the rows, where row[kept] lists the stores it kept:
    recall, the share of questions with a home kept; precision, the mean share of homes
    among the stores kept, over questions with any kept; none_rate, the share with none
    kept; mean_searched, the mean of `searched`. All None for no rows."""
    if not rows:
        return dict.fromkeys(MODE_SCORES)
    some = [r for r in rows if r[kept]]
    homes = lambda r: len(set(r["homes"]) & set(r[kept]))  # noqa: E731
    return {
        "recall": share([homes(r) > 0 for r in rows]),
        "precision": sum(homes(r) / len(r[kept]) for r in some) / len(some) if some else None,
        "none_rate": share([not r[kept] for r in rows]),
        "mean_searched": sum(searched) / len(rows),
    }
