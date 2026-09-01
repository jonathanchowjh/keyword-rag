"""Builds and refreshes every representation of a collection's items."""

import re
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from aiconnect.ai import AI
from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.datasets import source_for
from ragqa_core.keywords import assign, keyword_string, list_version, pick_all, tfidf_terms
from ragqa_core.models import Models
from ragqa_core.prompts import SUMMARY_TOKENS, parse_json, propose_prompt, summary_prompt
from ragqa_core.store import Store, item_hash
from ragqa_core.types import FACETS, CollectionSpec, Item, KeywordLists, QaType

NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
BATCH = 64
CHUNK_WORDS, CHUNK_STRIDE = 100, 75


class ModelMismatch(Exception):
    pass


class ListsBusy(Exception):
    pass


def chunk(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    starts = range(0, len(words) - CHUNK_WORDS + CHUNK_STRIDE, CHUNK_STRIDE)
    return [" ".join(words[s : s + CHUNK_WORDS]) for s in starts]


def dedupe(words: list[str]) -> list[str]:
    return list(dict.fromkeys(w.strip() for w in words if isinstance(w, str) and w.strip()))


class Indexer:
    def __init__(self, conn: Connection, store: Store, models: Models, ai: AI):
        self.conn = conn
        self.store = store
        self.models = models
        self.ai = ai
        # Keyword-list embeddings by list version: they change only when the
        # lists do, and re-embedding ~565 keywords per query dominated eval time.
        self._list_vectors: dict[str, tuple[dict[str, np.ndarray], list[str], np.ndarray]] = {}

    def create(self, name: str, qa_type: QaType, summarize: bool) -> CollectionSpec:
        if not NAME.match(name):
            raise ValueError(f"collection name {name!r} must match {NAME.pattern}")
        spec = CollectionSpec(
            name, qa_type, summarize, self.models.dense_model, self.models.dense_dim()
        )
        db.save_collection(self.conn, spec)
        self.store.create(name, spec.dense_dim)
        return spec

    def spec(self, name: str) -> CollectionSpec | None:
        return db.get_collection(self.conn, name)

    def sourced(self, items: list[Item], source: str) -> list[Item]:
        return [replace(i, meta={**i.meta, "source": source}) for i in items]

    def changed(
        self, spec: CollectionSpec, items: list[Item], source: str
    ) -> tuple[list[Item], dict[str, int]]:
        """The sourced items whose stored hash differs, with new/updated/unchanged counts."""
        items = self.sourced(items, source)
        stored = self.store.hashes(spec.name, [i.id for i in items])
        todo = [i for i in items if stored.get(i.id) != item_hash(i)]
        new = sum(1 for i in todo if i.id not in stored)
        counts = {"new": new, "updated": len(todo) - new, "unchanged": len(items) - len(todo)}
        return todo, counts

    def estimate_add_usd(self, spec: CollectionSpec, items: list[Item], source: str) -> float:
        if not spec.summarize:
            return 0.0
        todo, _ = self.changed(spec, items, source)
        batch = len(todo) > 50
        return sum(
            self.ai.estimate_usd(
                len(summary_prompt(i.text)), SUMMARY_TOKENS, role="summary", batch=batch
            )
            for i in todo
        )

    def add(
        self,
        spec: CollectionSpec,
        items: list[Item],
        source: str,
        progress: Callable[[float], None] = lambda p: None,
    ) -> dict[str, int]:
        self.check_model(spec)
        items, counts = self.changed(spec, items, source)
        if not items:
            progress(1.0)
            return counts
        summaries: list[str | None] = [None] * len(items)
        if spec.summarize:
            done = self.ai.complete_many(
                [summary_prompt(i.text) for i in items],
                role="summary",
                max_tokens=SUMMARY_TOKENS,
                purpose="summary",
            )
            summaries = [c.text.strip() for c in done]
        lists = db.get_keyword_lists(self.conn, spec.name)
        for start in range(0, len(items), BATCH):
            part = items[start : start + BATCH]
            sums = summaries[start : start + BATCH]
            texts = [i.text for i in part]
            self.store.upsert_items(
                spec.name, part, self.models.embed(texts), self.models.sparse(texts), sums
            )
            # An updated item may now have fewer chunks: drop the old ones first.
            self.store.delete_chunks(spec.name, [i.id for i in part])
            parents, pieces, sources = [], [], []
            for item in part:
                for piece in chunk(item.text):
                    parents.append(item.id)
                    pieces.append(piece)
                    sources.append(item.meta["source"])
            self.store.upsert_chunks(spec.name, parents, pieces, self.models.embed(pieces), sources)
            if lists is not None:
                keywords, vectors = self.keywords_for(spec, lists, texts, sums)
                self.store.set_keywords(spec.name, [i.id for i in part], keywords, vectors)
            # Last, so a batch that died part-way reads as changed and is redone.
            self.store.set_payloads(spec.name, {i.id: {"content_hash": item_hash(i)} for i in part})
            progress(min(start + BATCH, len(items)) / len(items))
        return counts

    def backfill(self, name: str, progress: Callable[[float], None] = lambda p: None) -> dict:
        """Label points that predate sources, in place: no embedding, no Claude."""
        self.store.ensure_indexes(name)
        datasets = db.loaded_datasets(self.conn, name)
        total, done = self.store.count(name), 0
        source_of: dict[str, str] = {}
        labelled_now = 0
        for hits in self.store.scroll(name):
            payloads = {}
            for h in hits:
                if h.meta.get("source"):
                    source_of[h.id] = h.meta["source"]
                    continue
                source = source_of[h.id] = source_for(h.id, datasets)
                meta = {**h.meta, "source": source}
                labelled = Item(h.id, h.text, h.answer, h.explanation, meta)
                labelled_now += 1
                payloads[h.id] = {"meta": meta, "content_hash": item_hash(labelled)}
            self.store.set_payloads(name, payloads)
            done += len(hits)
            progress(done / total)
        # Every item, not just this run's: an interrupted run may have labelled items only.
        self.store.set_chunk_sources(name, source_of)
        return {"labelled": labelled_now}

    def estimate_propose_usd(self, spec: CollectionSpec, sample: list[Item]) -> float:
        prompt = propose_prompt([i.text for i in sample], spec.qa_type)
        return self.ai.estimate_usd(len(prompt), 2000, role="keywords")

    def propose_lists(self, spec: CollectionSpec, sample: list[Item]) -> KeywordLists:
        reply = self.ai.complete(
            propose_prompt([i.text for i in sample], spec.qa_type),
            role="keywords",
            max_tokens=2000,
            purpose="propose",
        )
        proposed = parse_json(reply.text)
        topic = dedupe(proposed.get("topic") or [])
        method = dedupe(proposed.get("method") or [])
        if not topic or not method:
            raise ValueError(
                f"Claude's reply had no usable topic or method list: {reply.text[:200]!r}"
            )
        # A small sample can have no term in two texts; reuse the proposed phrases.
        common = tfidf_terms([i.text for i in sample]) or topic + method
        return KeywordLists(topic=topic, method=method, common=common)

    def save_lists(self, spec: CollectionSpec, lists: KeywordLists) -> str:
        """Validate, embed and store a version of the lists, without switching to it."""
        self.check_model(spec)  # the re-key would key items with another model's vectors
        for facet in FACETS:
            if not getattr(lists, facet):
                raise ValueError(f"keyword list '{facet}' is empty")
        # Embedded before saving: a collection must never point at lists it cannot use.
        self._lists_embedded(lists)
        version = list_version(lists)
        db.save_keyword_lists(self.conn, spec.name, lists, version)
        return version

    def rekey(
        self, name: str, version: str, progress: Callable[[float], None] = lambda p: None
    ) -> None:
        """Re-key every item with a saved version's lists, then switch to it.
        If anything raises, the old version stays in force."""
        spec = self.spec(name)
        lists = db.get_keyword_lists(self.conn, name, version)
        if spec is None or lists is None:
            raise ValueError(f"no keyword lists {version!r} for collection '{name}'")
        total, done = self.store.count(name), 0
        for hits in self.store.scroll(name):
            keywords, vectors = self.keywords_for(
                spec, lists, [h.text for h in hits], [h.summary for h in hits]
            )
            self.store.set_keywords(name, [h.id for h in hits], keywords, vectors)
            done += len(hits)
            progress(done / total)
        if total == 0:
            progress(1.0)
        db.set_keyword_version(self.conn, name, version)

    def set_lists(
        self,
        spec: CollectionSpec,
        lists: KeywordLists,
        progress: Callable[[float], None] = lambda p: None,
    ) -> str:
        version = self.save_lists(spec, lists)
        self.rekey(spec.name, version, progress)
        return version

    def keywords_for(
        self,
        spec: CollectionSpec,
        lists: KeywordLists,
        texts: list[str],
        summaries: list[str | None],
    ) -> tuple[list[dict], dict[str, np.ndarray]]:
        """Keywords and keyword vectors for items or queries — both sides alike."""
        item_vecs = self.models.embed(texts)
        facet_vecs, pooled, pooled_vecs = self._lists_embedded(lists)
        per_facet = assign(item_vecs, facet_vecs, lists)
        pooled_picks = pick_all(item_vecs, pooled_vecs, pooled)
        keywords = [{**kw, "all": picks} for kw, picks in zip(per_facet, pooled_picks, strict=True)]
        use_summary = [s if spec.summarize else None for s in summaries]
        # Every keyword string for every text, embedded in one pass: on a CPU the
        # per-call overhead of six separate passes dominated query time.
        groups: dict[str, list[str]] = {
            "kw_all": [
                keyword_string(kw["all"], s) for kw, s in zip(keywords, use_summary, strict=True)
            ],
            "kw_individual": [
                keyword_string(kw["topic"] + kw["method"] + kw["common"], s)
                for kw, s in zip(keywords, use_summary, strict=True)
            ],
            **{f"kw_{f}": [keyword_string(kw[f], None) for kw in keywords] for f in FACETS},
        }
        if spec.summarize:
            groups["kw_summary"] = [s or "" for s in summaries]
        flat = self.models.embed([text for strings in groups.values() for text in strings])
        vectors: dict[str, np.ndarray] = {}
        offset = 0
        for name, strings in groups.items():
            vectors[name] = flat[offset : offset + len(strings)]
            offset += len(strings)
        return keywords, vectors

    def _lists_embedded(
        self, lists: KeywordLists
    ) -> tuple[dict[str, np.ndarray], list[str], np.ndarray]:
        version = list_version(lists)
        if version not in self._list_vectors:
            pooled = list(dict.fromkeys(lists.topic + lists.method + lists.common))
            self._list_vectors[version] = (
                {f: self.models.embed(getattr(lists, f)) for f in FACETS},
                pooled,
                self.models.embed(pooled),
            )
        return self._list_vectors[version]

    def check_model(self, spec: CollectionSpec) -> None:
        if spec.dense_model != self.models.dense_model:
            raise ModelMismatch(
                f"collection '{spec.name}' was built with {spec.dense_model};"
                f" service is running {self.models.dense_model}"
            )
