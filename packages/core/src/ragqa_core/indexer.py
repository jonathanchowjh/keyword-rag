"""Builds and refreshes every representation of a collection's items."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from aiconnect.ai import AI, NoConnection
from ragqa_core import db, search_defaults
from ragqa_core.connection import Connection
from ragqa_core.datasets import source_for
from ragqa_core.keywords import assign, keyword_string, list_version, pick_all
from ragqa_core.models import Models
from ragqa_core.prompts import SUMMARY_TOKENS, parse_json, propose_prompt, summary_prompt
from ragqa_core.recipes import build_for, build_of, corpus_type_for, default_search, recipes_for
from ragqa_core.store import (
    CHUNK,
    SEARCH_DATA_SUFFIXES,
    SPARSE,
    Store,
    chunks_of,
    item_hash,
    rebuild_of,
)
from ragqa_core.types import (
    FACETS,
    BuildSettings,
    CollectionSpec,
    Item,
    KeywordLists,
    QaType,
)

BATCH = 64
PROPOSE_TOKENS = 2000


class ModelMismatch(Exception):
    pass


def chunk(text: str, words: int = 100, overlap: int = 25) -> list[str]:
    """Windows of `words` words sliding by `words - overlap`; words <= 0 keeps the text whole."""
    tokens = text.split()
    if words <= 0 or len(tokens) <= words:
        return [text]
    stride = words - overlap
    starts = range(0, len(tokens) - words + stride, stride)
    return [" ".join(tokens[s : s + words]) for s in starts]


def embed_text(spec: CollectionSpec, item: Item) -> str:
    """The text that is embedded for an item (dense vector and chunks); stored text is unchanged."""
    title = item.meta.get("title")
    build = build_of(spec)
    if build.added_text == "title_prefix" and isinstance(title, str) and title.strip():
        return f"{title.strip()}: {item.text}"
    return item.text


def dedupe(words: list[str]) -> list[str]:
    return list(dict.fromkeys(w.strip() for w in words if isinstance(w, str) and w.strip()))


def _as_list(value: object) -> list:
    return value if isinstance(value, list) else []


def sourced(items: list[Item], source: str) -> list[Item]:
    return [replace(i, meta={**i.meta, "source": source}) for i in items]


class Indexer:
    def __init__(self, conn: Connection, store: Store, models: Models, ai: AI):
        self.conn = conn
        self.store = store
        self.models = models
        self.ai = ai
        # Keyword-list embeddings by (model, list version): they change only when the
        # lists do, and re-embedding ~565 keywords per query dominated eval time. A copy
        # shares its source's lists version, so the model is in the key too.
        self._list_vectors: dict[
            tuple[str, str], tuple[dict[str, np.ndarray], list[str], np.ndarray]
        ] = {}

    def new_spec(
        self,
        name: str,
        qa_type: QaType,
        summarize: bool,
        recipe: str = "paper_2024",
        build: BuildSettings | None = None,
        dense_model: str | None = None,
    ) -> CollectionSpec:
        """A new collection's spec, not yet saved: the recipe's build, or the one given, and
        the dense model given, or the one new stores take."""
        if not db.NAME.match(name):
            raise ValueError(db.refused_name(name, "a store name"))
        if name.endswith(SEARCH_DATA_SUFFIXES):  # the names of a store's search data
            raise ValueError(
                f"store names may not end in {' or '.join(SEARCH_DATA_SUFFIXES)};"
                " those are kept for its search data"
            )
        build = build or build_for(recipe, summarize)
        allowed = recipes_for(qa_type)
        if recipe not in allowed:
            raise ValueError(
                f"a {qa_type} store takes the recipes {', '.join(allowed)}, not {recipe}"
            )
        model = dense_model or self.new_store_model()
        return CollectionSpec(
            name,
            qa_type,
            build.added_text == "summary",  # the build decides; the flag only moves paper 2024
            model,
            self.models.dense_dim(model),
            corpus_type=corpus_type_for(qa_type),
            recipe=recipe,
            build=build,
            search=default_search(qa_type, build),
        )

    def new_store_model(self) -> str:
        """The embedding model a new store takes: Settings', else the service's."""
        return str(search_defaults.in_force(self.conn)["dense_model"] or self.models.dense_model)

    def create(
        self,
        name: str,
        qa_type: QaType,
        summarize: bool,
        recipe: str = "paper_2024",
        build: BuildSettings | None = None,
        dense_model: str | None = None,
    ) -> CollectionSpec:
        spec = self.new_spec(name, qa_type, summarize, recipe, build, dense_model)
        db.save_collection(self.conn, spec)
        self.store.create(name, spec.dense_dim)
        return spec

    def changed(
        self, spec: CollectionSpec, items: list[Item], source: str
    ) -> tuple[list[Item], dict[str, int]]:
        """The sourced items whose stored hash differs, with new/updated/unchanged counts."""
        items = sourced(items, source)
        stored = self.store.hashes(spec.name, [i.id for i in items])
        todo = [i for i in items if stored.get(i.id) != item_hash(i)]
        new = sum(1 for i in todo if i.id not in stored)
        counts = {"new": new, "updated": len(todo) - new, "unchanged": len(items) - len(todo)}
        return todo, counts

    def plan_summaries(self, items: list[Item]) -> tuple[float | None, int]:
        """What summarizing the items would spend, and how many calls it would send (the
        cache misses). With no model to price it: None, and one call per item."""
        try:
            prompts = [summary_prompt(i.text) for i in items]
            return self.ai.plan_many(prompts, SUMMARY_TOKENS, role="summary")
        except NoConnection:
            return None, len(items)

    def add(
        self,
        spec: CollectionSpec,
        items: list[Item],
        source: str,
        progress: Callable[[float], None] = lambda p: None,
    ) -> dict[str, int]:
        self.check_model(spec)
        build = build_of(spec)
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
            whole = [embed_text(spec, i) for i in part]
            vectors = self.models.embed(whole, spec.dense_model)
            self.store.upsert_items(spec.name, part, vectors, self.models.sparse(texts), sums)
            # An updated item may now have fewer chunks: drop the old ones first.
            self.store.delete_chunks(spec.name, [i.id for i in part])
            parents, pieces, sources, own = [], [], [], []
            for n, item in enumerate(part):
                size = 0 if build.units == "whole" else build.chunk_words
                for piece in chunk(whole[n], size, build.chunk_overlap):
                    parents.append(item.id)
                    pieces.append(piece)
                    sources.append(item.meta["source"])
                    own.append(n if piece == whole[n] else None)
            # A TEXT THAT IS ITS OWN ONLY CHUNK (whole units, short items) REUSES ITS ITEM
            # VECTOR: embedding it again doubled the load's slowest step.
            # Only the rest are embedded, and never an empty list (the real model can't).
            rest = [p for p, o in zip(pieces, own, strict=True) if o is None]
            fresh = iter(self.models.embed(rest, spec.dense_model) if rest else [])
            chunk_vectors = np.stack([vectors[o] if o is not None else next(fresh) for o in own])
            self.store.upsert_chunks(spec.name, parents, pieces, chunk_vectors, sources)
            # THE INDUCED RECIPES KEEP THEIR TERMS AS TAGS, NOT AS THE THREE LISTS' VECTORS.
            if lists is not None and build.vocabulary == "lists":
                keywords, vectors = self.keywords_for(spec, lists, texts, sums)
                self.store.set_keywords(spec.name, [i.id for i in part], keywords, vectors)
            # Last, so a batch that died part-way reads as changed and is redone.
            self.store.set_payloads(spec.name, {i.id: {"content_hash": item_hash(i)} for i in part})
            progress(min(start + BATCH, len(items)) / len(items))
        return counts

    def backfill(self, name: str, progress: Callable[[float], None] = lambda p: None) -> dict:
        """Label points that predate sources, in place: no embedding, no AI call."""
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

    def plan_propose(self, spec: CollectionSpec, sample: list[Item]) -> tuple[float, int]:
        """What proposing lists from the sample would spend, and 1 call, or 0 when its reply
        is cached. Raises NoConnection("keywords") when no model can price it."""
        prompt = propose_prompt([i.text for i in sample], spec.qa_type)
        return self.ai.plan_many([prompt], PROPOSE_TOKENS, role="keywords")

    def propose_lists(self, spec: CollectionSpec, sample: list[Item]) -> KeywordLists:
        reply = self.ai.complete(
            propose_prompt([i.text for i in sample], spec.qa_type),
            role="keywords",
            max_tokens=PROPOSE_TOKENS,
            purpose="propose",
        )
        proposed = parse_json(reply.text)
        lists = {f: dedupe(_as_list(proposed.get(f))) for f in FACETS}
        empty = [f for f in FACETS if not lists[f]]
        if empty:
            raise ValueError(
                f"The AI's reply had no usable {' or '.join(empty)} list: {reply.text[:200]!r}"
            )
        return KeywordLists(**lists)

    def save_lists(self, spec: CollectionSpec, lists: KeywordLists) -> str:
        """Validate, embed and store a version of the lists, without switching to it."""
        self.check_model(spec)  # a model this service can't load can't embed the lists
        for facet in FACETS:
            if not getattr(lists, facet):
                raise ValueError(f"keyword list '{facet}' is empty")
        # Embedded before saving: a collection must never point at lists it cannot use.
        self._lists_embedded(spec.dense_model, lists)
        version = list_version(lists)
        db.save_keyword_lists(self.conn, spec.name, lists, version)
        return version

    def rekey(
        self, name: str, version: str, progress: Callable[[float], None] = lambda p: None
    ) -> None:
        """Re-key every item with a saved version's lists, then switch to it.
        If anything raises, the old version stays in force."""
        spec = db.get_collection(self.conn, name)
        lists = db.get_keyword_lists(self.conn, name, version)
        if spec is None or lists is None:
            raise ValueError(f"no keyword lists {version!r} for store '{name}'")
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
        item_vecs = self.models.embed(texts, spec.dense_model)
        facet_vecs, pooled, pooled_vecs = self._lists_embedded(spec.dense_model, lists)
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
        flat = self.models.embed(
            [text for strings in groups.values() for text in strings], spec.dense_model
        )
        vectors: dict[str, np.ndarray] = {}
        offset = 0
        for name, strings in groups.items():
            vectors[name] = flat[offset : offset + len(strings)]
            offset += len(strings)
        return keywords, vectors

    def _lists_embedded(
        self, model: str, lists: KeywordLists
    ) -> tuple[dict[str, np.ndarray], list[str], np.ndarray]:
        key = (model, list_version(lists))
        if key not in self._list_vectors:
            pooled = list(dict.fromkeys(lists.topic + lists.method + lists.common))
            self._list_vectors[key] = (
                {f: self.models.embed(getattr(lists, f), model) for f in FACETS},
                pooled,
                self.models.embed(pooled, model),
            )
        return self._list_vectors[key]

    def reembed(
        self,
        name: str,
        model: str,
        progress: Callable[[float], None] = lambda p: None,
        switching: bool = False,
        on_switch: Callable[[], None] = lambda: None,
    ) -> dict:
        """Rebuilds the store with another embedding model, then switches to it.

        Phase 1 embeds every item and chunk again into rebuild_of(name), keeping each point's
        id, payload, tags and sparse vector; keyword-list picks are recomputed with the new
        model. A stop there (Cancelled from progress included) drops the copy and leaves the
        store as it was. Phase 2, the switch, refills the store from the copy and records
        the model; it reports no progress, so it can't be cancelled, and `switching=True`
        resumes it after a stop."""
        spec = db.require_collection(self.conn, name)
        if not switching:
            self._rebuild(spec, model, progress)
            on_switch()
        elif not self.store.exists(rebuild_of(name)):
            if spec.dense_model == model:  # the switch finished; only the copy's drop was left
                self.store.drop_rebuild(name)  # its chunks may outlive a drop that died
                return {
                    "model": spec.dense_model,
                    "dim": spec.dense_dim,
                    "items": self.store.count(name),
                    "chunks": self.store.count(chunks_of(name)),
                }
            raise ValueError(
                f"the copy made for the switch to {model} is gone;"
                " delete the store and load it again"
            )
        dim = self.models.dense_dim(model)
        self.store.recreate(name, dim)
        items = self.store.copy(rebuild_of(name), name)
        chunks = self.store.copy(chunks_of(rebuild_of(name)), chunks_of(name))
        db.set_dense_model(self.conn, name, model, dim)
        self.store.drop_rebuild(name)
        return {"model": model, "dim": dim, "items": items, "chunks": chunks}

    def _rebuild(self, spec: CollectionSpec, model: str, progress: Callable[[float], None]) -> None:
        """Phase 1 of reembed. It never checks the old model: it reads only payloads and
        the sparse vector, so a store whose model can't load can be rebuilt with one that can."""
        target = replace(spec, dense_model=model, dense_dim=self.models.dense_dim(model))
        side = rebuild_of(spec.name)
        # WHATEVER THE VOCABULARY, AS REKEY: lists set by hand give any store kw_* vectors
        # that search reads, so they are recomputed with the new model whenever in force.
        lists = db.get_keyword_lists(self.conn, spec.name)
        total, done = self.store.count(spec.name), 0
        self.store.drop_rebuild(spec.name)
        try:
            self.store.create(side, target.dense_dim)
            for points in self.store.points(spec.name, BATCH):
                self._rebuild_batch(spec, target, lists, points)
                done += len(points)
                progress(0.95 * done / total)
        except BaseException:
            self.store.drop_rebuild(spec.name)
            raise

    def _rebuild_batch(
        self,
        spec: CollectionSpec,
        target: CollectionSpec,
        lists: KeywordLists | None,
        points: list[tuple[str, dict, dict]],
    ) -> None:
        """One batch of items, and their chunks, embedded with target's model into the copy."""
        model, side = target.dense_model, rebuild_of(spec.name)
        payloads = [dict(payload) for _, _, payload in points]
        items = [
            Item(
                p["item_id"], p["text"], p.get("answer"), p.get("explanation"), p.get("meta") or {}
            )
            for p in payloads
        ]
        whole = [embed_text(spec, i) for i in items]
        dense = self.models.embed(whole, model)
        vectors: list[dict] = [
            {"text_dense": dense[row], SPARSE: old[SPARSE]}
            for row, (_, old, _) in enumerate(points)
        ]
        if lists is not None:
            keywords, kw_vectors = self.keywords_for(
                target, lists, [i.text for i in items], [p.get("summary") for p in payloads]
            )
            for row, kw in enumerate(keywords):
                payloads[row]["keywords"] = kw
                vectors[row].update({v: vecs[row] for v, vecs in kw_vectors.items()})
        tagged = [row for row, p in enumerate(payloads) if p.get("tags")]
        if tagged:
            tags = [keyword_string(payloads[row]["tags"], None) for row in tagged]
            for row, vec in zip(tagged, self.models.embed(tags, model), strict=True):
                vectors[row]["kw_tags"] = vec
        self.store.write(
            side,
            [
                (pid, vecs, p)
                for (pid, _, _), vecs, p in zip(points, vectors, payloads, strict=True)
            ],
        )
        # A CHUNK THAT IS ITS ITEM'S WHOLE TEXT REUSES THE ITEM'S NEW VECTOR, AS IN add.
        row_of = {i.id: row for row, i in enumerate(items)}
        chunks = self.store.chunk_points(spec.name, [i.id for i in items])
        rows = [row_of[p["parent_id"]] for _, p in chunks]
        own = [r if p["text"] == whole[r] else None for (_, p), r in zip(chunks, rows, strict=True)]
        rest = [p["text"] for (_, p), o in zip(chunks, own, strict=True) if o is None]
        fresh = iter(self.models.embed(rest, model) if rest else [])
        self.store.write(
            chunks_of(side),
            [
                (pid, {CHUNK: dense[o] if o is not None else next(fresh)}, p)
                for (pid, p), o in zip(chunks, own, strict=True)
            ],
        )

    def check_model(self, spec: CollectionSpec) -> None:
        """Refuses a store this service can't search: a switch to another model is
        unfinished, its model is not one fastembed lists, or the dimension it was built at
        is not its model's. A store whose model is not the service's default is fine: it is
        embedded and searched with its own."""
        switching = db.switching_model(self.conn, spec.name)
        if switching:
            raise ModelMismatch(
                f"store '{spec.name}' is switching to {switching}; try again when its"
                " rebuild finishes, or resume it on the store page"
            )
        try:
            dim = self.models.dense_dim(spec.dense_model)
        except ValueError:
            raise ModelMismatch(
                f"store '{spec.name}' was built with {spec.dense_model},"
                " which this service can't load"
            ) from None
        if dim != spec.dense_dim:
            raise ModelMismatch(
                f"store '{spec.name}' was built with {spec.dense_model}"
                f" at {spec.dense_dim} dimensions; {spec.dense_model} gives {dim}"
            )
