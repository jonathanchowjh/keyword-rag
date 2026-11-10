"""The retrieval methods: the 2024 paper's seven, four modern ones (Keyword-RAG among
them), and two controls that pick examples without looking at the question's content.

Fusion is reciprocal rank fusion computed here, not by Qdrant, so the paper's
k=1 and today's k=60 behave identically in every storage mode.
"""

import random
from collections.abc import Callable, Collection

import numpy as np

from aiconnect.ai import AI, AwaitingReplies, BudgetExceeded, NoConnection, ProviderError
from ragqa_core import db
from ragqa_core.builder import TAG_TOKENS, Builder, build_of, picked_terms
from ragqa_core.connection import Connection
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.keywords import keyword_string, matched
from ragqa_core.models import Models
from ragqa_core.prompts import (
    QUERY_TASK,
    SUMMARY_TOKENS,
    hop_prompt,
    parse_json,
    query_tag_prompt,
    summary_prompt,
)
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Hit, SearchSettings, Skip, unknown_store

METHOD_NAMES = (
    "question_only",
    "vector",
    "keyword",
    "hybrid",
    "kw_all",
    "kw_individual",
    "kw_rrf",
    "hybrid_rerank",
    "kw_hybrid",
    "keyword_rag",
    "iterative",
    "random_example",
    "fixed_example",
)
FIXED_SEED = "fixed_example"
KEYWORD_METHODS = ("kw_all", "kw_individual", "kw_rrf", "kw_hybrid")
POOL = 50
NEIGHBOURS = 20  # the units whose tags vote for a question's (the classifier, D6)
MAX_HOPS = 3
HOP_TOKENS = 200


class MethodUnavailable(Exception):
    pass


class UnknownCollection(Exception):
    pass


# What one method's run may raise and still leave the others to run: it becomes that
# method's `error`, so it does not hide what the others spent.
PER_METHOD_ERRORS = (
    MethodUnavailable,
    NoConnection,
    BudgetExceeded,
    ModelMismatch,
    AwaitingReplies,
    ProviderError,  # a method's provider failing must not hide what the others spent
)


NEEDS_TAGS = "needs keyword tags: Add keyword tags on the store page"


def available_methods(
    spec: CollectionSpec, has_lists: bool, tagging: bool = False
) -> dict[str, str | None]:
    """Each method with why it can't run on the store, or None when it can. `tagging`: an
    Add keyword tags job is unfinished (db.tag_build_unfinished), so the vocabulary in
    force may have tagged only some passages."""
    out: dict[str, str | None] = {}
    for m in METHOD_NAMES:
        if m in KEYWORD_METHODS and not has_lists:
            out[m] = "needs keyword lists"
        elif m == "keyword_rag" and tagging:
            out[m] = NEEDS_TAGS
        elif m == "keyword_rag" and spec.vocabulary_version is None:
            out[m] = NEEDS_TAGS if spec.qa_type == "documents" else "needs a vocabulary"
        else:
            out[m] = None
    return out


def rrf(rankings: list[list[Hit]], k: int, weights: list[float] | None = None) -> list[Hit]:
    """Each hit scores the sum over rankings of weight / (k + rank); a weight defaults to 1."""
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    for n, ranking in enumerate(rankings):
        weight = weights[n] if weights is not None else 1.0
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + weight / (k + rank)
            first.setdefault(hit.id, hit)
    order = {hit_id: n for n, hit_id in enumerate(first)}
    fused = sorted(first, key=lambda i: (-scores[i], order[i]))
    out = []
    for hit_id in fused:
        hit = first[hit_id]
        out.append(Hit(**{**hit.__dict__, "score": scores[hit_id]}))
    return out


def shuffled(ids: list[str], seed: str) -> list[str]:
    """A seeded shuffle. Callers take a prefix, so the first pick (the one an
    mcq answer uses) is the same whatever k is."""
    out = list(ids)
    random.Random(seed).shuffle(out)
    return out


class Retriever:
    def __init__(
        self,
        conn: Connection,
        store: Store,
        models: Models,
        ai: AI,
        indexer: Indexer,
        builder: Builder,
    ) -> None:
        self.conn = conn
        self.store = store
        self.models = models
        self.ai = ai
        self.indexer = indexer
        self.builder = builder
        # Every item id by collection, with the count and skipped sources it was read at: the
        # controls need them per query, and an 11,656-item scroll each time is slow.
        self._ids: dict[str, tuple[tuple[int, tuple[str, ...]], list[str]]] = {}
        self._methods: dict[str, Callable[[CollectionSpec, str, int, int, Skip], list[Hit]]] = {
            "question_only": lambda spec, q, k, rk, skip: [],
            "vector": self._vector,
            "keyword": self._keyword,
            "hybrid": self._hybrid,
            "kw_all": self._kw_all,
            "kw_individual": self._kw_individual,
            "kw_rrf": self._kw_rrf,
            "hybrid_rerank": self._hybrid_rerank,
            "kw_hybrid": self._kw_hybrid,
            "iterative": self._iterative,
            "random_example": lambda spec, q, k, rk, skip: self._control(spec, q, k, skip),
            "fixed_example": lambda spec, q, k, rk, skip: self._control(spec, FIXED_SEED, k, skip),
        }

    def spec(self, name: str) -> CollectionSpec:
        spec = db.get_collection(self.conn, name)
        if spec is None:
            raise UnknownCollection(unknown_store(name))
        return spec

    def retrieve(
        self,
        collection: str,
        question: str,
        method: str,
        k: int = 5,
        rrf_k: int = 60,
        search: SearchSettings | None = None,
        tags: list[str] | None = None,
        exclude: Collection[str] = (),
        pool: int = POOL,
        rerank_model: str | None = None,
    ) -> list[Hit]:
        """`search` None means the collection's own settings; `tags` None means the query's
        tags are predicted as the settings say. Only keyword_rag uses either. `exclude` names
        items no channel may find, so every pool and k counts only the others. `pool` and
        `rerank_model` are read by hybrid_rerank alone: how many of hybrid's top it reranks,
        and with which model (None is the service's)."""
        if method not in METHOD_NAMES:
            raise ValueError(f"unknown method {method!r}; known: {', '.join(METHOD_NAMES)}")
        spec = self.spec(collection)
        # Every query is embedded with the store's own model; a store whose model this
        # service can't load, or whose dimension is not its model's, is refused by name.
        self.indexer.check_model(spec)
        skip = Skip(tuple(db.disabled_sources(self.conn, collection)), frozenset(exclude))
        if method == "keyword_rag":
            settings = search or spec.search or SearchSettings()
            return self._keyword_rag(spec, question, k, rrf_k, skip, settings, tags)
        if method == "hybrid_rerank":
            return self._hybrid_rerank(spec, question, k, rrf_k, skip, pool, rerank_model)
        return self._methods[method](spec, question, k, rrf_k, skip)

    def is_paid(self, spec: CollectionSpec, method: str) -> bool:
        return method == "iterative" or (method in KEYWORD_METHODS and spec.summarize)

    def prepare(self, spec: CollectionSpec, questions: list[str]) -> None:
        if spec.summarize and questions:
            self.ai.complete_many(
                [summary_prompt(q) for q in questions],
                role="query",
                max_tokens=SUMMARY_TOKENS,
                purpose="query_summary",
            )

    def tag_prompts(self, spec: CollectionSpec, questions: list[str]) -> list[str]:
        terms = self.vocabulary(spec)
        k = build_of(spec).tag_k
        return [query_tag_prompt(terms, q, k, spec.corpus_type) for q in questions]

    def prepare_tags(self, spec: CollectionSpec, questions: list[str]) -> None:
        """Asks the AI for every question's tags in one go, so a large run gets the Batch
        API's discount and a cancel between sends; query_tags then finds them cached."""
        if questions:
            self.ai.complete_many(
                self.tag_prompts(spec, questions),
                role="query",
                max_tokens=TAG_TOKENS,
                purpose="query_tags",
            )

    # -- the paper's methods ---------------------------------------------------

    def _vector(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        parents: dict[str, float] = {}
        for parent, score in self.store.search_chunks(
            spec.name, self._dense_query(spec, q), POOL, skip.sources, skip.items
        ):
            parents.setdefault(parent, score)
        top = list(parents)[:k]
        hits = self.store.get(spec.name, top)
        for h in hits:
            h.score = parents[h.id]
        return hits

    def _keyword(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        sparse = self.models.sparse_query([q])[0]
        return self.store.search_sparse(spec.name, sparse, k, skip.sources, skip.items)

    def _hybrid(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        dense = self.store.search(
            spec.name, "text_dense", self._dense_query(spec, q), POOL, skip.sources, skip.items
        )
        sparse = self.store.search_sparse(
            spec.name, self.models.sparse_query([q])[0], POOL, skip.sources, skip.items
        )
        return rrf([dense, sparse], rrf_k)[:k]

    def _kw_all(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        return self._kw_search(spec, q, ["kw_all"], k, rrf_k, skip)

    def _kw_individual(
        self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip
    ) -> list[Hit]:
        return self._kw_search(spec, q, ["kw_individual"], k, rrf_k, skip)

    def _kw_rrf(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        names = ["kw_topic", "kw_method", "kw_common"] + (["kw_summary"] if spec.summarize else [])
        return self._kw_search(spec, q, names, k, rrf_k, skip)

    # -- modern additions ------------------------------------------------------

    def _hybrid_rerank(
        self,
        spec: CollectionSpec,
        q: str,
        k: int,
        rrf_k: int,
        skip: Skip,
        pool: int = POOL,
        rerank_model: str | None = None,
    ) -> list[Hit]:
        top = self._hybrid(spec, q, pool, rrf_k, skip)
        if not top:
            return []
        scores = self.models.rerank(q, [h.text for h in top], rerank_model)
        ranked = sorted(zip(scores, range(len(top)), top, strict=True), key=lambda t: (-t[0], t[1]))
        out = []
        for score, _, hit in ranked[:k]:
            hit.score = score
            out.append(hit)
        return out

    def _kw_hybrid(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        query_kw, vectors = self._query_keywords(spec, q)
        keyword_hits = self._kw_rank(spec, ["kw_individual"], vectors, POOL, rrf_k, skip)
        fused = rrf([keyword_hits, self._hybrid(spec, q, POOL, rrf_k, skip)], rrf_k)[:k]
        return self._mark(query_kw, fused)

    def _keyword_rag(
        self,
        spec: CollectionSpec,
        q: str,
        k: int,
        rrf_k: int,
        skip: Skip,
        search: SearchSettings,
        tags: list[str] | None,
    ) -> list[Hit]:
        """Meaning, word match and the predicted keywords, fused by weighted RRF (D7). A
        channel weighted 0 is not searched; with no query tags, keywords finds nothing."""
        self.vocabulary(spec)
        if tags is None:
            tags = self.query_tags(spec, q, search, skip.sources, skip.items)
        weights = search.weights
        rankings: dict[str, list[Hit]] = {}
        if weights.get("meaning", 0.0) > 0:
            if spec.corpus_type == "reference":
                rankings["meaning"] = self._vector(spec, q, POOL, rrf_k, skip)
            else:
                dense = self._dense_query(spec, q)
                rankings["meaning"] = self.store.search(
                    spec.name, "text_dense", dense, POOL, skip.sources, skip.items
                )
        if weights.get("words", 0.0) > 0:
            sparse = self.models.sparse_query([q])[0]
            rankings["words"] = self.store.search_sparse(
                spec.name, sparse, POOL, skip.sources, skip.items
            )
        if weights.get("keywords", 0.0) > 0 and tags:
            vector = self.models.embed([keyword_string(tags, None)], spec.dense_model)[0]
            rankings["keywords"] = self.store.search(
                spec.name, "kw_tags", vector, POOL, skip.sources, skip.items
            )
        found: dict[str, list[str]] = {}
        for channel, ranking in rankings.items():
            for h in ranking:
                found.setdefault(h.id, []).append(channel)
        channels = list(rankings)
        fused = rrf([rankings[c] for c in channels], rrf_k, [weights[c] for c in channels])[:k]
        for h in fused:
            h.channels = found[h.id]
            h.matched = [t for t in tags if t in h.tags]
        return fused

    def query_tags(
        self,
        spec: CollectionSpec,
        question: str,
        search: SearchSettings,
        skip: tuple[str, ...],
        exclude: Collection[str] = (),
    ) -> list[str]:
        """The question's tags from the vocabulary: by the classifier's vote of the nearest
        units, by similarity to the terms, or picked by the AI (the query job's model). Oracle
        tags come from a question's gold, which a bare question does not have. Skipped
        sources and excluded items cast no vote."""
        if search.query_tags == "oracle":
            raise ValueError("oracle tags need a question's gold")
        terms = self.vocabulary(spec)
        self.indexer.check_model(spec)
        build = build_of(spec)
        if search.query_tags == "classifier":
            return self._classify(spec, question, build.tag_k, Skip(skip, frozenset(exclude)))
        if search.query_tags == "model":
            [prompt] = self.tag_prompts(spec, [question])
            reply = self.ai.complete(
                prompt, role="query", max_tokens=TAG_TOKENS, purpose="query_tags"
            )
            # A reply naming no term falls back to similarity, as an item's does: never none.
            picked = picked_terms(reply.text, terms, build.tag_k)
            model = spec.dense_model
            return picked or self.builder.similar(model, terms, [question], build.tag_k)[0]
        cutoff = build.tag_cutoff if build.tagging == "similarity_cutoff" else None
        return self.builder.similar(spec.dense_model, terms, [question], build.tag_k, cutoff)[0]

    def _classify(self, spec: CollectionSpec, q: str, k: int, skip: Skip) -> list[str]:
        """The nearest-neighbour vote (D6): each of the NEIGHBOURS nearest units adds its
        score to each of its tags; the k best win, ties to the tag seen first."""
        votes: dict[str, float] = {}
        neighbours = self.store.search(
            spec.name,
            "text_dense",
            self._dense_query(spec, q),
            NEIGHBOURS,
            skip.sources,
            skip.items,
        )
        for h in neighbours:
            for tag in h.tags:
                votes[tag] = votes.get(tag, 0.0) + h.score
        return sorted(votes, key=lambda t: -votes[t])[:k]

    def _iterative(self, spec: CollectionSpec, q: str, k: int, rrf_k: int, skip: Skip) -> list[Hit]:
        """Retrieve, then let the AI name the next query, up to MAX_HOPS calls.

        Reading of spec §5: rather than decomposing up front, each hop asks
        for the next missing fact given what has been found (≤3 calls)."""
        hops: list[list[Hit]] = []
        seen: set[str] = set()

        def hop(query: str) -> None:
            fresh = [
                h for h in self._hybrid_rerank(spec, query, 2, rrf_k, skip) if h.id not in seen
            ]
            seen.update(h.id for h in fresh)
            hops.append(fresh)

        query = q
        for _ in range(MAX_HOPS):
            hop(query)
            found = [h.text for hits in hops for h in hits]
            reply = self.ai.complete(
                hop_prompt(q, found), role="query", max_tokens=HOP_TOKENS, purpose="hop"
            )
            step = parse_json(reply.text)
            query = str(step.get("next_query") or "").strip()
            if step.get("done") or not query:
                break
        else:
            hop(query)
        # Round-robin across hops: the best passage of every hop comes before the
        # second-best of any, so a later hop's fact is never cut off by k.
        ordered = [hits[i] for i in range(2) for hits in hops if i < len(hits)]
        return ordered[:k]

    # -- controls ----------------------------------------------------------------

    def _control(self, spec: CollectionSpec, seed: str, k: int, skip: Skip) -> list[Hit]:
        """random_example seeds by the question, so each question always gets the
        same draw (and the same cached answer); fixed_example uses one seed for all.
        Against these, a method shows whether choosing the example helped at all."""
        ids = [i for i in self._all_ids(spec.name, skip.sources) if i not in skip.items]
        return self.store.get(spec.name, shuffled(ids, seed)[:k])

    def forget(self, name: str) -> None:
        """Drops what is cached for a collection that is gone."""
        self._ids.pop(name, None)

    def _all_ids(self, name: str, skip: tuple[str, ...]) -> list[str]:
        count = self.store.count(name)
        cached = self._ids.get(name)
        if cached is not None and cached[0] == (count, skip):
            return cached[1]
        ids = self.store.ids(name, skip=skip)
        self._ids[name] = ((count, skip), ids)
        return ids

    # -- shared ------------------------------------------------------------------

    def _dense_query(self, spec: CollectionSpec, q: str) -> np.ndarray:
        task = QUERY_TASK[spec.qa_type]
        return self.models.embed_query([q], task=task, model=spec.dense_model)[0]

    def _query_keywords(self, spec: CollectionSpec, q: str) -> tuple[dict, dict]:
        if db.job_running(self.conn, "rekey", spec.name):
            raise MethodUnavailable(
                "keyword lists are being re-keyed; try again when the job finishes"
            )
        lists = db.get_keyword_lists(self.conn, spec.name)
        if lists is None:
            raise MethodUnavailable(f"keyword lists not set for store '{spec.name}'")
        summary = None
        if spec.summarize:
            summary = self.ai.complete(
                summary_prompt(q), role="query", max_tokens=SUMMARY_TOKENS, purpose="query_summary"
            ).text.strip()
        keywords, vectors = self.indexer.keywords_for(spec, lists, [q], [summary])
        return keywords[0], {name: vecs[0] for name, vecs in vectors.items()}

    def vocabulary(self, spec: CollectionSpec) -> list[str]:
        """The terms in force; refused while a retag is switching them, after one stopped
        partway (items hold two versions' tags), until Add keyword tags finishes (its
        vocabulary is in force before every passage is tagged), or when there are none."""
        if db.job_running(self.conn, "retag", spec.name):
            raise MethodUnavailable(
                "the vocabulary is being re-tagged; try again when the job finishes"
            )
        if db.retag_partly_done(self.conn, spec.name):
            raise MethodUnavailable(
                f"the vocabulary of store '{spec.name}' is partly re-tagged; save it again"
            )
        if db.tag_build_unfinished(self.conn, spec.name):
            raise MethodUnavailable(NEEDS_TAGS)
        terms = db.get_vocabulary(self.conn, spec.name)
        if not terms and spec.qa_type == "documents":  # its other recipes are not for it
            raise MethodUnavailable(NEEDS_TAGS)
        if not terms:
            raise MethodUnavailable(
                f"no vocabulary for store '{spec.name}';"
                " build it with the worked examples or reference recipe"
            )
        return terms

    def _kw_search(
        self, spec: CollectionSpec, q: str, names: list[str], k: int, rrf_k: int, skip: Skip
    ) -> list[Hit]:
        query_kw, vectors = self._query_keywords(spec, q)
        return self._mark(query_kw, self._kw_rank(spec, names, vectors, k, rrf_k, skip))

    def _kw_rank(
        self,
        spec: CollectionSpec,
        names: list[str],
        vectors: dict,
        k: int,
        rrf_k: int,
        skip: Skip,
    ) -> list[Hit]:
        limit = k if len(names) == 1 else POOL
        rankings = [
            self.store.search(spec.name, n, vectors[n], limit, skip.sources, skip.items)
            for n in names
        ]
        return rankings[0][:k] if len(names) == 1 else rrf(rankings, rrf_k)[:k]

    @staticmethod
    def _mark(query_kw: dict, hits: list[Hit]) -> list[Hit]:
        for h in hits:
            h.matched = matched(query_kw, h.keywords)
        return hits
