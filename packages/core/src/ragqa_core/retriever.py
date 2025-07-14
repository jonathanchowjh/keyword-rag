"""The ten retrieval methods: the 2024 paper's seven and three modern ones.

Fusion is reciprocal rank fusion computed here, not by Qdrant, so the paper's
k=1 and today's k=60 behave identically in every storage mode.
"""

import sqlite3
from collections.abc import Callable

from ragqa_core import db
from ragqa_core.indexer import Indexer
from ragqa_core.keywords import matched
from ragqa_core.llm import Claude
from ragqa_core.models import Models
from ragqa_core.prompts import SUMMARY_TOKENS, hop_prompt, parse_json, summary_prompt
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Hit

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
    "iterative",
)
KEYWORD_METHODS = ("kw_all", "kw_individual", "kw_rrf", "kw_hybrid")
POOL = 50
MAX_HOPS = 3
HOP_TOKENS = 200


class MethodUnavailable(Exception):
    pass


class UnknownCollection(Exception):
    pass


def rrf(rankings: list[list[Hit]], k: int) -> list[Hit]:
    scores: dict[str, float] = {}
    first: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            first.setdefault(hit.id, hit)
    order = {hit_id: n for n, hit_id in enumerate(first)}
    fused = sorted(first, key=lambda i: (-scores[i], order[i]))
    out = []
    for hit_id in fused:
        hit = first[hit_id]
        out.append(Hit(**{**hit.__dict__, "score": scores[hit_id]}))
    return out


class Retriever:
    def __init__(
        self,
        conn: sqlite3.Connection,
        store: Store,
        models: Models,
        claude: Claude,
        indexer: Indexer,
    ) -> None:
        self.conn = conn
        self.store = store
        self.models = models
        self.claude = claude
        self.indexer = indexer
        self._methods: dict[str, Callable[[CollectionSpec, str, int, int], list[Hit]]] = {
            "question_only": lambda spec, q, k, rk: [],
            "vector": self._vector,
            "keyword": self._keyword,
            "hybrid": self._hybrid,
            "kw_all": self._kw_all,
            "kw_individual": self._kw_individual,
            "kw_rrf": self._kw_rrf,
            "hybrid_rerank": self._hybrid_rerank,
            "kw_hybrid": self._kw_hybrid,
            "iterative": self._iterative,
        }

    def spec(self, name: str) -> CollectionSpec:
        spec = db.get_collection(self.conn, name)
        if spec is None:
            raise UnknownCollection(f"no collection named '{name}'")
        return spec

    def retrieve(
        self, collection: str, question: str, method: str, k: int = 5, rrf_k: int = 60
    ) -> list[Hit]:
        if method not in self._methods:
            raise ValueError(f"unknown method {method!r}; known: {', '.join(METHOD_NAMES)}")
        spec = self.spec(collection)
        # A query embedded by another model is either a dimension crash or,
        # worse, silently meaningless; refuse it by name instead.
        self.indexer.check_model(spec)
        return self._methods[method](spec, question, k, rrf_k)

    def is_paid(self, spec: CollectionSpec, method: str) -> bool:
        return method == "iterative" or (method in KEYWORD_METHODS and spec.summarize)

    def prepare(self, spec: CollectionSpec, questions: list[str]) -> None:
        if spec.summarize and questions:
            self.claude.complete_many(
                [summary_prompt(q) for q in questions],
                max_tokens=SUMMARY_TOKENS,
                purpose="query_summary",
            )

    # -- the paper's methods ---------------------------------------------------

    def _vector(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        parents: dict[str, float] = {}
        for parent, score in self.store.search_chunks(
            spec.name, self.models.embed_query([q])[0], POOL
        ):
            parents.setdefault(parent, score)
        top = list(parents)[:k]
        hits = self.store.get(spec.name, top)
        for h in hits:
            h.score = parents[h.id]
        return hits

    def _keyword(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        return self.store.search_sparse(spec.name, self.models.sparse_query([q])[0], k)

    def _hybrid(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        dense = self.store.search(spec.name, "text_dense", self.models.embed_query([q])[0], POOL)
        sparse = self.store.search_sparse(spec.name, self.models.sparse_query([q])[0], POOL)
        return rrf([dense, sparse], rrf_k)[:k]

    def _kw_all(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        return self._kw_search(spec, q, ["kw_all"], k, rrf_k)

    def _kw_individual(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        return self._kw_search(spec, q, ["kw_individual"], k, rrf_k)

    def _kw_rrf(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        names = ["kw_topic", "kw_method", "kw_common"] + (["kw_summary"] if spec.summarize else [])
        return self._kw_search(spec, q, names, k, rrf_k)

    # -- modern additions ------------------------------------------------------

    def _hybrid_rerank(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        pool = self._hybrid(spec, q, POOL, rrf_k)
        if not pool:
            return []
        scores = self.models.rerank(q, [h.text for h in pool])
        ranked = sorted(
            zip(scores, range(len(pool)), pool, strict=True), key=lambda t: (-t[0], t[1])
        )
        out = []
        for score, _, hit in ranked[:k]:
            hit.score = score
            out.append(hit)
        return out

    def _kw_hybrid(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        query_kw, vectors = self._query_keywords(spec, q)
        keyword_hits = self._kw_rank(spec, ["kw_individual"], vectors, POOL, rrf_k)
        fused = rrf([keyword_hits, self._hybrid(spec, q, POOL, rrf_k)], rrf_k)[:k]
        return self._mark(query_kw, fused)

    def _iterative(self, spec: CollectionSpec, q: str, k: int, rrf_k: int) -> list[Hit]:
        """Retrieve, then let Claude name the next query, up to MAX_HOPS calls.

        Reading of spec §5: rather than decomposing up front, each hop asks
        for the next missing fact given what has been found (≤3 calls)."""
        hops: list[list[Hit]] = []
        seen: set[str] = set()

        def hop(query: str) -> None:
            fresh = [h for h in self._hybrid_rerank(spec, query, 2, rrf_k) if h.id not in seen]
            seen.update(h.id for h in fresh)
            hops.append(fresh)

        query = q
        for _ in range(MAX_HOPS):
            hop(query)
            found = [h.text for hits in hops for h in hits]
            reply = self.claude.complete(hop_prompt(q, found), max_tokens=HOP_TOKENS, purpose="hop")
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

    # -- shared ------------------------------------------------------------------

    def _query_keywords(self, spec: CollectionSpec, q: str) -> tuple[dict, dict]:
        lists = db.get_keyword_lists(self.conn, spec.name)
        if lists is None:
            raise MethodUnavailable(f"keyword lists not set for collection '{spec.name}'")
        summary = None
        if spec.summarize:
            summary = self.claude.complete(
                summary_prompt(q), max_tokens=SUMMARY_TOKENS, purpose="query_summary"
            ).text.strip()
        keywords, vectors = self.indexer.keywords_for(spec, lists, [q], [summary])
        return keywords[0], {name: vecs[0] for name, vecs in vectors.items()}

    def _kw_search(
        self, spec: CollectionSpec, q: str, names: list[str], k: int, rrf_k: int
    ) -> list[Hit]:
        query_kw, vectors = self._query_keywords(spec, q)
        return self._mark(query_kw, self._kw_rank(spec, names, vectors, k, rrf_k))

    def _kw_rank(
        self, spec: CollectionSpec, names: list[str], vectors: dict, k: int, rrf_k: int
    ) -> list[Hit]:
        limit = k if len(names) == 1 else POOL
        rankings = [self.store.search(spec.name, n, vectors[n], limit) for n in names]
        return rankings[0][:k] if len(names) == 1 else rrf(rankings, rrf_k)[:k]

    @staticmethod
    def _mark(query_kw: dict, hits: list[Hit]) -> list[Hit]:
        for h in hits:
            h.matched = matched(query_kw, h.keywords)
        return hits
