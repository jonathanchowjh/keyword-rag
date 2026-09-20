"""The build steps of the induced recipes (spec §15): a collection's vocabulary,
and each item's tags from it (the kw_tags vector) with the profile they add up to.

Every prompt here is written from an item's text or explanation only, never its
meta: ScienceQA's gold skill must not reach the vocabulary (spec §12).
"""

from collections import Counter
from collections.abc import Callable

import numpy as np

from aiconnect.ai import AI
from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.indexer import BATCH, Indexer
from ragqa_core.keywords import keyword_string
from ragqa_core.models import Models
from ragqa_core.prompts import induce_prompt, parse_json, tag_prompt
from ragqa_core.recipes import build_for
from ragqa_core.store import Store
from ragqa_core.types import BuildSettings, CollectionSpec, Item
from ragqa_core.vocabulary import clean_terms, describe_text, merge_terms, terms_version

INDUCE_BATCH = 20
INDUCE_TOKENS = 300
TAG_TOKENS = 100
CUTOFF_MOST = 10  # a cut-off keeps at most this many tags
AI_BATCH_OVER = 50  # more calls than this go to a provider's Batch API, as summaries do


def build_of(spec: CollectionSpec) -> BuildSettings:
    return spec.build or build_for("paper_2024", spec.summarize)


class Builder:
    def __init__(self, conn: Connection, store: Store, models: Models, ai: AI, indexer: Indexer):
        self.conn = conn
        self.store = store
        self.models = models
        self.ai = ai
        self.indexer = indexer
        # Term embeddings by vocabulary version, with the order they were embedded in:
        # a vocabulary changes only when it is saved again, and every batch needs it.
        self._term_vectors: dict[str, tuple[list[str], np.ndarray]] = {}

    def _induce_prompts(self, spec: CollectionSpec, items: list[Item]) -> list[str]:
        build = build_of(spec)
        texts = [describe_text(i, build.describes) for i in items]
        return [
            induce_prompt(texts[start : start + INDUCE_BATCH], spec.corpus_type)
            for start in range(0, len(texts), INDUCE_BATCH)
        ]

    def estimate_induce_usd(self, spec: CollectionSpec, items: list[Item]) -> float:
        """Raises NoConnection("keywords") when no model can price it."""
        prompts = self._induce_prompts(spec, items)
        batch = len(prompts) > AI_BATCH_OVER
        return sum(
            self.ai.estimate_usd(len(p), INDUCE_TOKENS, role="keywords", batch=batch)
            for p in prompts
        )

    def induce(
        self,
        spec: CollectionSpec,
        items: list[Item],
        progress: Callable[[float], None] = lambda p: None,
    ) -> list[str]:
        """Terms proposed batch by batch, then merged. A reply that is not usable JSON,
        or has no terms, adds nothing; the other batches still count."""
        done = self.ai.complete_many(
            self._induce_prompts(spec, items),
            role="keywords",
            max_tokens=INDUCE_TOKENS,
            purpose="induce",
        )
        terms: list[str] = []
        for reply in done:
            proposed = parse_json(reply.text).get("terms")
            if isinstance(proposed, list):
                terms += clean_terms(proposed)
        merged = self._merged(terms)
        progress(1.0)
        return merged

    def _merged(self, terms: list[str]) -> list[str]:
        if not terms:
            return []
        forms = list(dict.fromkeys(terms))
        vectors = self.models.embed(forms)
        row = {form: n for n, form in enumerate(forms)}
        return merge_terms(terms, vectors[[row[t] for t in terms]])

    def save_vocabulary(self, spec: CollectionSpec, terms: list[str]) -> str:
        """Clean and store a version of the vocabulary, without switching to it."""
        cleaned = clean_terms(terms)
        if not cleaned:
            raise ValueError("the vocabulary is empty")
        version = terms_version(cleaned)
        db.save_vocabulary(self.conn, spec.name, cleaned, version)
        return version

    def term_vectors(self, terms: list[str]) -> np.ndarray:
        """The terms' embeddings, one row per term in the order given."""
        version = terms_version(terms)
        if version not in self._term_vectors:
            self._term_vectors[version] = (list(terms), self.models.embed(terms))
        embedded, vectors = self._term_vectors[version]
        if embedded == terms:
            return vectors
        row = {t: n for n, t in enumerate(embedded)}
        return vectors[[row[t] for t in terms]]

    def _similar(
        self, terms: list[str], texts: list[str], k: int, cutoff: float | None = None
    ) -> list[list[str]]:
        """The k terms nearest each text; with a cut-off, every term at or above it (at
        most CUTOFF_MOST) and never none. Ties go to the earlier term."""
        if not texts:
            return []
        out = []
        for row in self.models.embed(texts) @ self.term_vectors(terms).T:
            order = np.argsort(-row, kind="stable")
            if cutoff is None:
                picked = list(order[:k])
            else:
                picked = [n for n in order[:CUTOFF_MOST] if row[n] >= cutoff] or [order[0]]
            out.append([terms[n] for n in picked])
        return out

    def tags_for(
        self,
        spec: CollectionSpec,
        terms: list[str],
        texts: list[str],
        progress: Callable[[float], None] = lambda p: None,
    ) -> list[list[str]]:
        """Each text's tags from the vocabulary, as the build's tagging says."""
        build = build_of(spec)
        if build.tagging != "ai":
            cutoff = build.tag_cutoff if build.tagging == "similarity_cutoff" else None
            tags: list[list[str]] = []
            for start in range(0, len(texts), BATCH):
                tags += self._similar(terms, texts[start : start + BATCH], build.tag_k, cutoff)
                progress(min(start + BATCH, len(texts)) / len(texts))
            return tags
        # One call for every text, so a large corpus gets the Batch API's discount.
        done = self.ai.complete_many(
            [tag_prompt(terms, t, build.tag_k) for t in texts],
            role="keywords",
            max_tokens=TAG_TOKENS,
            purpose="tag",
        )
        known = set(terms)
        tags = []
        for reply in done:
            replied = parse_json(reply.text).get("terms")
            picked = clean_terms(replied) if isinstance(replied, list) else []
            tags.append([t for t in picked if t in known][: build.tag_k])
        # A reply with no term of the vocabulary falls back to similarity: never no tags.
        unusable = [n for n, t in enumerate(tags) if not t]
        nearest = self._similar(terms, [texts[n] for n in unusable], build.tag_k)
        for n, fallback in zip(unusable, nearest, strict=True):
            tags[n] = fallback
        progress(1.0)
        return tags

    def estimate_tag_usd(self, spec: CollectionSpec, terms: list[str], items: list[Item]) -> float:
        """0 unless the build tags with the AI. Raises NoConnection("keywords") when no
        model can price it."""
        build = build_of(spec)
        if build.tagging != "ai":
            return 0.0
        batch = len(items) > AI_BATCH_OVER
        return sum(
            self.ai.estimate_usd(
                len(tag_prompt(terms, describe_text(i, build.describes), build.tag_k)),
                TAG_TOKENS,
                role="keywords",
                batch=batch,
            )
            for i in items
        )

    def _tag(
        self,
        spec: CollectionSpec,
        terms: list[str],
        items: list[Item],
        progress: Callable[[float], None],
    ) -> None:
        """Choosing the tags is the first half of the progress, writing them the second."""
        build = build_of(spec)
        texts = [describe_text(i, build.describes) for i in items]
        tags = self.tags_for(spec, terms, texts, lambda p: progress(p / 2))
        for start in range(0, len(items), BATCH):
            part = tags[start : start + BATCH]
            vectors = self.models.embed([keyword_string(t, None) for t in part])
            ids = [i.id for i in items[start : start + BATCH]]
            self.store.set_tags(spec.name, ids, part, vectors)
            progress(0.5 + min(start + BATCH, len(items)) / len(items) / 2)
        if not items:
            progress(1.0)

    def tag(
        self,
        spec: CollectionSpec,
        items: list[Item],
        progress: Callable[[float], None] = lambda p: None,
    ) -> None:
        """Tag the items with the vocabulary in force, then refresh the profile."""
        self.indexer.check_model(spec)
        terms = db.get_vocabulary(self.conn, spec.name)
        if not terms:
            raise ValueError(f"collection '{spec.name}' has no vocabulary in force")
        self._tag(spec, terms, items, progress)
        self.refresh_profile(spec)

    def stored_items(self, name: str) -> list[Item]:
        return [
            Item(h.id, h.text, h.answer, h.explanation, h.meta)
            for batch in self.store.scroll(name)
            for h in batch
        ]

    def retag(
        self, name: str, version: str, progress: Callable[[float], None] = lambda p: None
    ) -> None:
        """Tag every item with a saved version of the vocabulary, then switch to it.
        If anything raises, the old version stays in force."""
        spec = db.get_collection(self.conn, name)
        terms = db.get_vocabulary(self.conn, name, version)
        if spec is None or not terms:
            raise ValueError(f"no vocabulary {version!r} for collection '{name}'")
        self.indexer.check_model(spec)  # before any term is embedded with another model
        self._tag(spec, terms, self.stored_items(name), progress)
        db.set_vocabulary_version(self.conn, name, version)
        self.refresh_profile(spec)

    def refresh_profile(self, spec: CollectionSpec) -> dict[str, int]:
        """How many items carry each tag, most first, over every item; saved as the
        collection's profile when its build keeps one."""
        counts = Counter(t for batch in self.store.scroll(spec.name) for h in batch for t in h.tags)
        profile = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        if build_of(spec).profile:
            db.set_profile(self.conn, spec.name, profile)
        return profile
