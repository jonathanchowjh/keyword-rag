"""The build steps of the induced recipes (spec §15): a collection's vocabulary.

Every prompt here is written from an item's text or explanation only, never its
meta: ScienceQA's gold skill must not reach the vocabulary (spec §12).
"""

from collections.abc import Callable

from aiconnect.ai import AI
from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.indexer import Indexer
from ragqa_core.models import Models
from ragqa_core.prompts import induce_prompt, parse_json
from ragqa_core.recipes import build_for
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item
from ragqa_core.vocabulary import clean_terms, describe_text, merge_terms, terms_version

INDUCE_BATCH = 20
INDUCE_TOKENS = 300
AI_BATCH_OVER = 50  # more calls than this go to a provider's Batch API, as summaries do


class Builder:
    def __init__(self, conn: Connection, store: Store, models: Models, ai: AI, indexer: Indexer):
        self.conn = conn
        self.store = store
        self.models = models
        self.ai = ai
        self.indexer = indexer

    def _induce_prompts(self, spec: CollectionSpec, items: list[Item]) -> list[str]:
        build = spec.build or build_for("paper_2024", spec.summarize)
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
