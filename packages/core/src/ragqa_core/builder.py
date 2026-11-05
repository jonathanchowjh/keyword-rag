"""The build steps of the induced recipes (spec §15): a collection's vocabulary,
each item's tags from it (the kw_tags vector) with the profile they add up to, and
each worked example's abstracted form (D4). `build` runs them after items are added.

Every prompt here is written from an item's text or explanation only, never its
meta: ScienceQA's gold skill must not reach the vocabulary (spec §12).
"""

import re
from collections import Counter
from collections.abc import Callable

import numpy as np

from aiconnect.ai import AI, NoConnection
from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.indexer import BATCH, Indexer
from ragqa_core.keywords import keyword_string
from ragqa_core.models import Models
from ragqa_core.prompts import abstract_prompt, induce_prompt, parse_json, tag_prompt
from ragqa_core.recipes import build_for
from ragqa_core.settings import load_settings
from ragqa_core.store import Store
from ragqa_core.types import BuildSettings, CollectionSpec, Hit, Item
from ragqa_core.vocabulary import (
    MAX_TERM_CHARS,
    MAX_TERMS,
    clean_terms,
    describe_text,
    merge_terms,
    terms_version,
)

INDUCE_BATCH = 20
INDUCE_TOKENS = 300
TAG_TOKENS = 100
ABSTRACT_TOKENS = 300
CUTOFF_MOST = 10  # a cut-off keeps at most this many tags
# BEFORE INDUCTION THE TERMS ARE UNKNOWN: AI tagging is priced as if the vocabulary had
# as many terms as an induced one keeps, each as long as a term may be, so the estimate
# errs high, never low.
GUESSED_TERMS = [f"guessed term {n} ".ljust(MAX_TERM_CHARS, "x") for n in range(MAX_TERMS)]
ABSTRACTED = re.compile(r"Principle: (.*?)\nMethod: (.*)\nAnswer: (.*)", re.S)


def build_of(spec: CollectionSpec) -> BuildSettings:
    return spec.build or build_for("paper_2024", spec.summarize)


def builds(spec: CollectionSpec) -> bool:
    """Whether adding items runs the build steps: every recipe but paper 2024."""
    return spec.recipe != "paper_2024"


def picked_terms(text: str, terms: list[str], k: int) -> list[str]:
    """The vocabulary terms an AI reply of {"terms": [...]} picked, at most k; none when
    it named no term of the list."""
    replied = parse_json(text).get("terms")
    picked = clean_terms(replied) if isinstance(replied, list) else []
    known = set(terms)
    return [t for t in picked if t in known][:k]


def abstracted_text(reply: dict) -> str | None:
    """The stored form, "Principle: …\nMethod: …\nAnswer: …", of the AI's reply; None
    when a part is missing or empty. Steps given as a list are numbered on one line."""
    principle, method, answer = (reply.get(k) for k in ("principle", "method", "answer"))
    if isinstance(method, list) and all(isinstance(s, str) for s in method):
        method = " ".join(f"{n}. {s.strip()}" for n, s in enumerate(method, start=1))
    if isinstance(answer, int | float) and not isinstance(answer, bool):
        answer = str(answer)
    parts = [principle, method, answer]
    if not all(isinstance(p, str) and p.strip() for p in parts):
        return None
    principle, method, answer = (str(p).strip() for p in parts)
    return f"Principle: {principle}\nMethod: {method}\nAnswer: {answer}"


def abstracted_parts(text: str | None) -> dict[str, str | None]:
    """The principle, method and answer of a stored form (abstracted_text read back); all
    None without one."""
    found = ABSTRACTED.fullmatch(text or "")
    if not found:
        return {"principle": None, "method": None, "answer": None}
    principle, method, answer = found.groups()
    return {"principle": principle, "method": method, "answer": answer}


def as_item(h: Hit) -> Item:
    return Item(h.id, h.text, h.answer, h.explanation, h.meta)


def make_builder() -> "Builder":
    """The objects the service builds, from the same settings: for the owner's tools."""
    settings = load_settings()
    conn = db.connect(settings.home / "ragqa.db")
    store = Store(settings.qdrant_url, None if settings.qdrant_url else settings.home / "qdrant")
    models = Models(
        settings.dense_model, settings.judge_model, settings.rerank_model, settings.home / "models"
    )
    ai = AI(settings.home / "ragqa.db")
    return Builder(conn, store, models, ai, Indexer(conn, store, models, ai))


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
        return self.ai.estimate_many(
            self._induce_prompts(spec, items), INDUCE_TOKENS, role="keywords"
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

    def similar(
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
                tags += self.similar(terms, texts[start : start + BATCH], build.tag_k, cutoff)
                progress(min(start + BATCH, len(texts)) / len(texts))
            return tags
        # One call for every text, so a large corpus gets the Batch API's discount.
        done = self.ai.complete_many(
            [tag_prompt(terms, t, build.tag_k) for t in texts],
            role="keywords",
            max_tokens=TAG_TOKENS,
            purpose="tag",
        )
        tags = [picked_terms(reply.text, terms, build.tag_k) for reply in done]
        # A reply with no term of the vocabulary falls back to similarity: never no tags.
        unusable = [n for n, t in enumerate(tags) if not t]
        nearest = self.similar(terms, [texts[n] for n in unusable], build.tag_k)
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
        prompts = [tag_prompt(terms, describe_text(i, build.describes), build.tag_k) for i in items]
        return self.ai.estimate_many(prompts, TAG_TOKENS, role="keywords")

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
        self._write_tags(spec, [i.id for i in items], tags, lambda p: progress(0.5 + p / 2))

    def _write_tags(
        self,
        spec: CollectionSpec,
        ids: list[str],
        tags: list[list[str]],
        progress: Callable[[float], None] = lambda p: None,
    ) -> None:
        """Each item's tags and their kw_tags vector, a batch at a time."""
        for start in range(0, len(ids), BATCH):
            part = tags[start : start + BATCH]
            vectors = self.models.embed([keyword_string(t, None) for t in part])
            self.store.set_tags(spec.name, ids[start : start + BATCH], part, vectors)
            progress(min(start + BATCH, len(ids)) / len(ids))
        if not ids:
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
        """Every item the collection stores; none before its first add."""
        if not self.store.exists(name):
            return []
        return [as_item(h) for batch in self.store.scroll(name) for h in batch]

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

    def apply_outputs(
        self, spec: CollectionSpec, terms: list[str], outputs: dict[str, dict]
    ) -> dict:
        """Authored build outputs, put in place with no AI call. `terms` becomes the
        vocabulary in force; `outputs` maps every stored item's id to its tags (terms of
        that vocabulary) and, for a worked example with a solution, its principle, method
        and answer. Afterwards nothing is left to build, so a later load needs no model.

        THE WHOLE SET IS CHECKED BEFORE ANYTHING IS WRITTEN, so a bad file changes nothing:
        ValueError for an unknown or missing id, an item without tags, a tag outside the
        vocabulary, or an abstracted form that is not usable."""
        self.indexer.check_model(spec)  # before any tag is embedded with another model
        vocabulary, tags, abstracted = self.checked_outputs(
            spec, self.stored_items(spec.name), terms, outputs
        )
        version = self.save_vocabulary(spec, vocabulary)
        order = sorted(tags)
        self._write_tags(spec, order, [tags[i] for i in order])
        self.store.set_payloads(spec.name, abstracted)
        # Switched to last, as a retag does: a stop part-way leaves the old one in force.
        db.set_vocabulary_version(self.conn, spec.name, version)
        self.refresh_profile(spec)
        return {"vocabulary_version": version, "tagged": len(tags), "abstracted": len(abstracted)}

    def checked_outputs(
        self, spec: CollectionSpec, items: list[Item], terms: list[str], outputs: dict[str, dict]
    ) -> tuple[list[str], dict[str, list[str]], dict[str, dict]]:
        """(vocabulary, tags by id, abstracted payloads by id) of outputs that cover exactly
        `items`, or ValueError naming the first problem. Writes nothing, so a load can check a
        published build before it adds a single item."""
        vocabulary = clean_terms(terms)
        if not vocabulary:
            raise ValueError("the vocabulary is empty")
        ids = {i.id for i in items}
        unknown = sorted(set(outputs) - ids)
        if unknown:
            raise ValueError(
                f"{len(unknown)} outputs not in collection '{spec.name}': {unknown[:5]}"
            )
        missing = sorted(ids - set(outputs))
        if missing:
            raise ValueError(f"no output for {len(missing)} items: {missing[:5]}")
        abstractable = (
            {i.id for i in self._abstractable(spec, items)} if build_of(spec).abstracted else set()
        )
        known = set(vocabulary)
        tags: dict[str, list[str]] = {}
        abstracted: dict[str, dict] = {}
        for item_id, out in outputs.items():
            chosen = clean_terms(out.get("tags") or [])
            if not chosen:
                raise ValueError(f"item {item_id} without tags")
            outside = [t for t in chosen if t not in known]
            if outside:
                raise ValueError(f"item {item_id} has tags outside the vocabulary: {outside}")
            tags[item_id] = chosen
            if item_id in abstractable:
                text = abstracted_text(out)
                if not text:
                    raise ValueError(f"item {item_id} has no usable abstracted form")
                abstracted[item_id] = {"abstracted": text}
        return vocabulary, tags, abstracted

    def refresh_profile(self, spec: CollectionSpec) -> dict[str, int]:
        """How many items carry each tag, most first, over every item; saved as the
        collection's profile when its build keeps one."""
        counts = Counter(t for batch in self.store.scroll(spec.name) for h in batch for t in h.tags)
        profile = dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        if build_of(spec).profile:
            db.set_profile(self.conn, spec.name, profile)
        return profile

    def _abstractable(self, spec: CollectionSpec, items: list[Item]) -> list[Item]:
        """The worked examples that have a solution to abstract."""
        if spec.corpus_type != "worked":
            return []
        return [i for i in items if i.explanation and i.explanation.strip()]

    def _abstract_prompts(self, spec: CollectionSpec, items: list[Item]) -> list[str]:
        return [
            abstract_prompt(i.text, i.explanation or "", i.answer)
            for i in self._abstractable(spec, items)
        ]

    def estimate_abstract_usd(self, spec: CollectionSpec, items: list[Item]) -> float:
        """Raises NoConnection("keywords") when no model can price it."""
        return self.ai.estimate_many(
            self._abstract_prompts(spec, items), ABSTRACT_TOKENS, role="keywords"
        )

    def abstract(
        self,
        spec: CollectionSpec,
        items: list[Item],
        progress: Callable[[float], None] = lambda p: None,
    ) -> int:
        """Each worked example with a solution, rewritten once as principle, method and
        answer and stored as its payload `abstracted`. An unusable reply stores nothing,
        so the prompt shows that example in full. Returns how many were stored."""
        todo = self._abstractable(spec, items)
        if not todo:
            progress(1.0)
            return 0
        done = self.ai.complete_many(
            self._abstract_prompts(spec, todo),
            role="keywords",
            max_tokens=ABSTRACT_TOKENS,
            purpose="abstract",
        )
        written = {}
        for item, reply in zip(todo, done, strict=True):
            text = abstracted_text(parse_json(reply.text))
            if text:
                written[item.id] = {"abstracted": text}
        self.store.set_payloads(spec.name, written)
        progress(1.0)
        return len(written)

    def unbuilt(self, spec: CollectionSpec, items: list[Item]) -> dict[str, list[Item]]:
        """What `build` works on: the given items with every stored item not yet built, so a
        build that stopped part-way is finished by the next one. "induce": every item, while
        no vocabulary is in force (else none); "tag": those without tags; "abstract": the
        worked examples with a solution and no abstracted form, when the build makes them.
        Before the items are added it is what adding them would leave to build; before the
        collection is made (a copy's estimate) nothing is stored."""
        given = {i.id: i for i in items}
        stored = self.store.scroll(spec.name) if self.store.exists(spec.name) else []
        hits = [h for batch in stored for h in batch if h.id not in given]
        every = [*(as_item(h) for h in hits), *items]
        unabstracted = [*(as_item(h) for h in hits if h.abstracted is None), *items]
        return {
            "induce": [] if db.get_vocabulary(self.conn, spec.name) else every,
            "tag": [*(as_item(h) for h in hits if not h.tags), *items],
            "abstract": self._abstractable(spec, unabstracted) if build_of(spec).abstracted else [],
        }

    def estimate_build_usd(self, spec: CollectionSpec, items: list[Item]) -> float | None:
        """What `build` would spend after adding the items: induction while the collection
        has no vocabulary, then tagging and abstraction of what is not yet built. None when
        no model for keywords can price it: adding is then refused, as it is for summaries."""
        if not builds(spec):
            return 0.0
        todo = self.unbuilt(spec, items)
        terms = db.get_vocabulary(self.conn, spec.name)
        try:
            return (
                self.estimate_induce_usd(spec, todo["induce"])
                + self.estimate_tag_usd(spec, terms or GUESSED_TERMS, todo["tag"])
                + self.estimate_abstract_usd(spec, todo["abstract"])
            )
        except NoConnection:
            return None

    def build_calls(self, spec: CollectionSpec, items: list[Item]) -> int:
        """How many AI calls `build` would make after adding the items."""
        if not builds(spec):
            return 0
        todo = self.unbuilt(spec, items)
        calls = len(self._induce_prompts(spec, todo["induce"])) if todo["induce"] else 0
        calls += len(todo["tag"]) if build_of(spec).tagging == "ai" else 0
        return calls + len(todo["abstract"])

    def build(
        self,
        spec: CollectionSpec,
        items: list[Item],
        progress: Callable[[float], None] = lambda p: None,
    ) -> dict:
        """The build after adding items, for every recipe but paper 2024 (which gets {}),
        over the added items and any stored item an earlier build left unbuilt: induce and
        switch to a vocabulary while there is none, tag, abstract the worked examples when
        the build does, refresh the profile. An induction with no usable term leaves the
        items untagged; a vocabulary can be saved later."""
        if not builds(spec):
            return {}
        todo = self.unbuilt(spec, items)
        if not any(todo.values()):
            progress(1.0)
            fresh = db.get_collection(self.conn, spec.name)
            return {
                "vocabulary_version": fresh.vocabulary_version if fresh else None,
                "tagged": 0,
                "abstracted": 0,
            }
        self.indexer.check_model(spec)
        induces = bool(todo["induce"])
        abstracts = bool(todo["abstract"])
        steps = 1 + induces + abstracts

        def step(n: int) -> Callable[[float], None]:
            return lambda p: progress((n + p) / steps)

        if induces:
            terms = self.induce(spec, todo["induce"], step(0))
            # SWITCHED BEFORE TAGGING: tagging reads the vocabulary in force, not the spec.
            if terms:
                db.set_vocabulary_version(self.conn, spec.name, self.save_vocabulary(spec, terms))
        terms = db.get_vocabulary(self.conn, spec.name)
        if terms:
            self._tag(spec, terms, todo["tag"], step(induces))
        abstracted = self.abstract(spec, todo["abstract"], step(induces + 1)) if abstracts else 0
        self.refresh_profile(spec)
        progress(1.0)
        fresh = db.get_collection(self.conn, spec.name)
        return {
            "vocabulary_version": fresh.vocabulary_version if fresh else None,
            "tagged": len(todo["tag"]) if terms else 0,
            "abstracted": abstracted,
        }
