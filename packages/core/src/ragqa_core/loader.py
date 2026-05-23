"""The load and import jobs: index a dataset or an upload and give the collection lists."""

import random
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from ragqa_core import db, imports
from ragqa_core.connection import Connection
from ragqa_core.datasets import load as load_dataset
from ragqa_core.indexer import Indexer
from ragqa_core.keywords import tfidf_terms
from ragqa_core.llm import Claude
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item, KeywordLists, Question

PROPOSE_SAMPLE = 40


def limited(
    items: list[Item], questions: list[Question], limit: int | None
) -> tuple[list[Item], list[Question]]:
    """At most `limit` items, chosen so questions stay answerable: each
    question's gold passages first, in question order, while they fit; then
    the rest by id. Questions whose gold was left out are dropped."""
    if not limit:
        return items, questions
    chosen: set[str] = set()
    for q in questions:
        extra = set(q.gold_ids) - chosen
        if len(chosen) + len(extra) <= limit:
            chosen |= extra
    for item in sorted(items, key=lambda i: i.id):
        if len(chosen) >= limit:
            break
        chosen.add(item.id)
    kept = sorted((i for i in items if i.id in chosen), key=lambda i: i.id)
    return kept, [q for q in questions if set(q.gold_ids) <= chosen]


def sample_items(store: Store, name: str, skip: tuple[str, ...] = ()) -> list[Item]:
    hits = [h for batch in store.scroll(name, skip=skip) for h in batch]
    picked = random.Random(472).sample(hits, min(PROPOSE_SAMPLE, len(hits)))
    return [Item(h.id, h.text, h.answer, h.explanation, h.meta) for h in picked]


def propose_sample(items: list[Item]) -> list[Item]:
    """The items a load shows Claude when it proposes the keyword lists."""
    return random.Random(472).sample(items, min(PROPOSE_SAMPLE, len(items)))


def tfidf_lists(texts: list[str]) -> KeywordLists | None:
    """Lists without Claude: the corpus's own top terms. A small corpus has too
    few terms to fill all three facets, so a facet that would be empty reuses
    the head of the ranking rather than leaving its keyword vector undefined."""
    terms = tfidf_terms(texts, n=565)
    if not terms:
        return None
    return KeywordLists(
        topic=terms[:50],
        method=terms[50:65] or terms[:15],
        common=terms[65:565] or terms,
    )


def without_leaked(q: Question) -> Question:
    return replace(q, meta={k: v for k, v in q.meta.items() if k != "leaked"})


class Loader:
    def __init__(self, conn: Connection, indexer: Indexer, claude: Claude, home: Path):
        self.conn = conn
        self.indexer = indexer
        self.claude = claude
        self.home = home

    def run(self, job: dict, progress: Callable[[float], None]) -> dict:
        """Load the job's dataset. `progress(p)` records and logs the job's progress and may raise
        Cancelled. Adding is the whole of it, or the first half when the load makes the lists:
        then re-keying every item with them is the second. Returns the result dict."""
        conn, name = self.conn, job["collection"]
        spec = self.indexer.spec(name)
        if spec is None:
            raise ValueError(f"unknown collection {name!r}")

        items, questions = load_dataset(job["setup"]["dataset"], self.home)
        items, questions = limited(items, questions, job["setup"].get("limit"))
        makes_lists = db.get_keyword_lists(conn, name) is None
        share = 0.5 if makes_lists else 1.0
        dataset = job["setup"]["dataset"]
        counts = self.indexer.add(spec, items, dataset, progress=lambda p: progress(share * p))
        db.save_questions(conn, name, questions)
        source = None
        if makes_lists:
            source = self._make_lists(spec, items, lambda p: progress(0.5 + 0.5 * p))
        fresh = db.get_collection(conn, name)
        return {
            "added": counts["new"] + counts["updated"],
            **counts,
            "questions": len(questions),
            "leaked": self.mark_leaked(name),
            "lists_source": source,
            "keyword_version": fresh.keyword_version if fresh else None,
        }

    def _make_lists(
        self, spec: CollectionSpec, items: list[Item], progress: Callable[[float], None]
    ) -> str | None:
        """Give the collection lists, from Claude when there is a key, else its own top terms."""
        sample = propose_sample(items)
        lists, source = None, "tfidf"
        if self.claude.api_key:
            try:
                lists, source = self.indexer.propose_lists(spec, sample), "claude"
            except ValueError:
                source = "tfidf (claude proposal unusable)"
        if lists is None:
            lists = tfidf_lists([i.text for i in items])
        if lists is None:
            return None
        self.indexer.set_lists(spec, lists, progress=progress)
        return source

    def estimate_load(self, spec: CollectionSpec, dataset: str, limit: int | None) -> dict:
        """What a dataset load would spend. The dataset is read (and cached) to price it."""
        items, questions = load_dataset(dataset, self.home)
        items, _ = limited(items, questions, limit)
        return self._estimate(spec, items, dataset)

    def _estimate(self, spec: CollectionSpec, items: list[Item], source: str) -> dict:
        todo, _ = self.indexer.changed(spec, items, source)
        estimate = self.indexer.estimate_add_usd(spec, items, source)
        calls = len(todo) if spec.summarize else 0
        if self.claude.api_key and db.get_keyword_lists(self.conn, spec.name) is None:
            estimate += self.indexer.estimate_propose_usd(spec, propose_sample(items))
            calls += 1
        return {"est_usd": estimate, "items": len(items), "calls": calls}

    def _read_upload(self, spec: CollectionSpec, setup: dict) -> tuple[list, int]:
        """What the upload yields under the setup's mapping, and how many rows that leaves out."""
        path, meta = imports.open_upload(self.home, spec.name, setup["upload_id"])
        _, rows, _ = imports.read_rows(path, meta["filename"])
        if setup["kind"] == "items":
            built, _ = imports.build_items(rows, setup["mapping"])
        else:
            built, _ = imports.build_questions(rows, setup["mapping"], spec.qa_type)
            built = [replace(q, meta={**q.meta, "source": setup["source"]}) for q in built]
        return built, len(rows) - len(built)

    def _question_counts(self, name: str, questions: list[Question]) -> tuple[dict, list[Question]]:
        """New, updated, unchanged against the stored ones; a leaked mark is ours."""
        stored = {q.id: q for q in db.get_questions(self.conn, name)}
        counts, todo = {"new": 0, "updated": 0, "unchanged": 0}, []
        for q in questions:
            old = stored.get(q.id)
            if old is None:
                counts["new"] += 1
            elif asdict(q) == asdict(without_leaked(old)):
                counts["unchanged"] += 1
                continue
            else:
                counts["updated"] += 1
            todo.append(q)
        return counts, todo

    def estimate_import(self, spec: CollectionSpec, setup: dict) -> dict:
        built, skipped = self._read_upload(spec, setup)
        if setup["kind"] == "questions":
            counts, _ = self._question_counts(spec.name, built)
            return {"est_usd": 0.0, "calls": 0, **counts, "skipped": skipped}
        _, counts = self.indexer.changed(spec, built, setup["source"])
        return {**self._estimate(spec, built, setup["source"]), **counts, "skipped": skipped}

    def run_import(self, job: dict, progress: Callable[[float], None]) -> dict:
        """Import the job's upload as items (indexed under the source) or as questions."""
        conn, name, setup = self.conn, job["collection"], job["setup"]
        spec = self.indexer.spec(name)
        if spec is None:
            raise ValueError(f"unknown collection {name!r}")
        built, skipped = self._read_upload(spec, setup)
        if setup["kind"] == "items":
            makes_lists = db.get_keyword_lists(conn, name) is None
            share = 0.5 if makes_lists else 1.0
            counts = self.indexer.add(
                spec, built, setup["source"], progress=lambda p: progress(share * p)
            )
            if makes_lists:
                self._make_lists(spec, built, lambda p: progress(0.5 + 0.5 * p))
        else:
            counts, todo = self._question_counts(name, built)
            db.save_questions(conn, name, todo)
        progress(1.0)
        return {**counts, "skipped": skipped, "leaked": self.mark_leaked(name)}

    def mark_leaked(self, name: str) -> int:
        """Mark the imported or hand-written questions whose text sits inside an item of the
        collection; the built-in datasets' questions are never checked. Returns the leaked count."""
        questions = [
            q
            for q in db.get_questions(self.conn, name)
            if str(q.meta.get("source", "")).startswith("upload:") or q.meta.get("source") == "hand"
        ]
        if not questions:
            return 0
        texts = [h.text for batch in self.indexer.store.scroll(name) for h in batch]
        leaked = imports.leaked_ids(questions, texts)
        changed = []
        for q in questions:
            if (q.id in leaked) != bool(q.meta.get("leaked")):
                changed.append(
                    replace(q, meta={**q.meta, "leaked": True})
                    if q.id in leaked
                    else without_leaked(q)
                )
        db.save_questions(self.conn, name, changed)
        return len(leaked)
