"""The load and import jobs: index a dataset or an upload, then give the collection lists
or run its build steps."""

import random
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from aiconnect.ai import AI
from ragqa_core import db, imports
from ragqa_core.builder import Builder, build_of, builds
from ragqa_core.connection import Connection
from ragqa_core.datasets import load as load_dataset
from ragqa_core.indexer import Indexer
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item, Question

PROPOSE_SAMPLE = 40
UNUSABLE_PROPOSAL = "none (ai proposal unusable)"


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
    """The items a load shows the AI when it proposes the keyword lists."""
    return random.Random(472).sample(items, min(PROPOSE_SAMPLE, len(items)))


def without_leaked(q: Question) -> Question:
    return replace(q, meta={k: v for k, v in q.meta.items() if k != "leaked"})


class Loader:
    def __init__(self, conn: Connection, indexer: Indexer, builder: Builder, ai: AI, home: Path):
        self.conn = conn
        self.indexer = indexer
        self.builder = builder
        self.ai = ai
        self.home = home

    def run(self, job: dict, progress: Callable[[float], None]) -> dict:
        """Load the job's dataset. `progress(p)` records and logs the job's progress and may raise
        Cancelled. Adding is the whole of it, or the first half when the load makes the lists
        or builds: re-keying every item with the lists, or the build steps, are the second.
        Returns the result dict."""
        conn, name = self.conn, job["collection"]
        spec = self.indexer.spec(name)
        if spec is None:
            raise ValueError(f"unknown collection {name!r}")

        items, questions = load_dataset(job["setup"]["dataset"], self.home)
        items, questions = limited(items, questions, job["setup"].get("limit"))
        dataset = job["setup"]["dataset"]
        source, build_result, counts = self._add(spec, items, dataset, progress)
        db.save_questions(conn, name, questions)
        fresh = db.get_collection(conn, name)
        return {
            "added": counts["new"] + counts["updated"],
            **counts,
            "questions": len(questions),
            "leaked": self.mark_leaked(name),
            "lists_source": source,
            "keyword_version": fresh.keyword_version if fresh else None,
            **build_result,
        }

    def _add(
        self,
        spec: CollectionSpec,
        items: list[Item],
        source: str,
        progress: Callable[[float], None],
    ) -> tuple[str | None, dict, dict[str, int]]:
        """Add the items, then make the lists or run the build: (where the lists came
        from, the build's result, the add counts). Only the items this adds are built."""
        makes_lists, building = self.makes_lists(spec), builds(spec)
        todo, _ = self.indexer.changed(spec, items, source)
        share = 0.5 if makes_lists or building else 1.0
        counts = self.indexer.add(spec, items, source, progress=lambda p: progress(share * p))
        lists_source = None
        if makes_lists:
            lists_source = self._make_lists(spec, items, lambda p: progress(0.5 + 0.5 * p))
        build_result = self.builder.build(spec, todo, lambda p: progress(0.5 + 0.5 * p))
        return lists_source, build_result, counts

    def makes_lists(self, spec: CollectionSpec) -> bool:
        """Whether adding items also gives the collection lists: only a recipe that uses
        the three lists, and only while it has none. The induced recipes never use them."""
        return (
            build_of(spec).vocabulary == "lists"
            and db.get_keyword_lists(self.conn, spec.name) is None
        )

    def _make_lists(
        self, spec: CollectionSpec, items: list[Item], progress: Callable[[float], None]
    ) -> str | None:
        """Give the collection the AI's proposed lists, and say where they came from. With no
        model for keywords it gets none (spec §15: no TF-IDF fallback) and the source is
        None; a paid reply without all three lists is named, so its charge is explained.
        Either way its lists can be proposed or written later."""
        if not self.ai.ready("keywords"):
            progress(1.0)  # the step is over, with nothing to re-key
            return None
        try:
            lists = self.indexer.propose_lists(spec, propose_sample(items))
        except ValueError:
            progress(1.0)
            return UNUSABLE_PROPOSAL
        self.indexer.set_lists(spec, lists, progress=progress)
        return "ai"

    def estimate_load(self, spec: CollectionSpec, dataset: str, limit: int | None) -> dict:
        """What a dataset load would spend. The dataset is read (and cached) to price it."""
        items, questions = load_dataset(dataset, self.home)
        items, _ = limited(items, questions, limit)
        return self._estimate(spec, items, dataset)

    def _estimate(self, spec: CollectionSpec, items: list[Item], source: str) -> dict:
        todo, _ = self.indexer.changed(spec, items, source)
        estimate = self.indexer.estimate_add_usd(spec, items, source)
        calls = len(todo) if spec.summarize else 0
        if self.ai.ready("keywords") and self.makes_lists(spec):
            propose = self.indexer.estimate_propose_usd(spec, propose_sample(items))
            estimate = None if estimate is None else estimate + propose
            calls += 1
        build_usd = self.builder.estimate_build_usd(spec, todo)
        estimate = None if estimate is None or build_usd is None else estimate + build_usd
        calls += self.builder.build_calls(spec, todo)
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
        lists_source, build_result = None, {}
        if setup["kind"] == "items":
            lists_source, build_result, counts = self._add(spec, built, setup["source"], progress)
        else:
            counts, todo = self._question_counts(name, built)
            db.save_questions(conn, name, todo)
        progress(1.0)
        return {
            **counts,
            "skipped": skipped,
            "leaked": self.mark_leaked(name),
            "lists_source": lists_source,
            **build_result,
        }

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
