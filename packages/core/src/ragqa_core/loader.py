"""The dataset-load job: read a dataset, index it, and give the collection lists."""

import random
from collections.abc import Callable
from pathlib import Path

from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.datasets import load as load_dataset
from ragqa_core.indexer import Indexer
from ragqa_core.keywords import tfidf_terms
from ragqa_core.llm import Claude
from ragqa_core.store import Store
from ragqa_core.types import Item, KeywordLists, Question

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


def sample_items(store: Store, name: str) -> list[Item]:
    hits = [h for batch in store.scroll(name) for h in batch]
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


class Loader:
    def __init__(self, conn: Connection, indexer: Indexer, claude: Claude, home: Path):
        self.conn = conn
        self.indexer = indexer
        self.claude = claude
        self.home = home

    def run(self, job: dict, check: Callable[[], None]) -> dict:
        """Load the job's dataset; `check` may raise Cancelled. Returns the result dict."""
        conn, name = self.conn, job["collection"]
        spec = self.indexer.spec(name)
        if spec is None:
            raise ValueError(f"unknown collection {name!r}")

        def progress(p: float) -> None:
            check()
            db.update_job(conn, job["id"], progress=0.9 * p)

        items, questions = load_dataset(job["setup"]["dataset"], self.home)
        items, questions = limited(items, questions, job["setup"].get("limit"))
        self.indexer.add(spec, items, progress=progress)
        db.save_questions(conn, name, questions)
        source = None
        if db.get_keyword_lists(conn, name) is None:
            sample = propose_sample(items)
            lists, source = None, "tfidf"
            if self.claude.api_key:
                try:
                    lists, source = self.indexer.propose_lists(spec, sample), "claude"
                except ValueError:
                    source = "tfidf (claude proposal unusable)"
            if lists is None:
                lists = tfidf_lists([i.text for i in items])
            if lists is not None:
                self.indexer.set_lists(spec, lists)
            else:
                source = None
        fresh = db.get_collection(conn, name)
        return {
            "added": len(items),
            "questions": len(questions),
            "lists_source": source,
            "keyword_version": fresh.keyword_version if fresh else None,
        }
