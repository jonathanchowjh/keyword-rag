"""The load, import and copy jobs: index a dataset, an upload or another collection's items,
then give the collection lists or run its build steps."""

import random
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from aiconnect.ai import AI
from ragqa_core import db, imports
from ragqa_core.builder import Builder, as_item, build_of, builds
from ragqa_core.connection import Connection
from ragqa_core.datasets import load as load_dataset
from ragqa_core.hub import Hub
from ragqa_core.indexer import Indexer
from ragqa_core.recipes import build_for, build_settings
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
    """propose_sample over the stored items."""
    return propose_sample([as_item(h) for batch in store.scroll(name, skip=skip) for h in batch])


def propose_sample(items: list[Item]) -> list[Item]:
    """The items a load shows the AI when it proposes the keyword lists."""
    return random.Random(472).sample(items, min(PROPOSE_SAMPLE, len(items)))


def stored_exclusions(q: Question, stored: set[str]) -> Question:
    """The question with only the excluded ids the collection stores (BRIGHT lists
    thousands per question, most of them outside a limited load)."""
    if "excluded_ids" not in q.meta:
        return q
    kept = [i for i in q.meta["excluded_ids"] if i in stored]
    return replace(q, meta={**q.meta, "excluded_ids": kept})


def without_leaked(q: Question) -> Question:
    return replace(q, meta={k: v for k, v in q.meta.items() if k != "leaked"})


class Loader:
    def __init__(
        self,
        conn: Connection,
        indexer: Indexer,
        builder: Builder,
        ai: AI,
        home: Path,
        hub_repo: str = "",
    ):
        """`hub_repo` names the Hub repo of published builds; empty, a load never reads it."""
        self.conn = conn
        self.indexer = indexer
        self.builder = builder
        self.ai = ai
        self.home = home
        self.hub = Hub(hub_repo, home / "hub") if hub_repo else None

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
        limit = job["setup"].get("limit")
        items, questions = limited(items, questions, limit)
        dataset = job["setup"]["dataset"]
        published, hub = self.from_hub(spec, items, dataset, limit)
        held = job["setup"].get("hub_entry")
        if held and hub.get("entry") != held:
            # Queued unconfirmed because this entry made it free: nothing may be spent instead.
            why = hub.get("skipped") or f"the Hub now has {hub.get('entry')}"
            raise ValueError(
                f"the Hub entry {held} that made this load free is not usable now ({why});"
                " nothing was added; load again to see what it would spend"
            )
        if "skipped" in hub:
            db.log_job(conn, job["id"], f"Hub: {hub['skipped']}; building locally")
        source, build_result, counts = self._add(spec, items, dataset, progress, published)
        if hub:
            build_result = {**build_result, "hub": hub}
        # EVERY LOAD RE-READS THE DATASET'S FULL EXCLUDED LISTS, so a later, larger load
        # brings back an excluded id it now stores; the rest only slow every search.
        stored = set(self.indexer.store.ids(name))
        db.save_questions(conn, name, [stored_exclusions(q, stored) for q in questions])
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
        published: dict | None = None,
    ) -> tuple[str | None, dict, dict[str, int]]:
        """Add the items, then make the lists or run the build: (where the lists came
        from, the build's result, the add counts). Only the items this adds are built; with
        a published build that covers them all, it is applied instead, with no AI call."""
        makes_lists, building = self.makes_lists(spec), builds(spec)
        todo, _ = self.indexer.changed(spec, items, source)
        share = 0.5 if makes_lists or building else 1.0
        counts = self.indexer.add(spec, items, source, progress=lambda p: progress(share * p))
        lists_source = None
        if makes_lists:
            lists_source = self._make_lists(spec, items, lambda p: progress(0.5 + 0.5 * p))
        if published is not None:
            build_result = self.builder.apply_outputs(
                spec, published["terms"], published["outputs"]
            )
            progress(1.0)
        else:
            build_result = self.builder.build(spec, todo, lambda p: progress(0.5 + 0.5 * p))
        return lists_source, build_result, counts

    def from_hub(
        self, spec: CollectionSpec, items: list[Item], dataset: str, limit: int | None
    ) -> tuple[dict | None, dict]:
        """The published build to apply after adding the items, and what the load's result
        says of it: {entry, items}, or {skipped: why}; {} when building what the load adds
        needs no AI call (the recipe never builds, or it is all built), so the Hub is not read.
        ALL OR NOTHING: it is applied only when it covers exactly the items the collection
        will hold, built with its build settings, so no collection mixes two authors'
        vocabularies. Checked before any item is added: an unusable entry means a local
        build, not a failed load. ONLY A COLLECTION WITH NO VOCABULARY IN FORCE takes one:
        an entry never replaces a vocabulary built, authored or edited here."""
        todo, _ = self.indexer.changed(spec, items, dataset)
        if not self.builder.build_calls(spec, todo):
            return None, {}
        if db.get_vocabulary(self.conn, spec.name):
            return None, {"skipped": "the collection has a vocabulary of its own"}
        if self.hub is None:
            return None, {"skipped": "Hub off"}
        published, reason = self.hub.fetch(dataset, limit, spec.recipe)
        if published is None:
            return None, {"skipped": reason}
        if published["entry"].get("build") != asdict(build_of(spec)):
            return None, {"skipped": "the entry was built with other build settings"}
        every = {i.id: i for i in self.builder.stored_items(spec.name)}
        every.update({i.id: i for i in items})
        try:
            self.builder.checked_outputs(
                spec, list(every.values()), published["terms"], published["outputs"]
            )
        except ValueError as e:
            return None, {"skipped": f"the entry is unusable: {e}"}
        return published, {"entry": published["entry"]["id"], "items": len(every)}

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
        """What a dataset load would spend, with `hub` as its result will report it. The
        dataset is read (and cached) to price it."""
        items, questions = load_dataset(dataset, self.home)
        items, _ = limited(items, questions, limit)
        published, hub = self.from_hub(spec, items, dataset, limit)
        return {**self._estimate(spec, items, dataset, built=published is not None), "hub": hub}

    def _estimate(
        self, spec: CollectionSpec, items: list[Item], source: str, built: bool = False
    ) -> dict:
        """`built`: a published build will be applied, so the build steps cost nothing."""
        todo, _ = self.indexer.changed(spec, items, source)
        estimate = self.indexer.estimate_add_usd(spec, items, source)
        calls = len(todo) if spec.summarize else 0
        if self.ai.ready("keywords") and self.makes_lists(spec):
            propose = self.indexer.estimate_propose_usd(spec, propose_sample(items))
            estimate = None if estimate is None else estimate + propose
            calls += 1
        if not built:
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

    def copy_spec(
        self, source: CollectionSpec, name: str, recipe: str, override: dict | None
    ) -> CollectionSpec:
        """What a copy of the source under the recipe would be, not yet saved: the source's
        qa type and the recipe's build with the override merged in. Raises ValueError for a
        bad name, an unknown recipe or a bad override."""
        build = build_settings(build_for(recipe), override)
        return self.indexer.new_spec(name, source.qa_type, False, recipe, build)

    def estimate_copy(self, source: str, target: CollectionSpec) -> dict:
        """What copying the source's items into the target would spend: all of them for a
        target not yet made; for one a stopped copy left, only what it did not finish.
        Embedding is free: the target's summaries and build steps are what cost."""
        items = self.builder.stored_items(source)
        if self.indexer.store.exists(target.name):
            have = {h.id for batch in self.indexer.store.scroll(target.name) for h in batch}
            items = [i for i in items if i.id not in have]  # unbuilt stored ones: build's own
        estimate = self.indexer.estimate_summaries_usd(items) if target.summarize else 0.0
        build_usd = self.builder.estimate_build_usd(target, items)
        estimate = None if estimate is None or build_usd is None else estimate + build_usd
        calls = (len(items) if target.summarize else 0) + self.builder.build_calls(target, items)
        return {"est_usd": estimate, "items": len(items), "calls": calls}

    def start_copy(self, source: str, target: CollectionSpec) -> str:
        """Make the target with the source's questions, switched-off sources and leaked
        setting, and queue the copy job that fills it. Returns the job's id."""
        conn = self.conn
        self.indexer.create(
            target.name, target.qa_type, target.summarize, target.recipe, target.build
        )
        # Meta as it is: the copy stores the same item ids, so gold and excluded ids hold.
        db.save_questions(conn, target.name, db.get_questions(conn, source))
        for off in db.disabled_sources(conn, source):
            db.set_source_enabled(conn, target.name, off, False)
        db.set_include_leaked(conn, target.name, db.get_include_leaked(conn, source))
        return db.create_job(conn, "copy", target.name, {"source": source})

    def run_copy(self, job: dict, progress: Callable[[float], None]) -> dict:
        """Add the source collection's items to the job's, source by source, then build
        them. A target that uses keyword lists takes the source's lists in force first, so
        adding keys the items with them; it proposes none. Adding is the first half of the
        progress when the target builds. Returns the result dict."""
        conn, name, source = self.conn, job["collection"], job["setup"]["source"]
        spec = self.indexer.spec(name)
        if spec is None or self.indexer.spec(source) is None:
            raise ValueError(f"cannot copy {source!r} into {name!r}: no such collection")
        lists = db.get_keyword_lists(conn, source)
        if build_of(spec).vocabulary == "lists" and lists is not None:
            db.set_keyword_version(conn, name, self.indexer.save_lists(spec, lists))
        items = self.builder.stored_items(source)
        by_source: dict[str, list[Item]] = {}
        for item in items:
            by_source.setdefault(item.meta["source"], []).append(item)
        share = 0.5 if builds(spec) else 1.0
        added = done = 0
        for each, part in sorted(by_source.items()):

            def step(p: float, done: int = done, size: int = len(part)) -> None:
                progress(share * (done + p * size) / len(items))

            counts = self.indexer.add(spec, part, each, progress=step)
            added += counts["new"] + counts["updated"]
            done += len(part)
        build_result = self.builder.build(spec, items, lambda p: progress(share + (1 - share) * p))
        progress(1.0)
        return {"added": added, "sources": sorted(by_source), **build_result}

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
