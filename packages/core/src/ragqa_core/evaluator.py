"""Evaluation runs: estimate first, then queue and run (or resume), then tabulate
with intervals so a gap that is noise is shown as noise."""

import itertools
import math
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Literal

from aiconnect.ai import (
    AI,
    BATCH_THRESHOLD,
    AwaitingReplies,
    BudgetExceeded,
    Cancelled,
    NoConnection,
)
from ragqa_core import db, imports, verdict
from ragqa_core.builder import TAG_TOKENS, build_of
from ragqa_core.catalog import ANSWER_PAIR, CONTROLS, METRICS
from ragqa_core.connection import Connection
from ragqa_core.datasets import DEV_SEED, FILTER_NOTES, dev_slice
from ragqa_core.metrics import (
    bootstrap_ci,
    cosine,
    ndcg_at,
    recall_at,
    science_correct,
    separated,
    squad_em_f1,
    tatqa_em_f1,
)
from ragqa_core.models import Models
from ragqa_core.program import program_answer
from ragqa_core.prompts import (
    QUERY_TASK,
    SUMMARY_TOKENS,
    answer_line,
    answer_prompt,
    hop_prompt,
    parse_answer,
    query_tag_prompt,
    summary_prompt,
)
from ragqa_core.recipes import search_settings
from ragqa_core.retriever import HOP_TOKENS, KEYWORD_METHODS, MAX_HOPS, Retriever
from ragqa_core.types import EXAMPLE_TYPES, CollectionSpec, Hit, QaType, Question, SearchSettings

# What worked collections whose questions name relevant items (BRIGHT) gain, measured
# GOLD_DEPTH deep: how high the relevant examples rank, and how many of them come first.
GOLD_ID_METRICS = ("ndcg@10", "recall@1")
GOLD_DEPTH = 10
RETRIEVAL_METRICS: dict[QaType, tuple[str, ...]] = {
    "mcq": ("solution_sim@1", "same_skill@1", *GOLD_ID_METRICS),
    "science": ("solution_sim@1", "same_skill@1", *GOLD_ID_METRICS),
    "multihop": ("recall@2", "recall@5"),
    "table": ("recall@1", "recall@3"),
}
ANSWER_METRICS: dict[QaType, tuple[str, ...]] = {
    "mcq": ("accuracy",),
    "science": ("accuracy",),
    "multihop": ("em", "f1"),
    "table": ("em", "f1"),
}
# What counts as a retrieval miss when a run has no answers: a binary metric,
# because the continuous solution similarity is practically never exactly 0.
MISS_METRIC: dict[QaType, str] = {
    "mcq": "same_skill@1",
    "science": "same_skill@1",
    "multihop": "recall@2",
    "table": "recall@1",
}
PAIR_ANSWER_METRIC: dict[QaType, str] = {
    "mcq": "accuracy",
    "science": "accuracy",
    "multihop": "f1",
    "table": "f1",
}
# Multi-step science working runs long; a reply cut off before its Answer line scores wrong.
ANSWER_TOKENS: dict[QaType, int] = {"mcq": 400, "science": 1500, "multihop": 100, "table": 100}
# A plan and a program do not fit in a short answer's budget.
PROGRAM_TOKENS = 600
ESTIMATE_SAMPLE = 5


def clean(value: float) -> float | None:
    return None if math.isnan(value) else value


def answer_tokens(qa_type: QaType, program: bool) -> int:
    return max(ANSWER_TOKENS[qa_type], PROGRAM_TOKENS) if program else ANSWER_TOKENS[qa_type]


def answer_hits(qa_type: QaType, hits: list[Hit], example: str) -> list[Hit]:
    # The paper's one-shot: the single best example for worked types, none when the
    # example is off; passages otherwise.
    if qa_type in EXAMPLE_TYPES:
        return [] if example == "off" else hits[:1]
    return hits


def example_form(qa_type: QaType, shown: list[Hit], example: str) -> str | None:
    """How the worked example was shown: "off" when none was, "full" when the abstracted
    form was asked for and not written; None for passages."""
    if qa_type not in EXAMPLE_TYPES:
        return None
    if not shown:
        return "off"
    return "abstracted" if example == "abstracted" and shown[0].abstracted else "full"


def retrieval_depth(qa_type: QaType, q: Question, k: int) -> int:
    """How many hits a question keeps: k, or GOLD_DEPTH when its gold ids score a worked
    collection's ranking. Passage types keep k, since every hit kept is shown to the answerer."""
    return max(k, GOLD_DEPTH) if qa_type in EXAMPLE_TYPES and q.gold_ids else k


NOTHING_TO_MEASURE = (
    "nothing to measure: this store holds none of the eval set's evidence;"
    " have the AI answer, or choose another store"
)


class EvalSetMismatch(Exception):
    """An eval set and a store of different question types: a run cannot ask one of the other."""


def runnable(questions: list[Question], leaked: set[str], include_leaked: bool) -> list[Question]:
    """The questions a run may ask: leaked ones (text inside the store's items) only when
    included."""
    return questions if include_leaked else [q for q in questions if q.id not in leaked]


def measurable(qa_type: QaType, fields: dict) -> dict[str, dict]:
    """Which of the type's metrics the collection's data can produce, with a note when it
    cannot or only in part. `fields` counts questions (total, solution, skill, gold_ids)
    and items (total, explanation, skill)."""
    q, i = fields["questions"], fields["items"]
    out: dict[str, dict] = {m: {"measurable": True, "note": None} for m in ANSWER_METRICS[qa_type]}
    if qa_type in EXAMPLE_TYPES:
        needs = {
            "solution_sim@1": (
                ("no question has a worked solution", q["solution"]),
                ("no item has an explanation", i["explanation"]),
            ),
            "same_skill@1": (
                ("no question has a skill", q["skill"]),
                ("no item has a skill", i["skill"]),
            ),
        }
        for metric, checks in needs.items():
            note = next((why for why, count in checks if count == 0), None)
            out[metric] = {"measurable": note is None, "note": note}
    named = q["gold_ids"]
    partial = f"on {named} of {q['total']} questions" if 0 < named < q["total"] else None
    if qa_type in EXAMPLE_TYPES:
        # Gold-id metrics are listed only where questions name relevant items.
        for metric in GOLD_ID_METRICS if named else ():
            out[metric] = {"measurable": True, "note": partial}
    else:
        for metric in RETRIEVAL_METRICS[qa_type]:
            if named == 0:
                out[metric] = {"measurable": False, "note": "no question names its evidence"}
            else:
                out[metric] = {"measurable": True, "note": partial}
    return out


def untracked(setup: dict) -> bool:
    """A migrated run: it has question ids and recorded no versions."""
    return "keyword_version" not in setup and "question_ids" in setup


def changes_since(
    setup: dict, spec: CollectionSpec, disabled: list[str], eval_set_version: str | None = None
) -> dict:
    """What differs between a run's recorded setup and the collection now, in plain terms.
    A migrated run records no versions, so it is untracked rather than unchanged."""
    if untracked(setup):
        return {"untracked": True, "changes": []}
    changes = []
    # A setup without the key (a run that never started) has nothing to compare; one that
    # recorded None ran before the collection had lists. Only keyword methods use the lists.
    if (
        "keyword_version" in setup
        and setup["keyword_version"] != spec.keyword_version
        and any(m in KEYWORD_METHODS for m in setup.get("methods", []))
    ):
        changes.append("keyword lists")
    # Only Keyword-RAG searches with the vocabulary; a run from before it recorded none.
    if (
        "vocabulary_version" in setup
        and setup["vocabulary_version"] != spec.vocabulary_version
        and "keyword_rag" in setup.get("methods", [])
    ):
        changes.append("vocabulary")
    if setup.get("dense_model") is not None and setup["dense_model"] != spec.dense_model:
        changes.append("dense model")
    if "keyword_version" in setup and setup.get("disabled_sources", []) != disabled:
        changes.append("sources")
    recorded = setup.get("eval_set_version")
    # None on either side: nothing recorded, or no version to compare it with.
    if None not in (recorded, eval_set_version) and recorded != eval_set_version:
        changes.append("eval set")
    return {"untracked": False, "changes": changes}


class Evaluator:
    def __init__(self, conn: Connection, retriever: Retriever, models: Models, ai: AI) -> None:
        self.conn = conn
        self.retriever = retriever
        self.models = models
        self.ai = ai
        # leaked_ids by (store, eval set): the store's item count and the set's version it was
        # worked out at, and the ids. A count or version that differs means scan again.
        self._leaked: dict[tuple[str, str], tuple[int, str | None, set[str]]] = {}

    def _set_name(self, spec: CollectionSpec, setup: dict) -> str | None:
        """The eval set the run asks: the one it recorded, else the store's default."""
        return setup.get("eval_set") or db.default_eval_set(self.conn, spec.name)

    def _chosen_set(self, spec: CollectionSpec, eval_set: str | None) -> str:
        """The eval set a new run asks: the one named, else the store's default. Refuses a
        store with neither, and a set of another question type."""
        name = eval_set or db.default_eval_set(self.conn, spec.name)
        if name is None:
            raise ValueError(f"store '{spec.name}' has no eval set; choose one")
        found = db.get_eval_set(self.conn, name)
        if found is None:
            raise ValueError(f"no eval set '{name}'")
        if found["qa_type"] != spec.qa_type:
            raise EvalSetMismatch(
                f"eval set '{name}' is {found['qa_type']}; store '{spec.name}' is {spec.qa_type}"
            )
        return name

    def leaked_ids(self, spec: CollectionSpec, eval_set: str) -> set[str]:
        """The set's imported or hand-written questions whose text sits inside one of the
        store's items; the built-in datasets' questions are never checked, so a set of only
        those needs no scan of the store. The scan is kept until the store's item count or the
        set's version changes: the store page asks for it on every poll."""
        found = db.get_eval_set(self.conn, eval_set)
        version = found["version"] if found else None
        count = self.retriever.store.count(spec.name)
        kept = self._leaked.get((spec.name, eval_set))
        if kept is not None and kept[:2] == (count, version):
            return set(kept[2])
        checked = [
            q
            for q in db.get_questions(self.conn, eval_set, version)
            if str(q.meta.get("source", "")).startswith("upload:") or q.meta.get("source") == "hand"
        ]
        leaked: set[str] = set()
        if checked:
            texts = [h.text for batch in self.retriever.store.scroll(spec.name) for h in batch]
            leaked = imports.leaked_ids(checked, texts)
        self._leaked[(spec.name, eval_set)] = (count, version, leaked)
        return set(leaked)

    def forget(self, name: str) -> None:
        """Drops the leak scans of a store that is gone or whose items changed. The keys are
        copied first: a request thread may add a scan meanwhile."""
        for key in [k for k in list(self._leaked) if k[0] == name]:
            self._leaked.pop(key, None)

    def _questions(self, spec: CollectionSpec, setup: dict) -> list[Question]:
        """What a run that has not recorded its questions asks: its set's, less those leaked
        into this store unless the set includes them, then the dev slice of the rest. Leaked
        go first, so the slice is what is asked."""
        name = self._set_name(spec, setup)
        if name is None:
            return []
        questions = self._set_questions(spec, setup)
        if not db.get_include_leaked(self.conn, name):  # included: nothing to drop, no scan
            leaked = setup.get("leaked_ids")
            leaked = self.leaked_ids(spec, name) if leaked is None else leaked
            questions = runnable(questions, set(leaked), False)
        return dev_slice(questions) if setup["split"] == "dev" else questions

    def _question_ids(self, job_id: str, spec: CollectionSpec, setup: dict) -> list[str]:
        if "question_ids" in setup:
            return setup["question_ids"]
        # Not recorded yet: an old run's are the ones it has rows for; a new
        # run's are its split's, as its first run will record them.
        return db.run_item_question_ids(self.conn, job_id) or [
            q.id for q in self._questions(spec, setup)
        ]

    def _asked(self, job_id: str, spec: CollectionSpec, setup: dict) -> list[Question]:
        """The run's questions in recorded order, as its eval set version has them; ids
        since deleted are skipped."""
        by_id = {q.id: q for q in self._set_questions(spec, setup)}
        return [by_id[i] for i in self._question_ids(job_id, spec, setup) if i in by_id]

    def _run_questions(self, job_id: str, spec: CollectionSpec, setup: dict) -> list[Question]:
        """The run's questions with their evidence cut to what the store holds: every
        metric reads these."""
        return self._held(spec, self._asked(job_id, spec, setup))

    def _held(self, spec: CollectionSpec, questions: list[Question]) -> list[Question]:
        """Each question with its evidence ids cut to those the store holds. Evidence the
        store lacks cannot be found in it, so it is unmeasured there, never missed."""
        if not any(q.gold_ids for q in questions):
            return questions
        held = self._held_ids(spec, questions)
        return [replace(q, gold_ids=[g for g in q.gold_ids if g in held]) for q in questions]

    def _held_ids(self, spec: CollectionSpec, questions: list[Question]) -> set[str]:
        """Which of the questions' evidence ids the store holds, in one read of the store."""
        wanted = sorted({g for q in questions for g in q.gold_ids})
        return self.retriever.store.held(spec.name, wanted) if wanted else set()

    def _evidence(self, spec: CollectionSpec, questions: list[Question]) -> tuple[int, int]:
        """How many questions name evidence, and how many of those the store holds any of."""
        held = self._held(spec, questions)
        return sum(1 for q in questions if q.gold_ids), sum(1 for q in held if q.gold_ids)

    def _measures_nothing(self, spec: CollectionSpec, questions: list[Question]) -> bool:
        """Whether a retrieval-only run of these questions would measure nothing: the store
        holds none of their evidence, and no other retrieval metric of the type applies (a
        worked solution against item explanations, or a skill against item skills)."""
        if self._evidence(spec, questions)[1]:
            return False
        if spec.qa_type not in EXAMPLE_TYPES:
            return True
        have = self.retriever.store.field_counts(spec.name)
        solution = have["explanation"] and any(q.meta.get("solution") for q in questions)
        skill = have["skill"] and any(q.meta.get("skill") for q in questions)
        return not (solution or skill)

    def _set_questions(self, spec: CollectionSpec, setup: dict) -> list[Question]:
        """The questions of the run's eval set, as of the version it recorded (the set's
        current ones before it recorded any)."""
        name = self._set_name(spec, setup)
        if name is None:
            return []
        return db.get_questions(self.conn, name, setup.get("eval_set_version"))

    def _eval_set_version(self, spec: CollectionSpec, setup: dict) -> str | None:
        """The current version of the run's eval set."""
        name = self._set_name(spec, setup)
        found = db.get_eval_set(self.conn, name) if name else None
        return found["version"] if found else None

    def _todo(self, job_id: str, method: str, questions: list[Question]) -> list[Question]:
        # A (method, question) pair with no row at all is incomplete too.
        complete = db.complete_question_ids(self.conn, job_id, method)
        return [q for q in questions if q.id not in complete]

    def estimate(
        self,
        collection: str,
        split: Literal["dev", "full"],
        methods: list[str],
        answer: bool,
        model: str | None,
        search: dict | None = None,
        eval_set: str | None = None,
    ) -> dict:
        """What the run would spend, with the counts the run form shows for this (store,
        eval set): its questions, its dev slice, and of the questions asked, those naming
        evidence and those whose evidence the store holds."""
        spec = self.retriever.spec(collection)
        self.retriever.indexer.check_model(spec)
        every = self._questions(
            spec, {"split": "full", "eval_set": self._chosen_set(spec, eval_set)}
        )
        questions = dev_slice(every) if split == "dev" else every
        named, held = self._evidence(spec, questions)
        parts = self._estimate_parts(
            spec,
            {m: len(questions) for m in methods},
            answer,
            model,
            questions[:ESTIMATE_SAMPLE],
            search_settings(spec.search or SearchSettings(), search),
        )
        return {
            "questions": len(every),
            "dev_questions": len(dev_slice(every)),
            "evidence_named": named,
            "evidence_held": held,
            **parts,
        }

    def estimate_resume(self, job_id: str) -> dict:
        """What resuming would cost: only the incomplete (method, question) pairs."""
        job = db.get_job(self.conn, job_id)
        if job is None:
            raise KeyError(job_id)
        setup = job["setup"]
        spec = self.retriever.spec(setup["collection"])
        self.retriever.indexer.check_model(spec)
        questions = self._run_questions(job_id, spec, setup)
        todo = {m: self._todo(job_id, m, questions) for m in setup["methods"]}
        pending = {q.id for qs in todo.values() for q in qs}
        remaining = [q for q in questions if q.id in pending]
        parts = self._estimate_parts(
            spec,
            {m: len(qs) for m, qs in todo.items()},
            setup["answer"],
            setup["model"],
            remaining[:ESTIMATE_SAMPLE],
            self._search(spec, setup),
        )
        return {"questions": len(remaining), **parts}

    def _estimate_parts(
        self,
        spec: CollectionSpec,
        n_by_method: dict[str, int],
        answer: bool,
        model: str | None,
        sample: list[Question],
        search: SearchSettings,
    ) -> dict:
        # (calls, avg prompt chars, output tokens, batched?, role, model) — each part
        # priced the way it is sent: one complete_many per part, batched only
        # above 50 calls; iterative hops are single real-time calls. Summaries, hops and
        # the AI's query tags run on the query job's model; only answers use the requested one.
        parts: list[tuple[int, float, int, bool, str, str | None]] = []
        summaries = max((n for m, n in n_by_method.items() if m in KEYWORD_METHODS), default=0)
        if spec.summarize and summaries:
            chars = avg(len(summary_prompt(q.text)) for q in sample)
            parts.append(
                (summaries, chars, SUMMARY_TOKENS, summaries > BATCH_THRESHOLD, "query", None)
            )
        asked = n_by_method.get("keyword_rag", 0) if search.query_tags == "model" else 0
        if asked:
            terms = db.get_vocabulary(self.conn, spec.name) or []
            k = build_of(spec).tag_k
            chars = avg(len(query_tag_prompt(terms, q.text, k, spec.corpus_type)) for q in sample)
            parts.append((asked, chars, TAG_TOKENS, asked > BATCH_THRESHOLD, "query", None))
        hop_chars = avg(len(hop_prompt(q.text, ["x" * 600] * 4)) for q in sample)
        parts.extend(
            (n * MAX_HOPS, hop_chars, HOP_TOKENS, False, "query", None)
            for m, n in n_by_method.items()
            if m == "iterative"
        )
        if answer and any(n_by_method.values()):
            chars = self._answer_chars(spec, sample, search)
            program = search.program and any(not q.choices for q in sample)
            tokens = answer_tokens(spec.qa_type, program)
            parts.extend(
                (n, chars, tokens, n > BATCH_THRESHOLD, "answer", model)
                for n in n_by_method.values()
            )
        calls = sum(part[0] for part in parts)
        if calls == 0:
            return {"calls": 0, "est_usd": 0.0, "roles": []}
        # Every job the run spends on, so a confirm can be refused before any is unusable.
        roles = list(dict.fromkeys(part[4] for part in parts if part[0]))
        try:
            usd = sum(
                c * self.ai.estimate_usd(int(chars), out, role=role, model=m, batch=batch)
                for c, chars, out, batch, role, m in parts
            )
        except NoConnection:
            return {"calls": calls, "est_usd": None, "roles": roles}  # a job has no model
        return {"calls": calls, "est_usd": usd, "roles": roles}

    def _answer_chars(
        self, spec: CollectionSpec, sample: list[Question], search: SearchSettings
    ) -> float:
        sizes = []
        for q in sample:
            hits = self.retriever.retrieve(spec.name, q.text, "hybrid")
            hits = answer_hits(spec.qa_type, hits, search.example)
            system, user = answer_prompt(
                spec.qa_type, q.text, q.choices, hits, search.example, search.program
            )
            sizes.append(len(system) + len(user))
        return avg(sizes)

    def queue(
        self,
        collection: str,
        split: Literal["dev", "full"],
        methods: list[str],
        answer: bool,
        model: str | None,
        k: int = 5,
        rrf_k: int = 60,
        search: dict | None = None,
        eval_set: str | None = None,
    ) -> str:
        """Queue a run of the eval set (the store's default when None) with the request only;
        the rest of its setup is recorded when it runs. `search` overrides the collection's
        search settings in part; a bad one raises ValueError."""
        spec = self.retriever.spec(collection)
        if not methods:
            raise ValueError("choose at least one method")
        search_settings(spec.search or SearchSettings(), search)
        name = self._chosen_set(spec, eval_set)
        questions = self._questions(spec, {"split": split, "eval_set": name})
        if not questions:
            raise ValueError(f"eval set '{name}' has no questions")
        if not answer and self._measures_nothing(spec, questions):
            raise ValueError(NOTHING_TO_MEASURE)
        setup = {
            "collection": collection,
            "split": split,
            "methods": methods,
            "answer": answer,
            # None when no AI is set up: a retrieval-only run needs none.
            "model": model or self.ai.role_model("answer"),
            "k": k,
            "rrf_k": rrf_k,
            "search": search or {},
            "eval_set": name,
        }
        return db.create_job(self.conn, "eval", collection, setup)

    def _record_setup(self, job_id: str, spec: CollectionSpec, setup: dict) -> dict:
        """On the first run only: the versions the run uses and the questions it asks.

        Whatever the setup already records wins: an old run keeps its versions,
        and its question ids are the ones it has rows for."""
        name = self._set_name(spec, setup)
        if name:  # resolved once: a run queued before the store had a default set
            setup = {**setup, "eval_set": name}
            if "eval_set_version" not in setup:
                setup["eval_set_version"] = self._eval_set_version(spec, setup)
            if "leaked_ids" not in setup:
                setup["leaked_ids"] = sorted(self.leaked_ids(spec, name))
        ids = self._question_ids(job_id, spec, setup)
        disabled = db.disabled_sources(self.conn, spec.name)
        asked = self._asked(job_id, spec, {**setup, "question_ids": ids})
        questions = self._held(spec, asked)
        off = self._gold_off(spec, questions, disabled)
        include = db.get_include_leaked(self.conn, name) if name else False
        setup = {
            "dense_model": spec.dense_model,
            "query_instruction": self.models.query_instruction(QUERY_TASK[spec.qa_type]),
            "judge_model": self.models.judge_model,
            "keyword_version": spec.keyword_version,
            "vocabulary_version": spec.vocabulary_version,
            "seed": DEV_SEED,
            "dataset_filter": "; ".join(
                FILTER_NOTES[d] for d in sorted(db.loaded_datasets(self.conn, spec.name))
            )
            or "items you added or imported",
            "question_ids": ids,
            "disabled_sources": disabled,
            "leaked_excluded": 0 if include else len(setup.get("leaked_ids", [])),
            "evidence_named": sum(1 for q in asked if q.gold_ids),
            "evidence_held": sum(1 for q in questions if q.gold_ids),
            "evidence_off": sum(1 for q in questions if q.gold_ids and set(q.gold_ids) <= off),
            **setup,
            # THE EFFECTIVE SETTINGS, NOT THE OVERRIDE: every run's table can then be compared
            # with any other's. Merging a full recorded set over the defaults gives it back.
            "search": asdict(self._search(spec, setup)),
        }
        db.set_job_setup(self.conn, job_id, setup)
        return setup

    @staticmethod
    def _search(spec: CollectionSpec, setup: dict) -> SearchSettings:
        return search_settings(spec.search or SearchSettings(), setup.get("search"))

    def _gold_off(
        self, spec: CollectionSpec, questions: list[Question], disabled: list[str]
    ) -> set[str]:
        """The gold item ids that sit in a disabled source."""
        if not disabled:
            return set()
        gold = sorted({g for q in questions for g in q.gold_ids})
        hits = self.retriever.store.get(spec.name, gold)
        return {h.id for h in hits if h.meta.get("source") in disabled}

    def _version_change(self, spec: CollectionSpec, setup: dict) -> str | None:
        """Why the recorded versions no longer hold, or None. A setup that records
        no versions yet (never ran) records today's when it runs."""
        if "keyword_version" not in setup:
            # Migrated runs have question ids and no versions: nothing to check a resume against.
            if untracked(setup):
                return "this run predates version tracking; start a new run"
            return None
        old, new = setup["keyword_version"], spec.keyword_version
        # As changes_since: only the keyword methods search with the lists.
        if old != new and any(m in KEYWORD_METHODS for m in setup["methods"]):
            return f"keyword lists changed since this run ({old} → {new}); start a new run"
        old, new = setup.get("vocabulary_version"), spec.vocabulary_version
        if "vocabulary_version" in setup and old != new and "keyword_rag" in setup["methods"]:
            return f"the vocabulary changed since this run ({old} → {new}); start a new run"
        old, new = setup.get("dense_model"), spec.dense_model
        if old != new:
            return f"the dense model changed since this run ({old} → {new}); start a new run"
        old, new = setup.get("judge_model"), self.models.judge_model
        if old != new:
            return f"the judge model changed since this run ({old} → {new}); start a new run"
        old, new = setup.get("disabled_sources", []), db.disabled_sources(self.conn, spec.name)
        if old != new:
            return f"sources switched since this run ({old} → {new}); start a new run"
        return None

    def resume_refusal(self, job_id: str) -> str | None:
        job = db.get_job(self.conn, job_id)
        if job is None:
            raise KeyError(job_id)
        if job["kind"] != "eval":
            return "only evals can be resumed"
        if job["status"] not in db.JOBS_RESUMABLE:
            return f"a {job['status']} run cannot be resumed"
        setup = job["setup"]
        return self._version_change(self.retriever.spec(setup["collection"]), setup)

    def run(self, job_id: str, check: Callable[[], None] = lambda: None) -> None:
        """Run, or resume, a queued eval. check() raises Cancelled to stop it."""
        job = db.get_job(self.conn, job_id)
        if job is None:
            return
        setup = job["setup"]
        db.update_job(self.conn, job_id, status="running")
        try:
            spec = self.retriever.spec(setup["collection"])
            # Checked again at claim: a re-key queued ahead of a resume must not mix versions.
            reason = self._version_change(spec, setup)
            if reason:
                db.update_job(self.conn, job_id, status="failed", error=reason)
                return
            if "question_ids" not in setup:
                setup = self._record_setup(job_id, spec, setup)
            questions = self._run_questions(job_id, spec, setup)
            methods = setup["methods"]
            paid = [m for m in methods if m in KEYWORD_METHODS and self.retriever.is_paid(spec, m)]
            summarise = {q.id for m in paid for q in self._todo(job_id, m, questions)}
            if summarise:
                self.retriever.prepare(spec, [q.text for q in questions if q.id in summarise])
            if "keyword_rag" in methods and self._search(spec, setup).query_tags == "model":
                asked = self._todo(job_id, "keyword_rag", questions)
                self.retriever.prepare_tags(spec, [q.text for q in asked])
            for n, method in enumerate(methods, start=1):
                check()
                todo = self._todo(job_id, method, questions)  # read now: a repeat finds none

                def report(done: float, n: int = n) -> None:  # inside this method, 0..1
                    db.update_job(self.conn, job_id, progress=(n - 1 + done) / len(methods))

                self._run_method(job_id, spec, todo, method, setup, check, report)
                db.update_job(self.conn, job_id, progress=n / len(methods))
            db.update_job(self.conn, job_id, status="done")
        except Exception as e:  # recorded on the job; the thread must not die silently
            db.update_job(self.conn, job_id, **ended(e))

    def _run_method(
        self,
        job_id: str,
        spec: CollectionSpec,
        questions: list[Question],
        method: str,
        setup: dict,
        check: Callable[[], None],
        report: Callable[[float], None],
    ) -> None:
        """One method over `questions`, the ones still incomplete for it. Every 10
        questions and after each AI step it reports and checks, so the job's
        updated_at stays fresh and a cancel does not wait for the method to end."""
        if not questions:
            return
        span = 0.5 if setup["answer"] else 0.99  # the share the retrieval step is worth
        search = self._search(spec, setup)
        skip = tuple(setup.get("disabled_sources", []))
        all_hits, all_tags = [], []
        for n, q in enumerate(questions, start=1):
            # EXCLUDED IDS (a BRIGHT question's own problem and its near copies) ARE LEFT OUT
            # BY EVERY SEARCH, the classifier's vote included, so they are never scored or
            # shown and each channel's pool and the depth count only the others.
            excluded = q.meta.get("excluded_ids", [])
            tags = None
            if method == "keyword_rag":
                if search.query_tags == "oracle":
                    tags = self.oracle_tags(spec, q)
                else:
                    tags = self.retriever.query_tags(spec, q.text, search, skip, excluded)
            all_tags.append(tags)
            all_hits.append(
                self.retriever.retrieve(
                    spec.name,
                    q.text,
                    method,
                    retrieval_depth(spec.qa_type, q, setup["k"]),
                    setup["rrf_k"],
                    search,
                    tags,
                    excluded,
                )
            )
            if n % 10 == 0 and n < len(questions):
                report(span * n / len(questions))
                check()
        if method == "question_only":  # no retrieval, so nothing to score it on
            rows = [
                {"hit_ids": [], "hit_skills": [], "retrieval": {}, "usd": 0.0} for _ in questions
            ]
        else:
            off = self._gold_off(spec, questions, setup.get("disabled_sources", []))
            rows = self._retrieval_rows(spec, questions, all_hits, off)
        if method == "keyword_rag":
            for row, tags in zip(rows, all_tags, strict=True):
                row["query_tags"] = tags
        for q, row in zip(questions, rows, strict=True):
            db.upsert_run_item(self.conn, job_id, method, q.id, row, complete=not setup["answer"])
        if not setup["answer"]:
            return
        report(span)
        check()
        shown = [answer_hits(spec.qa_type, hits, search.example) for hits in all_hits]
        prompts = [
            answer_prompt(spec.qa_type, q.text, q.choices, hits, search.example, search.program)
            for q, hits in zip(questions, shown, strict=True)
        ]
        programs = [search.program and not q.choices for q in questions]
        done = self.ai.complete_many(
            [user for _, user in prompts],
            role="answer",
            system=prompts[0][0] if prompts else "",
            model=setup["model"],
            max_tokens=answer_tokens(spec.qa_type, any(programs)),
            purpose="answer",
        )
        for n, (q, row, hits, program, completion) in enumerate(
            zip(questions, rows, shown, programs, done, strict=True), start=1
        ):
            # PROGRAMS RUN HERE, UP TO PROGRAM_TIMEOUT EACH, so a cancel must not wait for all
            # of them. It costs nothing: the answers are cached, and a resume finds them.
            if n % 10 == 0:
                report(span + (0.99 - span) * n / len(questions))
                check()
            pred, record = read_answer(spec.qa_type, completion.text, bool(q.choices), program)
            row.update(
                {
                    "pred": pred,
                    "raw": answer_line(completion.text),
                    "usd": completion.usd,
                    "answer": score(spec, q, pred),
                    "program": record,
                    "example_form": example_form(spec.qa_type, hits, search.example),
                }
            )
            db.upsert_run_item(self.conn, job_id, method, q.id, row, complete=True)
        report(0.99)
        check()

    def oracle_tags(self, spec: CollectionSpec, q: Question) -> list[str]:
        """The tags a perfect classifier would predict (eval only, X2): those stored on the
        question's gold units, in order; without any, the similarity tags of its worked
        solution; else none."""
        gold = self.retriever.store.get(spec.name, q.gold_ids) if q.gold_ids else []
        tags = list(dict.fromkeys(t for h in gold for t in h.tags))
        if tags:
            return tags
        solution = q.meta.get("solution")
        if isinstance(solution, str) and solution.strip():
            similarity = SearchSettings(query_tags="similarity")
            return self.retriever.query_tags(spec, solution, similarity, ())
        return []

    def _retrieval_rows(
        self,
        spec: CollectionSpec,
        questions: list[Question],
        all_hits: list[list[Hit]],
        off: set[str],
    ) -> list[dict]:
        rows = []
        mine = gold = None
        if spec.qa_type in EXAMPLE_TYPES:
            explanations = [hits[0].explanation or "" if hits else "" for hits in all_hits]
            mine = self.models.judge(explanations)
            gold = self.models.judge([q.meta.get("solution", "") for q in questions])
        for n, (q, hits) in enumerate(zip(questions, all_hits, strict=True)):
            ids = [h.id for h in hits]
            if mine is not None and gold is not None:
                # Only what can be measured: no explanation or solution, no score.
                metrics = {}
                # No hits scores 0 only where the question could have been measured.
                skill = q.meta.get("skill")
                if not hits:
                    if q.meta.get("solution"):
                        metrics["solution_sim@1"] = 0.0
                    if skill:
                        metrics["same_skill@1"] = 0.0
                else:
                    if hits[0].explanation and q.meta.get("solution"):
                        metrics["solution_sim@1"] = cosine(mine[n], gold[n])
                    if skill and hits[0].meta.get("skill"):
                        metrics["same_skill@1"] = float(hits[0].meta["skill"] == skill)
                if q.gold_ids and not set(q.gold_ids) <= off:
                    metrics["ndcg@10"] = ndcg_at(ids, q.gold_ids, GOLD_DEPTH)
                    metrics["recall@1"] = recall_at(ids, q.gold_ids, 1)
            elif not q.gold_ids or set(q.gold_ids) <= off:
                metrics = {}  # nothing to find: absent, never 0
            else:
                cuts = (2, 5) if spec.qa_type == "multihop" else (1, 3)
                metrics = {f"recall@{c}": recall_at(ids, q.gold_ids, c) for c in cuts}
            rows.append(
                {
                    "hit_ids": ids,
                    "hit_skills": [h.meta.get("skill") for h in hits],
                    "hit_matched": [h.matched for h in hits],
                    "retrieval": metrics,
                    "usd": 0.0,
                }
            )
        return rows

    def _rows_by_method(self, job_id: str) -> tuple[dict[str, dict[str, dict]], bool]:
        """A run's rows by method and question id, and whether any row has an answer."""
        rows = db.run_items(self.conn, job_id)
        by_method: dict[str, dict[str, dict]] = {}
        for r in rows:
            by_method.setdefault(r["method"], {})[r["question_id"]] = r
        return by_method, any("answer" in r for r in rows)

    def _headline(
        self,
        spec: CollectionSpec,
        by_method: dict[str, dict[str, dict]],
        has_answers: bool,
        trusted: bool | None,
    ) -> tuple[str, dict, dict[str, dict[str, float]]]:
        """The headline metric, its scale, and each method's per-question values of it."""
        names = RETRIEVAL_METRICS[spec.qa_type] + (
            ANSWER_METRICS[spec.qa_type] if has_answers else ()
        )
        measurable = {
            m
            for m in names
            if m in METRICS and any(metric_values(q.values(), m) for q in by_method.values())
        }
        headline = verdict.headline_metric(spec.qa_type, has_answers, trusted, measurable)
        per_method = {
            method: metric_by_question(per_q, headline) for method, per_q in by_method.items()
        }
        random = list(per_method.get("random_example", {}).values())
        return headline, verdict.scale(headline, avg(random) if random else None), per_method

    def _question_count(self, setup: dict, by_method: dict[str, dict[str, dict]]) -> int:
        if "question_ids" in setup:
            return len(setup["question_ids"])
        return len({q for per_q in by_method.values() for q in per_q})

    def summary(self, job_id: str, proxy_trusted: bool | None) -> dict | None:
        """What a list of runs needs: the headline, its scale and the declared pair's
        answer, without the per-method intervals, all pairs, best of each or proxy.
        `proxy_trusted` is the collection's (collection_proxy_trusted), computed once
        by the caller; it matters only for an unanswered exam run."""
        job = db.get_job(self.conn, job_id)
        if job is None:
            return None
        setup = job["setup"]
        spec = self.retriever.spec(setup["collection"])
        by_method, has_answers = self._rows_by_method(job_id)
        trusted = proxy_trusted if spec.qa_type in EXAMPLE_TYPES and not has_answers else None
        headline, scaling, per_method = self._headline(spec, by_method, has_answers, trusted)
        return {
            "headline": headline,
            "scale": scaling,
            "answer": verdict.answer(per_method, headline, scaling["factor"]),
            "proxy_trusted": trusted,
            "usd_total": self.ai.job_spend(job_id),
            "question_count": self._question_count(setup, by_method),
        }

    def table(self, job_id: str) -> dict:
        job = db.get_job(self.conn, job_id)
        if job is None:
            raise KeyError(job_id)
        setup = job["setup"]
        spec = self.retriever.spec(setup["collection"])
        by_method, has_answers = self._rows_by_method(job_id)
        metric_names = RETRIEVAL_METRICS[spec.qa_type] + (
            ANSWER_METRICS[spec.qa_type] if has_answers else ()
        )
        methods = {}
        for method, per_q in by_method.items():
            entry: dict = {"usd": sum(r.get("usd", 0.0) for r in per_q.values()), "n": len(per_q)}
            for metric in metric_names:
                values = metric_values(per_q.values(), metric)
                if values:
                    mean, lo, hi = bootstrap_ci(values)
                    entry[metric] = {
                        "mean": clean(mean),
                        "lo": clean(lo),
                        "hi": clean(hi),
                        "n": len(values),
                    }
            methods[method] = entry
        # The type's first retrieval metric that any method has: BRIGHT has no solutions or
        # skills, so its runs pair on nDCG@10, their headline.
        retrieval = RETRIEVAL_METRICS[spec.qa_type]
        measured = [m for m in retrieval if any(m in entry for entry in methods.values())]
        pair_metrics = measured[:1] or [retrieval[0]]
        if has_answers:
            pair_metrics.append(PAIR_ANSWER_METRIC[spec.qa_type])
        pairs = []
        for metric in pair_metrics:
            for a, b in itertools.combinations(by_method, 2):
                # By question id: absent values drop out per method, so position can't pair them.
                va, vb = (metric_by_question(by_method[m], metric) for m in (a, b))
                paired = verdict.paired_by_question(va, vb, 1.0)
                if paired is None:
                    continue
                diff, lo, hi = paired
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "metric": metric,
                        "diff": clean(diff),
                        "lo": clean(lo),
                        "hi": clean(hi),
                        "separated": separated(lo, hi),
                    }
                )
        proxy = self._proxy_for(job_id) if spec.qa_type in EXAMPLE_TYPES and has_answers else None
        trusted = None
        if spec.qa_type in EXAMPLE_TYPES and not has_answers:
            trusted = self.collection_proxy_trusted(spec.name)
        headline, scaling, per_method = self._headline(spec, by_method, has_answers, trusted)
        reference = ANSWER_PAIR[1]
        return {
            "setup": setup,
            "status": job["status"],
            "progress": job["progress"],
            "error": job["error"],
            "methods": methods,
            "pairs": pairs,
            "proxy": proxy,
            "headline": headline,
            "proxy_trusted": trusted,
            "scale": scaling,
            "reference": reference,
            "gaps": verdict.gaps(per_method, reference, scaling["factor"]),
            "answer": verdict.answer(per_method, headline, scaling["factor"]),
            "best_of_each": verdict.best_of_each(per_method, headline, scaling["factor"]),
            "usd_total": self.ai.job_spend(job_id),
            "question_count": self._question_count(setup, by_method),
            "since": changes_since(
                setup,
                spec,
                db.disabled_sources(self.conn, spec.name),
                self._eval_set_version(spec, setup),
            ),
        }

    def _done_evals(self, collection: str) -> list[dict]:
        """The store's done runs of its default eval set, newest first: a run of another set
        asks other questions, so it is not the store's answer."""
        default = db.default_eval_set(self.conn, collection)
        done = db.list_jobs(
            self.conn, kind="eval", collection=collection, status="done", limit=db.ALL
        )
        return [j for j in done if j["setup"].get("eval_set") == default]

    def collection_answer(self, name: str) -> dict | None:
        """The collection's answer: the newest answered run that has the pair, else the
        newest run that has it (spec §7.3: a run lacking the pair gives no answer)."""
        spec = self.retriever.spec(name)
        done = self._done_evals(name)
        paired = [j for j in done if set(ANSWER_PAIR) <= set(j["setup"]["methods"])]
        answered_runs = [j for j in paired if j["setup"].get("answer")]
        job = (answered_runs or paired or [None])[0]
        if job is None:
            return None
        answered = bool(job["setup"].get("answer"))
        # Only an unanswered run's headline depends on the collection's proxy check.
        trusted = None if answered else self.collection_proxy_trusted(name, done)
        summary = self.summary(job["id"], trusted)
        if summary is None or summary["answer"] is None:
            return None
        return {
            "run_id": job["id"],
            "at": job["created_at"],
            "answered": answered,
            "runs": len(done),
            "proxy_trusted": summary["proxy_trusted"],
            "scale": summary["scale"],
            "since": changes_since(
                job["setup"],
                spec,
                db.disabled_sources(self.conn, name),
                self._eval_set_version(spec, job["setup"]),
            ),
            **summary["answer"],
        }

    def _proxy_for(self, job_id: str) -> dict | None:
        """The proxy check of one answered mcq run, straight from its rows."""
        by_method: dict[str, list[dict]] = {}
        for r in db.run_items(self.conn, job_id):
            by_method.setdefault(r["method"], []).append(r)
        means = {}
        for method, rows in by_method.items():
            sims = metric_values(rows, "solution_sim@1")
            accs = metric_values(rows, "accuracy")
            if sims and accs:
                means[method] = {"solution_sim@1": avg(sims), "accuracy": avg(accs)}
        return verdict.proxy(means)

    def collection_proxy_trusted(self, name: str, done: list[dict] | None = None) -> bool | None:
        """The latest real proxy check among the done answered runs, newest first."""
        for job in self._done_evals(name) if done is None else done:
            if job["setup"].get("answer"):
                check = self._proxy_for(job["id"])
                if check is not None:
                    return check["trusted"]
        return None

    def items(self, job_id: str, method: str, wrong_only: bool) -> list[dict]:
        job = db.get_job(self.conn, job_id)
        if job is None:
            raise KeyError(job_id)
        spec = self.retriever.spec(job["setup"]["collection"])
        questions = {q.id: q for q in self._set_questions(spec, job["setup"])}
        primary, answering = MISS_METRIC[spec.qa_type], bool(job["setup"].get("answer"))
        rows = [
            (row, questions.get(row["question_id"]))
            for row in db.run_items(self.conn, job_id, method)
            if not wrong_only or is_wrong(row, primary, answering)
        ]
        held = self._held_ids(spec, [q for _, q in rows if q])
        return [
            {
                **row,
                "question": q.text if q else None,
                "choices": q.choices if q else None,
                "gold_answers": q.gold_answers if q else [],
                "gold_ids": q.gold_ids if q else [],
                "gold_held": [g for g in q.gold_ids if g in held] if q else [],
            }
            for row, q in rows
        ]

    def questions(self, job_id: str, disagree: bool) -> list[dict]:
        """Each question of a run with every method's row and its right/wrong outcome. With
        `disagree`, only the questions where the methods' outcomes differ; the controls are
        left out of that, and a method with no outcome has nothing to differ from. An
        answering run is judged on answers only, so a method not answered yet has none."""
        job = db.get_job(self.conn, job_id)
        if job is None:
            raise KeyError(job_id)
        spec = self.retriever.spec(job["setup"]["collection"])
        questions = {q.id: q for q in self._set_questions(spec, job["setup"])}
        primary, answering = MISS_METRIC[spec.qa_type], bool(job["setup"].get("answer"))
        by_question: dict[str, dict[str, dict]] = {}
        for r in db.run_items(self.conn, job_id):
            by_question.setdefault(r["question_id"], {})[r["method"]] = r
        shown = [questions[qid] for qid in by_question if qid in questions]
        held = self._held_ids(spec, shown)  # the evidence the metrics read, as _held cuts it
        out = []
        for qid, rows in by_question.items():
            methods = {}
            for method, r in rows.items():
                result = outcome(r, primary, answering)
                methods[method] = {**r, "outcome": result} if result else r
            judged = {
                m["outcome"]
                for name, m in methods.items()
                if name not in CONTROLS and name != "question_only" and "outcome" in m
            }
            if disagree and len(judged) < 2:
                continue
            q = questions.get(qid)
            out.append(
                {
                    "question_id": qid,
                    "question": q.text if q else None,
                    "choices": q.choices if q else None,
                    "gold_answers": q.gold_answers if q else [],
                    "gold_ids": q.gold_ids if q else [],
                    "gold_held": [g for g in q.gold_ids if g in held] if q else [],
                    "methods": methods,
                }
            )
        return out


def read_answer(
    qa_type: QaType, text: str, has_choices: bool, program: bool
) -> tuple[str, dict | None]:
    """The prediction in an answer reply, and the record of its program when one was asked
    for: the program's printed answer wins over the reply's own line. Runs and /query both."""
    pred = parse_answer(qa_type, text, has_choices)
    if not program:
        return pred, None
    printed, record = program_answer(text)
    return printed or pred, record


def ended(e: Exception) -> dict:
    """The status and error a job that raised e ends with: stopped (resumable) when it ran
    out of budget, a model or replies; cancelled; otherwise failed."""
    if isinstance(e, Cancelled):
        return {"status": "cancelled"}
    if isinstance(e, BudgetExceeded):
        return {"status": "stopped_budget", "error": str(e)}
    if isinstance(e, NoConnection):
        return {"status": "stopped_no_key", "error": no_ai(e.role)}
    if isinstance(e, AwaitingReplies):
        return {"status": "stopped_replies", "error": awaiting(e)}
    return {"status": "failed", "error": repr(e)}


def no_ai(role: str) -> str:
    return f"no AI model is set up for the '{role}' job; choose one on the Settings page"


def awaiting(e: AwaitingReplies) -> str:
    return (
        f"{e}: export them with `python -m ragqa_core.replies export`, import the replies,"
        " then resume the run or start the job again; replies already imported stay cached"
    )


def avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score(spec: CollectionSpec, q: Question, pred: str) -> dict:
    """The answer metrics of one prediction; none for a question with no gold answer
    (BRIGHT's), which is then unmeasured rather than wrong."""
    if not q.gold_answers:
        return {}
    if spec.qa_type == "mcq":
        return {"accuracy": float(pred == q.gold_answers[0])}
    if spec.qa_type == "science":
        return {"accuracy": float(science_correct(pred, q.gold_answers, bool(q.choices)))}
    if spec.qa_type == "multihop":
        em, f1 = squad_em_f1(pred, q.gold_answers)
    else:
        em, f1 = tatqa_em_f1(pred, q.gold_answers, q.meta.get("scale", ""))
    return {"em": em, "f1": f1}


def metric_values(rows, metric: str) -> list[float]:
    values = []
    for r in rows:
        source = r.get("answer", {}) if metric in ("accuracy", "em", "f1") else r["retrieval"]
        if metric in source:
            values.append(float(source[metric]))
    return values


def metric_by_question(per_q: dict[str, dict], metric: str) -> dict[str, float]:
    return {q: values[0] for q, row in per_q.items() if (values := metric_values([row], metric))}


def outcome(row: dict, primary: str, answering: bool = False) -> str | None:
    """ "right" or "wrong" on the row's primary measure; None when it has none (unmeasured).
    In an `answering` run the measure is the answer, so a row without one is unmeasured."""
    answer = row.get("answer")
    if answer:
        key = "accuracy" if "accuracy" in answer else "em"
        return "wrong" if answer[key] == 0 else "right"
    if answering:
        return None
    if primary not in row["retrieval"]:  # absent means unmeasured (or question_only), never 0
        return None
    return "wrong" if row["retrieval"][primary] == 0 else "right"


def is_wrong(row: dict, primary: str, answering: bool = False) -> bool:
    return outcome(row, primary, answering) == "wrong"
