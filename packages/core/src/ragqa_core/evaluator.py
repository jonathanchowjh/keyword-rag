"""Evaluation runs: estimate first, then queue and run (or resume), then tabulate
with intervals so a gap that is noise is shown as noise."""

import itertools
import math
import sqlite3
from collections.abc import Callable
from typing import Literal

from ragqa_core import db, verdict
from ragqa_core.catalog import ANSWER_PAIR, METRICS
from ragqa_core.datasets import DEV_SEED, FILTER_NOTES, dev_slice
from ragqa_core.llm import BATCH_THRESHOLD, BudgetExceeded, Cancelled, Claude, NoApiKey
from ragqa_core.metrics import (
    bootstrap_ci,
    cosine,
    paired_diff_ci,
    recall_at,
    separated,
    squad_em_f1,
    tatqa_em_f1,
)
from ragqa_core.models import Models
from ragqa_core.prompts import (
    QUERY_TASK,
    SUMMARY_TOKENS,
    answer_line,
    answer_prompt,
    hop_prompt,
    parse_answer,
    summary_prompt,
)
from ragqa_core.retriever import HOP_TOKENS, KEYWORD_METHODS, MAX_HOPS, Retriever
from ragqa_core.types import CollectionSpec, Hit, QaType, Question

RETRIEVAL_METRICS: dict[QaType, tuple[str, ...]] = {
    "mcq": ("solution_sim@1", "same_skill@1"),
    "multihop": ("recall@2", "recall@5"),
    "table": ("recall@1", "recall@3"),
}
ANSWER_METRICS: dict[QaType, tuple[str, ...]] = {
    "mcq": ("accuracy",),
    "multihop": ("em", "f1"),
    "table": ("em", "f1"),
}
# What counts as a retrieval miss when a run has no answers: a binary metric,
# because the continuous solution similarity is practically never exactly 0.
MISS_METRIC: dict[QaType, str] = {
    "mcq": "same_skill@1",
    "multihop": "recall@2",
    "table": "recall@1",
}
PAIR_ANSWER_METRIC: dict[QaType, str] = {"mcq": "accuracy", "multihop": "f1", "table": "f1"}
ANSWER_TOKENS: dict[QaType, int] = {"mcq": 400, "multihop": 100, "table": 100}
ESTIMATE_SAMPLE = 5


def clean(value: float) -> float | None:
    return None if math.isnan(value) else value


def answer_hits(qa_type: QaType, hits: list[Hit]) -> list[Hit]:
    # The paper's one-shot: the single best example for mcq; passages otherwise.
    return hits[:1] if qa_type == "mcq" else hits


class Evaluator:
    def __init__(
        self, conn: sqlite3.Connection, retriever: Retriever, models: Models, claude: Claude
    ) -> None:
        self.conn = conn
        self.retriever = retriever
        self.models = models
        self.claude = claude

    def _questions(self, spec: CollectionSpec, split: str) -> list[Question]:
        questions = db.get_questions(self.conn, spec.name)
        return dev_slice(questions) if split == "dev" else questions

    def _question_ids(self, job_id: str, spec: CollectionSpec, setup: dict) -> list[str]:
        if "question_ids" in setup:
            return setup["question_ids"]
        # Not recorded yet: an old run's are the ones it has rows for; a new
        # run's are its split's, as its first run will record them.
        return db.run_item_question_ids(self.conn, job_id) or [
            q.id for q in self._questions(spec, setup["split"])
        ]

    def _run_questions(self, job_id: str, spec: CollectionSpec, setup: dict) -> list[Question]:
        """The run's questions in recorded order; ids since deleted are skipped."""
        by_id = {q.id: q for q in db.get_questions(self.conn, spec.name)}
        return [by_id[i] for i in self._question_ids(job_id, spec, setup) if i in by_id]

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
    ) -> dict:
        spec = self.retriever.spec(collection)
        self.retriever.indexer.check_model(spec)
        questions = self._questions(spec, split)
        parts = self._estimate_parts(
            spec, {m: len(questions) for m in methods}, answer, model, questions[:ESTIMATE_SAMPLE]
        )
        return {"questions": len(questions), **parts}

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
        )
        return {"questions": len(remaining), **parts}

    def _estimate_parts(
        self,
        spec: CollectionSpec,
        n_by_method: dict[str, int],
        answer: bool,
        model: str | None,
        sample: list[Question],
    ) -> dict:
        # (calls, avg prompt chars, output tokens, batched?, model) — each part
        # priced the way it is sent: one complete_many per part, batched only
        # above 50 calls; iterative hops are single real-time calls. Summaries and
        # hops run on the default model (None); only answers use the requested one.
        parts: list[tuple[int, float, int, bool, str | None]] = []
        summaries = max((n for m, n in n_by_method.items() if m in KEYWORD_METHODS), default=0)
        if spec.summarize and summaries:
            chars = avg(len(summary_prompt(q.text)) for q in sample)
            parts.append((summaries, chars, SUMMARY_TOKENS, summaries > BATCH_THRESHOLD, None))
        hop_chars = avg(len(hop_prompt(q.text, ["x" * 600] * 4)) for q in sample)
        parts.extend(
            (n * MAX_HOPS, hop_chars, HOP_TOKENS, False, None)
            for m, n in n_by_method.items()
            if m == "iterative"
        )
        if answer and any(n_by_method.values()):
            chars = self._answer_chars(spec, sample)
            parts.extend(
                (n, chars, ANSWER_TOKENS[spec.qa_type], n > BATCH_THRESHOLD, model)
                for n in n_by_method.values()
            )
        calls = sum(part[0] for part in parts)
        if calls == 0:
            return {"calls": 0, "est_usd": 0.0}
        usd = sum(
            c * self.claude.estimate_usd(int(chars), out, model=m, batch=batch)
            for c, chars, out, batch, m in parts
        )
        return {"calls": calls, "est_usd": usd}

    def _answer_chars(self, spec: CollectionSpec, sample: list[Question]) -> float:
        sizes = []
        for q in sample:
            hits = answer_hits(spec.qa_type, self.retriever.retrieve(spec.name, q.text, "hybrid"))
            system, user = answer_prompt(spec.qa_type, q.text, q.choices, hits)
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
    ) -> str:
        """Queue a run with the request only; the rest of its setup is recorded when it runs."""
        spec = self.retriever.spec(collection)
        if not methods:
            raise ValueError("choose at least one method")
        if not self._questions(spec, split):
            raise ValueError(f"collection '{collection}' has no questions")
        setup = {
            "collection": collection,
            "split": split,
            "methods": methods,
            "answer": answer,
            "model": model or self.claude.default_model,
            "k": k,
            "rrf_k": rrf_k,
        }
        return db.create_job(self.conn, "eval", collection, setup)

    def _record_setup(self, job_id: str, spec: CollectionSpec, setup: dict) -> dict:
        """On the first run only: the versions the run uses and the questions it asks.

        Whatever the setup already records wins: an old run keeps its versions,
        and its question ids are the ones it has rows for."""
        ids = self._question_ids(job_id, spec, setup)
        setup = {
            "dense_model": self.models.dense_model,
            "query_instruction": self.models.query_instruction(QUERY_TASK[spec.qa_type]),
            "judge_model": self.models.judge_model,
            "keyword_version": spec.keyword_version,
            "seed": DEV_SEED,
            "dataset_filter": FILTER_NOTES[spec.qa_type],
            "question_ids": ids,
            **setup,
        }
        db.set_job_setup(self.conn, job_id, setup)
        return setup

    def _version_change(self, spec: CollectionSpec, setup: dict) -> str | None:
        """Why the recorded versions no longer hold, or None. A setup that records
        no versions yet (never ran) records today's when it runs."""
        if "keyword_version" not in setup:
            return None
        old, new = setup["keyword_version"], spec.keyword_version
        if old != new:
            return f"keyword lists changed since this run ({old} → {new}); start a new run"
        old, new = setup.get("dense_model"), spec.dense_model
        if old != new:
            return f"the dense model changed since this run ({old} → {new}); start a new run"
        old, new = setup.get("judge_model"), self.models.judge_model
        if old != new:
            return f"the judge model changed since this run ({old} → {new}); start a new run"
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
            for n, method in enumerate(methods, start=1):
                check()
                todo = self._todo(job_id, method, questions)  # read now: a repeat finds none
                self._run_method(job_id, spec, todo, method, setup, check)
                db.update_job(self.conn, job_id, progress=n / len(methods))
            db.update_job(self.conn, job_id, status="done")
        except Cancelled:
            db.update_job(self.conn, job_id, status="cancelled")
        except BudgetExceeded as e:
            db.update_job(self.conn, job_id, status="stopped_budget", error=str(e))
        except NoApiKey as e:
            db.update_job(self.conn, job_id, status="stopped_no_key", error=str(e))
        except Exception as e:  # recorded on the job; the thread must not die silently
            db.update_job(self.conn, job_id, status="failed", error=repr(e))

    def _run_method(
        self,
        job_id: str,
        spec: CollectionSpec,
        questions: list[Question],
        method: str,
        setup: dict,
        check: Callable[[], None],
    ) -> None:
        """One method over `questions`, the ones still incomplete for it."""
        if not questions:
            return
        all_hits = [
            self.retriever.retrieve(spec.name, q.text, method, setup["k"], setup["rrf_k"])
            for q in questions
        ]
        if method == "question_only":  # no retrieval, so nothing to score it on
            rows = [
                {"hit_ids": [], "hit_skills": [], "retrieval": {}, "usd": 0.0} for _ in questions
            ]
        else:
            rows = self._retrieval_rows(spec, questions, all_hits)
        for q, row in zip(questions, rows, strict=True):
            db.upsert_run_item(self.conn, job_id, method, q.id, row, complete=not setup["answer"])
        if not setup["answer"]:
            return
        check()
        prompts = [
            answer_prompt(spec.qa_type, q.text, q.choices, answer_hits(spec.qa_type, hits))
            for q, hits in zip(questions, all_hits, strict=True)
        ]
        done = self.claude.complete_many(
            [user for _, user in prompts],
            system=prompts[0][0] if prompts else "",
            model=setup["model"],
            max_tokens=ANSWER_TOKENS[spec.qa_type],
            purpose="answer",
        )
        for q, row, completion in zip(questions, rows, done, strict=True):
            pred = parse_answer(spec.qa_type, completion.text)
            row.update(
                {
                    "pred": pred,
                    "raw": answer_line(completion.text),
                    "usd": completion.usd,
                    "answer": score(spec, q, pred),
                }
            )
            db.upsert_run_item(self.conn, job_id, method, q.id, row, complete=True)

    def _retrieval_rows(
        self, spec: CollectionSpec, questions: list[Question], all_hits: list[list[Hit]]
    ) -> list[dict]:
        rows = []
        mine = gold = None
        if spec.qa_type == "mcq":
            explanations = [hits[0].explanation or "" if hits else "" for hits in all_hits]
            mine = self.models.judge(explanations)
            gold = self.models.judge([q.meta.get("solution", "") for q in questions])
        for n, (q, hits) in enumerate(zip(questions, all_hits, strict=True)):
            ids = [h.id for h in hits]
            if mine is not None and gold is not None:
                # Only what can be measured: no explanation or solution, no score.
                metrics = {}
                if not hits:
                    metrics = {"solution_sim@1": 0.0, "same_skill@1": 0.0}
                else:
                    if hits[0].explanation and q.meta.get("solution"):
                        metrics["solution_sim@1"] = cosine(mine[n], gold[n])
                    skill = q.meta.get("skill")
                    if skill and hits[0].meta.get("skill"):
                        metrics["same_skill@1"] = float(hits[0].meta["skill"] == skill)
            elif spec.qa_type == "multihop":
                metrics = {
                    "recall@2": recall_at(ids, q.gold_ids, 2),
                    "recall@5": recall_at(ids, q.gold_ids, 5),
                }
            else:
                metrics = {
                    "recall@1": recall_at(ids, q.gold_ids, 1),
                    "recall@3": recall_at(ids, q.gold_ids, 3),
                }
            rows.append(
                {
                    "hit_ids": ids,
                    "hit_skills": [h.meta.get("skill") for h in hits],
                    "retrieval": metrics,
                    "usd": 0.0,
                }
            )
        return rows

    def table(self, job_id: str) -> dict:
        job = db.get_job(self.conn, job_id)
        if job is None:
            raise KeyError(job_id)
        setup = job["setup"]
        spec = self.retriever.spec(setup["collection"])
        rows = db.run_items(self.conn, job_id)
        by_method: dict[str, dict[str, dict]] = {}
        for r in rows:
            by_method.setdefault(r["method"], {})[r["question_id"]] = r
        has_answers = any("answer" in r for r in rows)
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
                    entry[metric] = {"mean": clean(mean), "lo": clean(lo), "hi": clean(hi)}
            methods[method] = entry
        pair_metrics = [RETRIEVAL_METRICS[spec.qa_type][0]]
        if has_answers:
            pair_metrics.append(PAIR_ANSWER_METRIC[spec.qa_type])
        pairs = []
        for metric in pair_metrics:
            for a, b in itertools.combinations(by_method, 2):
                shared = sorted(set(by_method[a]) & set(by_method[b]))
                va = metric_values((by_method[a][q] for q in shared), metric)
                vb = metric_values((by_method[b][q] for q in shared), metric)
                if not va or len(va) != len(vb):
                    continue
                diff, lo, hi = paired_diff_ci(va, vb)
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
        measurable = {metric for entry in methods.values() for metric in entry} & set(METRICS)
        proxy = self._proxy_for(job_id) if spec.qa_type == "mcq" and has_answers else None
        trusted = None
        if spec.qa_type == "mcq" and not has_answers:
            trusted = self.collection_proxy_trusted(spec.name)
        headline = verdict.headline_metric(spec.qa_type, has_answers, trusted, measurable)
        random_mean = methods.get("random_example", {}).get(headline, {}).get("mean")
        scaling = verdict.scale(headline, random_mean)
        per_method = {
            method: {
                q: values[0]
                for q, row in per_q.items()
                if (values := metric_values([row], headline))
            }
            for method, per_q in by_method.items()
        }
        reference = ANSWER_PAIR[1]
        question_ids = {q for per_q in by_method.values() for q in per_q}
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
            "gaps": verdict.gaps(per_method, reference, headline, scaling["factor"]),
            "answer": verdict.answer(per_method, headline, scaling["factor"]),
            "best_of_each": verdict.best_of_each(per_method, headline, scaling["factor"]),
            "usd_total": db.job_spend(self.conn, job_id),
            "question_count": len(setup["question_ids"])
            if "question_ids" in setup
            else len(question_ids),
        }

    def _done_evals(self, collection: str) -> list[dict]:
        return db.list_jobs(
            self.conn, kind="eval", collection=collection, status="done", limit=1_000_000
        )

    def collection_answer(self, name: str) -> dict | None:
        """The collection's answer: the newest answered run's, else the newest run's."""
        done = self._done_evals(name)
        answered = [j for j in done if j["setup"].get("answer")]
        job = (answered or done or [None])[0]
        if job is None:
            return None
        result = self.table(job["id"])["answer"]
        if result is None:
            return None
        return {
            "run_id": job["id"],
            "at": job["created_at"],
            "answered": bool(job["setup"].get("answer")),
            "runs": len(done),
            **result,
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

    def collection_proxy_trusted(self, name: str) -> bool | None:
        """The latest real proxy check among the done answered runs, newest first."""
        for job in self._done_evals(name):
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
        questions = {q.id: q for q in db.get_questions(self.conn, spec.name)}
        primary = MISS_METRIC[spec.qa_type]
        out = []
        for row in db.run_items(self.conn, job_id, method):
            if wrong_only and not is_wrong(row, primary):
                continue
            q = questions.get(row["question_id"])
            out.append(
                {
                    **row,
                    "question": q.text if q else None,
                    "choices": q.choices if q else None,
                    "gold_answers": q.gold_answers if q else [],
                    "gold_ids": q.gold_ids if q else [],
                }
            )
        return out


def avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def score(spec: CollectionSpec, q: Question, pred: str) -> dict:
    if spec.qa_type == "mcq":
        return {"accuracy": float(pred == q.gold_answers[0])}
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


def is_wrong(row: dict, primary: str) -> bool:
    answer = row.get("answer")
    if answer:
        key = "accuracy" if "accuracy" in answer else "em"
        return answer[key] == 0
    if not row["retrieval"]:  # question_only: no retrieval, so nothing to miss
        return False
    return row["retrieval"].get(primary, 0.0) == 0
