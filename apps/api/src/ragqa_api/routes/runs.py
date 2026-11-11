"""The run and job routes: evals and their results, jobs, resumes and /status."""

from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import (
    NO_KEYWORDS_MODEL,
    PROBLEM,
    Guards,
    check_methods,
    check_shown_price,
    estimate_reply,
)
from ragqa_core import db, jobs, search_defaults
from ragqa_core.loader import QUESTION_IMPORT_GONE, roles_needed
from ragqa_core.recipes import resolve_search
from ragqa_core.retriever import DEFAULT_K, DEFAULT_RRF_K, POOL

NOT_RESUMABLE = "the job is no longer resumable"
STALL_AFTER = timedelta(minutes=10)


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    split: Literal["dev", "full"] = "dev"
    methods: list[str]
    answer: bool = False
    model: str | None = None
    k: int = Field(DEFAULT_K, ge=1, le=50)
    rrf_k: int = Field(DEFAULT_RRF_K, ge=0)
    search: dict | None = None  # overrides the collection's search settings in part
    eval_set: str | None = None  # the store's default eval set when None
    rerank_pool: int = Field(POOL, ge=1, le=POOL)  # hybrid_rerank: hybrid's top N it reranks
    rerank_model: str | None = None  # hybrid_rerank: None takes the Settings model
    judge: bool = False  # Judge answers: a Documents run's answers graded by the answer model
    confirm: bool = False
    # What the caller was shown: given, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False
    # What the caller was shown: given, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

    @app.post("/eval", status_code=202)
    def evaluate(body: EvalRequest):
        check_methods(body.methods)
        spec = state.retriever.spec(body.collection)
        resolve_search(spec, body.search)  # 422 before an estimate
        if body.rerank_model is not None:
            search_defaults.check("rerank_model", body.rerank_model)
        estimate = state.evaluator.estimate(
            **body.model_dump(
                include={
                    "collection",
                    "split",
                    "methods",
                    "answer",
                    "model",
                    "search",
                    "eval_set",
                    "judge",
                }
            )
        )
        if not body.confirm:  # spec §7.3: an eval without confirm only estimates
            return estimate_reply(estimate)
        check_shown_price(estimate, body.calls, body.est_usd)
        guards.need_models(estimate, body.model)
        if "keyword_rag" in body.methods:
            state.retriever.vocabulary(spec)  # refused here, not after the other methods spent
        with guards.not_adding(body.collection):
            job_id = state.evaluator.queue(
                **body.model_dump(exclude={"confirm", "calls", "est_usd"}),
                evidence_size=estimate.get("evidence_size"),  # the size the estimate priced
            )
        return {"job_id": job_id, "estimate": estimate}

    @app.get("/eval")
    def list_runs(collection: str | None = None) -> list[dict]:
        jobs = db.list_jobs(state.conn, kind="eval", collection=collection, limit=db.ALL)
        trusted = {  # the proxy check once per collection, not once per run
            name: state.evaluator.collection_proxy_trusted(name)
            for name in {j["collection"] for j in jobs}
        }
        out = []
        for job in jobs:
            summary = state.evaluator.summary(job["id"], trusted[job["collection"]])
            if summary is not None:  # deleted since it was listed
                out.append(run_summary(job, summary))
        return out

    @app.get("/eval/{job_id}")
    def eval_table(job_id: str) -> dict:
        job = db.get_job(state.conn, job_id)
        if job is None or job["kind"] != "eval":
            raise HTTPException(404, f"no run {job_id}")
        return {
            **state.evaluator.table(job_id),
            **eval_set_fields(job["setup"]),
            "log": job["log"],
            "created_at": job["created_at"],
        }

    @app.delete("/eval/{job_id}", status_code=204)
    def delete_run(job_id: str) -> Response:
        job = db.get_job(state.conn, job_id)
        if job is None or job["kind"] != "eval":
            raise HTTPException(404, f"no run {job_id}")
        if job["status"] in jobs.BUSY:
            raise HTTPException(409, f"the run is {job['status']}; cancel it first")
        db.delete_job(state.conn, job_id)
        return Response(status_code=204)

    @app.get("/eval/{job_id}/items")
    def eval_items(job_id: str, method: str, wrong: bool = False) -> list[dict]:
        return state.evaluator.items(job_id, method, wrong)

    # One row per question with every method's row, where /items is one method's rows.
    @app.get("/eval/{job_id}/questions")
    def eval_questions(job_id: str, disagree: bool = False) -> list[dict]:
        return state.evaluator.questions(job_id, disagree)

    @app.get("/jobs")
    def list_jobs(
        collection: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        return db.list_jobs(
            state.conn, kind=kind, collection=collection, status=status, limit=limit
        )

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        row = db.get_job(state.conn, job_id)
        if row is None:
            raise HTTPException(404, f"no job {job_id}")
        return row

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        row = job(job_id)
        if row["status"] not in jobs.BUSY:
            raise HTTPException(409, f"job is {row['status']}")
        return {"status": db.request_cancel(state.conn, job_id)}

    @app.post("/jobs/{job_id}/resume", status_code=202)
    def resume(job_id: str, body: ResumeRequest | None = None):
        """Without confirm, what resuming would cost (spec §5.4); with it, requeued."""
        row = db.get_job(state.conn, job_id)
        if row is not None and row["kind"] == "copy":
            return resume_copy(row, body)
        if row is not None and row["kind"] == "reembed":
            return resume_reembed(row)
        if row is not None and row["kind"] == "tag_build":
            return resume_tag_build(row, body)
        if row is not None and row["kind"] == "import" and row["setup"].get("kind") == "questions":
            raise HTTPException(409, QUESTION_IMPORT_GONE)
        refusal = state.evaluator.resume_refusal(job_id)
        if refusal:
            raise HTTPException(409, refusal)
        if body is None or not body.confirm:
            return estimate_reply(state.evaluator.estimate_resume(job_id))
        estimate = state.evaluator.estimate_resume(job_id)
        check_shown_price(estimate, body.calls, body.est_usd)
        guards.need_models(estimate, job(job_id)["setup"]["model"])
        if not db.requeue(state.conn, job_id):  # moved since the check: a double confirm
            raise HTTPException(409, NOT_RESUMABLE)
        return {"job_id": job_id}

    def resume_copy(row: dict, body: ResumeRequest | None):
        """A stopped copy runs again in place: adding skips the items it added, and the
        build finishes what it left unbuilt, so only the rest is priced."""
        if refusal := jobs.not_resumable(row):
            raise HTTPException(409, refusal)
        source = row["setup"]["source"]
        target = state.retriever.spec(row["collection"])
        state.retriever.spec(source)  # 404 when the source is gone
        guards.refuse_busy(source)
        estimate = state.loader.estimate_copy(source, target)
        if body and body.confirm:
            check_shown_price(estimate, body.calls, body.est_usd)
        for role in roles_needed(target, estimate):
            guards.need_ai(role)
        if not (body and body.confirm):
            return estimate_reply(estimate)
        with guards.not_adding(source):
            if not db.requeue(state.conn, row["id"]):  # moved since the check
                raise HTTPException(409, NOT_RESUMABLE)
        return {"job_id": row["id"]}

    def resume_tag_build(row: dict, body: ResumeRequest | None):
        """A stopped Add keyword tags runs again in place: the build finishes what it left
        unbuilt (the vocabulary when none is in force, the passages without tags), so only
        that is priced."""
        if refusal := jobs.not_resumable(row):
            raise HTTPException(409, refusal)
        name = row["collection"]
        spec = state.retriever.spec(name)  # 404 when the store is gone
        with guards.not_adding(name):  # no job is queued between these checks and the requeue
            # The keyword tags state reads the newest tag build only, so an older one must
            # not run again in its place.
            newest = db.list_jobs(state.conn, kind="tag_build", collection=name, limit=1)
            if newest[0]["id"] != row["id"]:
                raise HTTPException(409, f"a newer tag build of '{name}' exists; resume that one")
            guards.refuse_busy(name)
            estimate, calls = state.builder.plan_tag_build(spec)
            if estimate is None:
                raise HTTPException(503, NO_KEYWORDS_MODEL)
            if not (body and body.confirm):
                return estimate_reply({"est_usd": estimate, "calls": calls})
            check_shown_price({"est_usd": estimate, "calls": calls}, body.calls, body.est_usd)
            if calls > 0:  # roles_needed's rule: the role is needed when its step makes calls
                guards.need_ai("keywords")
            if not db.requeue(state.conn, row["id"]):  # moved since the check
                raise HTTPException(409, NOT_RESUMABLE)
        return {"job_id": row["id"]}

    def resume_reembed(row: dict) -> dict:
        """A stopped rebuild runs again: before its switch it embeds every item afresh, and
        during it (setup["switching"]) it finishes the switch. It is free: no estimate."""
        if refusal := jobs.not_resumable(row):
            raise HTTPException(409, refusal)
        name = row["collection"]
        state.retriever.spec(name)  # 404 when the store is gone
        with guards.not_adding(name):  # no job is queued between these checks and the requeue
            # switching_model reads the newest rebuild only, so an older one must not run
            # again; the newest is this one, so a stopped switch is its own to finish.
            newest = db.list_jobs(state.conn, kind="reembed", collection=name, limit=1)
            if newest[0]["id"] != row["id"]:
                raise HTTPException(
                    409, f"a newer rebuild of '{name}' replaced this one; start the rebuild again"
                )
            guards.refuse_busy(name, stopped_switch_ok=True)
            if not db.requeue(state.conn, row["id"]):  # moved since the check
                raise HTTPException(409, NOT_RESUMABLE)
        return {"job_id": row["id"]}

    @app.get("/status")
    def status() -> dict:
        worker, alive = state.worker, state.worker.alive()
        running = next(iter(db.list_jobs(state.conn, status="running", limit=1)), None)
        batches = state.ai.pending_batches()
        problems = []
        if not alive:
            problems.append("The worker thread has stopped; restart the service")
        if worker.error:
            problems.append(f"Worker error: {worker.error}")
        if line := stalled(running, batches, datetime.now(UTC)):
            problems.append(line)
        problems += job_problems(db.newest_jobs(state.conn))
        return {
            "worker": {"alive": alive, "running": running, "error": worker.error},
            "queue": db.list_jobs(state.conn, status="queued", limit=db.ALL)[::-1],
            "batches": batches,
            "problems": problems,
        }


def eval_set_fields(setup: dict) -> dict:
    """What a run asked: its eval set at a version, and how many of its questions name
    evidence and how many the store holds any of (None before it first runs)."""
    keys = ("eval_set", "eval_set_version", "evidence_named", "evidence_held")
    return {k: setup.get(k) for k in keys}


def run_summary(job: dict, summary: dict) -> dict:
    setup = job["setup"]
    return {
        "id": job["id"],
        "collection": job["collection"],
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"],
        "methods": setup.get("methods"),
        "split": setup.get("split"),
        "answer": setup.get("answer"),
        "model": setup.get("model"),
        "question_count": summary["question_count"],
        **eval_set_fields(setup),
        "usd_total": summary["usd_total"],
        # The verdict with what its notes need: the proxy check and the headline's scale.
        "answer_summary": summary["answer"]
        and {
            **summary["answer"],
            "proxy_trusted": summary["proxy_trusted"],
            "scale": summary["scale"],
        },
    }


def job_problems(newest: list[dict]) -> list[str]:
    """One line per collection and kind whose newest job went wrong."""
    lines = []
    for job in newest:
        # Progress is written after each re-keyed batch, so progress > 0 means items changed.
        # (The jobs row has no started_at, and every claimed job logs "started".)
        if job["kind"] == "rekey" and job["status"] in jobs.PARTLY_DONE and job["progress"] > 0:
            name = job["collection"]
            lines.append(f"Keyword lists on {name} are partly re-keyed; save them again")
        elif job["kind"] == "retag" and job["status"] in jobs.PARTLY_DONE and job["progress"] > 0:
            name = job["collection"]
            lines.append(f"The vocabulary on {name} is partly re-tagged; save it again")
        elif job["kind"] == "copy" and job["status"] in jobs.RESUMABLE:
            name, why = job["collection"], PROBLEM.get(job["status"], "was cancelled")
            lines.append(f"The copy into {name} {why}; delete {name} and copy it again")
        elif job["status"] in PROBLEM:
            lines.append(
                f"{job['kind'].capitalize()} {job['id'][:8]} on {job['collection']} "
                f"{PROBLEM[job['status']]}"
            )
    return lines


def stalled(job: dict | None, batches: list[dict], at: datetime) -> str | None:
    """A running job that has not reported for STALL_AFTER and is not waiting on a
    AI batch: the worker may be hung while its thread is still alive (spec §5.7)."""
    if job is None or any(b["job_id"] == job["id"] for b in batches):
        return None
    quiet = at - datetime.fromisoformat(job["updated_at"])
    if quiet < STALL_AFTER:
        return None
    # A duration, not a clock time: the server's clock is UTC and the console's is local.
    return (
        f"{job['kind'].capitalize()} {job['id'][:8]} on {job['collection']}"
        f" has made no progress for {int(quiet.total_seconds() // 60)} minutes"
    )
