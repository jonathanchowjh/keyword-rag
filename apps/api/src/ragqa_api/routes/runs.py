"""The run and job routes: evals and their results, jobs, resumes and /status."""

from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import (
    EstimateReply,
    Guards,
    check_methods,
    check_shown_price,
    estimate_reply,
)
from ragqa_core import db, jobs, search_defaults
from ragqa_core.connection import Connection
from ragqa_core.evaluator import retrieval_depth
from ragqa_core.loader import QUESTION_IMPORT_GONE, roles_needed
from ragqa_core.recipes import resolve_search
from ragqa_core.retriever import DEFAULT_K, DEFAULT_RRF_K, POOL
from ragqa_core.types import Question

NOT_RESUMABLE = "the job is no longer resumable"
STALL_AFTER = timedelta(minutes=10)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str
    question: str
    choices: list[str] | None = None
    methods: list[str] | None = None  # None: the Settings method
    k: int = Field(5, ge=1, le=50)
    rrf_k: int | None = Field(None, ge=0)  # None: the Settings RRF k
    answer: bool = False
    model: str | None = None
    search: dict | None = None  # overrides the collection's search settings in part


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
    rerank_pool: int | None = Field(None, ge=1, le=POOL)  # hybrid_rerank: None, the server picks
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

    @app.post(
        "/eval",
        status_code=202,
        responses={
            200: {
                "model": EstimateReply,
                "description": "Without confirm: what it would spend; nothing is queued",
            }
        },
    )
    def evaluate(body: EvalRequest):
        check_methods(body.methods)
        spec = db.require_collection(state.conn, body.collection)
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
            return estimate_reply(guards.with_missing(estimate, estimate["roles"], body.model))
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

    @app.post("/query")
    def query(body: QueryRequest) -> dict:
        """Each method's hits for a question; with no methods or rrf_k, Settings' own."""
        defaults: dict[str, Any] = search_defaults.in_force(state.conn)
        methods = body.methods if body.methods is not None else [str(defaults["method"])]
        rrf_k = body.rrf_k if body.rrf_k is not None else int(defaults["rrf_k"])
        check_methods(methods)
        spec = db.require_collection(state.conn, body.collection)
        # A store mid-switch is emptied, so every method is refused at once (409); a model
        # it can't load fails only the methods that embed, each with its error.
        if db.switching_model(state.conn, spec.name):
            state.indexer.check_model(spec)
        search = resolve_search(spec, body.search)
        if "keyword_rag" in methods and search.query_tags == "oracle":
            raise ValueError("oracle tags need a question's gold")  # eval only
        skip = tuple(db.disabled_sources(state.conn, body.collection))
        # Answered as a run answers: a Documents store's over Settings' evidence size.
        size = int(defaults["evidence_size"])
        q = Question(
            id="lab", text=body.question, choices=body.choices or [], gold_answers=[], gold_ids=[]
        )
        # An answer on a Documents store reads at least the evidence size, so retrieve that many.
        reads = size if spec.qa_type == "documents" and body.answer else 0
        depth = retrieval_depth(spec.qa_type, q, body.k, reads)
        results = {
            method: state.evaluator.query(
                spec, q, method, depth, rrf_k, search, skip, body.answer, body.model, size
            )
            for method in methods
        }
        return {"results": results}

    @app.get("/jobs")
    def list_jobs(
        collection: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        found = db.list_jobs(
            state.conn, kind=kind, collection=collection, status=status, limit=limit
        )
        return [job_row(j) for j in found]

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        return job_row(db.require_job(state.conn, job_id))

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        row = job(job_id)
        if row["status"] not in jobs.BUSY:
            raise HTTPException(409, f"job is {row['status']}")
        return {"status": db.request_cancel(state.conn, job_id)}

    @app.post(
        "/jobs/{job_id}/resume",
        status_code=202,
        responses={
            200: {
                "model": EstimateReply,
                "description": "Without confirm: what it would spend; nothing is queued. "
                "A rebuild is free and is requeued at once.",
            }
        },
    )
    def resume(job_id: str, body: ResumeRequest | None = None):
        """Without confirm, what resuming would cost (spec §5.4); with it, requeued."""
        row = job(job_id)
        match row["kind"]:
            case "copy":
                return resume_copy(row, body)
            case "reembed":
                return resume_reembed(row)
            case "tag_build":
                return resume_tag_build(row, body)
            case "import" if row["setup"].get("kind") == "questions":
                raise HTTPException(409, QUESTION_IMPORT_GONE)
            case "eval":
                if refusal := state.evaluator.resume_refusal(job_id):
                    raise HTTPException(409, refusal)
            case _:
                raise HTTPException(409, jobs.not_resumable(row))
        estimate = state.evaluator.estimate_resume(job_id)
        model = row["setup"]["model"]
        if body is None or not body.confirm:
            return estimate_reply(guards.with_missing(estimate, estimate["roles"], model))
        check_shown_price(estimate, body.calls, body.est_usd)
        guards.need_models(estimate, model)
        if not db.requeue(state.conn, job_id):  # moved since the check: a double confirm
            raise HTTPException(409, NOT_RESUMABLE)
        return {"job_id": job_id}

    def resume_copy(row: dict, body: ResumeRequest | None):
        """A stopped copy runs again in place: adding skips the items it added, and the
        build finishes what it left unbuilt, so only the rest is priced."""
        if refusal := jobs.not_resumable(row):
            raise HTTPException(409, refusal)
        source = row["setup"]["source"]
        target = db.require_collection(state.conn, row["collection"])
        db.require_collection(state.conn, source)  # 404 when the source is gone
        guards.refuse_busy(source)
        estimate = state.loader.estimate_copy(source, target)
        roles = roles_needed(target, estimate)
        if not (body and body.confirm):
            return estimate_reply(guards.with_missing(estimate, roles))
        check_shown_price(estimate, body.calls, body.est_usd)
        for role in roles:
            guards.need_ai(role)
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
        spec = db.require_collection(state.conn, name)  # 404 when the store is gone
        with guards.not_adding(name):  # no job is queued between these checks and the requeue
            # The keyword tags state reads the newest tag build only, so an older one must
            # not run again in its place.
            newest = db.list_jobs(state.conn, kind="tag_build", collection=name, limit=1)
            if newest[0]["id"] != row["id"]:
                raise HTTPException(409, f"a newer tag build of '{name}' exists; resume that one")
            guards.refuse_busy(name)
            if reply := guards.tag_build_gate(spec, body):
                return reply
            if not db.requeue(state.conn, row["id"]):  # moved since the check
                raise HTTPException(409, NOT_RESUMABLE)
        return {"job_id": row["id"]}

    def resume_reembed(row: dict) -> dict:
        """A stopped rebuild runs again: before its switch it embeds every item afresh, and
        during it (setup["switching"]) it finishes the switch. It is free: no estimate."""
        if refusal := jobs.not_resumable(row):
            raise HTTPException(409, refusal)
        name = row["collection"]
        db.require_collection(state.conn, name)  # 404 when the store is gone
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
        found = db.list_jobs(state.conn, status="running", limit=1)
        running = job_row(found[0]) if found else None
        batches = state.ai.pending_batches()
        problems = []
        if not alive:
            problems.append(
                {"text": "The worker thread has stopped; restart the service", "href": None}
            )
        if worker.error:
            problems.append({"text": f"Worker error: {worker.error}", "href": None})
        if line := stalled(running, batches, datetime.now(UTC)):
            problems.append(line)
        problems += job_problems(state.conn, db.newest_jobs(state.conn))
        return {
            "worker": {"alive": alive, "running": running, "error": worker.error},
            "queue": [
                job_row(j) for j in db.list_jobs(state.conn, status="queued", limit=db.ALL)[::-1]
            ],
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


def job_href(job: dict) -> str:
    """Where a job's results live: an eval's run page, or its store's page."""
    store = f"/c/{quote(job['collection'])}"
    return f"{store}/runs/{job['id']}" if job["kind"] == "eval" else store


def job_row(job: dict) -> dict:
    return {
        **job,
        "label": jobs.label(job),
        "resumable": jobs.resumable(job),
        "href": job_href(job),
    }


def job_problems(conn: Connection, newest: list[dict]) -> list[dict]:
    """One {text, href} per collection and kind whose newest job went wrong."""
    lines = []
    for job in newest:
        name = job["collection"]
        # Progress is written after each re-keyed batch, so progress > 0 means items changed.
        # (The jobs row has no started_at, and every claimed job logs "started".)
        if job["kind"] == "rekey" and job["status"] in jobs.PARTLY_DONE and job["progress"] > 0:
            text = f"Keyword lists on {name} are partly re-keyed; save them again"
            lines.append({"text": text, "href": job_href(job)})
        elif (
            job["kind"] == "retag"
            and job["status"] in jobs.PARTLY_DONE
            and db.retag_partly_done(conn, name)
        ):
            text = f"The vocabulary on {name} is partly re-tagged; save it again"
            lines.append({"text": text, "href": job_href(job)})
        elif job["kind"] == "copy" and job["status"] in jobs.RESUMABLE:
            why = jobs.PROBLEM.get(job["status"], "was cancelled")
            text = f"The copy into {name} {why}; resume it, or delete {name} and copy it again"
            lines.append({"text": text, "href": job_href(job)})
        elif job["status"] in jobs.PROBLEM:
            lines.append(problem(job, jobs.PROBLEM[job["status"]]))
    return lines


def problem(job: dict, why: str) -> dict:
    return {"text": f"{jobs.label(job)} ({job['id'][:8]}) {why}", "href": job_href(job)}


def stalled(job: dict | None, batches: list[dict], at: datetime) -> dict | None:
    """A running job that has not reported for STALL_AFTER and is not waiting on a
    AI batch: the worker may be hung while its thread is still alive (spec §5.7)."""
    if job is None or any(b["job_id"] == job["id"] for b in batches):
        return None
    quiet = at - datetime.fromisoformat(job["updated_at"])
    if quiet < STALL_AFTER:
        return None
    # A duration, not a clock time: the server's clock is UTC and the console's is local.
    return problem(job, f"has made no progress for {int(quiet.total_seconds() // 60)} minutes")
