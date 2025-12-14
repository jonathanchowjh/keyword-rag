"""The HTTP API. Thin: every decision lives in ragqa_core; this maps it to HTTP.

Routes never run long work: loads, re-keys and evals are queued as jobs, and
the worker thread (ragqa_worker), started by the lifespan, runs them in order.

Run with `uvicorn --factory ragqa_api.app:create_app`.
"""

import base64
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ragqa_core import db
from ragqa_core.catalog import catalog
from ragqa_core.datasets import DATASETS, dev_slice
from ragqa_core.datasets import load as load_dataset
from ragqa_core.evaluator import ANSWER_TOKENS, Evaluator, answer_hits
from ragqa_core.indexer import Indexer, ListsBusy, ModelMismatch
from ragqa_core.llm import BudgetExceeded, Claude, NoApiKey, UnknownModel
from ragqa_core.loader import Loader, limited, propose_sample, sample_items
from ragqa_core.models import Models
from ragqa_core.prompts import answer_prompt, parse_answer
from ragqa_core.retriever import (
    KEYWORD_METHODS,
    METHOD_NAMES,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
)
from ragqa_core.settings import Settings, load_settings
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item, KeywordLists, QaType
from ragqa_worker.worker import Worker

STATIC = Path(__file__).parent / "static"
PER_METHOD_ERRORS = (MethodUnavailable, NoApiKey, BudgetExceeded, ModelMismatch)
ALL = 1_000_000  # list_jobs limit meaning "every job"
BUSY_STATE = {"load": "loading", "rekey": "rekeying", "eval": "evaluating"}
STALL_AFTER = timedelta(minutes=10)
PROBLEM = {
    "failed": "failed",
    "interrupted": "was interrupted by a restart",
    "stopped_budget": "stopped at the monthly cap",
    "stopped_no_key": "stopped: needs ANTHROPIC_API_KEY",
}
REKEY_PARTLY_DONE = ("failed", "cancelled", "interrupted")


class NewCollection(BaseModel):
    name: str
    qa_type: QaType
    summarize: bool = False


class ItemIn(BaseModel):
    id: str
    text: str
    answer: str | None = None
    explanation: str | None = None
    meta: dict = Field(default_factory=dict)


class Documents(BaseModel):
    items: list[ItemIn]
    confirm: bool = False


class LoadRequest(BaseModel):
    dataset: str
    limit: int | None = None
    confirm: bool = False


class ProposeRequest(BaseModel):
    confirm: bool = False


class Lists(BaseModel):
    topic: list[str]
    method: list[str]
    common: list[str]


class QueryRequest(BaseModel):
    collection: str
    question: str
    choices: list[str] | None = None
    methods: list[str] = Field(default_factory=lambda: ["hybrid", "kw_individual"])
    k: int = Field(5, ge=1, le=50)
    rrf_k: int = Field(60, ge=0)
    answer: bool = False
    model: str | None = None


class EvalRequest(BaseModel):
    collection: str
    split: Literal["dev", "full"] = "dev"
    methods: list[str]
    answer: bool = False
    model: str | None = None
    k: int = Field(5, ge=1, le=50)
    rrf_k: int = Field(60, ge=0)
    confirm: bool = False


class ResumeRequest(BaseModel):
    confirm: bool = False


def spec_json(spec: CollectionSpec) -> dict:
    return asdict(spec)


def check_methods(methods: list[str]) -> None:
    unknown = [m for m in methods if m not in METHOD_NAMES]
    if unknown:
        raise HTTPException(422, f"unknown methods {unknown}; known: {list(METHOD_NAMES)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Jobs left running by a stopped process are interrupted; queued ones stay queued."""
    db.interrupt_running(app.state.conn)
    app.state.worker.start()
    try:
        yield
    finally:
        app.state.worker.stop()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="rag-qa", version="0.1.0", lifespan=lifespan)
    state = app.state
    state.settings = settings
    state.conn = db.connect(settings.home / "ragqa.db")
    state.store = Store(
        settings.qdrant_url, None if settings.qdrant_url else settings.home / "qdrant"
    )
    state.models = Models(
        settings.dense_model,
        settings.judge_model,
        settings.rerank_model,
        settings.home / "models",
    )
    state.claude = Claude(state.conn, settings.api_key, settings.cap_usd, settings.model)
    state.indexer = Indexer(state.conn, state.store, state.models, state.claude)
    state.retriever = Retriever(state.conn, state.store, state.models, state.claude, state.indexer)
    state.evaluator = Evaluator(state.conn, state.retriever, state.models, state.claude)
    state.loader = Loader(state.conn, state.indexer, state.claude, settings.home)
    state.worker = Worker(state.conn, state.claude, state.indexer, state.loader, state.evaluator)

    # -- auth ------------------------------------------------------------------

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        s: Settings = request.app.state.settings
        if bool(s.username) != bool(s.password):
            return JSONResponse(
                {"detail": "set both RAGQA_USERNAME and RAGQA_PASSWORD, or neither"}, 503
            )
        if s.username and s.password and not authorised(request, s.username, s.password):
            return JSONResponse(
                {"detail": "authentication required"},
                401,
                headers={"WWW-Authenticate": "Basic"},
            )
        return await call_next(request)

    # -- errors ----------------------------------------------------------------

    def error(status: int):
        def handler(request: Request, exc: Exception) -> JSONResponse:
            return JSONResponse({"detail": str(exc)}, status)

        return handler

    app.add_exception_handler(UnknownCollection, error(404))
    app.add_exception_handler(ValueError, error(422))
    app.add_exception_handler(UnknownModel, error(422))
    for exc in (MethodUnavailable, ModelMismatch, ListsBusy):
        app.add_exception_handler(exc, error(409))
    app.add_exception_handler(BudgetExceeded, error(402))
    app.add_exception_handler(NoApiKey, error(503))

    # -- routes ----------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "api_key": bool(state.claude.api_key)}

    @app.post("/collections", status_code=201)
    def create_collection(body: NewCollection) -> dict:
        if db.get_collection(state.conn, body.name):
            raise HTTPException(409, f"collection '{body.name}' already exists")
        return spec_json(state.indexer.create(body.name, body.qa_type, body.summarize))

    @app.get("/collections")
    def list_collections() -> list[dict]:
        return [spec_json(s) for s in db.list_collections(state.conn)]

    @app.get("/collections/{name}")
    def get_collection(name: str) -> dict:
        spec = state.retriever.spec(name)
        lists = db.get_keyword_lists(state.conn, name)
        available = [m for m in METHOD_NAMES if lists is not None or m not in KEYWORD_METHODS]
        count = state.store.count(name)
        jobs = db.list_jobs(state.conn, collection=name, limit=ALL)
        busy, job = collection_state(jobs, count, lists is not None)
        questions = db.get_questions(state.conn, name)
        return {
            **spec_json(spec),
            "count": count,
            "keyword_lists": asdict(lists) if lists else None,
            "methods_available": available,
            "state": busy,
            "progress": job["progress"] if job and job["status"] == "running" else None,
            "job": job,
            "questions": len(questions),
            "dev_questions": len(dev_slice(questions)),
            "answer": state.evaluator.collection_answer(name),
            "runs": sum(1 for j in jobs if j["kind"] == "eval"),
        }

    @app.delete("/collections/{name}", status_code=204)
    def delete_collection(name: str) -> Response:
        state.retriever.spec(name)
        running = db.list_jobs(state.conn, collection=name, status="running", limit=1)
        if running:
            raise HTTPException(
                409, f"a {running[0]['kind']} job is running on '{name}'; cancel it first"
            )
        for job in db.list_jobs(state.conn, collection=name, status="queued", limit=ALL):
            if db.request_cancel(state.conn, job["id"]) == "running":  # claimed meanwhile
                raise HTTPException(
                    409, f"a {job['kind']} job started on '{name}'; it is being cancelled"
                )
        state.store.drop(name)
        db.delete_collection_rows(state.conn, name)
        return Response(status_code=204)

    @app.post("/collections/{name}/documents")
    def add_documents(name: str, body: Documents) -> dict:
        spec = state.retriever.spec(name)
        items = [Item(**i.model_dump()) for i in body.items]
        estimate = state.indexer.estimate_add_usd(spec, items)
        if estimate > 0 and not body.confirm:
            return {"estimate": {"est_usd": estimate, "items": len(items)}}
        state.indexer.add(spec, items)
        return {"added": len(items)}

    @app.post("/collections/{name}/load", status_code=202)
    def load(name: str, body: LoadRequest):
        spec = state.retriever.spec(name)
        if body.dataset not in DATASETS:
            raise HTTPException(422, f"unknown dataset {body.dataset!r}; known: {list(DATASETS)}")
        if DATASETS[body.dataset] != spec.qa_type:
            raise HTTPException(
                409,
                f"dataset {body.dataset} is {DATASETS[body.dataset]};"
                f" collection '{name}' is {spec.qa_type}",
            )
        if spec.summarize and not state.claude.api_key:
            raise NoApiKey()
        # What the loader spends on: summaries, and a Claude proposal when there are no lists.
        proposes = bool(state.claude.api_key) and db.get_keyword_lists(state.conn, name) is None
        if (spec.summarize or proposes) and not body.confirm:
            # The dataset is read (and cached) to price them.
            items, questions = load_dataset(body.dataset, settings.home)
            items, _ = limited(items, questions, body.limit)
            estimate = state.indexer.estimate_add_usd(spec, items)
            calls = len(items) if spec.summarize else 0
            if proposes:
                estimate += state.indexer.estimate_propose_usd(spec, propose_sample(items))
                calls += 1
            return JSONResponse(
                {"estimate": {"est_usd": estimate, "items": len(items), "calls": calls}}, 200
            )
        job_id = db.create_job(
            state.conn, "load", name, {"dataset": body.dataset, "limit": body.limit}
        )
        return {"job_id": job_id}

    @app.post("/collections/{name}/keyword-lists/propose")
    def propose(name: str, body: ProposeRequest | None = None):
        spec = state.retriever.spec(name)
        if not state.claude.api_key:
            raise NoApiKey()
        sample = sample_items(state.store, name)
        if not (body and body.confirm):
            estimate = state.indexer.estimate_propose_usd(spec, sample)
            return JSONResponse({"estimate": {"est_usd": estimate, "calls": 1}}, 200)
        return asdict(state.indexer.propose_lists(spec, sample))

    @app.put("/collections/{name}/keyword-lists", status_code=202)
    def put_lists(name: str, body: Lists) -> dict:
        """Saves the lists as a version; a queued rekey job switches to it once every
        item is re-keyed (spec §5.5)."""
        spec = state.retriever.spec(name)
        if db.running_eval(state.conn, name):
            raise ListsBusy(f"an eval is running on '{name}'; wait for it to finish")
        version = state.indexer.save_lists(spec, KeywordLists(**body.model_dump()))
        job_id = db.create_job(state.conn, "rekey", name, {"version": version})
        return {"version": version, "job_id": job_id}

    @app.post("/query")
    def query(body: QueryRequest) -> dict:
        check_methods(body.methods)
        spec = state.retriever.spec(body.collection)
        results = {}
        for method in body.methods:
            out: dict = {"hits": [], "answer": None, "usd": 0.0, "error": None}
            try:
                hits = state.retriever.retrieve(
                    body.collection, body.question, method, body.k, body.rrf_k
                )
                out["hits"] = [asdict(h) for h in hits]
                if body.answer:
                    system, user = answer_prompt(
                        spec.qa_type, body.question, body.choices, answer_hits(spec.qa_type, hits)
                    )
                    done = state.claude.complete(
                        user,
                        system=system,
                        model=body.model,
                        max_tokens=ANSWER_TOKENS[spec.qa_type],
                        purpose="query",
                    )
                    out["answer"] = parse_answer(spec.qa_type, done.text)
                    out["usd"] = done.usd
            except PER_METHOD_ERRORS as e:
                out["error"] = str(e)
            results[method] = out
        return {"results": results}

    @app.post("/eval", status_code=202)
    def evaluate(body: EvalRequest):
        check_methods(body.methods)
        estimate = state.evaluator.estimate(
            body.collection, body.split, body.methods, body.answer, body.model
        )
        if not body.confirm:  # spec §7.3: an eval without confirm only estimates
            return JSONResponse({"estimate": estimate}, 200)
        job_id = state.evaluator.queue(
            body.collection, body.split, body.methods, body.answer, body.model, body.k, body.rrf_k
        )
        return {"job_id": job_id, "estimate": estimate}

    @app.get("/eval")
    def list_runs(collection: str | None = None) -> list[dict]:
        jobs = db.list_jobs(state.conn, kind="eval", collection=collection, limit=ALL)
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
            "log": job["log"],
            "created_at": job["created_at"],
        }

    @app.delete("/eval/{job_id}", status_code=204)
    def delete_run(job_id: str) -> Response:
        job = db.get_job(state.conn, job_id)
        if job is None or job["kind"] != "eval":
            raise HTTPException(404, f"no run {job_id}")
        if job["status"] in ("queued", "running"):
            raise HTTPException(409, f"the run is {job['status']}; cancel it first")
        db.delete_job(state.conn, job_id)
        return Response(status_code=204)

    @app.get("/eval/{job_id}/items")
    def eval_items(job_id: str, method: str, wrong: bool = False) -> list[dict]:
        try:
            return state.evaluator.items(job_id, method, wrong)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None

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
        if row["status"] not in ("queued", "running"):
            raise HTTPException(409, f"job is {row['status']}")
        return {"status": db.request_cancel(state.conn, job_id)}

    @app.post("/jobs/{job_id}/resume", status_code=202)
    def resume(job_id: str, body: ResumeRequest | None = None):
        """Without confirm, what resuming would cost (spec §5.4); with it, requeued."""
        try:
            refusal = state.evaluator.resume_refusal(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None
        if refusal:
            raise HTTPException(409, refusal)
        if body is None or not body.confirm:
            return JSONResponse({"estimate": state.evaluator.estimate_resume(job_id)}, 200)
        if not db.requeue(state.conn, job_id):  # moved since the check: a double confirm
            raise HTTPException(409, "the job is no longer resumable")
        return {"job_id": job_id}

    @app.get("/status")
    def status() -> dict:
        worker, alive = state.worker, state.worker.alive()
        running = next(iter(db.list_jobs(state.conn, status="running", limit=1)), None)
        batches = db.pending_batches(state.conn)
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
            "queue": db.list_jobs(state.conn, status="queued", limit=ALL)[::-1],
            "batches": batches,
            "problems": problems,
        }

    @app.get("/catalog")
    def get_catalog() -> dict:
        return catalog()

    @app.get("/budget")
    def budget() -> dict:
        return {
            "month": datetime.now(UTC).strftime("%Y-%m"),
            "spent_usd": state.claude.month_spend(),
            "cap_usd": state.claude.cap_usd,
            "note": "This service's own spend; the Console shows the credit balance.",
        }

    # -- pages: plain HTML plus the shared ui.js, served from /static (behind the same auth) --

    pages = {
        "/": "collections",
        "/c/{name}": "collection",
        "/c/{name}/runs/{id}": "run",
        "/worker": "worker",
        "/lab": "lab",
    }
    for route, page in pages.items():
        app.add_api_route(
            route, page_route(STATIC / f"{page}.html"), methods=["GET"], include_in_schema=False
        )
    app.add_api_route(
        "/board", lambda: RedirectResponse("/", status_code=307), include_in_schema=False
    )
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    return app


def page_route(path: Path):
    def serve() -> FileResponse:
        return FileResponse(path, media_type="text/html")

    return serve


def authorised(request: Request, username: str, password: str) -> bool:
    header = request.headers.get("authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        given_user, _, given_pass = base64.b64decode(header[6:]).decode().partition(":")
    except (ValueError, UnicodeDecodeError):
        return False
    # Compared as bytes: compare_digest refuses str with non-ASCII characters.
    return secrets.compare_digest(given_user.encode(), username.encode()) & secrets.compare_digest(
        given_pass.encode(), password.encode()
    )


def collection_state(jobs: list[dict], count: int, has_lists: bool) -> tuple[str, dict | None]:
    """A collection's state and the job behind it; `jobs` are its jobs, newest first."""
    running = next((j for j in jobs if j["status"] == "running"), None)
    if running:
        return BUSY_STATE.get(running["kind"], "busy"), running
    queued = [j for j in jobs if j["status"] == "queued"]
    if queued:
        return "queued", queued[-1]  # the one the worker takes first
    if count == 0:
        return "empty", None
    return ("ready" if has_lists else "no_lists"), None


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
        "usd_total": summary["usd_total"],
        "answer_summary": summary["answer"],
    }


def job_problems(newest: list[dict]) -> list[str]:
    """One line per collection and kind whose newest job went wrong."""
    lines = []
    for job in newest:
        if job["kind"] == "rekey" and job["status"] in REKEY_PARTLY_DONE:
            name = job["collection"]
            lines.append(f"Keyword lists on {name} are partly re-keyed; save them again")
        elif job["status"] in PROBLEM:
            lines.append(
                f"{job['kind'].capitalize()} {job['id'][:4]} on {job['collection']} "
                f"{PROBLEM[job['status']]}"
            )
    return lines


def stalled(job: dict | None, batches: list[dict], at: datetime) -> str | None:
    """A running job that has not reported for STALL_AFTER and is not waiting on a
    Claude batch: the worker may be hung while its thread is still alive (spec §5.7)."""
    if job is None or any(b["job_id"] == job["id"] for b in batches):
        return None
    updated = datetime.fromisoformat(job["updated_at"])
    if at - updated < STALL_AFTER:
        return None
    return (
        f"{job['kind'].capitalize()} {job['id'][:8]} on {job['collection']}"
        f" has made no progress since {updated:%H:%M}"
    )
