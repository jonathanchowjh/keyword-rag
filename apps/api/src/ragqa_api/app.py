"""The HTTP API. Thin: every decision lives in ragqa_core; this maps it to HTTP.

Loads, imports, re-keys and evals are queued as jobs, and the worker thread
(ragqa_worker), started by the lifespan, runs them in order. A few routes still
work in the request: /documents indexes its items, and /query and a confirmed
proposal call Claude.

Run with `uvicorn --factory ragqa_api.app:create_app`.
"""

import base64
import json
import secrets
import shutil
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from anyio import from_thread
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aiconnect.ai import AI, BudgetExceeded, NoConnection, UnknownModel
from ragqa_core import db, imports
from ragqa_core.catalog import catalog
from ragqa_core.datasets import DATASETS, dev_slice
from ragqa_core.evaluator import ANSWER_TOKENS, Evaluator, answer_hits, measurable, runnable
from ragqa_core.imports import FIELDS, SOURCE_NAME, TooLarge
from ragqa_core.indexer import Indexer, ListsBusy, ModelMismatch
from ragqa_core.loader import Loader, sample_items
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
PER_METHOD_ERRORS = (MethodUnavailable, NoConnection, BudgetExceeded, ModelMismatch)
BUSY_STATE = {
    "load": "loading",
    "rekey": "rekeying",
    "eval": "evaluating",
    "import": "importing",
    "backfill": "labelling",
}
DOCUMENT_FIELDS = ("id", "text", "answer", "explanation", "meta")
SYNC_ITEMS = 20  # more items than this are imported by the worker
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


class ImportRequest(BaseModel):
    upload_id: str
    kind: Literal["items", "questions"]
    mapping: dict[str, str | None] | None = None
    source_name: str
    confirm: bool = False


class SourceSwitch(BaseModel):
    enabled: bool


class LeakedSwitch(BaseModel):
    include: bool


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


def check_mapping(kind: str, mapping: dict | None) -> None:
    unknown = [k for k in mapping or {} if k not in FIELDS[kind]]
    if unknown:
        raise HTTPException(422, f"unknown fields {unknown}; known: {list(FIELDS[kind])}")


def body_chunks(request: Request):
    """The request body, chunk by chunk, for a plain (threaded) route."""
    stream = request.stream().__aiter__()

    async def next_chunk() -> bytes:
        return await stream.__anext__()

    while True:
        try:
            chunk = from_thread.run(next_chunk)
        except StopAsyncIteration:
            return
        yield chunk


def check_methods(methods: list[str]) -> None:
    unknown = [m for m in methods if m not in METHOD_NAMES]
    if unknown:
        raise HTTPException(422, f"unknown methods {unknown}; known: {list(METHOD_NAMES)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Jobs left running by a stopped process are interrupted; queued ones stay queued."""
    db.interrupt_running(app.state.conn)
    # Collections whose points predate sources are labelled in place by a queued job.
    for spec in db.list_collections(app.state.conn):
        if not app.state.store.client.collection_exists(spec.name):
            continue
        app.state.store.ensure_indexes(spec.name)  # collections older than the indexes lack them
        if app.state.store.unsourced(spec.name) > 0 and not db.job_waiting_or_running(
            app.state.conn, "backfill", spec.name
        ):
            db.create_job(app.state.conn, "backfill", spec.name, {})
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
    # The AI's tables share the file; the old ledger, cache and batches move into them once.
    state.ai = AI(settings.home / "ragqa.db")
    db.move_ai_tables(state.conn)
    state.ai.apply_env(api_key=settings.api_key, model=settings.model, cap_usd=settings.cap_usd)
    state.indexer = Indexer(state.conn, state.store, state.models, state.ai)
    state.retriever = Retriever(state.conn, state.store, state.models, state.ai, state.indexer)
    state.evaluator = Evaluator(state.conn, state.retriever, state.models, state.ai)
    state.loader = Loader(state.conn, state.indexer, state.ai, settings.home)
    state.worker = Worker(state.conn, state.ai, state.indexer, state.loader, state.evaluator)

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
    app.add_exception_handler(TooLarge, error(413))
    app.add_exception_handler(UnknownModel, error(422))
    for exc in (MethodUnavailable, ModelMismatch, ListsBusy):
        app.add_exception_handler(exc, error(409))
    app.add_exception_handler(BudgetExceeded, error(402))

    def no_connection(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, NoConnection)
        return JSONResponse({"detail": no_ai(exc.role)}, 503)

    app.add_exception_handler(NoConnection, no_connection)

    # -- helpers ---------------------------------------------------------------

    def need_ai(role: str) -> None:
        """Refuse at once a request that would spend on a job with no usable model."""
        if not state.ai.ready(role):
            raise NoConnection(role)

    def source_rows(name: str) -> list[dict]:
        counts, disabled = state.store.source_counts(name), db.disabled_sources(state.conn, name)
        asked = db.question_sources(state.conn, name)
        return [
            {
                "source": s,
                "items": counts.get(s, 0),
                "questions": asked.get(s, 0),
                "enabled": s not in disabled,
            }
            for s in sorted(set(counts) | set(disabled) | set(asked))
        ]

    def open_upload(name: str, upload_id: str) -> tuple[Path, dict]:
        try:
            return imports.open_upload(state.settings.home, name, upload_id)
        except KeyError:
            raise HTTPException(404, f"no upload {upload_id!r} on '{name}'") from None

    def preview_upload(
        spec: CollectionSpec,
        path: Path,
        filename: str,
        kind: str,
        mapping: dict | None,
        source: str,
    ) -> dict:
        """The preview; a file that cannot be read is removed along with its upload."""
        known = set(state.store.ids(spec.name)) if kind == "questions" else None

        def owner_of(ids: list[str]) -> dict[str, str | None]:
            if kind == "items":
                return state.store.sources_of(spec.name, ids)
            asked = {q.id: q.meta.get("source") for q in db.get_questions(state.conn, spec.name)}
            return {i: asked[i] for i in ids if i in asked}

        try:
            return imports.preview(
                path, filename, kind, spec.qa_type, mapping, known, source, owner_of
            )
        except ValueError:
            shutil.rmtree(path.parent, ignore_errors=True)
            raise

    def estimate_upload(spec: CollectionSpec, upload_id: str, kind: str, shown: dict, source: str):
        """What importing would do; None until the mapping is complete."""
        if not shown["ready"]:
            return None
        setup = {
            "upload_id": upload_id,
            "kind": kind,
            "mapping": shown["mapping"],
            "source": source,
        }
        return state.loader.estimate_import(spec, setup)

    def leaked_count(name: str) -> int:
        return sum(1 for q in db.get_questions(state.conn, name) if q.meta.get("leaked"))

    def field_report(name: str, questions: list) -> dict:
        have = state.store.field_counts(name)
        return {
            "questions": {
                "total": len(questions),
                "solution": sum(1 for q in questions if q.meta.get("solution")),
                "skill": sum(1 for q in questions if q.meta.get("skill")),
                "gold_ids": sum(1 for q in questions if q.gold_ids),
            },
            "items": {"total": state.store.count(name), **have},
        }

    # -- routes ----------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "api_key": state.ai.ready("answer")}

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
        jobs = db.list_jobs(state.conn, collection=name, limit=db.ALL)
        busy, job = collection_state(jobs, count, lists is not None)
        questions = runnable(
            db.get_questions(state.conn, name), db.get_include_leaked(state.conn, name)
        )
        fields = field_report(name, questions)
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
            "sources": source_rows(name),
            "unlabelled": state.store.unsourced(name),
            "leaked": leaked_count(name),
            "include_leaked": db.get_include_leaked(state.conn, name),
            "fields": fields,
            "measurable": measurable(spec.qa_type, fields),
        }

    @app.delete("/collections/{name}", status_code=204)
    def delete_collection(name: str) -> Response:
        state.retriever.spec(name)
        running = db.list_jobs(state.conn, collection=name, status="running", limit=1)
        if running:
            raise HTTPException(
                409, f"a {running[0]['kind']} job is running on '{name}'; cancel it first"
            )
        for job in db.list_jobs(state.conn, collection=name, status="queued", limit=db.ALL):
            if db.request_cancel(state.conn, job["id"]) == "running":  # claimed meanwhile
                raise HTTPException(
                    409, f"a {job['kind']} job started on '{name}'; it is being cancelled"
                )
        state.store.drop(name)
        state.retriever.forget(name)
        db.delete_collection_rows(state.conn, name)
        imports.delete_uploads(state.settings.home, name)
        return Response(status_code=204)

    @app.post("/collections/{name}/documents", status_code=202)
    def add_documents(name: str, body: Documents):
        spec = state.retriever.spec(name)
        # An item added now would be keyed under mixed lists, miss a version or skew a run.
        for kind in ("load", "rekey", "eval", "import", "backfill"):
            if db.job_waiting_or_running(state.conn, kind, name):
                raise HTTPException(409, f"a {kind} job is queued or running on '{name}'; wait")
        if len(body.items) > SYNC_ITEMS:
            return queue_documents(spec, body)
        items = [Item(**i.model_dump()) for i in body.items]
        estimate = state.indexer.estimate_add_usd(spec, items, "hand")
        if estimate != 0 and not body.confirm:  # None: no model to price it, so not free
            return JSONResponse({"estimate": {"est_usd": estimate, "items": len(items)}}, 200)
        state.indexer.add(spec, items, "hand")
        return JSONResponse({"added": len(items)}, 200)

    def queue_documents(spec: CollectionSpec, body: Documents):
        """Many items are written as an upload and imported by the worker, as source 'hand'."""
        rows = [json.dumps(i.model_dump()).encode() + b"\n" for i in body.items]
        upload_id = imports.save_upload(state.settings.home, spec.name, "documents.jsonl", rows)
        mapping = {f: (f if f in DOCUMENT_FIELDS else None) for f in FIELDS["items"]}
        setup = {"upload_id": upload_id, "kind": "items", "mapping": mapping, "source": "hand"}
        queued = False
        try:  # the upload is the job's input: without a job it goes, whatever stopped it
            estimate = state.loader.estimate_import(spec, setup)
            # Only summaries can be unpaid-for: a proposal is priced only when it can be made.
            ready = not spec.summarize or state.ai.ready("summary")
            if estimate["est_usd"] != 0 and (not body.confirm or not ready):
                if body.confirm:
                    raise NoConnection("summary")
                return JSONResponse({"estimate": estimate}, 200)
            job_id = db.create_job(state.conn, "import", spec.name, setup)
            queued = True
            return {"job_id": job_id}
        finally:
            if not queued:
                shutil.rmtree(state.settings.home / "uploads" / upload_id, ignore_errors=True)

    @app.post("/collections/{name}/uploads")
    def add_upload(name: str, request: Request, filename: str, kind: Literal["items", "questions"]):
        """The raw file as the body; answers with a preview under a guessed mapping."""
        spec = state.retriever.spec(name)
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > imports.UPLOAD_LIMIT:
            raise HTTPException(413, f"the file is over {imports.UPLOAD_LIMIT} bytes")
        upload_id = imports.save_upload(state.settings.home, name, filename, body_chunks(request))
        path, _ = open_upload(name, upload_id)
        stem = Path(filename).stem
        source = "upload:" + (stem if SOURCE_NAME.match(stem) else "upload")
        shown = preview_upload(spec, path, filename, kind, None, source)
        try:
            estimate = estimate_upload(spec, upload_id, kind, shown, source)
        except Exception:  # the caller never learns the upload's id: nothing could reuse it
            shutil.rmtree(path.parent, ignore_errors=True)
            raise
        return {
            "upload_id": upload_id,
            "filename": filename,
            "preview": shown,
            "estimate": estimate,
        }

    @app.post("/collections/{name}/imports", status_code=202)
    def start_import(name: str, body: ImportRequest):
        spec = state.retriever.spec(name)
        path, meta = open_upload(name, body.upload_id)
        if not SOURCE_NAME.match(body.source_name):
            raise HTTPException(422, "the source name needs 1-64 letters, digits, space, _ . -")
        check_mapping(body.kind, body.mapping)
        source = "upload:" + body.source_name
        shown = preview_upload(spec, path, meta["filename"], body.kind, body.mapping, source)
        estimate = estimate_upload(spec, body.upload_id, body.kind, shown, source)
        if not body.confirm:
            return JSONResponse({"preview": shown, "estimate": estimate}, 200)
        if estimate is None:
            why = (
                f"map these fields first: {shown['missing']}"
                if shown["missing"]
                else "no usable rows"
            )
            raise HTTPException(422, why)
        if estimate["est_usd"] != 0 and spec.summarize:  # None: no model to price it
            need_ai("summary")
        setup = {
            "upload_id": body.upload_id,
            "kind": body.kind,
            "mapping": shown["mapping"],
            "source": source,
        }
        return {"job_id": db.create_job(state.conn, "import", name, setup)}

    @app.get("/collections/{name}/sources")
    def list_sources(name: str) -> list[dict]:
        state.retriever.spec(name)
        return source_rows(name)

    def known_source(name: str, source: str) -> dict:
        state.retriever.spec(name)
        row = next((r for r in source_rows(name) if r["source"] == source), None)
        if row is None:
            raise HTTPException(404, f"no source {source!r} on '{name}'")
        return row

    @app.patch("/collections/{name}/sources/{source}")
    def switch_source(name: str, source: str, body: SourceSwitch) -> dict:
        row = known_source(name, source)
        if db.job_running(state.conn, "eval", name):
            raise ListsBusy(f"an eval is running on '{name}'; wait for it to finish")
        if db.job_waiting_or_running(state.conn, "backfill", name):
            raise HTTPException(409, "sources are still being labelled; try again shortly")
        if not body.enabled and row["enabled"]:
            others = [
                r
                for r in source_rows(name)
                if r["enabled"] and r["items"] > 0 and r["source"] != source
            ]
            if row["items"] > 0 and not others:
                raise HTTPException(409, "at least one source must stay on")
        db.set_source_enabled(state.conn, name, source, body.enabled)
        return {**row, "enabled": body.enabled}

    @app.delete("/collections/{name}/sources/{source}", status_code=204)
    def delete_source(name: str, source: str) -> Response:
        known_source(name, source)
        for status in ("running", "queued"):
            busy = db.list_jobs(state.conn, collection=name, status=status, limit=1)
            if busy:
                raise HTTPException(409, f"a {busy[0]['kind']} job is {status} on '{name}'; wait")
        left = [r for r in source_rows(name) if r["source"] != source and r["items"] > 0]
        if left and not any(r["enabled"] for r in left):
            raise HTTPException(409, "at least one source must stay on")
        state.store.delete_source(name, source)
        db.delete_source_questions(state.conn, name, source)
        db.set_source_enabled(state.conn, name, source, True)
        state.retriever.forget(name)
        return Response(status_code=204)

    @app.get("/collections/{name}/questions")
    def list_questions(
        name: str, leaked: bool | None = None, limit: int = 50, offset: int = 0
    ) -> dict:
        state.retriever.spec(name)
        everything = db.get_questions(state.conn, name)
        shown = [q for q in everything if leaked is None or bool(q.meta.get("leaked")) == leaked]
        page = shown[max(offset, 0) : max(offset, 0) + max(limit, 0)]
        return {
            "total": len(shown),
            "leaked": sum(1 for q in everything if q.meta.get("leaked")),
            "include_leaked": db.get_include_leaked(state.conn, name),
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "source": q.meta.get("source"),
                    "leaked": bool(q.meta.get("leaked")),
                    "has_gold": bool(q.gold_ids),
                    "has_solution": bool(q.meta.get("solution")),
                }
                for q in page
            ],
        }

    @app.patch("/collections/{name}/questions/leaked")
    def switch_leaked(name: str, body: LeakedSwitch) -> dict:
        state.retriever.spec(name)
        db.set_include_leaked(state.conn, name, body.include)
        return {"include_leaked": body.include}

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
        if spec.summarize:
            need_ai("summary")
        # What the loader spends on: summaries, and a proposal when there are no lists.
        proposes = state.ai.ready("keywords") and db.get_keyword_lists(state.conn, name) is None
        if (spec.summarize or proposes) and not body.confirm:
            estimate = state.loader.estimate_load(spec, body.dataset, body.limit)
            return JSONResponse({"estimate": estimate}, 200)
        job_id = db.create_job(
            state.conn, "load", name, {"dataset": body.dataset, "limit": body.limit}
        )
        return {"job_id": job_id}

    @app.post("/collections/{name}/keyword-lists/propose")
    def propose(name: str, body: ProposeRequest | None = None):
        spec = state.retriever.spec(name)
        need_ai("keywords")
        sample = sample_items(state.store, name, tuple(db.disabled_sources(state.conn, name)))
        if not (body and body.confirm):
            estimate = state.indexer.estimate_propose_usd(spec, sample)
            return JSONResponse({"estimate": {"est_usd": estimate, "calls": 1}}, 200)
        return asdict(state.indexer.propose_lists(spec, sample))

    @app.get("/collections/{name}/keyword-lists")
    def list_versions(name: str) -> list[dict]:
        """Every saved version of the lists, newest first, with a count per list."""
        spec = state.retriever.spec(name)
        return [
            {
                "version": v["version"],
                "created_at": v["created_at"],
                "counts": {k: len(items) for k, items in asdict(v["lists"]).items()},
                "active": v["version"] == spec.keyword_version,
            }
            for v in db.list_keyword_versions(state.conn, name)
        ]

    @app.get("/collections/{name}/keyword-lists/{version}")
    def get_version(name: str, version: str) -> dict:
        spec = state.retriever.spec(name)
        v = db.get_keyword_version(state.conn, name, version)
        if v is None:
            raise HTTPException(404, f"no keyword-list version {version!r} on '{name}'")
        return {
            "version": version,
            "created_at": v["created_at"],
            "active": version == spec.keyword_version,
            "lists": asdict(v["lists"]),
        }

    @app.put("/collections/{name}/keyword-lists", status_code=202)
    def put_lists(name: str, body: Lists) -> dict:
        """Saves the lists as a version; a queued rekey job switches to it once every
        item is re-keyed (spec §5.5)."""
        spec = state.retriever.spec(name)
        if db.job_running(state.conn, "eval", name):
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
            # Every call the method makes: query summaries, hops and the answer.
            with state.ai.tally() as spent:
                try:
                    hits = state.retriever.retrieve(
                        body.collection, body.question, method, body.k, body.rrf_k
                    )
                    out["hits"] = [asdict(h) for h in hits]
                    if body.answer:
                        system, user = answer_prompt(
                            spec.qa_type,
                            body.question,
                            body.choices,
                            answer_hits(spec.qa_type, hits),
                        )
                        done = state.ai.complete(
                            user,
                            role="answer",
                            system=system,
                            model=body.model,
                            max_tokens=ANSWER_TOKENS[spec.qa_type],
                            purpose="query",
                        )
                        out["answer"] = parse_answer(spec.qa_type, done.text, bool(body.choices))
                except PER_METHOD_ERRORS as e:
                    out["error"] = no_ai(e.role) if isinstance(e, NoConnection) else str(e)
            out["usd"] = spent["usd"]
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
        if estimate["est_usd"] != 0:  # without a model for each job it would stop at once
            for role in estimate["roles"]:
                need_ai(role)
        job_id = state.evaluator.queue(
            body.collection, body.split, body.methods, body.answer, body.model, body.k, body.rrf_k
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

    # One row per question with every method's row, where /items is one method's rows.
    @app.get("/eval/{job_id}/questions")
    def eval_questions(job_id: str, disagree: bool = False) -> list[dict]:
        try:
            return state.evaluator.questions(job_id, disagree)
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
        estimate = state.evaluator.estimate_resume(job_id)
        if estimate["est_usd"] != 0:
            for role in estimate["roles"]:
                need_ai(role)
        if not db.requeue(state.conn, job_id):  # moved since the check: a double confirm
            raise HTTPException(409, "the job is no longer resumable")
        return {"job_id": job_id}

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

    @app.get("/catalog")
    def get_catalog() -> dict:
        return catalog()

    @app.get("/budget")
    def budget() -> dict:
        return {
            "month": datetime.now(UTC).strftime("%Y-%m"),
            "spent_usd": state.ai.month_spend(),
            "cap_usd": state.ai.cap_usd,
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
        if job["kind"] == "rekey" and job["status"] in REKEY_PARTLY_DONE and job["progress"] > 0:
            name = job["collection"]
            lines.append(f"Keyword lists on {name} are partly re-keyed; save them again")
        elif job["status"] in PROBLEM:
            lines.append(
                f"{job['kind'].capitalize()} {job['id'][:8]} on {job['collection']} "
                f"{PROBLEM[job['status']]}"
            )
    return lines


def no_ai(role: str) -> str:
    return f"no AI model is set up for the '{role}' job; choose one on the Settings page"


def stalled(job: dict | None, batches: list[dict], at: datetime) -> str | None:
    """A running job that has not reported for STALL_AFTER and is not waiting on a
    Claude batch: the worker may be hung while its thread is still alive (spec §5.7)."""
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
