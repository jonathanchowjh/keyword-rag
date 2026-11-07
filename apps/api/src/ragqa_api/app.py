"""The HTTP API. Thin: every decision lives in ragqa_core; this maps it to HTTP.

Loads, imports, copies, re-keys and evals are queued as jobs, and the worker thread
(ragqa_worker), started by the lifespan, runs them in order. A few routes still
work in the request: /documents indexes its items, and /query and a confirmed
proposal call the AI.

Run with `uvicorn --factory ragqa_api.app:create_app`.
"""

import base64
import json
import secrets
import shutil
import threading
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from anyio import from_thread
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from aiconnect.ai import (
    AI,
    AwaitingReplies,
    BudgetExceeded,
    Conflict,
    NoConnection,
    ProviderError,
    UnknownModel,
    UnknownPrice,
)
from aiconnect.presets import PRESETS
from ragqa_core import db, imports
from ragqa_core.builder import Builder, builds
from ragqa_core.catalog import catalog
from ragqa_core.chats import MAX_EVIDENCE, AlreadyAnswered, Chats, UnknownChat
from ragqa_core.datasets import DATASETS, dev_slice
from ragqa_core.evaluator import (
    EvalSetMismatch,
    Evaluator,
    answer_hits,
    answer_tokens,
    measurable,
    no_ai,
    read_answer,
    runnable,
)
from ragqa_core.imports import FIELDS, SOURCE_NAME, TooLarge
from ragqa_core.indexer import Indexer, ListsBusy, ModelMismatch
from ragqa_core.loader import Loader, sample_items
from ragqa_core.models import Models
from ragqa_core.prompts import (
    CHAT_PASSAGE_CHARS,
    CHAT_TOKENS,
    CHAT_TURNS,
    answer_prompt,
    chat_prompt,
    cited_numbers,
    edge_order,
)
from ragqa_core.recipes import search_settings
from ragqa_core.retriever import (
    METHOD_NAMES,
    PER_METHOD_ERRORS,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
    available_methods,
)
from ragqa_core.router import Router
from ragqa_core.settings import Settings, load_settings
from ragqa_core.store import Store
from ragqa_core.types import (
    CollectionSpec,
    Hit,
    Item,
    KeywordLists,
    QaType,
    Question,
    SearchSettings,
)
from ragqa_core.vocabulary import clean_terms
from ragqa_worker.worker import Worker

STATIC = Path(__file__).parent / "static"
BUSY_STATE = {
    "load": "loading",
    "rekey": "rekeying",
    "retag": "retagging",
    "copy": "copying",
    "eval": "evaluating",
    "import": "importing",
    "backfill": "labelling",
}
DOCUMENT_FIELDS = ("id", "text", "answer", "explanation", "meta")
SYNC_ITEMS = 20  # more items than this are imported by the worker
TEXT_SUFFIXES = (".txt", ".md")  # what the chat can add to a store
STALL_AFTER = timedelta(minutes=10)
PROBLEM = {
    "failed": "failed",
    "interrupted": "was interrupted by a restart",
    "stopped_budget": "stopped at the monthly cap",
    "stopped_no_key": "stopped: no AI model is set up for this job (see the Settings page)",
    "stopped_replies": "stopped: its prompts wait for replies by hand",
}
KNOWN_ROLES = ["default", "answer", "summary", "keywords", "query", "rewrite"]
REKEY_PARTLY_DONE = ("failed", "cancelled", "interrupted")
COPY_UNFINISHED = (*REKEY_PARTLY_DONE, "stopped_budget", "stopped_no_key", "stopped_replies")


class NewCollection(BaseModel):
    name: str
    qa_type: QaType
    summarize: bool = False
    recipe: str | None = None


class NewEvalSet(BaseModel):
    name: str
    qa_type: QaType


class QuestionIn(BaseModel):
    id: str
    text: str
    choices: list[str] | None = None
    gold_answers: list[str] = Field(default_factory=list)
    gold_ids: list[str] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


class QuestionsPut(BaseModel):
    questions: list[QuestionIn]
    base_version: str | None = None


class EvalSetPatch(BaseModel):
    include_leaked: bool


class CollectionPatch(BaseModel):
    """Only the fields sent change."""

    eval_set: str | None = None  # null: the set named after the store, if any
    route_profile: str | None = None  # null or blank: the router's draft


class EvalSetImport(BaseModel):
    upload_id: str
    mapping: dict[str, str | None] | None = None
    source_name: str


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


class CopyRequest(BaseModel):
    name: str
    recipe: str
    build: dict | None = None  # overrides the recipe's build settings in part
    confirm: bool = False


class ProposeRequest(BaseModel):
    confirm: bool = False


class Lists(BaseModel):
    topic: list[str]
    method: list[str]
    common: list[str]


class Vocabulary(BaseModel):
    terms: list[str]
    confirm: bool = False


class QueryRequest(BaseModel):
    collection: str
    question: str
    choices: list[str] | None = None
    methods: list[str] = Field(default_factory=lambda: ["hybrid", "kw_individual"])
    k: int = Field(5, ge=1, le=50)
    rrf_k: int = Field(60, ge=0)
    answer: bool = False
    model: str | None = None
    search: dict | None = None  # overrides the collection's search settings in part


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    collection: str
    messages: list[ChatTurn] = Field(min_length=1)  # the last is the user's question
    method: str | None = None  # keyword_rag with a vocabulary, else hybrid
    k: int = Field(5, ge=1, le=20)
    model: str | None = None


class NewChat(BaseModel):
    stores: list[str]
    title: str | None = None


class ChatPatch(BaseModel):
    """An absent field changes nothing; an explicit null is refused, not ignored."""

    title: str | None = None
    stores: list[str] | None = None
    evidence_size: int | None = Field(None, ge=1, le=MAX_EVIDENCE)
    rewrite: bool | None = None
    routing: Literal["all", "profiles"] | None = None

    @model_validator(mode="before")
    @classmethod
    def no_nulls(cls, data: object) -> object:
        if isinstance(data, dict):
            nulls = [f for f, value in data.items() if value is None and f in cls.model_fields]
            if nulls:
                raise ValueError(f"{', '.join(nulls)} cannot be null: leave a field out to keep it")
        return data


class NewMessage(BaseModel):
    content: str = Field(min_length=1)
    model: str | None = None


class StoreOverride(BaseModel):
    """null for method, k or bar means the store's own; an absent field changes nothing."""

    method: str | None = None
    k: int | None = Field(None, ge=1, le=MAX_EVIDENCE)
    search: dict | None = None
    bar: float | None = Field(None, ge=-1, le=1)
    keep: bool = False


class NewAttempt(BaseModel):
    overrides: dict[str, StoreOverride] = {}
    # fields of the chat's held overrides to drop, so they are what is in force again;
    # dropped before the overrides are merged in
    drop: dict[str, list[Literal["method", "k", "search", "bar", "keep"]]] = {}


class RoutingEval(BaseModel):
    stores: list[str] = Field(min_length=2)
    split: Literal["held_out", "dev", "full"] = "held_out"  # held_out: not the bars' dev slice


class AnswerRequest(BaseModel):
    model: str | None = None


class EvalRequest(BaseModel):
    collection: str
    split: Literal["dev", "full"] = "dev"
    methods: list[str]
    answer: bool = False
    model: str | None = None
    k: int = Field(5, ge=1, le=50)
    rrf_k: int = Field(60, ge=0)
    search: dict | None = None  # overrides the collection's search settings in part
    eval_set: str | None = None  # the store's default eval set when None
    confirm: bool = False


class ResumeRequest(BaseModel):
    confirm: bool = False


class NewConnection(BaseModel):
    preset: str | None = None
    id: str
    label: str | None = None
    kind: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class ConnectionEdit(BaseModel):
    label: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class NewModel(BaseModel):
    model_id: str
    label: str | None = None
    price_in: float | None = 0.0  # $0 unless given; null leaves it unpriced
    price_out: float | None = 0.0


class ModelEdit(BaseModel):
    label: str | None = None
    price_in: float | None = None
    price_out: float | None = None
    enabled: bool | None = None


class RoleSet(BaseModel):
    model: str | None


class BudgetSet(BaseModel):
    cap_usd: float


def spec_json(spec: CollectionSpec) -> dict:
    return asdict(spec)


def type_clash(eval_set: str, set_type: str, store: str, store_type: str) -> EvalSetMismatch:
    return EvalSetMismatch(f"eval set '{eval_set}' is {set_type}; store '{store}' is {store_type}")


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
    state.builder = Builder(state.conn, state.store, state.models, state.ai, state.indexer)
    state.retriever = Retriever(
        state.conn, state.store, state.models, state.ai, state.indexer, state.builder
    )
    state.evaluator = Evaluator(state.conn, state.retriever, state.models, state.ai)
    state.router = Router(state.conn, state.retriever, state.evaluator)
    state.chats = Chats(state.conn, state.retriever, state.router, state.ai)
    state.loader = Loader(
        state.conn, state.indexer, state.builder, state.ai, settings.home, settings.build_hub
    )
    state.worker = Worker(
        state.conn, state.ai, state.indexer, state.loader, state.evaluator, state.builder
    )
    # Collections a sync add is writing to now. It runs outside the worker, so the jobs
    # table can't show it: a job queued under it is refused instead (one lock for both).
    state.adding = set()
    state.adding_lock = threading.Lock()

    @contextmanager
    def not_adding(name: str):
        """Held while a job on the collection is queued; refused while items are added."""
        with state.adding_lock:
            if name in state.adding:
                raise HTTPException(409, f"items are being added to '{name}'; try again shortly")
            yield

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
    for exc in (UnknownModel, UnknownPrice):
        app.add_exception_handler(exc, error(422))
    app.add_exception_handler(UnknownChat, error(404))
    for exc in (
        AlreadyAnswered,
        MethodUnavailable,
        ModelMismatch,
        ListsBusy,
        Conflict,
        AwaitingReplies,
        EvalSetMismatch,
    ):
        app.add_exception_handler(exc, error(409))
    app.add_exception_handler(ProviderError, error(502))
    app.add_exception_handler(BudgetExceeded, error(402))

    def no_connection(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, NoConnection)
        return JSONResponse({"detail": no_ai(exc.role)}, 503)

    app.add_exception_handler(NoConnection, no_connection)

    # -- helpers ---------------------------------------------------------------

    def need_ai(role: str, model: str | None = None) -> None:
        """Refuse at once a request that would spend on a job with no usable model.
        A model the request names is checked itself, not the job's."""
        if not state.ai.ready(role, model):
            raise NoConnection(role)

    def need_models(estimate: dict, model: str | None) -> None:
        """Every job a run spends on, with its answers on `model` when one is named. A
        named model is checked even when it is free: priced 0, it can still be off."""
        for role in estimate["roles"]:
            named = model if role == "answer" else None
            if estimate["est_usd"] != 0 or named:  # with no model each would stop at once
                need_ai(role, named)

    def source_rows(name: str) -> list[dict]:
        counts, disabled = state.store.source_counts(name), db.disabled_sources(state.conn, name)
        return [
            {"source": s, "items": counts.get(s, 0), "enabled": s not in disabled}
            for s in sorted(set(counts) | set(disabled))
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
            default = db.default_eval_set(state.conn, spec.name) or spec.name
            asked = {q.id: q.meta.get("source") for q in db.get_questions(state.conn, default)}
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

    def store_questions(name: str) -> tuple[str | None, list]:
        """The store's default eval set and its questions (none, with no default)."""
        eval_set = db.default_eval_set(state.conn, name)
        return eval_set, db.get_questions(state.conn, eval_set) if eval_set else []

    def check_default_set(spec: CollectionSpec) -> None:
        """Refuses a store whose default eval set is of another type, before anything is
        estimated or queued. Reads only: no set is made and no pointer written."""
        name = db.default_eval_set(state.conn, spec.name)
        found = db.get_eval_set(state.conn, name) if name else None
        if name and found and found["qa_type"] != spec.qa_type:
            raise type_clash(name, found["qa_type"], spec.name, spec.qa_type)

    def leaked_ids(spec: CollectionSpec, eval_set: str | None) -> set[str]:
        """The eval set's questions whose text is inside the store's items."""
        return state.evaluator.leaked_ids(spec, eval_set) if eval_set else set()

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
        return {"ok": True, "ai": state.ai.ready("answer")}

    @app.post("/collections", status_code=201)
    def create_collection(body: NewCollection) -> dict:
        if db.get_collection(state.conn, body.name):
            raise HTTPException(409, f"collection '{body.name}' already exists")
        found = db.get_eval_set(state.conn, body.name)
        if found and found["qa_type"] != body.qa_type:  # it would be the store's default
            raise type_clash(body.name, found["qa_type"], body.name, body.qa_type)
        return spec_json(
            state.indexer.create(
                body.name, body.qa_type, body.summarize, body.recipe or "paper_2024"
            )
        )

    @app.get("/collections")
    def list_collections() -> list[dict]:
        return [spec_json(s) for s in db.list_collections(state.conn)]

    @app.get("/collections/{name}")
    def get_collection(name: str) -> dict:
        spec = state.retriever.spec(name)
        lists = db.get_keyword_lists(state.conn, name)
        available = [m for m, why in available_methods(spec, lists is not None).items() if not why]
        count = state.store.count(name)
        jobs = db.list_jobs(state.conn, collection=name, limit=db.ALL)
        busy, job = collection_state(jobs, count, lists is not None)
        eval_set, everything = store_questions(name)
        include_leaked = db.get_include_leaked(state.conn, eval_set) if eval_set else False
        leaked = leaked_ids(spec, eval_set)
        questions = runnable(everything, leaked, include_leaked)
        fields = field_report(name, questions)
        route_profile, drafted = state.router.route_profile(spec)
        return {
            **spec_json(spec),
            "count": count,
            "keyword_lists": asdict(lists) if lists else None,
            "methods_available": available,
            # A chat's search defaults, as Set as default saves them; null is the store's own.
            "chat_method": db.get_chat_method(state.conn, name),
            "chat_k": db.get_chat_k(state.conn, name),
            "chat_bar": db.get_chat_bar(state.conn, name),
            "state": busy,
            "progress": job["progress"] if job and job["status"] == "running" else None,
            "job": job,
            "questions": len(questions),
            "dev_questions": len(dev_slice(questions)),
            "answer": state.evaluator.collection_answer(name),
            "runs": sum(1 for j in jobs if j["kind"] == "eval"),
            "sources": source_rows(name),
            "unlabelled": state.store.unsourced(name),
            "leaked": len(leaked),
            "include_leaked": include_leaked,
            "eval_set": eval_set,
            "fields": fields,
            "measurable": measurable(spec.qa_type, fields),
            "route_profile": route_profile,
            "route_profile_drafted": drafted,
        }

    @app.patch("/collections/{name}")
    def patch_collection(name: str, body: CollectionPatch) -> dict:
        """Makes the eval set the store's default, null clearing it, and saves the routing
        profile, null or blank going back to the draft. A field left out stays as it is."""
        spec = state.retriever.spec(name)
        sent = body.model_fields_set
        if "eval_set" in sent:
            if body.eval_set is not None:
                found = known_set(body.eval_set)
                if found["qa_type"] != spec.qa_type:
                    raise type_clash(body.eval_set, found["qa_type"], name, spec.qa_type)
            if store_is_loading(name):
                raise HTTPException(
                    409, f"a load or question import is under way on '{name}'; wait"
                )
            db.set_default_eval_set(state.conn, name, body.eval_set)
        if "route_profile" in sent:
            text = (body.route_profile or "").strip()
            db.set_route_profile(state.conn, name, text or None)
        return get_collection(name)

    @app.post("/collections/{name}/route-profile/draft")
    def draft_route_profile(name: str) -> dict:
        """What the router would draft for the store now; nothing is saved."""
        return {"route_profile": state.router.draft_route_profile(state.retriever.spec(name))}

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
        state.evaluator.forget(name)
        state.router.forget(name)
        db.delete_collection_rows(state.conn, name)
        imports.delete_uploads(state.settings.home, name)
        return Response(status_code=204)

    @app.post("/collections/{name}/documents", status_code=202)
    def add_documents(name: str, body: Documents):
        spec = state.retriever.spec(name)
        # An item added now would be keyed under mixed lists, miss a version or skew a run;
        # under the lock, so no job is queued between this check and the add's end.
        now = len(body.items) <= SYNC_ITEMS
        with state.adding_lock:
            for kind in ("load", "rekey", "retag", "copy", "eval", "import", "backfill"):
                if db.job_waiting_or_running(state.conn, kind, name):
                    raise HTTPException(409, f"a {kind} job is queued or running on '{name}'; wait")
            if name in state.adding:
                raise HTTPException(409, f"items are being added to '{name}'; try again shortly")
            if now:
                state.adding.add(name)
        if not now:
            return queue_documents(spec, body)  # queued as an import, which not_adding guards
        try:
            return add_now(spec, body)
        finally:
            with state.adding_lock:
                state.adding.discard(name)

    def add_now(spec: CollectionSpec, body: Documents):
        items = [Item(**i.model_dump()) for i in body.items]
        todo, _ = state.indexer.changed(spec, items, "hand")
        estimate = state.indexer.estimate_add_usd(spec, items, "hand")
        build_usd = state.builder.estimate_build_usd(spec, todo)
        estimate = None if estimate is None or build_usd is None else estimate + build_usd
        if estimate != 0 and not body.confirm:  # None: no model to price it, so not free
            return JSONResponse({"estimate": {"est_usd": estimate, "items": len(items)}}, 200)
        if builds(spec) and estimate != 0:  # before adding: a refused build adds nothing
            need_ai("keywords")
        state.indexer.add(spec, items, "hand")
        state.builder.build(spec, todo)
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
            # Summaries and the build can be unpaid-for (a proposal is priced only when it
            # can be made); a collection has one or the other, never both.
            role = "keywords" if builds(spec) else "summary"
            ready = not (spec.summarize or builds(spec)) or state.ai.ready(role)
            if estimate["est_usd"] != 0 and (not body.confirm or not ready):
                if body.confirm:
                    raise NoConnection(role)
                return JSONResponse({"estimate": estimate}, 200)
            with not_adding(spec.name):
                job_id = db.create_job(state.conn, "import", spec.name, setup)
            queued = True
            return {"job_id": job_id}
        finally:
            if not queued:
                shutil.rmtree(state.settings.home / "uploads" / upload_id, ignore_errors=True)

    @app.post("/collections/{name}/uploads")
    def add_upload(
        name: str,
        request: Request,
        filename: str,
        kind: Literal["items", "questions"],
        accept: Literal["text"] | None = None,
    ):
        """The raw file as the body; answers with a preview under a guessed mapping.
        With accept=text only a .txt or .md file is taken."""
        spec = state.retriever.spec(name)
        if accept == "text" and Path(filename).suffix.lower() not in TEXT_SUFFIXES:
            raise HTTPException(422, "only .txt and .md files can be added here")
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > imports.UPLOAD_LIMIT:
            raise HTTPException(413, f"the file is over {imports.UPLOAD_LIMIT} bytes")
        upload_id = imports.save_upload(state.settings.home, name, filename, body_chunks(request))
        path, _ = open_upload(name, upload_id)
        source_name = imports.source_name(filename)
        source = "upload:" + source_name
        shown = preview_upload(spec, path, filename, kind, None, source)
        try:
            estimate = estimate_upload(spec, upload_id, kind, shown, source)
        except Exception:  # the caller never learns the upload's id: nothing could reuse it
            shutil.rmtree(path.parent, ignore_errors=True)
            raise
        return {
            "upload_id": upload_id,
            "filename": filename,
            "source_name": source_name,
            "preview": shown,
            "estimate": estimate,
        }

    @app.post("/collections/{name}/imports", status_code=202)
    def start_import(name: str, body: ImportRequest):
        spec = state.retriever.spec(name)
        if body.kind == "questions":
            check_default_set(spec)
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
        if estimate["est_usd"] != 0 and builds(spec):
            need_ai("keywords")
        if body.kind == "questions":
            state.loader.questions_set(spec)  # a clash of types is refused before queueing
        setup = {
            "upload_id": body.upload_id,
            "kind": body.kind,
            "mapping": shown["mapping"],
            "source": source,
        }
        with not_adding(name):
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
        state.evaluator.forget(name)  # what the store holds changed: scan for leaks again
        state.router.forget(name)
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
        db.set_source_enabled(state.conn, name, source, True)
        state.retriever.forget(name)
        state.evaluator.forget(name)  # a re-import of as many items must scan again
        state.router.forget(name)  # and set its bar again
        return Response(status_code=204)

    @app.get("/collections/{name}/questions")
    def list_questions(
        name: str, leaked: bool | None = None, limit: int = 50, offset: int = 0
    ) -> dict:
        spec = state.retriever.spec(name)
        eval_set = state.loader.questions_set(spec)
        everything = db.get_questions(state.conn, eval_set)
        inside = leaked_ids(spec, eval_set)
        shown = [q for q in everything if leaked is None or (q.id in inside) == leaked]
        page = shown[max(offset, 0) : max(offset, 0) + max(limit, 0)]
        return {
            "total": len(shown),
            "leaked": len(inside),
            "include_leaked": db.get_include_leaked(state.conn, eval_set),
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "source": q.meta.get("source"),
                    "leaked": q.id in inside,
                    "has_gold": bool(q.gold_ids),
                    "has_solution": bool(q.meta.get("solution")),
                }
                for q in page
            ],
        }

    @app.patch("/collections/{name}/questions/leaked")
    def switch_leaked(name: str, body: LeakedSwitch) -> dict:
        spec = state.retriever.spec(name)
        db.set_include_leaked(state.conn, state.loader.questions_set(spec), body.include)
        return {"include_leaked": body.include}

    # -- eval sets -------------------------------------------------------------

    def known_set(name: str) -> dict:
        row = db.get_eval_set(state.conn, name)
        if row is None:
            raise HTTPException(404, f"no eval set {name!r}")
        return row

    def eval_runs(name: str) -> list[dict]:
        return [
            j
            for j in db.list_jobs(state.conn, kind="eval", limit=db.ALL)
            if j["setup"].get("eval_set") == name
        ]

    def store_is_loading(store: str) -> bool:
        """Whether a load or a question import is queued or running on the store: it writes
        to the store's default eval set."""
        for status in ("queued", "running"):
            for job in db.list_jobs(state.conn, collection=store, status=status, limit=db.ALL):
                if job["kind"] == "load" or (
                    job["kind"] == "import" and job["setup"].get("kind") == "questions"
                ):
                    return True
        return False

    def set_is_loading(name: str) -> bool:
        """Whether a load or a question import is queued or running on a store whose default
        is this set: it would write to the set while it is saved."""
        stores = [c.name for c in db.list_collections(state.conn)]
        return any(
            store_is_loading(s) for s in stores if db.default_eval_set(state.conn, s) == name
        )

    def not_loading(name: str) -> None:
        if set_is_loading(name):
            raise HTTPException(
                409, f"a load or question import is under way on a store of '{name}'"
            )

    @app.get("/eval-sets", response_model=None)
    def list_eval_sets(request: Request):
        # The API's list and the console's page share this path: a browser asks for HTML. Vary
        # keeps the browser from answering one with the other from its cache.
        vary = {"Vary": "Accept"}
        if "text/html" in request.headers.get("accept", ""):
            return FileResponse(STATIC / "eval_sets.html", media_type="text/html", headers=vary)
        return JSONResponse(db.list_eval_sets(state.conn), headers=vary)

    @app.post("/eval-sets", status_code=201)
    def create_eval_set(body: NewEvalSet) -> dict:
        store = db.get_collection(state.conn, body.name)
        if store and store.qa_type != body.qa_type:  # it would be the store's default
            raise type_clash(body.name, body.qa_type, body.name, store.qa_type)
        try:
            db.create_eval_set(state.conn, body.name, body.qa_type)
        except ValueError as e:
            if "already exists" in str(e):
                raise HTTPException(409, str(e)) from None
            raise
        return {"name": body.name}

    @app.get("/eval-sets/{name}")
    def get_eval_set(name: str) -> dict:
        return {
            **known_set(name),
            "questions": [asdict(q) for q in db.get_questions(state.conn, name)],
            "versions": db.eval_set_versions(state.conn, name),
        }

    @app.put("/eval-sets/{name}/questions")
    def put_eval_questions(name: str, body: QuestionsPut) -> dict:
        row = known_set(name)
        questions = [Question(**q.model_dump()) for q in body.questions]
        # A question with no source whose id no version of the set has held is written by
        # hand: checked for leaks. One the set held before (a restored older version) is not.
        held = db.eval_set_ids_ever(state.conn, name)
        questions = [
            q
            if q.id in held or "source" in q.meta
            else replace(q, meta={**q.meta, "source": "hand"})
            for q in questions
        ]
        seen: set[str] = set()
        for q in questions:
            if q.id in seen:
                raise HTTPException(422, f"question id '{q.id}' appears more than once")
            seen.add(q.id)
            if problem := imports.question_problem(q, row["qa_type"]):
                raise HTTPException(422, f"question '{q.id}': {problem}")
        not_loading(name)
        try:
            version = db.replace_questions(state.conn, name, questions, body.base_version)
        except ValueError as e:  # the questions were checked above: only a stale base is left
            raise HTTPException(409, str(e)) from None
        return {"version": version}

    @app.patch("/eval-sets/{name}")
    def patch_eval_set(name: str, body: EvalSetPatch) -> dict:
        known_set(name)
        db.set_include_leaked(state.conn, name, body.include_leaked)
        return known_set(name)

    @app.delete("/eval-sets/{name}")
    def delete_eval_set(name: str, confirm: bool = False):
        known_set(name)
        runs = eval_runs(name)
        if busy := next((j for j in runs if j["status"] in ("queued", "running")), None):
            raise HTTPException(409, f"an eval that uses '{name}' is {busy['status']}; wait")
        not_loading(name)
        if not confirm:
            return {"runs": len(runs)}
        db.delete_eval_set(state.conn, name)
        imports.delete_uploads(state.settings.home, f"eval-set:{name}")
        return Response(status_code=204)

    @app.post("/eval-sets/{name}/uploads")
    def add_eval_set_upload(name: str, request: Request, filename: str):
        """The raw file as the body, questions only; answers with a preview."""
        row = known_set(name)
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > imports.UPLOAD_LIMIT:
            raise HTTPException(413, f"the file is over {imports.UPLOAD_LIMIT} bytes")
        owner = f"eval-set:{name}"
        upload_id = imports.save_upload(state.settings.home, owner, filename, body_chunks(request))
        path, _ = imports.open_upload(state.settings.home, owner, upload_id)
        source_name = imports.source_name(filename)
        source = "upload:" + source_name
        try:
            shown = imports.preview(path, filename, "questions", row["qa_type"], None, None, source)
        except ValueError:
            shutil.rmtree(path.parent, ignore_errors=True)
            raise
        return {
            "upload_id": upload_id,
            "filename": filename,
            "source_name": source_name,
            "preview": shown,
        }

    @app.post("/eval-sets/{name}/imports")
    def import_eval_set_questions(name: str, body: EvalSetImport) -> dict:
        row = known_set(name)
        not_loading(name)
        try:
            path, meta = imports.open_upload(
                state.settings.home, f"eval-set:{name}", body.upload_id
            )
        except KeyError:
            raise HTTPException(404, f"no upload {body.upload_id!r} on eval set '{name}'") from None
        if not SOURCE_NAME.match(body.source_name):
            raise HTTPException(422, "the source name needs 1-64 letters, digits, space, _ . -")
        check_mapping("questions", body.mapping)
        source = "upload:" + body.source_name
        shown = imports.preview(
            path, meta["filename"], "questions", row["qa_type"], body.mapping, None, source
        )
        if not shown["ready"]:
            raise HTTPException(
                422,
                f"map these fields first: {shown['missing']}"
                if shown["missing"]
                else "no usable rows",
            )
        _, rows, _ = imports.read_rows(path, meta["filename"])
        built, _ = imports.build_questions(rows, shown["mapping"], row["qa_type"])
        built = [replace(q, meta={**q.meta, "source": source}) for q in built]
        counts, todo = state.loader.question_counts(name, built)
        db.save_questions(state.conn, name, todo)
        return {
            **counts,
            "skipped": len(rows) - len(built),
            "version": known_set(name)["version"],
        }

    @app.put("/collections/{name}/search")
    def put_search(name: str, body: dict) -> dict:
        spec = state.retriever.spec(name)
        merged = search_settings(spec.search or SearchSettings(), body)
        db.set_search(state.conn, name, merged)
        return asdict(merged)

    @app.post("/collections/{name}/load", status_code=202)
    def load(name: str, body: LoadRequest):
        spec = state.retriever.spec(name)
        check_default_set(spec)
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
        # What the loader spends on: summaries, a proposal when there are no lists, and
        # the build steps. Whether the build spends depends on the items (and on the Hub), so
        # a load that builds is priced even when confirmed.
        proposes = state.ai.ready("keywords") and state.loader.makes_lists(spec)
        setup = {"dataset": body.dataset, "limit": body.limit}
        if builds(spec) or ((spec.summarize or proposes) and not body.confirm):
            estimate = state.loader.estimate_load(spec, body.dataset, body.limit)
            if builds(spec) and estimate["est_usd"] != 0:  # None: no model to price it
                need_ai("keywords")
            spends = spec.summarize or proposes or estimate["est_usd"] != 0
            if spends and not body.confirm:
                return JSONResponse({"estimate": estimate}, 200)
            if not body.confirm and estimate["hub"].get("entry"):
                # Free only because of this entry: the run fails rather than spend unasked.
                setup["hub_entry"] = estimate["hub"]["entry"]
        with not_adding(name):
            job_id = db.create_job(state.conn, "load", name, setup)
        return {"job_id": job_id}

    @app.post("/collections/{name}/copy", status_code=202)
    def copy_collection(name: str, body: CopyRequest):
        """A new collection of this one's items, questions and switched-off sources, built
        under another recipe. Without confirm, what the copy would spend; with it, the
        collection is made and a queued copy job adds and builds its items."""
        source = state.retriever.spec(name)
        if db.get_collection(state.conn, body.name):
            raise HTTPException(409, f"collection '{body.name}' already exists")
        found = db.get_eval_set(state.conn, body.name)
        if found and found["qa_type"] != source.qa_type:  # one name, one question type
            raise type_clash(body.name, found["qa_type"], body.name, source.qa_type)
        target = state.loader.copy_spec(source, body.name, body.recipe, body.build)
        # A job on the source could change the items while the copy reads them.
        for status in ("running", "queued"):
            busy = db.list_jobs(state.conn, collection=name, status=status, limit=1)
            if busy:
                raise HTTPException(409, f"a {busy[0]['kind']} job is {status} on '{name}'; wait")
        if target.summarize:
            need_ai("summary")
        estimate = state.loader.estimate_copy(name, target)
        if builds(target) and estimate["est_usd"] != 0:  # None: no model to price it
            need_ai("keywords")
        if not body.confirm:
            return JSONResponse({"estimate": estimate}, 200)
        with not_adding(name):  # the copy reads the source's items
            return {"name": body.name, "job_id": state.loader.start_copy(name, target)}

    @app.post("/collections/{name}/keyword-lists/propose")
    def propose(name: str, body: ProposeRequest | None = None):
        spec = state.retriever.spec(name)
        need_ai("keywords")
        sample = sample_items(state.store, name, tuple(db.disabled_sources(state.conn, name)))
        if not (body and body.confirm):
            estimate, calls = state.indexer.plan_propose(spec, sample)
            return JSONResponse({"estimate": {"est_usd": estimate, "calls": calls}}, 200)
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
        with not_adding(name):  # a refused rekey saves no version either
            version = state.indexer.save_lists(spec, KeywordLists(**body.model_dump()))
            job_id = db.create_job(state.conn, "rekey", name, {"version": version})
        return {"version": version, "job_id": job_id}

    @app.get("/collections/{name}/vocabulary")
    def get_vocabulary(name: str) -> dict:
        """The vocabulary in force and every saved version. This body is the export:
        PUT it back, edited, to import a version (spec §16)."""
        spec = state.retriever.spec(name)
        return {
            "version": spec.vocabulary_version,
            "terms": db.get_vocabulary(state.conn, name),
            "versions": db.list_vocabularies(state.conn, name),
        }

    @app.put("/collections/{name}/vocabulary", status_code=202)
    def put_vocabulary(name: str, body: Vocabulary):
        """Without confirm, what re-tagging every item would cost; with it, saves the terms
        as a version and a queued retag job switches to it once every item is tagged."""
        spec = state.retriever.spec(name)
        build = spec.build
        if build is None or build.vocabulary == "lists":
            raise HTTPException(
                409, f"'{name}' uses keyword lists, not a vocabulary; edit its lists instead"
            )
        # Items tagged now would mix versions, or skew a run that is reading them.
        busy = db.busy_job(state.conn, name)
        if busy:
            raise HTTPException(409, f"a {busy} job is queued or running on '{name}'; wait")
        state.indexer.check_model(spec)  # the retag would embed with another model
        terms = clean_terms(body.terms)
        if not terms:
            raise ValueError("the vocabulary is empty")
        if build.tagging == "ai":
            need_ai("keywords")
        if not body.confirm:
            items = state.builder.stored_items(name)
            estimate, calls = state.builder.plan_tag(spec, terms, items)
            return JSONResponse(
                {"estimate": {"est_usd": estimate, "items": len(items), "calls": calls}}, 200
            )
        with not_adding(name):
            version = state.builder.save_vocabulary(spec, terms)
            job_id = db.create_job(state.conn, "retag", name, {"version": version})
        return {"version": version, "job_id": job_id}

    def run_method(
        spec: CollectionSpec,
        question: str,
        method: str,
        k: int,
        rrf_k: int,
        search: SearchSettings,
        skip: tuple[str, ...],
        answer=None,
    ) -> dict:
        """One method's reply to a question: its hits, the answer `answer(hits)` gives (a
        dict merged into the reply) and every call's cost. A provider or budget failure is the
        reply's `error`, so it does not hide what the other methods spent."""
        out: dict = {"hits": [], "answer": None, "usd": 0.0, "error": None, "program": None}
        with state.ai.tally() as spent:
            try:
                tags = None
                if method == "keyword_rag":
                    tags = state.retriever.query_tags(spec, question, search, skip)
                    out["query_tags"] = tags
                hits = state.retriever.retrieve(spec.name, question, method, k, rrf_k, search, tags)
                out["hits"] = [asdict(h) for h in hits]
                if answer:
                    out.update(answer(hits))
            except PER_METHOD_ERRORS as e:
                out["error"] = no_ai(e.role) if isinstance(e, NoConnection) else str(e)
        out["usd"] = spent["usd"]
        return out

    # -- chats -----------------------------------------------------------------
    # Chats tallies its own spending, so these routes never open a tally.

    @app.get("/collections/{name}/bar")
    def collection_bar(name: str) -> dict:
        return asdict(state.router.bar(state.retriever.spec(name)))

    @app.get("/chats")
    def list_chats() -> list[dict]:
        return state.chats.list()

    @app.post("/chats", status_code=201)
    def create_chat(body: NewChat) -> dict:
        return state.chats.create(body.stores, body.title)

    @app.get("/chats/{chat_id}")
    def get_chat(chat_id: str) -> dict:
        return state.chats.get(chat_id)

    @app.patch("/chats/{chat_id}")
    def patch_chat(chat_id: str, body: ChatPatch) -> dict:
        return state.chats.update(chat_id, **body.model_dump(exclude_unset=True))

    @app.delete("/chats/{chat_id}", status_code=204)
    def delete_chat(chat_id: str) -> Response:
        state.chats.delete(chat_id)
        return Response(status_code=204)

    @app.post("/chats/{chat_id}/messages")
    def chat_message(chat_id: str, body: NewMessage) -> dict:
        return state.chats.ask(chat_id, body.content, body.model)

    @app.post("/chats/{chat_id}/messages/{n}/attempts")
    def chat_attempt(chat_id: str, n: int, body: NewAttempt) -> dict:
        overrides = {s: o.model_dump(exclude_unset=True) for s, o in body.overrides.items()}
        drop = {s: list(fields) for s, fields in body.drop.items()}
        return state.chats.rerun(chat_id, n, overrides, drop)

    @app.post("/chats/{chat_id}/messages/{n}/attempts/{attempt}/answer")
    def chat_answer(chat_id: str, n: int, attempt: int, body: AnswerRequest | None = None) -> dict:
        return state.chats.answer(chat_id, n, attempt, body.model if body else None)

    @app.post("/chats/{chat_id}/stores/{store}/default")
    def chat_default(chat_id: str, store: str) -> dict:
        return state.chats.set_default(chat_id, store)

    @app.post("/routing/eval")
    def routing_eval(body: RoutingEval) -> dict:
        """The gate and the profile router scored on the stores' own eval set questions, in
        the request: embeddings only, so it spends nothing."""
        return state.router.evaluate(body.stores, body.split)

    @app.post("/query")
    def query(body: QueryRequest) -> dict:
        check_methods(body.methods)
        spec = state.retriever.spec(body.collection)
        search = search_settings(spec.search or SearchSettings(), body.search)
        if "keyword_rag" in body.methods and search.query_tags == "oracle":
            raise ValueError("oracle tags need a question's gold")  # eval only
        skip = tuple(db.disabled_sources(state.conn, body.collection))
        results = {}
        for method in body.methods:

            def answer(hits: list[Hit]) -> dict:
                program = search.program and not body.choices
                system, user = answer_prompt(
                    spec.qa_type,
                    body.question,
                    body.choices,
                    answer_hits(spec.qa_type, hits, search.example),
                    search.example,
                    search.program,
                )
                done = state.ai.complete(
                    user,
                    role="answer",
                    system=system,
                    model=body.model,
                    max_tokens=answer_tokens(spec.qa_type, program),
                    purpose="query",
                )
                text, prog = read_answer(spec.qa_type, done.text, bool(body.choices), program)
                return {"answer": text, "program": prog}

            results[method] = run_method(
                spec,
                body.question,
                method,
                body.k,
                body.rrf_k,
                search,
                skip,
                answer if body.answer else None,
            )
        return {"results": results}

    @app.post("/chat")
    def chat(body: ChatRequest) -> dict:
        """An answer from a store's passages, citing them as [n]. Spends like the Lab: no
        estimate, the cost is in the reply, the budget cap holds."""
        *earlier, last = body.messages
        if last.role != "user":
            raise ValueError("the last message must be the user's")
        spec = state.retriever.spec(body.collection)
        reasons = available_methods(spec, db.get_keyword_lists(state.conn, spec.name) is not None)
        method = body.method or ("keyword_rag" if reasons["keyword_rag"] is None else "hybrid")
        check_methods([method])
        if reasons[method]:
            raise HTTPException(409, f"{method} {reasons[method]} on '{body.collection}'")
        search = spec.search or SearchSettings()
        if search.query_tags == "oracle":  # it needs a question's gold; chat has none
            search = replace(search, query_tags="classifier")
        skip = tuple(db.disabled_sources(state.conn, body.collection))
        history = [t.model_dump() for t in earlier][-CHAT_TURNS:]
        cited: list[dict] = []

        def answer(hits: list[Hit]) -> dict:
            system, user = chat_prompt(last.content, history, hits)
            done = state.ai.complete(
                user,
                role="answer",
                system=system,
                model=body.model,
                max_tokens=CHAT_TOKENS,
                purpose="chat",
            )
            ordered = edge_order(hits)
            for n in cited_numbers(done.text, len(ordered)):
                hit = ordered[n - 1]
                cited.append(
                    {
                        "n": n,
                        "id": hit.id,
                        "text": hit.text[:CHAT_PASSAGE_CHARS],
                        "source": hit.meta.get("source"),
                        "part": hit.meta.get("part"),
                        "parts": hit.meta.get("parts"),
                    }
                )
            return {"answer": done.text}

        out = run_method(spec, last.content, method, body.k, 60, search, skip, answer)
        return {
            "answer": out["answer"],
            "citations": cited,
            "hit_ids": [h["id"] for h in out["hits"]],
            "method": method,
            "query_tags": out.get("query_tags"),
            "usd": out["usd"],
            "error": out["error"],
        }

    @app.post("/eval", status_code=202)
    def evaluate(body: EvalRequest):
        check_methods(body.methods)
        spec = state.retriever.spec(body.collection)
        search_settings(spec.search or SearchSettings(), body.search)  # 422 before an estimate
        estimate = state.evaluator.estimate(
            body.collection,
            body.split,
            body.methods,
            body.answer,
            body.model,
            body.search,
            body.eval_set,
        )
        if not body.confirm:  # spec §7.3: an eval without confirm only estimates
            return JSONResponse({"estimate": estimate}, 200)
        need_models(estimate, body.model)
        if "keyword_rag" in body.methods:
            state.retriever.vocabulary(spec)  # refused here, not after the other methods spent
        with not_adding(body.collection):
            job_id = state.evaluator.queue(
                body.collection,
                body.split,
                body.methods,
                body.answer,
                body.model,
                body.k,
                body.rrf_k,
                body.search,
                body.eval_set,
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
        row = db.get_job(state.conn, job_id)
        if row is not None and row["kind"] == "copy":
            return resume_copy(row, body is not None and body.confirm)
        try:
            refusal = state.evaluator.resume_refusal(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None
        if refusal:
            raise HTTPException(409, refusal)
        if body is None or not body.confirm:
            return JSONResponse({"estimate": state.evaluator.estimate_resume(job_id)}, 200)
        estimate = state.evaluator.estimate_resume(job_id)
        need_models(estimate, job(job_id)["setup"]["model"])
        if not db.requeue(state.conn, job_id):  # moved since the check: a double confirm
            raise HTTPException(409, "the job is no longer resumable")
        return {"job_id": job_id}

    def resume_copy(row: dict, confirm: bool):
        """A stopped copy runs again in place: adding skips the items it added, and the
        build finishes what it left unbuilt, so only the rest is priced."""
        if row["status"] not in db.JOBS_RESUMABLE:
            raise HTTPException(409, f"a {row['status']} copy cannot be resumed")
        source = row["setup"]["source"]
        target = state.retriever.spec(row["collection"])
        state.retriever.spec(source)  # 404 when the source is gone
        busy = db.list_jobs(state.conn, collection=source, status="running", limit=1)
        busy = busy or db.list_jobs(state.conn, collection=source, status="queued", limit=1)
        if busy:
            raise HTTPException(
                409, f"a {busy[0]['kind']} job is {busy[0]['status']} on '{source}'; wait"
            )
        estimate = state.loader.estimate_copy(source, target)
        if target.summarize and estimate["est_usd"] != 0:
            need_ai("summary")
        if builds(target) and estimate["est_usd"] != 0:
            need_ai("keywords")
        if not confirm:
            return JSONResponse({"estimate": estimate}, 200)
        with not_adding(source):
            if not db.requeue(state.conn, row["id"]):  # moved since the check
                raise HTTPException(409, "the job is no longer resumable")
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

    @app.get("/catalog")
    def get_catalog() -> dict:
        return {**catalog(), "datasets": DATASETS}

    @app.get("/budget")
    def budget() -> dict:
        return {
            "month": datetime.now(UTC).strftime("%Y-%m"),
            "spent_usd": state.ai.month_spend(),
            "cap_usd": state.ai.cap_usd,
            "note": "This service's own spend; your provider's dashboard shows the rest.",
        }

    # -- AI settings ------------------------------------------------------------

    @app.get("/ai")
    def get_ai() -> dict:
        return {
            "connections": state.ai.connections(),
            "models": state.ai.models(),
            "roles": state.ai.roles(),
            "default_from_env": state.ai.env_default,
            "known_roles": KNOWN_ROLES,
            "budget": {
                "cap_usd": state.ai.cap_usd,
                "spent_usd": state.ai.month_spend(),
                "from_env": state.ai.env_cap,
            },
            "presets": [
                {k: p[k] for k in ("label", "kind", "base_url")} | {"id": id}
                for id, p in PRESETS.items()
            ],
            # The list prices a model can be given on the Settings page, besides $0 and typed.
            "prices": [
                {"label": label, "price_in": price_in, "price_out": price_out}
                for p in PRESETS.values()
                for label, price_in, price_out in p["models"].values()
            ],
        }

    @app.post("/ai/connections", status_code=201)
    def add_connection(body: NewConnection) -> dict:
        if body.preset is not None and body.preset not in PRESETS:
            raise HTTPException(422, f"unknown preset {body.preset!r}; known: {list(PRESETS)}")
        preset = PRESETS.get(body.preset or "", {})
        kind = body.kind or preset.get("kind")
        if not kind:
            raise HTTPException(422, "give a preset or a kind")
        state.ai.add_connection(
            body.id,
            kind,
            body.label or preset.get("label") or body.id,
            base_url=body.base_url or preset.get("base_url"),
            api_key=body.api_key,
        )
        for model_id, (label, price_in, price_out) in preset.get("models", {}).items():
            state.ai.add_model(
                body.id, model_id, label=label, price_in=price_in, price_out=price_out
            )
        return {"id": body.id}

    @app.patch("/ai/connections/{id}")
    def edit_connection(id: str, body: ConnectionEdit) -> dict:
        state.ai.update_connection(id, **body.model_dump(exclude_unset=True))
        return {"id": id}

    @app.delete("/ai/connections/{id}", status_code=204)
    def remove_connection(id: str) -> Response:
        state.ai.remove_connection(id)
        return Response(status_code=204)

    @app.post("/ai/connections/{id}/test")
    def test_connection(id: str) -> dict:
        return state.ai.test_connection(id)

    @app.post("/ai/connections/{id}/models/fetch")
    def fetch_models(id: str) -> dict:
        return {"added": state.ai.fetch_models(id)}

    @app.post("/ai/connections/{id}/models")
    def add_model(id: str, body: NewModel) -> dict:
        state.ai.add_model(
            id, body.model_id, label=body.label, price_in=body.price_in, price_out=body.price_out
        )
        return {"ref": f"{id}/{body.model_id}"}

    @app.put("/ai/models/{connection_id}/{model_id:path}")
    def edit_model(connection_id: str, model_id: str, body: ModelEdit) -> dict:
        changes = body.model_dump(exclude_unset=True)
        if any(changes.get(f, 1) is None for f in ("label", "enabled")):
            raise HTTPException(422, "label and enabled cannot be null")
        state.ai.update_model(connection_id, model_id, **changes)
        return {"ref": f"{connection_id}/{model_id}"}

    @app.put("/ai/roles/{role}")
    def set_role(role: str, body: RoleSet) -> dict:
        if role not in KNOWN_ROLES:
            raise HTTPException(422, f"unknown role {role!r}; known: {KNOWN_ROLES}")
        state.ai.set_role(role, body.model)
        return {"role": role, "model": state.ai.roles().get(role)}

    @app.put("/ai/budget")
    def set_budget(body: BudgetSet) -> dict:
        state.ai.set_cap(body.cap_usd)
        return {"cap_usd": state.ai.cap_usd}

    # -- pages: plain HTML plus the shared ui.js, served from /static (behind the same auth) --

    pages = {
        "/": "collections",
        "/c/{name}": "collection",
        "/e/{name}": "eval_set",
        "/c/{name}/runs/{id}": "run",
        "/worker": "worker",
        "/lab": "lab",
        "/chat": "chat",
        "/settings": "settings",
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
        if job["kind"] == "rekey" and job["status"] in REKEY_PARTLY_DONE and job["progress"] > 0:
            name = job["collection"]
            lines.append(f"Keyword lists on {name} are partly re-keyed; save them again")
        elif job["kind"] == "retag" and job["status"] in REKEY_PARTLY_DONE and job["progress"] > 0:
            name = job["collection"]
            lines.append(f"The vocabulary on {name} is partly re-tagged; save it again")
        elif job["kind"] == "copy" and job["status"] in COPY_UNFINISHED:
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
