"""The HTTP API. Thin: every decision lives in ragqa_core; this maps it to HTTP.

Loads, imports, copies, re-keys, rebuilds and evals are queued as jobs, and the worker thread
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
from typing import Any, Literal

from anyio import from_thread
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from ragqa_core import db, imports, search_defaults
from ragqa_core.builder import Builder, builds
from ragqa_core.catalog import catalog
from ragqa_core.chats import AlreadyAnswered, Chats, UnknownChat
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
    CHAT_TURNS,
    answer_prompt,
)
from ragqa_core.recipes import search_settings
from ragqa_core.retriever import (
    METHOD_NAMES,
    PER_METHOD_ERRORS,
    POOL,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
    available_methods,
)
from ragqa_core.search_defaults import MAX_CANDIDATES, MAX_EVIDENCE
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
    "reembed": "rebuilding",
    "tag_build": "tagging",
}
DOCUMENT_FIELDS = ("id", "text", "answer", "explanation", "meta")
SYNC_ITEMS = 20  # more items than this are imported by the worker
# The jobs a sync add of items waits for: each adds, keys, tags or reads the items.
ADD_WAITS_FOR = (
    "load",
    "rekey",
    "retag",
    "copy",
    "eval",
    "import",
    "backfill",
    "reembed",
    "tag_build",
)
TEXT_SUFFIXES = (".txt", ".md")  # what the chat can add to a store, and a Documents store takes
DOCUMENTS_ONLY = "A Documents store takes .md and .txt files"
NOT_UNTAGGED = "only a Documents store without keyword tags can add them"
NO_KEYWORDS_MODEL = "connect a model for keywords first"
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
    dense_model: str | None = None  # None takes the Settings model, else the service's


class ReembedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


class NewEvalSet(BaseModel):
    name: str
    qa_type: QaType
    store: str | None = None  # the store its labels are made against


class PoolRequest(BaseModel):
    """Find passages: the question, and the passages already marked on it (`labelled`), so
    those the pool misses still come back."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    labelled: list[str] = []


class ReferenceAnswerRequest(BaseModel):
    """The AI's answer to a labelled question from the passages marked y: priced without
    `confirm`, asked with it."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    passages: list[str] = Field(min_length=1, max_length=MAX_EVIDENCE)
    confirm: bool = False


class QuestionIn(BaseModel):
    id: str
    text: str
    choices: list[str] | None = None
    gold_answers: list[str] = Field(default_factory=list)
    gold_ids: list[str] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    judged_ids: list[str] | None = None


class QuestionsPut(BaseModel):
    questions: list[QuestionIn]
    base_version: str | None = None


GONE_QUESTION = "This question was deleted since you opened it"


class NewQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    choices: list[str] | None = None
    gold_answers: list[str] = []
    gold_ids: list[str] = []
    judged_ids: list[str] | None = None
    meta: dict = {}


class NewQuestionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: NewQuestion


class QuestionPut(BaseModel):
    """One question and the question as read, its conflict check."""

    model_config = ConfigDict(extra="forbid")

    question: QuestionIn
    base: QuestionIn


class EvalSetPatch(BaseModel):
    """Only the fields sent change; an unknown field is refused."""

    model_config = ConfigDict(extra="forbid")

    include_leaked: bool | None = None  # not nullable
    store: str | None = None  # null: unlink


class CollectionPatch(BaseModel):
    """Only the fields sent change; an unknown field is refused."""

    model_config = ConfigDict(extra="forbid")

    eval_set: str | None = None  # null: the set named after the store, if any
    chat_method: str | None = None  # a chat's method on the store; null: Settings'
    chat_k: int | None = Field(None, ge=1, le=MAX_CANDIDATES)  # candidates; null: Settings'


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
    methods: list[str] | None = None  # None: the Settings method
    k: int = Field(5, ge=1, le=50)
    rrf_k: int | None = Field(None, ge=0)  # None: the Settings RRF k
    answer: bool = False
    model: str | None = None
    search: dict | None = None  # overrides the collection's search settings in part


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    collection: str
    messages: list[ChatTurn] = Field(min_length=1)  # the last is the user's question
    method: str | None = None  # None: the store's chat method, as a chat inherits it
    k: int | None = Field(None, ge=1, le=MAX_EVIDENCE)  # passages answered from; None: Settings
    model: str | None = None


class NewChat(BaseModel):
    """Stores to chat over, or the id of a chat to copy (from). The route checks that exactly
    one is given, so the 422's detail is a plain string."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    stores: list[str] | None = None
    from_: str | None = Field(None, alias="from")
    title: str | None = None


class ChatPatch(BaseModel):
    """An absent field changes nothing; null for evidence_size, rewrite or rerank inherits
    (from Settings). A null title or stores, or an unknown field, is refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    stores: list[str] | None = None
    evidence_size: int | None = Field(None, ge=1, le=MAX_EVIDENCE)
    rewrite: bool | None = None
    rerank: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def no_nulls(cls, data: object) -> object:
        if isinstance(data, dict):
            nulls = [f for f, value in data.items() if value is None and f in ("title", "stores")]
            if nulls:
                raise ValueError(f"{', '.join(nulls)} cannot be null: leave a field out to keep it")
        return data


class NewMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelt answer must not pay for one

    content: str = Field(min_length=1)
    model: str | None = None
    answer: bool = True  # False: the passages only, nothing is paid for an answer


class ChatSettingsIn(BaseModel):
    """The chat's own values a re-run saves; null inherits, an absent field keeps the chat's."""

    model_config = ConfigDict(extra="forbid")

    evidence_size: int | None = Field(default=None, ge=1, le=MAX_EVIDENCE)
    rerank: bool | None = None
    rewrite: bool | None = None


class StoreSettingsIn(BaseModel):
    """A store's values on the chat; null inherits, an absent field keeps the chat's."""

    model_config = ConfigDict(extra="forbid")

    method: str | None = None
    candidates: int | None = Field(None, ge=1, le=MAX_CANDIDATES)


class NewAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    settings: ChatSettingsIn = ChatSettingsIn()
    stores: dict[str, StoreSettingsIn] = {}


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
    rerank_pool: int = Field(POOL, ge=1, le=POOL)  # hybrid_rerank: hybrid's top N it reranks
    rerank_model: str | None = None  # hybrid_rerank: None takes the Settings model
    judge: bool = False  # Judge answers: a Documents run's answers graded by the answer model
    confirm: bool = False


class ResumeRequest(BaseModel):
    confirm: bool = False


class KeywordTagsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
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


class SearchDefaultsIn(BaseModel):
    """A partial change: a field left out keeps its value, null puts it back to recommended.
    Strict, so "yes" is not read as true: search_defaults.check says what each field takes."""

    model_config = ConfigDict(extra="forbid", strict=True)

    method: str | None = None
    candidates: int | None = None
    evidence_size: int | None = None
    rerank: bool | None = None
    rewrite: bool | None = None
    follow_up_turns: int | None = None
    rrf_k: int | None = None
    rerank_model: str | None = None
    rerank_pool: int | None = None
    dense_model: str | None = None


def spec_json(spec: CollectionSpec) -> dict:
    return asdict(spec)


def type_clash(eval_set: str, set_type: str, store: str, store_type: str) -> EvalSetMismatch:
    return EvalSetMismatch(f"eval set '{eval_set}' is {set_type}; store '{store}' is {store_type}")


def brief_reply(message: dict) -> dict:
    """A message's first attempt, short: the answer, its citations and the passages, for an
    app that calls the API. The raw score is left out; relevance is 0 to 1 or null."""
    one = message["attempts"][0]
    pipeline = one["pipeline"]
    keys = ("n", "store", "id", "text", "source", "part", "parts", "relevance")
    return {
        "n": message["n"],
        "answer": one["answer"],
        "citations": one["citations"],
        "passages": [{k: e[k] for k in keys} for e in one["evidence"]],
        "passages_only": pipeline["passages_only"] or pipeline.get("answer_skipped", False),
        "error": one["error"],
        "usd": one["usd"],
    }


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
    # Chat's first rerank need not wait for it; None is the service's own reranker.
    reranker = search_defaults.in_force(app.state.conn)["rerank_model"]
    app.state.models.warm_rerank_in_background(reranker if isinstance(reranker, str) else None)
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
    state.chats = Chats(state.conn, state.retriever, state.ai)
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

    def refuse_busy(name: str, stopped_switch_ok: bool = False) -> None:
        """409 while any job is running or queued on the collection, or while a rebuild's
        switch stopped part-way: the store is then emptied or partly refilled, and the
        resume refills it from the rebuilt copy. Only resuming that switch may pass it."""
        for status in ("running", "queued"):
            busy = db.list_jobs(state.conn, collection=name, status=status, limit=1)
            if busy:
                raise HTTPException(409, f"a {busy[0]['kind']} job is {status} on '{name}'; wait")
        # switching_model reads the newest rebuild: a new one would hide the stopped switch.
        switching = db.switching_model(state.conn, name)
        if switching and not stopped_switch_ok:
            raise HTTPException(409, f"the switch to {switching} stopped part-way; resume it first")

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
        model = body.dense_model
        known = (*search_defaults.supported_embedders(), state.models.dense_model)
        if model is not None and model not in known:
            raise HTTPException(422, f"unknown embedding model {model}")
        recipe = body.recipe or ("documents" if body.qa_type == "documents" else "paper_2024")
        return spec_json(
            state.indexer.create(body.name, body.qa_type, body.summarize, recipe, None, model)
        )

    @app.get("/collections")
    def list_collections() -> list[dict]:
        """Every spec, with the store's default eval set (null for none), as one store's
        GET names it."""
        defaults = db.default_eval_sets(state.conn)
        return [
            {**spec_json(s), "eval_set": defaults.get(s.name)}
            for s in db.list_collections(state.conn)
        ]

    @app.get("/collections/{name}")
    def get_collection(name: str) -> dict:
        spec = state.retriever.spec(name)
        lists = db.get_keyword_lists(state.conn, name)
        reasons = available_methods(
            spec, lists is not None, db.tag_build_unfinished(state.conn, name)
        )
        available = [m for m, why in reasons.items() if not why]
        count = state.store.count(name)
        jobs = db.list_jobs(state.conn, collection=name, limit=db.ALL)
        busy, job = collection_state(jobs, count, lists is not None or spec.qa_type == "documents")
        eval_set, everything = store_questions(name)
        include_leaked = db.get_include_leaked(state.conn, eval_set) if eval_set else False
        leaked = leaked_ids(spec, eval_set)
        questions = runnable(everything, leaked, include_leaked)
        fields = field_report(name, questions)
        return {
            **spec_json(spec),
            "count": count,
            "keyword_lists": asdict(lists) if lists else None,
            "methods_available": available,
            "methods_unavailable": {m: why for m, why in reasons.items() if why},
            # A chat's method and candidates on the store, as Set as default saves them;
            # null inherits from Settings. chat_settings says what is in force, and whence.
            # one saved before it left the chat list (hybrid_rerank) reads as null
            "chat_method": search_defaults.chat_method(db.get_chat_method(state.conn, name)),
            "chat_k": db.get_chat_k(state.conn, name),
            "chat_settings": state.chats.store_in_force(name),
            "state": busy,
            "progress": job["progress"] if job and job["status"] == "running" else None,
            "job": job,
            "load_error": load_error(jobs),
            "rebuild": rebuild_state(jobs),
            "keyword_tags": keyword_tags_state(spec, jobs, db.get_vocabulary(state.conn, name)),
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
        }

    @app.patch("/collections/{name}")
    def patch_collection(name: str, body: CollectionPatch) -> dict:
        """Makes the eval set the store's default, null clearing it, and sets a chat's
        method and candidates on the store, null inheriting. Every field is checked before
        any is saved; a field left out stays as it is."""
        spec = state.retriever.spec(name)
        sent = body.model_fields_set
        if "chat_method" in sent and body.chat_method is not None:
            search_defaults.check("method", body.chat_method)  # not a chat method: 422
            lists = db.get_keyword_lists(state.conn, name)
            tagging = db.tag_build_unfinished(state.conn, name)
            why = available_methods(spec, lists is not None, tagging)[body.chat_method]
            if why:
                raise MethodUnavailable(f"{body.chat_method} {why} on '{name}'")
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
        if "chat_method" in sent:
            db.set_chat_method(state.conn, name, body.chat_method)
        if "chat_k" in sent:
            db.set_chat_k(state.conn, name, body.chat_k)
        return get_collection(name)

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
            for kind in ADD_WAITS_FOR:
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
        # Each file becomes passages: a zip, a table or json lines is not a Documents file.
        is_text = Path(filename).suffix.lower() in TEXT_SUFFIXES
        if kind == "items" and spec.qa_type == "documents" and not is_text:
            raise HTTPException(422, DOCUMENTS_ONLY)
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
            # It was priced on the recipe now; a queued tag_build would switch it first.
            if db.job_waiting_or_running(state.conn, "tag_build", name):
                raise HTTPException(409, f"a tag_build job is queued or running on '{name}'; wait")
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
        return {**row, "enabled": body.enabled}

    @app.delete("/collections/{name}/sources/{source}", status_code=204)
    def delete_source(name: str, source: str) -> Response:
        known_source(name, source)
        refuse_busy(name)
        left = [r for r in source_rows(name) if r["source"] != source and r["items"] > 0]
        if left and not any(r["enabled"] for r in left):
            raise HTTPException(409, "at least one source must stay on")
        state.store.delete_source(name, source)
        db.set_source_enabled(state.conn, name, source, True)
        state.retriever.forget(name)
        state.evaluator.forget(name)  # a re-import of as many items must scan again
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

    def check_link(name: str, qa_type: str, store: str) -> None:
        spec = state.retriever.spec(store)  # unknown: 404
        if spec.qa_type != qa_type:
            raise type_clash(name, qa_type, store, spec.qa_type)

    @app.post("/eval-sets", status_code=201)
    def create_eval_set(body: NewEvalSet) -> dict:
        store = db.get_collection(state.conn, body.name)
        if store and store.qa_type != body.qa_type:  # it would be the store's default
            raise type_clash(body.name, body.qa_type, body.name, store.qa_type)
        if body.store is not None:
            check_link(body.name, body.qa_type, body.store)
        try:
            # One lock from reading the store's default to setting it, so two creates
            # can't both become it, and no load is queued in between.
            with state.conn.lock:
                first = False  # the linked store has no default set: this one becomes it
                if body.store is not None:
                    first = db.default_eval_set(state.conn, body.store) is None
                    if first and store_is_loading(body.store):
                        raise HTTPException(
                            409, f"a load or question import is under way on '{body.store}'; wait"
                        )
                db.create_eval_set(state.conn, body.name, body.qa_type, body.store)
                if first and body.store is not None:
                    db.set_default_eval_set(state.conn, body.store, body.name)
        except ValueError as e:
            if "already exists" in str(e):
                raise HTTPException(409, str(e)) from None
            raise
        return {"name": body.name}

    @app.get("/eval-sets/{name}")
    def get_eval_set(name: str, version: str | None = None) -> dict:
        row = known_set(name)
        try:
            questions = db.get_questions(state.conn, name, version)
        except KeyError:
            raise HTTPException(404, f"That version of '{name}' doesn't exist") from None
        missing: dict[str, list[str]] = {}
        wanted = sorted({i for q in questions for i in q.gold_ids})
        if row["store"] and wanted:
            held = state.store.held(row["store"], wanted)
            missing = {
                q.id: gone for q in questions if (gone := [i for i in q.gold_ids if i not in held])
            }
        return {
            **row,
            "questions": [asdict(q) for q in questions],
            "versions": db.eval_set_versions(state.conn, name),
            "missing": missing,
            "questions_version": version or row["version"],
            "answer_available": state.ai.ready("answer"),
        }

    @app.post("/eval-sets/{name}/questions", status_code=201)
    def add_eval_question(name: str, body: NewQuestionIn) -> dict:
        row = known_set(name)
        not_loading(name)
        new = body.question
        meta = {"source": "hand", **new.meta}
        q = Question(
            imports.make_id("q-", new.text),
            new.text,
            new.choices,
            new.gold_answers,
            new.gold_ids,
            meta,
            new.judged_ids,
        )
        if problem := imports.question_problem(q, row["qa_type"]):
            raise HTTPException(422, problem)
        question_id, version = db.add_question(state.conn, name, q)
        return {"id": question_id, "version": version}

    # :path, so an imported id with a slash (hr/leave-1) still reaches the route.
    @app.put("/eval-sets/{name}/questions/{question_id:path}")
    def put_eval_question(name: str, question_id: str, body: QuestionPut) -> dict:
        row = known_set(name)
        not_loading(name)
        if body.question.id != question_id or body.base.id != question_id:
            raise HTTPException(422, "the question's id must match the address")
        q = Question(**body.question.model_dump())
        if problem := imports.question_problem(q, row["qa_type"]):
            raise HTTPException(422, f"question '{q.id}': {problem}")
        try:
            version = db.upsert_question(state.conn, name, q, Question(**body.base.model_dump()))
        except KeyError:
            raise HTTPException(404, GONE_QUESTION) from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"version": version}

    @app.delete("/eval-sets/{name}/questions/{question_id:path}")
    def delete_eval_question(name: str, question_id: str) -> dict:
        known_set(name)
        not_loading(name)
        try:
            return {"version": db.delete_question(state.conn, name, question_id)}
        except KeyError:
            raise HTTPException(404, GONE_QUESTION) from None

    @app.post("/eval-sets/{name}/pool")
    def eval_set_pool(name: str, body: PoolRequest) -> dict:
        row = known_set(name)
        if row["qa_type"] != "documents":
            raise HTTPException(422, "Find passages is for Documents eval sets")
        if row["store"] is None:
            raise HTTPException(409, "pick a store for this eval set first")
        return state.chats.labelling_pool(row["store"], body.question, body.labelled)

    @app.post("/eval-sets/{name}/answer")
    def eval_set_answer(name: str, body: ReferenceAnswerRequest) -> dict:
        row = known_set(name)
        if row["qa_type"] != "documents":
            raise HTTPException(422, "The AI's answer is for Documents eval sets")
        if row["store"] is None:
            raise HTTPException(409, "pick a store for this eval set first")
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "a question needs some text")
        return state.chats.reference_answer(row["store"], question, body.passages, body.confirm)

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
        row = known_set(name)
        sent = body.model_fields_set
        if "include_leaked" in sent and body.include_leaked is None:
            raise HTTPException(422, "include_leaked cannot be null: leave it out to keep it")
        if "store" in sent and body.store is not None:
            check_link(name, row["qa_type"], body.store)
        if body.include_leaked is not None:
            db.set_include_leaked(state.conn, name, body.include_leaked)
        if "store" in sent:
            db.set_eval_set_store(state.conn, name, body.store)
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
        with state.conn.lock:  # no labeller's save between reading the set and writing it
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
        refuse_busy(name)
        if target.summarize:
            need_ai("summary")
        estimate = state.loader.estimate_copy(name, target)
        if builds(target) and estimate["est_usd"] != 0:  # None: no model to price it
            need_ai("keywords")
        if not body.confirm:
            return JSONResponse({"estimate": estimate}, 200)
        with not_adding(name):  # the copy reads the source's items
            return {"name": body.name, "job_id": state.loader.start_copy(name, target)}

    @app.post("/collections/{name}/reembed", status_code=202)
    def reembed(name: str, body: ReembedRequest) -> dict:
        """Queues a rebuild of the store with another embedding model. It is free: every
        item is embedded again on this machine, so there is nothing to estimate or confirm."""
        spec = state.retriever.spec(name)
        model = body.model
        if model not in (*search_defaults.supported_embedders(), state.models.dense_model):
            raise HTTPException(422, f"unknown embedding model {model}")
        # Under the lock, so no job is queued between these checks and this one's queueing.
        with not_adding(name):
            refuse_busy(name)  # items added or changed meanwhile would miss the rebuilt copy
            if model == spec.dense_model:
                raise HTTPException(409, f"'{name}' is already built with {model}")
            setup = {"model": model, "from": spec.dense_model}
            return {"job_id": db.create_job(state.conn, "reembed", name, setup)}

    @app.post("/collections/{name}/keyword-tags", status_code=202)
    def add_keyword_tags(name: str, body: KeywordTagsRequest | None = None):
        """Add keyword tags: a Documents store switches to documents_tagged in place, and a
        queued tag_build job induces its vocabulary and tags every passage. Without confirm,
        what that would spend."""
        spec = state.retriever.spec(name)
        if spec.qa_type != "documents" or spec.recipe != "documents":
            raise HTTPException(409, NOT_UNTAGGED)
        # Under the lock, so no job is queued and no passage added between these checks and
        # this one's queueing.
        with not_adding(name):
            if state.store.count(name) == 0:
                raise HTTPException(409, "add passages first")
            refuse_busy(name)
            estimate, calls = state.builder.plan_tag_build(spec)
            if estimate is None:
                raise HTTPException(409, NO_KEYWORDS_MODEL)
            if not (body and body.confirm):
                return JSONResponse({"estimate": {"est_usd": estimate, "calls": calls}}, 200)
            if estimate != 0:  # all cached: the build sends nothing
                need_ai("keywords")
            return {"job_id": db.create_job(state.conn, "tag_build", name, {})}

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
        if spec.recipe == "documents":  # a vocabulary comes with Add keyword tags
            raise HTTPException(
                409, f"'{name}' has no keyword tags yet; use Add keyword tags on the store page"
            )
        build = spec.build
        if build is None or build.vocabulary == "lists":
            raise HTTPException(
                409, f"'{name}' uses keyword lists, not a vocabulary; edit its lists instead"
            )
        # Items tagged now would mix versions, or skew a run that is reading them.
        busy = db.busy_job(state.conn, name)
        if busy:
            raise HTTPException(409, f"a {busy} job is queued or running on '{name}'; wait")
        state.indexer.check_model(spec)  # the retag would embed with a model it can't load
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

    @app.get("/chats")
    def list_chats() -> list[dict]:
        return state.chats.list()

    @app.post("/chats", status_code=201)
    def create_chat(body: NewChat) -> dict:
        if (body.stores is None) == (body.from_ is None):
            if body.stores is None:
                raise ValueError("a chat needs stores, or from: a chat to copy")
            raise ValueError("send stores or from, not both")
        if body.from_ is not None:
            return state.chats.create_like(body.from_, body.title)
        return state.chats.create(body.stores or [], body.title)

    @app.get("/chats/{chat_id}")
    def get_chat(chat_id: str) -> dict:
        return state.chats.get(chat_id)

    @app.patch("/chats/{chat_id}")
    def patch_chat(chat_id: str, body: ChatPatch) -> dict:
        return state.chats.update(chat_id, body.model_dump(exclude_unset=True))

    @app.delete("/chats/{chat_id}", status_code=204)
    def delete_chat(chat_id: str) -> Response:
        state.chats.delete(chat_id)
        return Response(status_code=204)

    @app.post("/chats/{chat_id}/messages")
    def chat_message(chat_id: str, body: NewMessage, brief: bool = False) -> dict:
        message = state.chats.ask(chat_id, body.content, body.model, body.answer)
        return brief_reply(message) if brief else message

    @app.post("/chats/{chat_id}/messages/{n}/attempts")
    def chat_attempt(chat_id: str, n: int, body: NewAttempt) -> dict:
        settings = body.settings.model_dump(exclude_unset=True)
        stores = {s: v.model_dump(exclude_unset=True) for s, v in body.stores.items()}
        return state.chats.rerun(chat_id, n, settings, stores)

    @app.post("/chats/{chat_id}/messages/{n}/attempts/{attempt}/answer")
    def chat_answer(chat_id: str, n: int, attempt: int, body: AnswerRequest | None = None) -> dict:
        return state.chats.answer(chat_id, n, attempt, body.model if body else None)

    @app.post("/chats/{chat_id}/stores/{store}/default")
    def chat_default(chat_id: str, store: str) -> dict:
        return state.chats.set_default(chat_id, store)

    @app.post("/query")
    def query(body: QueryRequest) -> dict:
        """Each method's hits for a question; with no methods or rrf_k, Settings' own."""
        defaults: dict[str, Any] = search_defaults.in_force(state.conn)
        methods = body.methods if body.methods is not None else [str(defaults["method"])]
        rrf_k = body.rrf_k if body.rrf_k is not None else int(defaults["rrf_k"])
        check_methods(methods)
        spec = state.retriever.spec(body.collection)
        # A store mid-switch is emptied, so every method is refused at once (409); a model
        # it can't load fails only the methods that embed, each with its error.
        if db.switching_model(state.conn, spec.name):
            state.indexer.check_model(spec)
        search = search_settings(spec.search or SearchSettings(), body.search)
        if "keyword_rag" in methods and search.query_tags == "oracle":
            raise ValueError("oracle tags need a question's gold")  # eval only
        skip = tuple(db.disabled_sources(state.conn, body.collection))
        results = {}
        for method in methods:

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
                rrf_k,
                search,
                skip,
                answer if body.answer else None,
            )
        return {"results": results}

    @app.post("/chat")
    def chat(body: ChatRequest) -> dict:
        """An answer from a store's passages, citing them as [n]. Searched as a chat searches
        one store: its method and candidates, then the pool reranked against the question,
        with Settings' rerank, RRF k, reranker and pool cap. Spends like the Lab: no
        estimate, the cost is in the reply, the budget cap holds."""
        *earlier, last = body.messages
        if last.role != "user":
            raise ValueError("the last message must be the user's")
        spec = state.retriever.spec(body.collection)
        chain = state.chats.store_in_force(spec.name)
        method = body.method or str(chain["method"]["value"])
        if body.method is not None:
            check_methods([method])
            if method not in search_defaults.CHAT_METHODS:
                raise HTTPException(422, f"{method} is not a chat method")
            why = state.chats.why_not(spec, method)
            if why:
                raise HTTPException(409, f"{method} {why} on '{body.collection}'")
        defaults: dict[str, Any] = search_defaults.in_force(state.conn)
        reranker = defaults["rerank_model"] or state.models.rerank_model
        found = state.chats.search(
            {spec.name: {"method": method, "candidates": chain["candidates"]["value"]}},
            last.content,
            last.content,
            body.k or int(defaults["evidence_size"]),
            bool(defaults["rerank"]),
            int(defaults["rrf_k"]),
            str(reranker),
            int(defaults["rerank_pool"]),
        )
        decision, evidence = found["decisions"][0], found["evidence"]
        out: dict = {"answer": None, "citations": [], "error": decision["error"]}
        if decision["error"] is None:  # a store that failed is not answered from
            history = [t.model_dump() for t in earlier][-CHAT_TURNS:]
            answered = state.chats.answer_from(last.content, history, evidence, body.model)
            out.update(answer=answered["answer"], error=answered["error"])
            out["citations"] = [  # one store: its name is not repeated
                {k: v for k, v in c.items() if k != "store"} for c in answered["citations"]
            ]
            found["usd"] += answered["usd"]
        notes = [decision["note"]] if decision["note"] else []
        return {
            **out,
            "hit_ids": [e["id"] for e in evidence],
            "method": decision["method"],
            "query_tags": decision["tags"],
            "usd": found["usd"],
            "notes": notes + found["notes"],
        }

    @app.post("/eval", status_code=202)
    def evaluate(body: EvalRequest):
        check_methods(body.methods)
        spec = state.retriever.spec(body.collection)
        search_settings(spec.search or SearchSettings(), body.search)  # 422 before an estimate
        if body.rerank_model is not None:
            search_defaults.check("rerank_model", body.rerank_model)
        estimate = state.evaluator.estimate(
            body.collection,
            body.split,
            body.methods,
            body.answer,
            body.model,
            body.search,
            body.eval_set,
            judge=body.judge,
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
                body.rerank_pool,
                body.rerank_model,
                estimate.get("evidence_size"),  # the size the estimate priced
                judge=body.judge,
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
        if row is not None and row["kind"] == "reembed":
            return resume_reembed(row)
        if row is not None and row["kind"] == "tag_build":
            return resume_tag_build(row, body is not None and body.confirm)
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
        refuse_busy(source)
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

    def resume_tag_build(row: dict, confirm: bool):
        """A stopped Add keyword tags runs again in place: the build finishes what it left
        unbuilt (the vocabulary when none is in force, the passages without tags), so only
        that is priced."""
        if row["status"] not in db.JOBS_RESUMABLE:
            raise HTTPException(409, f"a {row['status']} tag build cannot be resumed")
        name = row["collection"]
        spec = state.retriever.spec(name)  # 404 when the store is gone
        with not_adding(name):  # no job is queued between these checks and the requeue
            # The keyword tags state reads the newest tag build only, so an older one must
            # not run again in its place.
            newest = db.list_jobs(state.conn, kind="tag_build", collection=name, limit=1)
            if newest[0]["id"] != row["id"]:
                raise HTTPException(409, f"a newer tag build of '{name}' exists; resume that one")
            refuse_busy(name)
            estimate, calls = state.builder.plan_tag_build(spec)
            if estimate is None:
                raise HTTPException(409, NO_KEYWORDS_MODEL)
            if not confirm:
                return JSONResponse({"estimate": {"est_usd": estimate, "calls": calls}}, 200)
            if estimate != 0:
                need_ai("keywords")
            if not db.requeue(state.conn, row["id"]):  # moved since the check
                raise HTTPException(409, "the job is no longer resumable")
        return {"job_id": row["id"]}

    def resume_reembed(row: dict) -> dict:
        """A stopped rebuild runs again: before its switch it embeds every item afresh, and
        during it (setup["switching"]) it finishes the switch. It is free: no estimate."""
        if row["status"] not in db.JOBS_RESUMABLE:
            raise HTTPException(409, f"a {row['status']} rebuild cannot be resumed")
        name = row["collection"]
        state.retriever.spec(name)  # 404 when the store is gone
        with not_adding(name):  # no job is queued between these checks and the requeue
            # switching_model reads the newest rebuild only, so an older one must not run
            # again; the newest is this one, so a stopped switch is its own to finish.
            newest = db.list_jobs(state.conn, kind="reembed", collection=name, limit=1)
            if newest[0]["id"] != row["id"]:
                raise HTTPException(
                    409, f"a newer rebuild of '{name}' replaced this one; start the rebuild again"
                )
            refuse_busy(name, stopped_switch_ok=True)
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

    # -- search defaults (Settings -> Search defaults) ----------------------------

    def search_defaults_reply() -> dict:
        models = state.models

        def named(values: dict) -> dict:  # None is the service's model: say which
            return {
                **values,
                "rerank_model": values["rerank_model"] or models.rerank_model,
                "dense_model": values["dense_model"] or models.dense_model,
            }

        saved = search_defaults.saved(state.conn)
        in_force = named(search_defaults.in_force(state.conn))
        reranker = str(in_force["rerank_model"])
        status = models.rerank_status(reranker)
        # The short list, then the service's model and the saved one if not on it.
        offered = [*search_defaults.EMBEDDERS, models.dense_model, saved["dense_model"]]
        embedders = [str(name) for name in dict.fromkeys(offered) if name is not None]
        return {
            "saved": saved,
            "recommended": named(search_defaults.RECOMMENDED),
            "in_force": in_force,
            # The in-force reranker's warm-up: idle, loading, ready or failed, and why it failed.
            "rerank_status": status,
            "rerank_error": models.rerank_errors.get(reranker) if status == "failed" else None,
            "limits": {field: list(bounds) for field, bounds in search_defaults.LIMITS.items()},
            "methods": list(search_defaults.CHAT_METHODS),
            "rerankers": [
                {"name": name, "size_gb": size}
                for name, size in search_defaults.supported_rerankers().items()
            ],
            "embedders": [
                {
                    "name": name,
                    "dim": models.dense_dim(name),
                    "size_gb": search_defaults.supported_embedders().get(name, {}).get("size_gb"),
                    "note": search_defaults.EMBEDDER_NOTES.get(name),
                    "seconds_per_1000": search_defaults.EMBED_SECONDS.get(name),
                }
                for name in embedders
            ],
            "embed_machine": search_defaults.EMBED_MACHINE,
            "notes": {"rerank": search_defaults.RERANK_OFF_WHY},
        }

    @app.get("/search-defaults")
    def get_search_defaults() -> dict:
        return search_defaults_reply()

    @app.put("/search-defaults")
    def put_search_defaults(body: SearchDefaultsIn) -> dict:
        changes = body.model_dump(exclude_unset=True)
        models = state.models
        before = str(search_defaults.in_force(state.conn)["rerank_model"] or models.rerank_model)
        search_defaults.save(state.conn, changes)  # a bad value is a ValueError: 422
        picked = changes.get("rerank_model")
        # Loaded before the next message needs it. Saving the one in force again retries only
        # a failed load (say, the Hub was down), so a Save of other fields starts nothing.
        if "rerank_model" in changes and (
            (picked or models.rerank_model) != before or models.rerank_status(picked) == "failed"
        ):
            models.retry_rerank(picked)
        return search_defaults_reply()

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
    stopped = rebuild_state(jobs)
    if stopped and stopped["switching"]:  # emptied or part-filled: search is refused
        return "rebuilding", next(j for j in jobs if j["id"] == stopped["job_id"])
    if count == 0:
        return "empty", None
    return ("ready" if has_lists else "no_lists"), None


def load_error(jobs: list[dict]) -> str | None:
    """Why the store's newest load stopped, or None while it is queued, running, done or
    cancelled (a later load that finished clears it). `jobs` are its jobs, newest first."""
    job = next((j for j in jobs if j["kind"] == "load"), None)
    if job is None or job["status"] in ("done", "queued", "running", "cancelled"):
        return None
    return job["error"] or f"the load {PROBLEM.get(job['status'], job['status'])}"


def keyword_tags_state(spec: CollectionSpec, jobs: list[dict], terms: list[str] | None):
    """A Documents store's keyword tags, the first that applies: building while its newest
    tag_build is queued or running, stopped while that is otherwise not done, built while a
    vocabulary (`terms`) is in force, else none. None for any other store. `jobs` are its
    jobs, newest first."""
    if spec.qa_type != "documents":
        return None
    job = next((j for j in jobs if j["kind"] == "tag_build"), None)
    if job and job["status"] in ("queued", "running"):
        return {"state": "building", "job_id": job["id"], "progress": job["progress"]}
    if job and job["status"] != "done":
        return {"state": "stopped", "job_id": job["id"]}
    if terms:
        return {"state": "built", "terms": len(terms)}
    return {"state": "none"}


def rebuild_state(jobs: list[dict]) -> dict | None:
    """The store's newest rebuild while it is unfinished; `jobs` are its jobs, newest first."""
    job = next((j for j in jobs if j["kind"] == "reembed"), None)
    if job is None or job["status"] == "done":
        return None
    setup = job["setup"]
    return {
        "job_id": job["id"],
        "status": job["status"],
        "model": setup.get("model"),
        "from": setup.get("from"),
        "progress": job["progress"],
        "switching": setup.get("switching", False),
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
