"""The HTTP API's wiring: the app's state, the login, the mapping of errors to HTTP statuses,
and the pages. The routes live in ragqa_api.routes, one module per resource, and are added
here in order: /health and /catalog, then each module's, then the pages and /static last.

Loads, imports, copies, re-keys, rebuilds and evals are queued as jobs, and the worker thread
(ragqa_worker), started by the lifespan, runs them in order. A few routes still
work in the request: /documents indexes its items, and /query and a confirmed
proposal call the AI.

Run with `uvicorn --factory ragqa_api.app:create_app`.
"""

import base64
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

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
from ragqa_api.guards import STATIC, Guards
from ragqa_api.routes import ai_settings, chats, documents, eval_sets, keyword_lists, runs, stores
from ragqa_core import db, search_defaults
from ragqa_core.builder import Builder
from ragqa_core.catalog import catalog
from ragqa_core.chats import AlreadyAnswered, Chats, NeedsAI, UnknownChat
from ragqa_core.datasets import DATASETS
from ragqa_core.evaluator import EvalSetMismatch, Evaluator
from ragqa_core.imports import TooLarge
from ragqa_core.indexer import Indexer, ListsBusy, ModelMismatch
from ragqa_core.jobs import no_ai
from ragqa_core.loader import Loader
from ragqa_core.models import Models
from ragqa_core.retriever import MethodUnavailable, Retriever
from ragqa_core.settings import Settings, load_settings
from ragqa_core.store import Store
from ragqa_core.types import AlreadyExists, NotFound, UnknownCollection
from ragqa_worker.worker import Worker


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
    app = FastAPI(title="keyword-rag", version="0.2.0", lifespan=lifespan)
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
    guards = Guards(state)

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

    for exc in (UnknownCollection, NotFound):
        app.add_exception_handler(exc, error(404))
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
        AlreadyExists,
    ):
        app.add_exception_handler(exc, error(409))
    app.add_exception_handler(NeedsAI, error(503))
    app.add_exception_handler(ProviderError, error(502))
    app.add_exception_handler(BudgetExceeded, error(402))

    def no_connection(request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, NoConnection)
        return JSONResponse({"detail": no_ai(exc.role)}, 503)

    app.add_exception_handler(NoConnection, no_connection)

    # -- routes ----------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "ai": state.ai.ready("answer")}

    @app.get("/catalog")
    def get_catalog() -> dict:
        return {**catalog(), "datasets": DATASETS}

    stores.add_routes(app, guards)
    documents.add_routes(app, guards)
    eval_sets.add_routes(app, guards)
    keyword_lists.add_routes(app, guards)
    chats.add_routes(app, guards)
    runs.add_routes(app, guards)
    ai_settings.add_routes(app, guards)

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
