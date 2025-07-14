"""The HTTP API. Thin: every decision lives in ragqa_core; this maps it to HTTP.

Run with `uvicorn --factory ragqa_api.app:create_app`.
"""

import base64
import random
import secrets
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ragqa_api.config import Settings, load_settings
from ragqa_core import db
from ragqa_core.datasets import DATASETS
from ragqa_core.datasets import load as load_dataset
from ragqa_core.evaluator import ANSWER_TOKENS, Evaluator, answer_hits
from ragqa_core.indexer import Indexer, ListsBusy, ModelMismatch
from ragqa_core.keywords import tfidf_terms
from ragqa_core.llm import BudgetExceeded, Claude, NoApiKey, UnknownModel
from ragqa_core.models import Models
from ragqa_core.prompts import answer_prompt, parse_answer
from ragqa_core.retriever import (
    KEYWORD_METHODS,
    METHOD_NAMES,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
)
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item, KeywordLists, QaType

STATIC = Path(__file__).parent / "static"
PROPOSE_SAMPLE = 40
PER_METHOD_ERRORS = (MethodUnavailable, NoApiKey, BudgetExceeded, ModelMismatch)


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


def spec_json(spec: CollectionSpec) -> dict:
    return asdict(spec)


def check_methods(methods: list[str]) -> None:
    unknown = [m for m in methods if m not in METHOD_NAMES]
    if unknown:
        raise HTTPException(422, f"unknown methods {unknown}; known: {list(METHOD_NAMES)}")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="rag-qa", version="0.1.0")
    state = app.state
    state.settings = settings
    state.conn = db.connect(settings.home / "ragqa.db")
    db.mark_interrupted(state.conn)
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

    def rewire() -> None:
        """Rebuild the objects that hold store/models/claude, after a swap."""
        state.indexer = Indexer(state.conn, state.store, state.models, state.claude)
        state.retriever = Retriever(
            state.conn, state.store, state.models, state.claude, state.indexer
        )
        state.evaluator = Evaluator(state.conn, state.retriever, state.models, state.claude)

    rewire()
    state.rewire = rewire

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
        return {
            **spec_json(spec),
            "count": state.store.count(name),
            "keyword_lists": asdict(lists) if lists else None,
            "methods_available": available,
        }

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
        if spec.summarize and not body.confirm:
            # Summaries cost; the dataset is read (and cached) to price them.
            items, _ = load_dataset(body.dataset, settings.home)
            items = limited(items, body.limit)
            estimate = state.indexer.estimate_add_usd(spec, items)
            return JSONResponse({"estimate": {"est_usd": estimate, "items": len(items)}}, 200)
        job_id = db.create_job(
            state.conn, "load", name, {"dataset": body.dataset, "limit": body.limit}
        )
        threading.Thread(
            target=run_load, args=(state, settings, job_id, spec, body), daemon=True
        ).start()
        return {"job_id": job_id}

    @app.post("/collections/{name}/keyword-lists/propose")
    def propose(name: str) -> dict:
        spec = state.retriever.spec(name)
        sample = sample_items(state.store, name)
        return asdict(state.indexer.propose_lists(spec, sample))

    @app.put("/collections/{name}/keyword-lists")
    def put_lists(name: str, body: Lists) -> dict:
        spec = state.retriever.spec(name)
        return {"version": state.indexer.set_lists(spec, KeywordLists(**body.model_dump()))}

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
        job_id = state.evaluator.start(
            body.collection, body.split, body.methods, body.answer, body.model, body.k, body.rrf_k
        )
        return {"job_id": job_id, "estimate": estimate}

    @app.get("/eval/{job_id}")
    def eval_table(job_id: str) -> dict:
        try:
            return state.evaluator.table(job_id)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None

    @app.get("/eval/{job_id}/items")
    def eval_items(job_id: str, method: str, wrong: bool = False) -> list[dict]:
        try:
            return state.evaluator.items(job_id, method, wrong)
        except KeyError:
            raise HTTPException(404, f"no job {job_id}") from None

    @app.get("/jobs/{job_id}")
    def job(job_id: str) -> dict:
        row = db.get_job(state.conn, job_id)
        if row is None:
            raise HTTPException(404, f"no job {job_id}")
        return row

    @app.get("/budget")
    def budget() -> dict:
        return {
            "month": datetime.now(UTC).strftime("%Y-%m"),
            "spent_usd": state.claude.month_spend(),
            "cap_usd": state.claude.cap_usd,
            "note": "This service's own spend; the Console shows the credit balance.",
        }

    for page in ("lab", "board"):
        path = STATIC / f"{page}.html"
        if path.exists():
            app.add_api_route(
                f"/{page}", page_route(path), methods=["GET"], include_in_schema=False
            )

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
    return secrets.compare_digest(given_user, username) & secrets.compare_digest(
        given_pass, password
    )


def limited(items: list[Item], limit: int | None) -> list[Item]:
    ordered = sorted(items, key=lambda i: i.id)
    return ordered[:limit] if limit else ordered


def sample_items(store: Store, name: str) -> list[Item]:
    hits = [h for batch in store.scroll(name) for h in batch]
    picked = random.Random(472).sample(hits, min(PROPOSE_SAMPLE, len(hits)))
    return [Item(h.id, h.text, h.answer, h.explanation, h.meta) for h in picked]


def tfidf_lists(texts: list[str]) -> KeywordLists | None:
    """Lists without Claude: the corpus's own top terms. A small corpus has too
    few terms to fill all three facets, so a facet that would be empty reuses
    the head of the ranking rather than leaving its keyword vector undefined."""
    terms = tfidf_terms(texts, n=565)
    if not terms:
        return None
    return KeywordLists(
        topic=terms[:50],
        method=terms[50:65] or terms[:15],
        common=terms[65:565] or terms,
    )


def run_load(state, settings: Settings, job_id: str, spec: CollectionSpec, body: LoadRequest):
    conn = state.conn
    db.update_job(conn, job_id, status="running")
    try:
        items, questions = load_dataset(body.dataset, settings.home)
        items = limited(items, body.limit)
        state.indexer.add(
            spec, items, progress=lambda p: db.update_job(conn, job_id, progress=0.9 * p)
        )
        db.save_questions(conn, spec.name, questions)
        source = None
        if db.get_keyword_lists(conn, spec.name) is None:
            sample = random.Random(472).sample(items, min(PROPOSE_SAMPLE, len(items)))
            lists, source = None, "tfidf"
            if state.claude.api_key:
                try:
                    lists, source = state.indexer.propose_lists(spec, sample), "claude"
                except ValueError:
                    source = "tfidf (claude proposal unusable)"
            if lists is None:
                lists = tfidf_lists([i.text for i in items])
            if lists is not None:
                state.indexer.set_lists(spec, lists)
            else:
                source = None
        fresh = db.get_collection(conn, spec.name)
        db.update_job(
            conn,
            job_id,
            status="done",
            progress=1.0,
            result={
                "added": len(items),
                "questions": len(questions),
                "lists_source": source,
                "keyword_version": fresh.keyword_version if fresh else None,
            },
        )
    except BudgetExceeded as e:
        db.update_job(conn, job_id, status="stopped_budget", error=str(e))
    except NoApiKey as e:
        db.update_job(conn, job_id, status="stopped_no_key", error=str(e))
    except Exception as e:  # recorded on the job; the thread must not die silently
        db.update_job(conn, job_id, status="failed", error=repr(e))
