"""The store routes: a store's spec, state and sources, its loads, copies and rebuilds."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import PROBLEM, Guards, check_shown_price, estimate_reply, type_clash
from ragqa_core import db, imports, search_defaults
from ragqa_core.builder import builds
from ragqa_core.datasets import DATASETS, dev_slice
from ragqa_core.evaluator import measurable, runnable
from ragqa_core.indexer import ListsBusy
from ragqa_core.jobs import BUSY, BUSY_STATE
from ragqa_core.loader import roles_needed
from ragqa_core.recipes import resolve_search
from ragqa_core.retriever import MethodUnavailable, method_reasons
from ragqa_core.search_defaults import MAX_CANDIDATES
from ragqa_core.types import CollectionSpec, QaType


class NewCollection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    qa_type: QaType
    summarize: bool = False
    recipe: str | None = None
    dense_model: str | None = None  # None takes the Settings model, else the service's


class ReembedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


class CollectionPatch(BaseModel):
    """Only the fields sent change; an unknown field is refused."""

    model_config = ConfigDict(extra="forbid")

    eval_set: str | None = None  # null: the set named after the store, if any
    chat_method: str | None = None  # a chat's method on the store; null: Settings'
    chat_k: int | None = Field(None, ge=1, le=MAX_CANDIDATES)  # candidates; null: Settings'


class SourceSwitch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class LoadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str
    limit: int | None = None
    confirm: bool = False
    # What the caller was shown: given, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


class CopyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    recipe: str
    build: dict | None = None  # overrides the recipe's build settings in part
    confirm: bool = False
    # What the caller was shown: on confirm, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


def spec_json(spec: CollectionSpec) -> dict:
    return asdict(spec)


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

    def source_rows(name: str) -> list[dict]:
        counts, disabled = state.store.source_counts(name), db.disabled_sources(state.conn, name)
        return [
            {"source": s, "items": counts.get(s, 0), "enabled": s not in disabled}
            for s in sorted(set(counts) | set(disabled))
        ]

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

    @app.post("/collections", status_code=201)
    def create_collection(body: NewCollection) -> dict:
        if db.get_collection(state.conn, body.name):
            raise HTTPException(409, f"store '{body.name}' already exists")
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
        reasons = method_reasons(state.conn, spec)
        available = [m for m, why in reasons.items() if not why]
        count = state.store.count(name)
        jobs = db.list_jobs(state.conn, collection=name, limit=db.ALL)
        busy, job = collection_state(jobs, count, lists is not None or spec.qa_type == "documents")
        eval_set, everything = guards.store_questions(name)
        include_leaked = db.get_include_leaked(state.conn, eval_set) if eval_set else False
        leaked = guards.leaked_ids(spec, eval_set)
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
            # The search a chat uses, as a run names it: a Documents store is scored on it.
            "chat_search": search_defaults.store_search(state.conn, name),
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
            why = method_reasons(state.conn, spec)[body.chat_method]
            if why:
                raise MethodUnavailable(f"{body.chat_method} {why} on '{name}'")
        if "eval_set" in sent:
            if body.eval_set is not None:
                found = guards.known_set(body.eval_set)
                if found["qa_type"] != spec.qa_type:
                    raise type_clash(body.eval_set, found["qa_type"], name, spec.qa_type)
            if guards.store_is_loading(name):
                raise HTTPException(409, f"a load is under way on '{name}'; wait")
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
        state.evaluator.forget_store(name)
        db.delete_collection_rows(state.conn, name)
        imports.delete_uploads(state.settings.home, name)
        return Response(status_code=204)

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
        guards.refuse_busy(name)
        left = [r for r in source_rows(name) if r["source"] != source and r["items"] > 0]
        if left and not any(r["enabled"] for r in left):
            raise HTTPException(409, "at least one source must stay on")
        state.store.delete_source(name, source)
        db.set_source_enabled(state.conn, name, source, True)
        state.evaluator.forget_store(name)  # a re-import of as many items must scan again
        return Response(status_code=204)

    @app.put("/collections/{name}/search")
    def put_search(name: str, body: dict) -> dict:
        spec = state.retriever.spec(name)
        merged = resolve_search(spec, body)
        db.set_search(state.conn, name, merged)
        return asdict(merged)

    @app.post("/collections/{name}/load", status_code=202)
    def load(name: str, body: LoadRequest):
        spec = state.retriever.spec(name)
        guards.check_default_set(spec)
        if body.dataset not in DATASETS:
            raise HTTPException(422, f"unknown dataset {body.dataset!r}; known: {list(DATASETS)}")
        if DATASETS[body.dataset] != spec.qa_type:
            raise HTTPException(
                409,
                f"dataset {body.dataset} is {DATASETS[body.dataset]};"
                f" store '{name}' is {spec.qa_type}",
            )
        # What the loader spends on: summaries, a proposal when there are no lists, and
        # the build steps. Whether the build spends depends on the items (and on the Hub), so
        # a load that builds is priced even when confirmed, as is one that says what it saw.
        priced = builds(spec) or body.calls is not None or body.est_usd is not None
        if not priced:  # before the dataset is read
            for role in roles_needed(spec, None):
                guards.need_ai(role)
        propose = guards.proposes(spec)
        setup = {"dataset": body.dataset, "limit": body.limit, "propose": propose}
        if priced or ((spec.summarize or propose) and not body.confirm):
            estimate = state.loader.estimate_load(spec, body.dataset, body.limit)
            spends = spec.summarize or propose or estimate["est_usd"] != 0
            if body.confirm or not spends:  # it would queue; otherwise the fresh estimate
                check_shown_price(estimate, body.calls, body.est_usd)  # before any 503
            for role in roles_needed(spec, estimate):
                guards.need_ai(role)
            if spends and not body.confirm:
                return estimate_reply(estimate)
            if not body.confirm and estimate["hub"].get("entry"):
                # Free only because of this entry: the run fails rather than spend unasked.
                setup["hub_entry"] = estimate["hub"]["entry"]
        with guards.not_adding(name):
            job_id = db.create_job(state.conn, "load", name, setup)
        return {"job_id": job_id}

    @app.post("/collections/{name}/copy", status_code=202)
    def copy_collection(name: str, body: CopyRequest):
        """A new collection of this one's items, questions and switched-off sources, built
        under another recipe. Without confirm, what the copy would spend; with it, the
        collection is made and a queued copy job adds and builds its items."""
        source = state.retriever.spec(name)
        if db.get_collection(state.conn, body.name):
            raise HTTPException(409, f"store '{body.name}' already exists")
        found = db.get_eval_set(state.conn, body.name)
        if found and found["qa_type"] != source.qa_type:  # one name, one question type
            raise type_clash(body.name, found["qa_type"], body.name, source.qa_type)
        target = state.loader.copy_spec(source, body.name, body.recipe, body.build)
        # A job on the source could change the items while the copy reads them.
        guards.refuse_busy(name)
        estimate = state.loader.estimate_copy(name, target)
        if body.confirm:
            check_shown_price(estimate, body.calls, body.est_usd)
        for role in roles_needed(target, estimate):
            guards.need_ai(role)
        if not body.confirm:
            return estimate_reply(estimate)
        with guards.not_adding(name):  # the copy reads the source's items
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
        with guards.not_adding(name):
            guards.refuse_busy(name)  # items added or changed meanwhile would miss the rebuilt copy
            if model == spec.dense_model:
                raise HTTPException(409, f"'{name}' is already built with {model}")
            setup = {"model": model, "from": spec.dense_model}
            return {"job_id": db.create_job(state.conn, "reembed", name, setup)}


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
    if job and job["status"] in BUSY:
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
