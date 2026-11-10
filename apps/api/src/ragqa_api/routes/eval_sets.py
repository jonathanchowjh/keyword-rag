"""The eval-set routes: sets, their questions and labels, and question imports."""

import shutil
from dataclasses import asdict, replace

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import (
    STATIC,
    Guards,
    body_chunks,
    check_mapping,
    check_shown_price,
    type_clash,
)
from ragqa_core import db, imports
from ragqa_core.imports import SOURCE_NAME
from ragqa_core.search_defaults import MAX_EVIDENCE
from ragqa_core.types import NotFound, QaType, Question


class NewEvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    """The AI's answer to a labelled question from the passages marked Right: priced without
    `confirm`, asked with it."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    passages: list[str] = Field(min_length=1, max_length=MAX_EVIDENCE)
    confirm: bool = False
    # What the caller was shown: on confirm, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


class QuestionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    choices: list[str] | None = None
    gold_answers: list[str] = Field(default_factory=list)
    gold_ids: list[str] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)
    judged_ids: list[str] | None = None


class QuestionsPut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[QuestionIn]
    base_version: str | None = None


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


class EvalSetImport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str
    mapping: dict[str, str | None] | None = None
    source_name: str


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

    def eval_runs(name: str) -> list[dict]:
        return [
            j
            for j in db.list_jobs(state.conn, kind="eval", limit=db.ALL)
            if j["setup"].get("eval_set") == name
        ]

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
        # One lock from reading the store's default to setting it, so two creates
        # can't both become it, and no load is queued in between.
        with state.conn.lock:
            first = False  # the linked store has no default set: this one becomes it
            if body.store is not None:
                first = db.default_eval_set(state.conn, body.store) is None
                if first and guards.store_is_loading(body.store):
                    raise HTTPException(409, f"a load is under way on '{body.store}'; wait")
            db.create_eval_set(state.conn, body.name, body.qa_type, body.store)
            if first and body.store is not None:
                db.set_default_eval_set(state.conn, body.store, body.name)
        return {"name": body.name}

    @app.get("/eval-sets/{name}")
    def get_eval_set(name: str, version: str | None = None) -> dict:
        row = guards.known_set(name)
        questions = db.get_questions(state.conn, name, version)
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
        row = guards.known_set(name)
        guards.not_loading(name)
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
        row = guards.known_set(name)
        guards.not_loading(name)
        if body.question.id != question_id or body.base.id != question_id:
            raise HTTPException(422, "the question's id must match the address")
        q = Question(**body.question.model_dump())
        if problem := imports.question_problem(q, row["qa_type"]):
            raise HTTPException(422, f"question '{q.id}': {problem}")
        try:
            version = db.upsert_question(state.conn, name, q, Question(**body.base.model_dump()))
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return {"version": version}

    @app.delete("/eval-sets/{name}/questions/{question_id:path}")
    def delete_eval_question(name: str, question_id: str) -> dict:
        guards.known_set(name)
        guards.not_loading(name)
        return {"version": db.delete_question(state.conn, name, question_id)}

    @app.post("/eval-sets/{name}/pool")
    def eval_set_pool(name: str, body: PoolRequest) -> dict:
        row = guards.known_set(name)
        if row["qa_type"] != "documents":
            raise HTTPException(409, "Find passages is for Documents eval sets")
        if row["store"] is None:
            raise HTTPException(409, "pick a store for this eval set first")
        return state.chats.labelling_pool(row["store"], body.question, body.labelled)

    @app.post("/eval-sets/{name}/answer")
    def eval_set_answer(name: str, body: ReferenceAnswerRequest) -> dict:
        row = guards.known_set(name)
        if row["qa_type"] != "documents":
            raise HTTPException(409, "The AI's answer is for Documents eval sets")
        if row["store"] is None:
            raise HTTPException(409, "pick a store for this eval set first")
        question = body.question.strip()
        if not question:
            raise HTTPException(422, "a question needs some text")
        if body.confirm and (body.calls is not None or body.est_usd is not None):
            priced = state.chats.reference_answer(row["store"], question, body.passages, False)
            check_shown_price(priced["estimate"], body.calls, body.est_usd)
        return state.chats.reference_answer(row["store"], question, body.passages, body.confirm)

    @app.put("/eval-sets/{name}/questions")
    def put_eval_questions(name: str, body: QuestionsPut) -> dict:
        row = guards.known_set(name)
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
        guards.not_loading(name)
        try:
            version = db.replace_questions(state.conn, name, questions, body.base_version)
        except ValueError as e:  # the questions were checked above: only a stale base is left
            raise HTTPException(409, str(e)) from None
        return {"version": version}

    @app.patch("/eval-sets/{name}")
    def patch_eval_set(name: str, body: EvalSetPatch) -> dict:
        row = guards.known_set(name)
        sent = body.model_fields_set
        if "include_leaked" in sent and body.include_leaked is None:
            raise HTTPException(422, "include_leaked cannot be null: leave it out to keep it")
        if "store" in sent and body.store is not None:
            check_link(name, row["qa_type"], body.store)
        if body.include_leaked is not None:
            db.set_include_leaked(state.conn, name, body.include_leaked)
        if "store" in sent:
            db.set_eval_set_store(state.conn, name, body.store)
        return guards.known_set(name)

    @app.delete("/eval-sets/{name}")
    def delete_eval_set(name: str, confirm: bool = False):
        guards.known_set(name)
        runs = eval_runs(name)
        if busy := next((j for j in runs if j["status"] in ("queued", "running")), None):
            raise HTTPException(409, f"an eval that uses '{name}' is {busy['status']}; wait")
        guards.not_loading(name)
        if not confirm:
            return {"runs": len(runs)}
        db.delete_eval_set(state.conn, name)
        imports.delete_uploads(state.settings.home, f"eval-set:{name}")
        return Response(status_code=204)

    @app.post("/eval-sets/{name}/uploads")
    def add_eval_set_upload(name: str, request: Request, filename: str):
        """The raw file as the body, questions only; answers with a preview."""
        row = guards.known_set(name)
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
        row = guards.known_set(name)
        guards.not_loading(name)
        try:
            path, meta = imports.open_upload(
                state.settings.home, f"eval-set:{name}", body.upload_id
            )
        except NotFound:
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
            "version": guards.known_set(name)["version"],
        }
