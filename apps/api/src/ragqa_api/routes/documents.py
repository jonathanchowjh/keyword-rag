"""The document routes: items added by hand, uploaded or imported into a store."""

import json
import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import (
    ADDING_BUSY,
    EstimateReply,
    Guards,
    body_chunks,
    check_mapping,
    check_ready,
    check_shown_price,
    check_source_name,
    check_upload_size,
    estimate_reply,
)
from ragqa_core import db, imports
from ragqa_core.imports import FIELDS, TEXT_SUFFIXES
from ragqa_core.loader import add_usd, roles_needed
from ragqa_core.types import CollectionSpec, Item

DOCUMENT_FIELDS = ("id", "text", "answer", "explanation", "meta")
SYNC_ITEMS = 20  # more items than this are imported by the worker
DOCUMENTS_ONLY = "A Documents store takes .md and .txt files"


class ItemIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    answer: str | None = None
    explanation: str | None = None
    meta: dict = Field(default_factory=dict)


class Documents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ItemIn]
    confirm: bool = False
    # What the caller was shown: given, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


class ImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str
    kind: Literal["items"] = "items"
    mapping: dict[str, str | None] | None = None
    source_name: str
    confirm: bool = False
    # What the caller was shown: on confirm, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

    def preview_upload(
        spec: CollectionSpec,
        path: Path,
        filename: str,
        mapping: dict | None,
        source: str,
    ) -> dict:
        """The preview; a file that cannot be read is removed along with its upload."""

        def owner_of(ids: list[str]) -> dict[str, str | None]:
            return state.store.sources_of(spec.name, ids)

        try:
            return imports.preview(
                path, filename, "items", spec.qa_type, mapping, None, source, owner_of
            )
        except ValueError:
            shutil.rmtree(path.parent, ignore_errors=True)
            raise

    def estimate_upload(spec: CollectionSpec, upload_id: str, shown: dict, source: str):
        """What importing would do; None until the mapping is complete."""
        if not shown["ready"]:
            return None
        setup = {
            "upload_id": upload_id,
            "kind": "items",
            "mapping": shown["mapping"],
            "source": source,
        }
        estimate = state.loader.estimate_import(spec, setup)
        return guards.with_missing(estimate, roles_needed(spec, estimate))

    @app.post(
        "/collections/{name}/documents",
        status_code=202,
        responses={
            200: {
                "description": "Without confirm: {estimate} when it spends or a model is missing; "
                "a free add answers {added}"
            }
        },
    )
    def add_documents(name: str, body: Documents):
        spec = db.require_collection(state.conn, name)
        # An item added now would be keyed under mixed lists, miss a version or skew a run;
        # under the lock, so no job is queued between this check and the add's end.
        now = len(body.items) <= SYNC_ITEMS
        with state.adding_lock:
            guards.refuse_busy(name)
            if name in state.adding:
                raise HTTPException(409, ADDING_BUSY.format(name=name))
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
        usd, summary_calls = state.indexer.plan_summaries(todo) if spec.summarize else (0.0, 0)
        # Decision H (2026-11-09): a sync add makes no keyword lists; the queued path does.
        build_usd, keyword_calls = state.builder.plan_build(spec, todo)
        estimate = {
            "est_usd": add_usd(usd, build_usd),
            "items": len(items),
            "calls": summary_calls + keyword_calls,
            "summary_calls": summary_calls,
            "keyword_calls": keyword_calls,
        }
        roles = roles_needed(spec, estimate)
        shown = guards.with_missing(estimate, roles)
        if not body.confirm and (estimate["est_usd"] != 0 or shown["missing"]):  # None: not free
            return estimate_reply(shown)
        if estimate is not None:  # always, once ready; the type checker cannot see it
            check_shown_price(estimate, body.calls, body.est_usd)
        for role in roles:  # before adding: a refusal adds nothing
            guards.need_ai(role)
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
            roles = roles_needed(spec, estimate)
            shown = guards.with_missing(estimate, roles)
            if not body.confirm and (estimate["est_usd"] != 0 or shown["missing"]):
                return estimate_reply(shown)
            check_shown_price(estimate, body.calls, body.est_usd)
            for role in roles:
                guards.need_ai(role)
            setup["propose"] = guards.proposes(spec)
            with guards.not_adding(spec.name):
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
        kind: Literal["items"] = "items",
        accept: Literal["text"] | None = None,
    ):
        """The raw file as the body; answers with a preview under a guessed mapping.
        With accept=text only a .txt or .md file is taken."""
        spec = db.require_collection(state.conn, name)
        if accept == "text" and Path(filename).suffix.lower() not in TEXT_SUFFIXES:
            raise HTTPException(422, "only .txt and .md files can be added here")
        # Each file becomes passages: a zip, a table or json lines is not a Documents file.
        is_text = Path(filename).suffix.lower() in TEXT_SUFFIXES
        if spec.qa_type == "documents" and not is_text:
            raise HTTPException(422, DOCUMENTS_ONLY)
        check_upload_size(request)
        upload_id = imports.save_upload(state.settings.home, name, filename, body_chunks(request))
        path, _ = imports.open_upload(state.settings.home, name, upload_id)
        source_name = imports.source_name(filename)
        source = "upload:" + source_name
        shown = preview_upload(spec, path, filename, None, source)
        try:
            estimate = estimate_upload(spec, upload_id, shown, source)
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

    @app.post(
        "/collections/{name}/imports",
        status_code=202,
        responses={
            200: {
                "model": EstimateReply,
                "description": "Without confirm: what it would spend; nothing is queued",
            }
        },
    )
    def start_import(name: str, body: ImportRequest):
        spec = db.require_collection(state.conn, name)
        path, meta = imports.open_upload(state.settings.home, name, body.upload_id)
        check_source_name(body.source_name)
        check_mapping("items", body.mapping)
        source = "upload:" + body.source_name
        shown = preview_upload(spec, path, meta["filename"], body.mapping, source)
        estimate = estimate_upload(spec, body.upload_id, shown, source)
        if not body.confirm:
            return estimate_reply(estimate, preview=shown)
        check_ready(shown)
        if estimate is not None:  # always, once ready; the type checker cannot see it
            check_shown_price(estimate, body.calls, body.est_usd)
        for role in roles_needed(spec, estimate):
            guards.need_ai(role)
        setup = {
            "upload_id": body.upload_id,
            "kind": "items",
            "mapping": shown["mapping"],
            "source": source,
            "propose": guards.proposes(spec),
        }
        with guards.not_adding(name):
            guards.refuse_tag_build(name)
            return {"job_id": db.create_job(state.conn, "import", name, setup)}
