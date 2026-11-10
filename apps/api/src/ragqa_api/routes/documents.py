"""The document routes: items added by hand, uploaded or imported into a store."""

import json
import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import Guards, body_chunks, check_mapping, check_shown_price, estimate_reply
from ragqa_core import db, imports, jobs
from ragqa_core.imports import FIELDS, SOURCE_NAME, TEXT_SUFFIXES
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
        return state.loader.estimate_import(spec, setup)

    @app.post("/collections/{name}/documents", status_code=202)
    def add_documents(name: str, body: Documents):
        spec = state.retriever.spec(name)
        # An item added now would be keyed under mixed lists, miss a version or skew a run;
        # under the lock, so no job is queued between this check and the add's end.
        now = len(body.items) <= SYNC_ITEMS
        with state.adding_lock:
            for kind in jobs.BUSY_STATE:
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
        if estimate["est_usd"] != 0 and not body.confirm:  # None: not priced, so not free
            return estimate_reply(estimate)
        check_shown_price(estimate, body.calls, body.est_usd)
        for role in roles_needed(spec, estimate):  # before adding: a refusal adds nothing
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
            if estimate["est_usd"] != 0 and not body.confirm:  # None: not priced, so not free
                return estimate_reply(estimate)
            check_shown_price(estimate, body.calls, body.est_usd)
            for role in roles_needed(spec, estimate):
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
        spec = state.retriever.spec(name)
        if accept == "text" and Path(filename).suffix.lower() not in TEXT_SUFFIXES:
            raise HTTPException(422, "only .txt and .md files can be added here")
        # Each file becomes passages: a zip, a table or json lines is not a Documents file.
        is_text = Path(filename).suffix.lower() in TEXT_SUFFIXES
        if spec.qa_type == "documents" and not is_text:
            raise HTTPException(422, DOCUMENTS_ONLY)
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > imports.UPLOAD_LIMIT:
            raise HTTPException(413, f"the file is over {imports.UPLOAD_LIMIT} bytes")
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

    @app.post("/collections/{name}/imports", status_code=202)
    def start_import(name: str, body: ImportRequest):
        spec = state.retriever.spec(name)
        path, meta = imports.open_upload(state.settings.home, name, body.upload_id)
        if not SOURCE_NAME.match(body.source_name):
            raise HTTPException(422, "the source name needs 1-64 letters, digits, space, _ . -")
        check_mapping("items", body.mapping)
        source = "upload:" + body.source_name
        shown = preview_upload(spec, path, meta["filename"], body.mapping, source)
        estimate = estimate_upload(spec, body.upload_id, shown, source)
        if not body.confirm:
            return estimate_reply(estimate, preview=shown)
        if estimate is not None:
            check_shown_price(estimate, body.calls, body.est_usd)
        if estimate is None:
            why = (
                f"map these fields first: {shown['missing']}"
                if shown["missing"]
                else "no usable rows"
            )
            raise HTTPException(422, why)
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
            # It was priced on the recipe now; a queued tag_build would switch it first.
            if db.job_waiting_or_running(state.conn, "tag_build", name):
                raise HTTPException(409, f"a tag_build job is queued or running on '{name}'; wait")
            return {"job_id": db.create_job(state.conn, "import", name, setup)}
