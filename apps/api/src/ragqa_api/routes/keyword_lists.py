"""The keyword routes: a store's keyword tags, keyword lists and vocabulary."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from aiconnect.ai import NoConnection, UnknownModel
from ragqa_api.guards import EstimateReply, Guards, check_shown_price, estimate_reply
from ragqa_core import db
from ragqa_core.loader import sample_items
from ragqa_core.types import KeywordLists
from ragqa_core.vocabulary import clean_terms

NOT_UNTAGGED = "only a Documents store without keyword tags can add them"


class ProposeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirm: bool = False
    # What the caller was shown: given, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


class Lists(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: list[str]
    method: list[str]
    common: list[str]


class Vocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[str]
    confirm: bool = False
    # What the caller was shown: on confirm of a priced retag, other calls or a higher
    # price are refused.
    calls: int | None = None
    est_usd: float | None = None
    # The GET body's other fields, accepted and ignored, so the export PUTs back as the import.
    version: str | None = None
    versions: list | None = None


class KeywordTagsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: bool = False
    # What the caller was shown: given, other calls or a higher price are refused.
    calls: int | None = None
    est_usd: float | None = None


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

    @app.post(
        "/collections/{name}/keyword-tags",
        status_code=202,
        responses={
            200: {
                "model": EstimateReply,
                "description": "Without confirm: what it would spend; nothing is queued",
            }
        },
    )
    def add_keyword_tags(name: str, body: KeywordTagsRequest | None = None):
        """Add keyword tags: a Documents store switches to documents_tagged in place, and a
        queued tag_build job induces its vocabulary and tags every passage. Without confirm,
        what that would spend."""
        spec = db.require_collection(state.conn, name)
        if spec.qa_type != "documents" or spec.recipe != "documents":
            raise HTTPException(409, NOT_UNTAGGED)
        # Under the lock, so no job is queued and no passage added between these checks and
        # this one's queueing.
        with guards.not_adding(name):
            if state.store.count(name) == 0:
                raise HTTPException(409, "add passages first")
            guards.refuse_busy(name)
            if reply := guards.tag_build_gate(spec, body):
                return reply
            return {"job_id": db.create_job(state.conn, "tag_build", name, {})}

    @app.post(
        "/collections/{name}/keyword-lists/propose",
        responses={200: {"description": "Without confirm: {estimate}; with confirm: the result"}},
    )
    def propose(name: str, body: ProposeRequest | None = None):
        spec = db.require_collection(state.conn, name)
        skip = tuple(db.disabled_sources(state.conn, name))
        if not (body and body.confirm):
            sample = sample_items(state.store, name, skip)
            try:
                estimate, calls = state.indexer.plan_propose(spec, sample)
            except (NoConnection, UnknownModel):
                estimate, calls = None, 1  # nothing can price it: the same as vocabulary
            price = {"est_usd": estimate, "calls": calls}
            return estimate_reply(guards.with_missing(price, ["keywords"]))
        sample = sample_items(state.store, name, skip)
        if body.calls is not None or body.est_usd is not None:
            try:
                estimate, calls = state.indexer.plan_propose(spec, sample)
            except (NoConnection, UnknownModel):
                estimate, calls = None, 1  # nothing can price it: the shown price is checked first
            check_shown_price({"est_usd": estimate, "calls": calls}, body.calls, body.est_usd)
        guards.need_ai("keywords")
        return asdict(state.indexer.propose_lists(spec, sample))

    @app.get("/collections/{name}/keyword-lists")
    def list_versions(name: str) -> list[dict]:
        """Every saved version of the lists, newest first, with a count per list."""
        spec = db.require_collection(state.conn, name)
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
        spec = db.require_collection(state.conn, name)
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
        spec = db.require_collection(state.conn, name)
        guards.refuse_running_eval(name)
        with guards.not_adding(name):  # a refused rekey saves no version either
            version = state.indexer.save_lists(spec, KeywordLists(**body.model_dump()))
            job_id = db.create_job(state.conn, "rekey", name, {"version": version})
        return {"version": version, "job_id": job_id}

    @app.get("/collections/{name}/vocabulary")
    def get_vocabulary(name: str) -> dict:
        """The vocabulary in force and every saved version. This body is the export:
        PUT it back, edited, to import a version (spec §16)."""
        spec = db.require_collection(state.conn, name)
        return {
            "version": spec.vocabulary_version,
            "terms": db.get_vocabulary(state.conn, name),
            "versions": db.list_vocabularies(state.conn, name),
        }

    @app.put(
        "/collections/{name}/vocabulary",
        status_code=202,
        responses={
            200: {
                "model": EstimateReply,
                "description": "Without confirm: what it would spend; nothing is queued",
            }
        },
    )
    def put_vocabulary(name: str, body: Vocabulary):
        """Without confirm, what re-tagging every item would cost; with it, saves the terms
        as a version and a queued retag job switches to it once every item is tagged."""
        spec = db.require_collection(state.conn, name)
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
        guards.refuse_item_jobs(name)
        state.indexer.check_model(spec)  # the retag would embed with a model it can't load
        terms = clean_terms(body.terms)
        if not terms:
            raise ValueError("the vocabulary is empty")
        if not body.confirm or build.tagging == "ai":  # an estimate, or an AI-tagging retag
            items = state.builder.stored_items(name)
            try:
                estimate, calls = state.builder.plan_tag(spec, terms, items)
            except (NoConnection, UnknownModel):
                estimate, calls = None, len(items)
            roles = ["keywords"] if calls > 0 else []  # roles_needed's rule: its step makes calls
            if not body.confirm:
                shown = {"est_usd": estimate, "items": len(items), "calls": calls}
                return estimate_reply(guards.with_missing(shown, roles))
            check_shown_price({"est_usd": estimate, "calls": calls}, body.calls, body.est_usd)
            for role in roles:
                guards.need_ai(role)
        with guards.not_adding(name):
            version = state.builder.save_vocabulary(spec, terms)
            job_id = db.create_job(state.conn, "retag", name, {"version": version})
        return {"version": version, "job_id": job_id}
