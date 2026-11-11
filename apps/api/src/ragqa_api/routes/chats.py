"""The chat routes: chats and their messages, and the one-shot /chat."""

from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from ragqa_api.guards import Guards, check_methods
from ragqa_core import db, search_defaults
from ragqa_core.chats import NEEDS_AI
from ragqa_core.prompts import CHAT_TURNS
from ragqa_core.search_defaults import MAX_CANDIDATES, MAX_EVIDENCE


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    model: str | None = None


def brief_reply(message: dict) -> dict:
    """A message's first attempt, short: the answer, its citations and the passages, for an
    app that calls the API. The raw score is left out; relevance is 0 to 1 or null."""
    one = message["attempts"][0]
    pipeline = one["pipeline"]
    keys = ("n", "store", "id", "text", "file", "source", "part", "parts", "relevance")
    return {
        "n": message["n"],
        "answer": one["answer"],
        "citations": one["citations"],
        "passages": [{k: e[k] for k in keys} for e in one["evidence"]],
        "passages_only": pipeline["passages_only"] or pipeline.get("answer_skipped", False),
        "error": one["error"],
        "usd": one["usd"],
    }


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

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
        nulls = [
            f
            for f in ("title", "stores")
            if f in body.model_fields_set and getattr(body, f) is None
        ]
        if nulls:
            raise ValueError(f"{', '.join(nulls)} cannot be null: leave a field out to keep it")
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

    @app.post("/chat")
    def chat(body: ChatRequest) -> dict:
        """An answer from a store's passages, citing them as [n]. Searched as a chat searches
        one store: its method and candidates, then the pool reranked against the question,
        with Settings' rerank, RRF k, reranker and pool cap. Spends like the Lab: no
        estimate, the cost is in the reply, the budget cap holds."""
        *earlier, last = body.messages
        if last.role != "user":
            raise ValueError("the last message must be the user's")
        spec = db.require_collection(state.conn, body.collection)
        chain = search_defaults.store_in_force(state.conn, spec.name)
        method = body.method or str(chain["method"]["value"])
        if body.method is not None:
            check_methods([method])
            if method not in search_defaults.CHAT_METHODS:
                raise HTTPException(422, f"{method} is not a chat method")
            why = state.chats.why_not(spec, method)
            if why:  # no model is 503; a method the store can't run, 409
                status = 503 if why == NEEDS_AI else 409
                raise HTTPException(status, f"{method} {why} on '{body.collection}'")
        defaults: dict[str, Any] = search_defaults.in_force(state.conn)
        found = state.chats.search(
            {spec.name: {"method": method, "candidates": chain["candidates"]["value"]}},
            last.content,
            last.content,
            body.k or int(defaults["evidence_size"]),
            bool(defaults["rerank"]),
            search_defaults.search_settings(defaults, state.models.rerank_model),
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
