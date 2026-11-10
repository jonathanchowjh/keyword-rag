"""Saved chats: each message is searched in every store on the chat. The stores' candidates
pool round-robin by rank (a passage two stores hold takes one slot), the reranker scores the
pool against the question, and the best answer the message. With no reranker the pool keeps
its order, which is RRF's over separate lists. Stores built with different embedding models
pool together: each store is searched in its own model's space, and neither the reranker nor
the merge by rank compares vectors across stores. With no answer model the attempt is
passages only. Each attempt is saved with its decisions, its pipeline and a snapshot of its
evidence, so an old chat reads the same after a store is rebuilt or deleted.

A chat's settings are saved with it: its evidence size, rerank and rewrite, and for each
store a method and candidates (the k it is searched with). Null means inherit, highest level
first: a store's fields from the chat, the store, Settings, then the recommended value; the
chat's own from the chat, Settings, then recommended. Only set_default writes to a store.
"""

import copy
import hashlib
import math
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any

from aiconnect.ai import AI, NoConnection, UnknownModel, UnknownPrice
from ragqa_core import catalog, db, search_defaults
from ragqa_core.connection import Connection
from ragqa_core.evaluator import no_ai
from ragqa_core.prompts import (
    CHAT_PASSAGE_CHARS,
    CHAT_TOKENS,
    CHAT_TURNS,
    REWRITE_TOKENS,
    chat_prompt,
    cited_numbers,
    edge_order,
    rewrite_prompt,
    strip_citations,
)
from ragqa_core.retriever import (
    PER_METHOD_ERRORS,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
    available_methods,
)
from ragqa_core.search_defaults import RECOMMENDED, STORE_FIELDS, inherited
from ragqa_core.types import CollectionSpec, Hit, SearchSettings, unknown_store

# What a store's step or an answer may raise and still leave a saved attempt: it becomes
# the decision's or the attempt's `error`.
CHAT_ERRORS = PER_METHOD_ERRORS + (UnknownPrice, UnknownModel)
NEW_TITLE = "New chat"
TITLE_CHARS = 60
CHAT_FIELDS = ("evidence_size", "rerank", "rewrite")  # the chat's own; null: Settings
UPDATE_FIELDS = ("title", "stores", *CHAT_FIELDS)
RERANK_LOADING = "reranker still loading: merged by rank"
RERANK_UNAVAILABLE = "reranker unavailable: merged by rank"
# Find passages, the labelling pool: the top POOL_DEPTH of each free method a store can run.
POOL_DEPTH = 10
POOL_METHODS = ("vector", "keyword", "hybrid", "keyword_rag")
POOL_LOADING = "The ranking model is still loading, so passages are in search order, not best first"
POOL_UNAVAILABLE = (
    "The ranking model is unavailable, so passages are in search order, not best first"
)
POOL_MODEL_TAGS = (
    "Keyword-tag search was left out: its tags would use AI, and Find passages is free"
)


class AlreadyAnswered(Exception):
    """An attempt that already has an answer; it is not answered twice."""


class UnknownChat(KeyError):
    """An unknown chat, message or attempt."""

    def __str__(self) -> str:  # a KeyError quotes its message
        return str(self.args[0]) if self.args else ""


def error_text(e: Exception) -> str:
    return no_ai(e.role) if isinstance(e, NoConnection) else str(e)


def known(fields, allowed: tuple[str, ...], what: str) -> None:
    unknown = [f for f in fields if f not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; known: {list(allowed)}")


def chat_search(spec: CollectionSpec) -> SearchSettings:
    """The store's search settings as a chat runs them: oracle tags need a question's gold,
    which a chat has none of."""
    search = spec.search or SearchSettings()
    return replace(search, query_tags="classifier") if search.query_tags == "oracle" else search


def text_key(text: str) -> str:
    """A passage's key: the same text in two stores (a copied store) is one passage."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def round_robin(
    lists: dict[str, list[Hit]],
    cap: int | None,
    key: Callable[[Hit], str] = lambda h: text_key(h.text),
) -> list[tuple[str, Hit, int]]:
    """The lists pooled as (list's name, hit, rank in its list): rank 1 of each list in the
    dict's order, then rank 2, and so on, skipping a hit whose key is already taken, up to
    `cap` (None: every hit). Chat pools stores by text; the labelling pool pools methods by
    id. With every key distinct this is RRF's order over separate lists."""
    pool: list[tuple[str, Hit, int]] = []
    taken: set[str] = set()
    longest = max((len(hits) for hits in lists.values()), default=0)
    for i in range(longest):
        for name, hits in lists.items():
            if i >= len(hits):
                continue
            k = key(hits[i])
            if k in taken:
                continue
            taken.add(k)
            pool.append((name, hits[i], i + 1))
            if cap is not None and len(pool) >= cap:
                return pool
    return pool


def relevance(score: float) -> float:
    """A reranker score as 0..1: the logistic of the score, written so it never overflows."""
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    e = math.exp(score)
    return e / (1 + e)


def evidence_of(merged: list[tuple[str, Hit, int, float, float | None]]) -> list[dict]:
    """The merged passages in merged order, each with `n`, its number in the prompt, which
    chat_prompt gives in edge order: the number the answer cites."""
    numbers = {i: n for n, i in enumerate(edge_order(list(range(len(merged)))), start=1)}
    return [
        {
            "n": numbers[i],
            "id": hit.id,
            "text": hit.text[:CHAT_PASSAGE_CHARS],
            "file": hit.meta.get("file"),
            "source": hit.meta.get("source"),
            "part": hit.meta.get("part"),
            "parts": hit.meta.get("parts"),
            "store": store,
            "rank": rank,
            "score": score,
            "relevance": rel,
            "channels": list(hit.channels),
        }
        for i, (store, hit, rank, score, rel) in enumerate(merged)
    ]


def pool_passage(hit: Hit, rel: float | None) -> dict:
    """A labelling pool's passage: where it is in its file, and no store, rank or method."""
    meta = hit.meta
    text = hit.text[:CHAT_PASSAGE_CHARS]
    found = {"part": meta.get("part"), "parts": meta.get("parts"), "relevance": rel}
    return {"id": hit.id, "text": text, "file": meta.get("file"), **found}


class Chats:
    def __init__(self, conn: Connection, retriever: Retriever, ai: AI):
        self.conn = conn
        self.retriever = retriever
        self.ai = ai
        self._answering: set[tuple[str, int, int]] = set()  # (chat, n, attempt) being answered
        # Held over each read, change and write of a chat's settings, so no edit is lost.
        self._lock = threading.Lock()

    # -- chats -------------------------------------------------------------------

    def create(self, stores: list[str], title: str | None = None) -> dict:
        stores = list(dict.fromkeys(stores))
        if not stores:
            raise ValueError("a chat needs at least one store")
        for store in stores:
            self.retriever.spec(store)
        settings = {"evidence_size": None, "rewrite": None, "rerank": None, "stores": {}}
        return self._chat(db.create_chat(self.conn, title or NEW_TITLE, stores, settings))

    def create_like(self, chat_id: str, title: str | None = None) -> dict:
        """A new chat with the template's stores that still exist and a copy of its saved
        settings, nulls kept so the copy keeps following the store and Settings. No messages."""
        template = self._chat(chat_id)
        stores = [s for s in template["stores"] if db.get_collection(self.conn, s)]
        if not stores:
            raise ValueError("a chat needs at least one store")
        settings = copy.deepcopy(template["settings"])
        chosen = settings.get("stores") or {}
        settings["stores"] = {s: v for s, v in chosen.items() if s in stores}
        return self._chat(db.create_chat(self.conn, title or NEW_TITLE, stores, settings))

    def get(self, chat_id: str) -> dict:
        chat = self._chat(chat_id)
        attempts: dict[int, list[dict]] = {}
        named = list(chat["stores"])  # every store the chat or its old rows name
        for a in db.chat_attempts(self.conn, chat_id):
            attempts.setdefault(a["n"], []).append(a)
            named += [d["store"] for d in a["decisions"]] + [e["store"] for e in a["evidence"]]
        messages = [
            {"n": m["n"], "content": m["content"], "attempts": attempts.get(m["n"], [])}
            for m in db.chat_messages(self.conn, chat_id)
        ]
        named = list(dict.fromkeys(named))
        deleted = [s for s in named if db.get_collection(self.conn, s) is None]
        in_force = self._in_force(chat)  # what the next search uses
        return {**chat, "deleted_stores": deleted, "messages": messages, "in_force": in_force}

    def in_force(self, chat_id: str) -> dict[str, Any]:
        return self._in_force(self._chat(chat_id))

    def store_in_force(self, store: str, chosen: dict | None = None) -> dict[str, Any]:
        """The store's method and candidates, each with the level it comes from. `chosen`
        holds the chat's values for the store."""
        return search_defaults.store_in_force(self.conn, store, chosen)

    def update(self, chat_id: str, changes: dict[str, object]) -> dict:
        """Only stores being added must exist, so a list naming a deleted store still saves.
        None for evidence_size, rewrite or rerank inherits; a store taken off the chat drops
        its values."""
        known(changes, UPDATE_FIELDS, "chat fields")
        for field in ("title", "stores"):
            if field in changes and changes[field] is None:
                raise ValueError(f"{field} cannot be null: leave it out to keep it")
        for field in CHAT_FIELDS:
            if field in changes:
                search_defaults.check(field, changes[field])
        title, stores = changes.get("title"), changes.get("stores")
        if title is not None and (not isinstance(title, str) or not title.strip()):
            raise ValueError("a chat needs a title")
        with self._lock:
            chat = self._chat(chat_id)
            if stores is not None:
                if not isinstance(stores, list) or not all(isinstance(s, str) for s in stores):
                    raise ValueError("stores must be a list of store names")
                stores = list(dict.fromkeys(stores))
                if not stores:
                    raise ValueError("a chat needs at least one store")
                for store in stores:
                    if store not in chat["stores"]:
                        self.retriever.spec(store)
            settings = {**chat["settings"], **{f: changes[f] for f in CHAT_FIELDS if f in changes}}
            if stores is not None:
                chosen = settings.get("stores") or {}
                settings["stores"] = {s: v for s, v in chosen.items() if s in stores}
            db.update_chat(self.conn, chat_id, title=title, stores=stores, settings=settings)
        return self._chat(chat_id)

    def delete(self, chat_id: str) -> None:
        self._chat(chat_id)
        db.delete_chat(self.conn, chat_id)

    # -- messages and attempts -----------------------------------------------------

    def search_text(self, chat_id: str, n: int, content: str, rewrite: bool, turns: int) -> dict:
        """Message n's search text and the text its pool is reranked against, where they came
        from, what the rewrite spent and its error. A first message is itself. A follow-up is
        the last `turns` user messages before n and `content`, one per line, reranked against
        `content` alone; the assistant's replies are never searched. With `rewrite` and a
        rewrite model, it is the AI's standalone question, both searched and reranked against;
        a failed rewrite falls back to the join. With no rewrite model nothing is tried."""
        earlier = [m["content"] for m in db.chat_messages(self.conn, chat_id) if m["n"] < n]
        out: dict[str, Any] = {"search_text": content, "rerank_text": content, "from": "message"}
        out.update(usd=0.0, error=None)
        if not earlier:
            return out
        out["search_text"] = "\n".join([*(earlier[-turns:] if turns else []), content])
        out["from"] = "joined"
        if not (rewrite and self._resolves("rewrite")):
            return out
        system, user = rewrite_prompt(earlier[-CHAT_TURNS:], content)
        with self.ai.tally() as spent:
            try:
                done = self.ai.complete(
                    user,
                    role="rewrite",
                    system=system,
                    max_tokens=REWRITE_TOKENS,
                    purpose="chat_rewrite",
                )
                text = done.text.strip()
                if text:
                    out.update(search_text=text, rerank_text=text)
                    out["from"] = "rewrite"
            except CHAT_ERRORS as e:
                out["error"] = error_text(e)
        out["usd"] = spent["usd"]
        return out

    def ask(
        self, chat_id: str, content: str, model: str | None = None, answer: bool = True
    ) -> dict:
        """Saves the message and its first attempt, answered unless answer is False. An
        answer model that does not resolve or has no price is refused first, so nothing is
        saved."""
        chat = self._chat(chat_id)
        if not content.strip():
            raise ValueError("a message needs some text")
        self._check_model(model)
        n = db.add_chat_message(self.conn, chat_id, content)
        try:
            attempt = self._attempt(chat, n, content, answer, model)
        except BaseException:
            db.delete_chat_message(self.conn, chat_id, n)  # never a message with no attempt
            raise
        if n == 1 and chat["title"] == NEW_TITLE:
            db.update_chat(self.conn, chat_id, title=" ".join(content.split())[:TITLE_CHARS])
        return {"n": n, "content": content, "attempts": [attempt]}

    def rerun(
        self,
        chat_id: str,
        n: int,
        settings: dict | None = None,
        stores: dict[str, dict] | None = None,
    ) -> dict:
        """Checks every value first (a bad one saves none), saves them onto the chat, then
        adds an unanswered attempt to message n, searched with the latest attempt's text
        unless the rewrite or follow-up turns it was made with changed. `settings`
        holds the chat's own fields, `stores` each store's {method?, candidates?}; None
        inherits, and an absent field keeps the chat's value."""
        settings, stores = settings or {}, stores or {}
        self._chat(chat_id)
        message = self._message(chat_id, n)
        self._attempt_row(chat_id, n, 1)
        with self._lock:  # checked and saved in one hold, so no other edit lands between
            chat = self._chat(chat_id)
            db.update_chat(self.conn, chat_id, settings=self._changed(chat, settings, stores))
        chat = self._chat(chat_id)
        text = self._rerun_text(chat, n, message["content"])
        return self._attempt(chat, n, message["content"], False, None, text)

    def answer(self, chat_id: str, n: int, attempt: int, model: str | None = None) -> dict:
        """Answers an attempt from its evidence snapshot. One already answered, or being
        answered by another request, is refused before anything is paid. The attempt keeps
        its search's error (a failed rewrite's note); the answer's own replaces the last."""
        self._chat(chat_id)
        message = self._message(chat_id, n)
        key = (chat_id, n, attempt)
        with self._lock:  # claimed before the paid call, so a second request pays nothing
            if key in self._answering:
                raise AlreadyAnswered(f"attempt {attempt} of message {n} is being answered")
            self._answering.add(key)
        try:
            found = self._attempt_row(chat_id, n, attempt)
            if found["answer"] is not None:
                raise AlreadyAnswered(f"attempt {attempt} of message {n} is already answered")
            answered = self._answer(chat_id, n, message["content"], found["evidence"], model)
            answered["usd"] += found["usd"]
            pipeline = found["pipeline"]  # None on an old attempt: it stays None
            if pipeline is not None:
                pipeline = {**pipeline, "answer_cached": answered["cached"]}
            if not db.answer_chat_attempt(self.conn, chat_id, n, attempt, answered, pipeline):
                raise AlreadyAnswered(f"attempt {attempt} of message {n} is already answered")
        finally:
            with self._lock:
                self._answering.discard(key)
        return self._attempt_row(chat_id, n, attempt)

    def set_default(self, chat_id: str, store: str) -> dict:
        """Saves the chat's method and candidates for the store, those it sets, as the
        store's own, then clears them on the chat. With none set it changes nothing; a method
        the store can't run is MethodUnavailable and saves nothing."""
        with self._lock:
            chat = self._chat(chat_id)
            if store not in chat["stores"]:
                raise UnknownCollection(f"'{store}' is not on this chat")
            spec = self.retriever.spec(store)
            stores = dict(chat["settings"].get("stores") or {})
            chosen = stores.get(store) or {}
            method = chosen.get("method")
            why = self._reasons(spec)[method] if method is not None else None
            if why:  # saved on the store, it would fail every chat that inherits it
                raise MethodUnavailable(f"{method} {why} on '{store}'")
            if any(chosen.get(f) is not None for f in STORE_FIELDS):
                stores[store] = dict.fromkeys(STORE_FIELDS)
                with self.conn.lock:
                    if chosen.get("method") is not None:
                        db.set_chat_method(self.conn, store, chosen["method"])
                    if chosen.get("candidates") is not None:
                        db.set_chat_k(self.conn, store, chosen["candidates"])
                    settings = {**chat["settings"], "stores": stores}
                    db.update_chat(self.conn, chat_id, settings=settings)
        return {
            "chat_method": search_defaults.chat_method(db.get_chat_method(self.conn, store)),
            "chat_k": db.get_chat_k(self.conn, store),
            "in_force": self.store_in_force(store),
        }

    # -- one attempt -------------------------------------------------------------

    def _rerun_text(self, chat: dict, n: int, content: str) -> dict:
        """The search text for a re-run of message n: the latest attempt's, unless the way it
        was made differs from the way in force (message, rewrite, or joined with so many
        turns), then made again. Follow-up turns don't count while rewriting: the rewrite
        reads CHAT_TURNS messages, so no rewrite is paid for nothing. An attempt from before
        the pipeline, or a re-run of one ("legacy"), keeps its text whatever is in force."""
        force = self._in_force(chat)
        rewrite = bool(force["rewrite"]["value"])
        turns = int(force["follow_up_turns"])
        latest = db.chat_attempts(self.conn, chat["id"], n)[-1]
        first = not any(m["n"] < n for m in db.chat_messages(self.conn, chat["id"]))
        pipeline = latest.get("pipeline")
        if pipeline is None or pipeline["search_from"] == "legacy":  # its text as it was
            rerank_text = content if pipeline is None else pipeline["rerank_text"]
            reused = {"search_text": latest["search_text"], "rerank_text": rerank_text}
            return {**reused, "from": "legacy", "usd": 0.0, "error": None}
        if first:
            wanted: tuple = ("message",)
        elif rewrite and force["rewrite_available"]:
            wanted = ("rewrite",)
        else:
            wanted = ("joined", turns)
        made = pipeline["search_from"]
        had = (made,) if made in ("message", "rewrite") else ("joined", pipeline["follow_up_turns"])
        if wanted != had:
            return self.search_text(chat["id"], n, content, rewrite, turns)
        reused = {"search_text": latest["search_text"], "rerank_text": pipeline["rerank_text"]}
        return {**reused, "from": made, "usd": 0.0, "error": None}

    def _attempt(
        self,
        chat: dict,
        n: int,
        content: str,
        answer: bool,
        model: str | None,
        text: dict | None = None,
    ) -> dict:
        """Searches every store on the chat for message n, pools and reranks their passages,
        answers when asked, and saves the attempt. Its usd is the rewrite's, every store's
        and the answer's, tallied one after another (tallies never nest). With no answer
        model, the attempt is passages only: nothing is asked and nothing is an error."""
        force = self._in_force(chat)
        if text is None:
            rewrite, turns = bool(force["rewrite"]["value"]), int(force["follow_up_turns"])
            text = self.search_text(chat["id"], n, content, rewrite, turns)
        errors = []
        if text["error"]:
            errors.append(f"the rewrite failed, so the joined text was searched: {text['error']}")
        live = force["stores"]  # a deleted store has no values: its decision says so
        plan = {
            s: {f: live[s][f]["value"] if s in live else None for f in STORE_FIELDS}
            for s in chat["stores"]
        }
        size = int(force["evidence_size"]["value"])
        rerank, rrf_k = bool(force["rerank"]["value"]), int(force["rrf_k"])
        reranker, pool_cap = str(force["rerank_model"]), int(force["rerank_pool"])
        found = self.search(
            plan, text["search_text"], text["rerank_text"], size, rerank, rrf_k, reranker, pool_cap
        )
        # No answer model was available when this attempt was made, re-runs included.
        passages_only = model is None and not self._resolves("answer")
        row = {
            "search_text": text["search_text"],
            "decisions": found["decisions"],
            "evidence": found["evidence"],
            "answer": None,
            "citations": [],
            "usd": text["usd"] + found["usd"],
            "search_error": "; ".join(errors) or None,  # kept when the attempt is answered later
            "error": None,
            "model": None,
            "pipeline": {
                "search_from": text["from"],
                "rerank_text": text["rerank_text"],
                "merged_by": found["merged_by"],
                "rerank": rerank,
                "rerank_model": reranker,
                "rerank_pool": pool_cap,
                "rewrite": bool(force["rewrite"]["value"]),
                "follow_up_turns": int(force["follow_up_turns"]),
                "evidence_size": size,
                "rrf_k": rrf_k,
                "pool": found["pool"],
                "notes": found["notes"],
                "passages_only": passages_only,
                "answer_skipped": not answer,
                "answer_cached": False,  # True once the AI cache serves its answer
            },
        }
        if answer and not passages_only:
            answered = self._answer(chat["id"], n, content, found["evidence"], model)
            row.update(answer=answered["answer"], citations=answered["citations"])
            row.update(model=answered["model"], usd=row["usd"] + answered["usd"])
            row["pipeline"]["answer_cached"] = answered["cached"]
            if answered["error"]:
                errors.append(answered["error"])
        row["error"] = "; ".join(errors) or None
        number = db.add_chat_attempt(self.conn, chat["id"], n, row)
        return self._attempt_row(chat["id"], n, number)

    def search(
        self,
        plan: dict[str, dict],
        search_text: str,
        rerank_text: str,
        evidence_size: int,
        rerank: bool,
        rrf_k: int,
        rerank_model: str,
        rerank_pool: int,
    ) -> dict:
        """Searches each store in `plan` ({store: {method, candidates}}) with search_text,
        pools their passages round-robin by rank up to rerank_pool, reranks the pool against
        rerank_text when `rerank` is set and the reranker is ready, and cuts to
        evidence_size. The pool is never smaller than evidence_size. Without a rerank the
        pool keeps its order, scored 1 / (rrf_k + rank); a loading or failed reranker leaves
        a note, and is never waited for."""
        decisions, lists = [], {}
        for store, chosen in plan.items():
            decision, hits = self._decide(
                store, search_text, chosen["method"], chosen["candidates"], rrf_k
            )
            decisions.append(decision)
            if hits:
                lists[store] = hits
        pool = round_robin(lists, max(rerank_pool, evidence_size))  # never fewer passages
        notes: list[str] = []
        merged: list[tuple[str, Hit, int, float, float | None]] = [
            (store, hit, rank, 1 / (rrf_k + rank), None) for store, hit, rank in pool
        ]
        merged_by = "rrf"
        scores = self._rerank(rerank_text, pool, rerank_model, notes) if rerank and pool else None
        if scores is not None:
            order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
            merged = [(*pool[i], scores[i], relevance(scores[i])) for i in order]
            merged_by = "rerank"
        evidence = evidence_of(merged[:evidence_size])
        used = Counter(e["store"] for e in evidence)
        for decision in decisions:
            decision["used"] = used[decision["store"]]
        return {
            "decisions": decisions,
            "evidence": evidence,
            "merged_by": merged_by,
            "pool": len(pool),
            "notes": notes,
            "usd": sum(d["usd"] for d in decisions),
        }

    def labelling_pool(self, store: str, question: str, labelled: list[str]) -> dict:
        """Find passages: the top POOL_DEPTH of each free method the store can run, searched
        as a chat searches it, pooled once by id and reranked against the question with the
        Settings reranker, whatever chat's rerank says. A loading or failed reranker keeps
        round-robin order, with a note. keyword_rag whose tags would be the AI's is left out
        with a note, AI or none; so is a method whose search fails. `labelled` passages the
        pool missed come back under `labelled`; those the store no longer holds, `missing`."""
        spec = db.get_collection(self.conn, store)
        if spec is None:
            raise UnknownCollection(unknown_store(store))
        settings: dict[str, Any] = search_defaults.in_force(self.conn)
        rrf_k = int(settings["rrf_k"])
        reranker = str(settings["rerank_model"] or self.retriever.models.rerank_model)
        notes: list[str] = []
        lists: dict[str, list[Hit]] = {}
        reasons = self._reasons(spec)
        search = chat_search(spec)  # searched as checked: a settings change mid-pool can't spend
        for method in POOL_METHODS:
            ready = reasons[method] is None
            if method == "keyword_rag" and ready and search.query_tags == "model":
                notes.append(POOL_MODEL_TAGS)  # checked first: why_not would say "needs AI"
                continue
            if self.why_not(spec, method) is not None or self.retriever.is_paid(spec, method):
                continue
            try:
                hits, _ = self._retrieve(spec, question, method, POOL_DEPTH, rrf_k, search)
            except CHAT_ERRORS as e:
                label = catalog.METHODS[method]["label"]
                notes.append(f"{label} was left out: {error_text(e)}")
                continue
            lists[method] = hits
        pool = round_robin(lists, None, key=lambda h: h.id)
        merged: list[tuple[Hit, float | None]] = [(hit, None) for _, hit, _ in pool]
        merged_by = "rank"
        if not pool:
            notes.append(f"No passages found in {store}")
        else:
            said: list[str] = []
            scores = self._rerank(question, pool, reranker, said)
            if scores is None:
                notes.append(POOL_LOADING if said == [RERANK_LOADING] else POOL_UNAVAILABLE)
            else:
                order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
                merged = [(pool[i][1], relevance(scores[i])) for i in order]
                merged_by = "rerank"
        pooled = {hit.id for hit, _ in merged}
        wanted = [i for i in dict.fromkeys(labelled) if i not in pooled]
        held = self.retriever.store.get(store, wanted) if wanted else []
        held_ids = {hit.id for hit in held}
        return {
            "store": store,
            "methods": list(lists),
            "passages": [pool_passage(hit, rel) for hit, rel in merged],
            "merged_by": merged_by,
            "notes": notes,
            "labelled": [pool_passage(hit, None) for hit in held],
            "missing": [i for i in wanted if i not in held_ids],
        }

    def reference_prompt(
        self, store: str, question: str, passage_ids: list[str]
    ) -> tuple[str, str]:
        """The labelling answer's prompt: chat's, over the store's held passages among
        `passage_ids`, each once in the order sent, with no history."""
        if db.get_collection(self.conn, store) is None:
            raise UnknownCollection(unknown_store(store))
        hits = self.retriever.store.get(store, list(dict.fromkeys(passage_ids)))
        if not hits:
            raise ValueError(f"mark at least one passage Right that is in {store} first")
        return chat_prompt(question, [], hits)

    def reference_answer(
        self, store: str, question: str, passage_ids: list[str], confirm: bool
    ) -> dict:
        """The AI's answer to a labelled question from its Right passages, one `complete`, cached.
        Without `confirm`, only its estimate: {"estimate": {"est_usd", "calls"}} (est_usd None
        when no answer model is set up). Nothing is stored."""
        system, user = self.reference_prompt(store, question, passage_ids)
        if not confirm:
            try:
                usd, calls = self.ai.plan_many([user], CHAT_TOKENS, role="answer", system=system)
            except NoConnection:
                return {"estimate": {"est_usd": None, "calls": 1}}
            return {"estimate": {"est_usd": usd, "calls": calls}}
        with self.ai.tally() as spent:
            done = self.ai.complete(
                user,
                role="answer",
                system=system,
                max_tokens=CHAT_TOKENS,
                purpose="reference_answer",
            )
        return {"answer": strip_citations(done.text), "usd": spent["usd"]}

    def _rerank(
        self, text: str, pool: list[tuple[str, Hit, int]], model: str, notes: list[str]
    ) -> list[float] | None:
        """The reranker's score of each pooled passage against `text`, or None, with a note
        saying why, when it is still loading or unavailable. A failed one is retried in the
        background; an idle one loads now."""
        models = self.retriever.models
        status = models.rerank_status(model)
        if status == "loading":
            notes.append(RERANK_LOADING)
            return None
        if status == "failed":
            notes.append(RERANK_UNAVAILABLE)
            models.warm_rerank_in_background(model)
            return None
        try:
            return models.rerank(text, [hit.text for _, hit, _ in pool], model)
        except Exception:
            notes.append(RERANK_UNAVAILABLE)
            return None

    def _decide(
        self, store: str, search_text: str, method: str | None, candidates: int | None, rrf_k: int
    ) -> tuple[dict, list[Hit]]:
        """One store's decision and its passages, searched with `method` and k = candidates.
        A method the store can't run is replaced by hybrid, with a note saying why. A failure
        in its retrieval is the decision's error, and the other stores still run. `used` is
        filled in once the evidence is chosen."""
        decision = {
            "store": store,
            "method": None,
            "candidates": None,
            "search": None,
            "tags": None,  # the query tags keyword_rag picked; None for other methods
            "found": 0,
            "used": 0,
            "note": "deleted",
            "usd": 0.0,
            "error": None,
        }
        spec = db.get_collection(self.conn, store)
        if spec is None:
            return decision, []
        decision["note"] = None
        if method is None or candidates is None:  # not given: the store's own chain
            force = self.store_in_force(store)
            method = method or str(force["method"]["value"])
            candidates = candidates or int(force["candidates"]["value"])
        search = chat_search(spec)
        why = self.why_not(spec, method)
        if why:
            decision["note"] = f"{method} {why}, so hybrid was used"
            method = "hybrid"
        decision.update(method=method, candidates=candidates, search=asdict(search))
        hits: list[Hit] = []
        with self.ai.tally() as spent:
            try:
                hits, decision["tags"] = self._retrieve(
                    spec, search_text, method, candidates, rrf_k, search
                )
            except CHAT_ERRORS as e:
                decision["error"] = error_text(e)
                hits = []
        decision["usd"] = spent["usd"]
        decision["found"] = len(hits)
        return decision, hits

    def _retrieve(
        self,
        spec: CollectionSpec,
        text: str,
        method: str,
        k: int,
        rrf_k: int,
        search: SearchSettings,
    ) -> tuple[list[Hit], list[str] | None]:
        """The store's passages and, for keyword_rag, the query tags it picked first, as
        POST /chat runs them."""
        tags = None
        if method == "keyword_rag":
            skip = tuple(db.disabled_sources(self.conn, spec.name))
            tags = self.retriever.query_tags(spec, text, search, skip)
        return self.retriever.retrieve(spec.name, text, method, k, rrf_k, search, tags), tags

    def _answer(
        self, chat_id: str, n: int, content: str, evidence: list[dict], model: str | None
    ) -> dict:
        """The answer to message n from an evidence snapshot, after its history."""
        return self.answer_from(content, self._history(chat_id, n), evidence, model)

    def answer_from(
        self, content: str, history: list, evidence: list[dict], model: str | None
    ) -> dict:
        """The answer to `content` after `history` ([{role, content}]) from evidence (kept in
        merged order, so the prompt numbers it as `n` says), with its citations, usd, error,
        model, and whether the AI cache served it (False when it failed). POST /chat answers
        through it too."""
        hits = [Hit(e["id"], e["text"], e["score"], {}, [], None, None, {}) for e in evidence]
        system, user = chat_prompt(content, history, hits)
        out: dict = {"answer": None, "citations": [], "error": None, "model": None}
        out["cached"] = False
        with self.ai.tally() as spent:
            try:
                out["model"] = model or self.ai.model_for("answer")
                done = self.ai.complete(
                    user,
                    role="answer",
                    system=system,
                    model=model,
                    max_tokens=CHAT_TOKENS,
                    purpose="chat",
                )
                out.update(answer=done.text, cached=done.cached)
            except CHAT_ERRORS as e:
                out.update(error=error_text(e), model=None)
        out["usd"] = spent["usd"]
        if out["answer"] is not None:
            by_n = {e["n"]: e for e in evidence}
            keys = ("n", "id", "text", "file", "source", "part", "parts", "store")
            out["citations"] = [  # .get: evidence saved before it named its file has none
                {k: by_n[c].get(k) for k in keys}
                for c in cited_numbers(out["answer"], len(evidence))
            ]
        return out

    def _history(self, chat_id: str, n: int) -> list[dict]:
        """The messages before n, each followed by its latest answered attempt's answer,
        cut to the last CHAT_TURNS turns."""
        answers = {
            a["n"]: a["answer"]
            for a in db.chat_attempts(self.conn, chat_id)  # by attempt, so the latest wins
            if a["n"] < n and a["answer"] is not None
        }
        history = []
        for m in db.chat_messages(self.conn, chat_id):
            if m["n"] >= n:
                break
            history.append({"role": "user", "content": m["content"]})
            if m["n"] in answers:
                history.append({"role": "assistant", "content": answers[m["n"]]})
        return history[-CHAT_TURNS:]

    # -- checks and reads --------------------------------------------------------

    def _check_model(self, model: str | None) -> None:
        """Refuses, as POST /chat does, an answer model that does not resolve (UnknownModel)
        or has no price (UnknownPrice). With no answer model at all, the attempt is
        passages only."""
        try:
            ref = model or self.ai.model_for("answer")
        except NoConnection:
            return
        self.ai.need_rates(*self.ai.resolve(ref))

    def _changed(self, chat: dict, settings: dict, stores: dict[str, dict]) -> dict:
        """The chat's settings with `settings` and `stores` saved over them, every value
        checked first. A method the store can't run, or that needs AI when none resolves,
        is MethodUnavailable."""
        known(settings, CHAT_FIELDS, "chat settings")
        for field, value in settings.items():
            search_defaults.check(field, value)
        for store, values in stores.items():
            if store not in chat["stores"]:
                raise UnknownCollection(f"'{store}' is not on this chat")
            known(values, STORE_FIELDS, f"settings for '{store}'")
            for field, value in values.items():
                search_defaults.check(field, value)
            spec = self.retriever.spec(store)
            method = values.get("method")
            why = self.why_not(spec, method) if method is not None else None
            if why:
                raise MethodUnavailable(f"{method} {why} on '{store}'")
        chosen = dict(chat["settings"].get("stores") or {})
        for store, values in stores.items():
            chosen[store] = {**dict.fromkeys(STORE_FIELDS), **chosen.get(store, {}), **values}
        return {**chat["settings"], **settings, "stores": chosen}

    def why_not(self, spec: CollectionSpec, method: str) -> str | None:
        """Why the chat can't run `method` on the store, or None when it can: what
        available_methods says, else "needs AI" for a method that would call AI (a paid one,
        or keyword_rag picking its tags by AI) when no AI resolves."""
        why = self._reasons(spec)[method]
        paid = self.retriever.is_paid(spec, method)
        paid = paid or (method == "keyword_rag" and chat_search(spec).query_tags == "model")
        if why is None and paid and not self._resolves("query"):
            why = "needs AI"
        return why

    def _resolves(self, role: str) -> bool:
        try:
            self.ai.model_for(role)
        except NoConnection:
            return False
        return True

    def _in_force(self, chat: dict) -> dict[str, Any]:
        """What the chat's next search uses, each chat field with the level it comes from."""
        own = chat["settings"]
        saved, settings = search_defaults.saved(self.conn), search_defaults.in_force(self.conn)
        out: dict[str, Any] = {
            f: inherited({"chat": own.get(f), "settings": saved[f], "recommended": RECOMMENDED[f]})
            for f in CHAT_FIELDS
        }
        for field in ("follow_up_turns", "rrf_k", "rerank_pool"):
            out[field] = settings[field]
        out["rerank_model"] = settings["rerank_model"] or self.retriever.models.rerank_model
        out["rewrite_available"] = self._resolves("rewrite")
        out["answer_available"] = self._resolves("answer")
        chosen = own.get("stores") or {}
        live = [s for s in chat["stores"] if db.get_collection(self.conn, s) is not None]
        out["stores"] = {s: self.store_in_force(s, chosen.get(s)) for s in live}
        return out

    def _reasons(self, spec: CollectionSpec) -> dict[str, str | None]:
        has_lists = db.get_keyword_lists(self.conn, spec.name) is not None
        return available_methods(spec, has_lists, db.tag_build_unfinished(self.conn, spec.name))

    def _chat(self, chat_id: str) -> dict:
        chat = db.get_chat(self.conn, chat_id)
        if chat is None:
            raise UnknownChat(f"no chat {chat_id}")
        return chat

    def _message(self, chat_id: str, n: int) -> dict:
        found = [m for m in db.chat_messages(self.conn, chat_id) if m["n"] == n]
        if not found:
            raise UnknownChat(f"no message {n} in chat {chat_id}")
        return found[0]

    def _attempt_row(self, chat_id: str, n: int, attempt: int) -> dict:
        found = [a for a in db.chat_attempts(self.conn, chat_id, n) if a["attempt"] == attempt]
        if not found:
            raise UnknownChat(f"no attempt {attempt} of message {n} in chat {chat_id}")
        return found[0]

    def list(self) -> list[dict]:
        """Every chat, newest first. Last in the class, so `list` in the annotations above
        is the builtin."""
        fields = ("id", "title", "stores", "updated_at", "messages")
        return [{f: c[f] for f in fields} for c in db.list_chats(self.conn)]
