"""Saved chats: each message is searched across the chat's stores. A store is kept when its
best match clears its bar, it has no bar, the last message kept it, or it is searched
anyway; the kept stores' passages merge by RRF and answer the message. Each attempt is saved
with its decisions and a snapshot of its evidence, so an old chat reads the same after a
store is rebuilt or deleted.

Overrides, the edits made to a store in the retrieval step, live here in memory until a
restart. Only set_default writes them to the store.
"""

import copy
import math
import threading
from dataclasses import asdict, replace

from aiconnect.ai import AI, NoConnection, UnknownModel, UnknownPrice
from ragqa_core import db
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
)
from ragqa_core.recipes import search_settings
from ragqa_core.retriever import (
    METHOD_NAMES,
    PER_METHOD_ERRORS,
    MethodUnavailable,
    Retriever,
    UnknownCollection,
    available_methods,
)
from ragqa_core.router import Router
from ragqa_core.types import Bar, CollectionSpec, Hit, SearchSettings

# What a store's step or an answer may raise and still leave a saved attempt: it becomes
# the decision's or the attempt's `error`.
CHAT_ERRORS = PER_METHOD_ERRORS + (UnknownPrice, UnknownModel)
EVIDENCE_SIZE = 8
MAX_EVIDENCE = 20  # also the most passages an override's k may ask of one store
FOLLOW_UP_TURNS = 2  # the previous user messages joined to a follow-up's search text
RRF_K = 60
NEW_TITLE = "New chat"
TITLE_CHARS = 60
OVERRIDE_FIELDS = ("method", "k", "search", "bar", "keep")
ON_MERIT = ("search anyway", "clears its bar", "no bar")  # a store kept so is sticky


class UnknownChat(KeyError):
    """An unknown chat, message or attempt."""

    def __str__(self) -> str:  # a KeyError quotes its message
        return str(self.args[0]) if self.args else ""


def error_text(e: Exception) -> str:
    return no_ai(e.role) if isinstance(e, NoConnection) else str(e)


def whole(value: object, low: int, high: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def merge(lists: dict[str, list[Hit]], size: int) -> list[tuple[str, Hit, int, float]]:
    """The stores' lists fused by RRF and cut to `size`, as (store, hit, rank in its store's
    list, fused score). Passages are keyed by (store, id): a copied store keeps the same ids.
    Ties keep the stores' order, so equal ranks interleave."""
    scores: dict[tuple[str, str], float] = {}
    first: dict[tuple[str, str], tuple[Hit, int]] = {}
    for store, hits in lists.items():
        for rank, hit in enumerate(hits, start=1):
            key = (store, hit.id)
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)
            first.setdefault(key, (hit, rank))
    order = {key: i for i, key in enumerate(first)}
    fused = sorted(first, key=lambda key: (-scores[key], order[key]))[:size]
    return [(key[0], first[key][0], first[key][1], scores[key]) for key in fused]


def evidence_of(merged: list[tuple[str, Hit, int, float]]) -> list[dict]:
    """The merged passages in merged order, each with `n`, its number in the prompt, which
    chat_prompt gives in edge order: the number the answer cites."""
    numbers = {i: n for n, i in enumerate(edge_order(list(range(len(merged)))), start=1)}
    return [
        {
            "n": numbers[i],
            "id": hit.id,
            "text": hit.text[:CHAT_PASSAGE_CHARS],
            "source": hit.meta.get("source"),
            "part": hit.meta.get("part"),
            "parts": hit.meta.get("parts"),
            "store": store,
            "rank": rank,
            "score": score,
            "channels": list(hit.channels),
        }
        for i, (store, hit, rank, score) in enumerate(merged)
    ]


class Chats:
    def __init__(self, conn: Connection, retriever: Retriever, router: Router, ai: AI):
        self.conn = conn
        self.retriever = retriever
        self.router = router
        self.ai = ai
        # chat id -> store -> {method?, k?, search?, bar?, keep?}; None means the store's own.
        self.overrides: dict[str, dict[str, dict]] = {}
        self._lock = threading.Lock()

    # -- chats -------------------------------------------------------------------

    def create(self, stores: list[str], title: str | None = None) -> dict:
        stores = list(dict.fromkeys(stores))
        if not stores:
            raise ValueError("a chat needs at least one store")
        for store in stores:
            self.retriever.spec(store)
        settings = {"evidence_size": EVIDENCE_SIZE, "rewrite": False, "routing": "all"}
        return self._chat(db.create_chat(self.conn, title or NEW_TITLE, stores, settings))

    def get(self, chat_id: str) -> dict:
        chat = self._chat(chat_id)
        attempts: dict[int, list[dict]] = {}
        for a in db.chat_attempts(self.conn, chat_id):
            attempts.setdefault(a["n"], []).append(a)
        messages = [
            {"n": m["n"], "content": m["content"], "attempts": attempts.get(m["n"], [])}
            for m in db.chat_messages(self.conn, chat_id)
        ]
        deleted = [s for s in chat["stores"] if db.get_collection(self.conn, s) is None]
        return {**chat, "deleted_stores": deleted, "messages": messages}

    def update(
        self,
        chat_id: str,
        *,
        title: str | None = None,
        stores: list[str] | None = None,
        evidence_size: int | None = None,
        rewrite: bool | None = None,
    ) -> dict:
        """Only stores being added must exist, so a list naming a deleted store still saves."""
        chat = self._chat(chat_id)
        if title is not None and not title.strip():
            raise ValueError("a chat needs a title")
        if stores is not None:
            stores = list(dict.fromkeys(stores))
            if not stores:
                raise ValueError("a chat needs at least one store")
            for store in stores:
                if store not in chat["stores"]:
                    self.retriever.spec(store)
        settings = dict(chat["settings"])
        if evidence_size is not None:
            if not whole(evidence_size, 1, MAX_EVIDENCE):
                raise ValueError(f"evidence size must be a whole number from 1 to {MAX_EVIDENCE}")
            settings["evidence_size"] = evidence_size
        if rewrite is not None:
            if not isinstance(rewrite, bool):
                raise ValueError("rewrite must be true or false")
            settings["rewrite"] = rewrite
        db.update_chat(self.conn, chat_id, title=title, stores=stores, settings=settings)
        if stores is not None:
            with self._lock:
                edits = self.overrides.get(chat_id, {})
                for store in [s for s in edits if s not in stores]:
                    del edits[store]
        return self._chat(chat_id)

    def delete(self, chat_id: str) -> None:
        self._chat(chat_id)
        db.delete_chat(self.conn, chat_id)
        with self._lock:
            self.overrides.pop(chat_id, None)

    # -- messages and attempts -----------------------------------------------------

    def search_text(self, chat_id: str, n: int, content: str) -> tuple[str, float, str | None]:
        """Message n's search text, what its rewrite spent, and the rewrite's error. By
        default the last FOLLOW_UP_TURNS user messages before n and `content`, one per line;
        the assistant's replies are never searched. With rewrite on, a follow-up is rewritten
        by the AI as one standalone question, and a failed rewrite falls back to the join."""
        chat = self._chat(chat_id)
        earlier = [m["content"] for m in db.chat_messages(self.conn, chat_id) if m["n"] < n]
        joined = "\n".join([*earlier[-FOLLOW_UP_TURNS:], content])
        if not chat["settings"]["rewrite"] or not earlier:
            return joined, 0.0, None
        system, user = rewrite_prompt(earlier, content)
        text, error = joined, None
        with self.ai.tally() as spent:
            try:
                done = self.ai.complete(
                    user,
                    role="rewrite",
                    system=system,
                    max_tokens=REWRITE_TOKENS,
                    purpose="chat_rewrite",
                )
                text = done.text.strip() or joined
            except CHAT_ERRORS as e:
                error = error_text(e)
        return text, spent["usd"], error

    def ask(self, chat_id: str, content: str, model: str | None = None) -> dict:
        """Saves the message and its first attempt, answered. An answer model that does not
        resolve or has no price is refused first, so nothing is saved."""
        chat = self._chat(chat_id)
        if not content.strip():
            raise ValueError("a message needs some text")
        self._check_model(model)
        n = db.add_chat_message(self.conn, chat_id, content)
        try:
            attempt = self._attempt(chat, n, content, True, model)
        except BaseException:
            db.delete_chat_message(self.conn, chat_id, n)  # never a message with no attempt
            raise
        if n == 1 and chat["title"] == NEW_TITLE:
            db.update_chat(self.conn, chat_id, title=" ".join(content.split())[:TITLE_CHARS])
        return {"n": n, "content": content, "attempts": [attempt]}

    def rerun(self, chat_id: str, n: int, overrides: dict[str, dict]) -> dict:
        """Merges the overrides into the chat's (all checked first: a bad one keeps none),
        then adds an unanswered attempt to message n, searched with attempt 1's text."""
        chat = self._chat(chat_id)
        message = self._message(chat_id, n)
        with self._lock:
            current = copy.deepcopy(self.overrides.get(chat_id, {}))
        merged = {
            store: self._merged(chat, store, current.get(store, {}), override)
            for store, override in overrides.items()
        }
        with self._lock:
            self.overrides.setdefault(chat_id, {}).update(merged)
        first = db.chat_attempts(self.conn, chat_id, n)[0]
        return self._attempt(chat, n, message["content"], False, None, first["search_text"])

    def answer(self, chat_id: str, n: int, attempt: int, model: str | None = None) -> dict:
        """Answers an attempt from its evidence snapshot."""
        self._chat(chat_id)
        message = self._message(chat_id, n)
        found = self._attempt_row(chat_id, n, attempt)
        answered = self._answer(chat_id, n, message["content"], found["evidence"], model)
        answered["usd"] += found["usd"]
        db.answer_chat_attempt(self.conn, chat_id, n, attempt, answered)
        return self._attempt_row(chat_id, n, attempt)

    def set_default(self, chat_id: str, store: str) -> dict:
        """Saves the fields the chat's override for the store names (None clears the saved
        value), then drops the override. "Search anyway" is chat-only and is not saved."""
        chat = self._chat(chat_id)
        if store not in chat["stores"]:
            raise UnknownCollection(f"'{store}' is not on this chat")
        spec = self.retriever.spec(store)
        with self._lock:
            override = dict(self.overrides.get(chat_id, {}).get(store, {}))
        with self.conn.lock:
            if "search" in override:
                base = spec.search or SearchSettings()
                db.set_search(self.conn, store, search_settings(base, override["search"]))
            if "method" in override:
                db.set_chat_method(self.conn, store, override["method"])
            if "k" in override:
                db.set_chat_k(self.conn, store, override["k"])
            if "bar" in override:
                db.set_chat_bar(self.conn, store, override["bar"])
        with self._lock:
            self.overrides.get(chat_id, {}).pop(store, None)
        search = self.retriever.spec(store).search or SearchSettings()
        return {
            "search": asdict(search),
            "chat_method": db.get_chat_method(self.conn, store),
            "chat_k": db.get_chat_k(self.conn, store),
            "chat_bar": db.get_chat_bar(self.conn, store),
        }

    # -- one attempt -------------------------------------------------------------

    def _attempt(
        self,
        chat: dict,
        n: int,
        content: str,
        answer: bool,
        model: str | None,
        search_text: str | None = None,
    ) -> dict:
        """Searches every store on the chat for message n, merges the kept stores' passages,
        answers when asked, and saves the attempt. Its usd is the rewrite's, every store's
        and the answer's, tallied one after another (tallies never nest)."""
        rewrite_usd, errors = 0.0, []
        if search_text is None:
            search_text, rewrite_usd, rewrite_error = self.search_text(chat["id"], n, content)
            if rewrite_error:
                errors.append(
                    f"the rewrite failed, so the joined text was searched: {rewrite_error}"
                )
        with self._lock:
            overrides = copy.deepcopy(self.overrides.get(chat["id"], {}))
        sticky = self._sticky(chat["id"], n)
        size = chat["settings"]["evidence_size"]
        decisions, lists = [], {}
        for store in chat["stores"]:
            decision, hits = self._decide(
                store, content, search_text, overrides.get(store, {}), store in sticky, size
            )
            decisions.append(decision)
            if hits:
                lists[store] = hits
        evidence = evidence_of(merge(lists, size))
        row = {
            "search_text": search_text,
            "decisions": decisions,
            "evidence": evidence,
            "answer": None,
            "citations": [],
            "usd": rewrite_usd + sum(d["usd"] for d in decisions),
            "error": None,
            "model": None,
        }
        if answer:
            answered = self._answer(chat["id"], n, content, evidence, model)
            row.update(answer=answered["answer"], citations=answered["citations"])
            row.update(model=answered["model"], usd=row["usd"] + answered["usd"])
            if answered["error"]:
                errors.append(answered["error"])
        row["error"] = "; ".join(errors) or None
        number = db.add_chat_attempt(self.conn, chat["id"], n, row)
        return self._attempt_row(chat["id"], n, number)

    def _decide(
        self, store: str, content: str, search_text: str, override: dict, sticky: bool, size: int
    ) -> tuple[dict, list[Hit]]:
        """One store's decision and, when kept, its passages. A failure in its best match,
        bar or retrieval is the decision's error, and the other stores still run."""
        decision = {
            "store": store,
            "method": None,
            "k": None,
            "search": None,
            "best": None,
            "bar": None,
            "bar_source": None,
            "kept": False,
            "reason": "deleted",
            "usd": 0.0,
            "error": None,
        }
        spec = db.get_collection(self.conn, store)
        if spec is None:
            return decision, []
        reasons = self._reasons(spec)
        method = override.get("method") or db.get_chat_method(self.conn, store)
        method = method or ("keyword_rag" if reasons["keyword_rag"] is None else "hybrid")
        k = override.get("k") or db.get_chat_k(self.conn, store) or size
        search = search_settings(spec.search or SearchSettings(), override.get("search"))
        if search.query_tags == "oracle":  # it needs a question's gold; a chat has none
            search = replace(search, query_tags="classifier")
        decision.update(method=method, k=k, search=asdict(search))
        hits: list[Hit] = []
        with self.ai.tally() as spent:
            try:
                # best_match checks the store's model first; bar does not.
                texts = list(dict.fromkeys([content, search_text]))
                best = decision["best"] = self.router.best_match(spec, texts)
                if override.get("bar") is not None:
                    bar = Bar(override["bar"], "override", None, 0)
                else:
                    bar = self.router.bar(spec)
                decision.update(bar=bar.value, bar_source=bar.source)
                if override.get("keep"):
                    reason = "search anyway"
                elif sticky:
                    reason = "kept from the last message"
                elif self.router.kept(best, bar):
                    reason = "no bar" if bar.value is None else "clears its bar"
                else:
                    reason = "below its bar"
                decision.update(kept=reason != "below its bar", reason=reason)
                if decision["kept"]:
                    if reasons[method]:
                        raise MethodUnavailable(f"{method} {reasons[method]} on '{store}'")
                    hits = self._retrieve(spec, search_text, method, k, search)
            except CHAT_ERRORS as e:
                decision.update(kept=False, reason="error", error=error_text(e))
                hits = []
        decision["usd"] = spent["usd"]
        return decision, hits

    def _retrieve(
        self, spec: CollectionSpec, text: str, method: str, k: int, search: SearchSettings
    ) -> list[Hit]:
        tags = None
        if method == "keyword_rag":  # its query tags first, as POST /chat runs them
            skip = tuple(db.disabled_sources(self.conn, spec.name))
            tags = self.retriever.query_tags(spec, text, search, skip)
        return self.retriever.retrieve(spec.name, text, method, k, RRF_K, search, tags)

    def _answer(
        self, chat_id: str, n: int, content: str, evidence: list[dict], model: str | None
    ) -> dict:
        """The answer to message n from an evidence snapshot (kept in merged order, so the
        prompt numbers it as `n` says), with its citations, cost, error and model."""
        hits = [Hit(e["id"], e["text"], e["score"], {}, [], None, None, {}) for e in evidence]
        system, user = chat_prompt(content, self._history(chat_id, n), hits)
        out: dict = {"answer": None, "citations": [], "error": None, "model": None}
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
                out["answer"] = done.text
            except CHAT_ERRORS as e:
                out.update(error=error_text(e), model=None)
        out["usd"] = spent["usd"]
        if out["answer"] is not None:
            by_n = {e["n"]: e for e in evidence}
            out["citations"] = [
                {k: by_n[c][k] for k in ("n", "id", "text", "source", "part", "parts", "store")}
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

    def _sticky(self, chat_id: str, n: int) -> set[str]:
        """The stores message n-1's latest answered attempt kept on their own merit. One
        kept there only by stickiness is not sticky again, so the list never grows alone."""
        attempts = db.chat_attempts(self.conn, chat_id, n - 1)
        answered = [a for a in attempts if a["answer"] is not None]
        if not answered:
            return set()
        return {
            d["store"] for d in answered[-1]["decisions"] if d["kept"] and d["reason"] in ON_MERIT
        }

    # -- checks and reads --------------------------------------------------------

    def _check_model(self, model: str | None) -> None:
        """Refuses, as POST /chat does, an answer model that does not resolve (UnknownModel)
        or has no price (UnknownPrice). With no answer model at all, the attempt records
        it, as POST /chat's reply does."""
        try:
            ref = model or self.ai.model_for("answer")
        except NoConnection:
            return
        self.ai.need_rates(*self.ai.resolve(ref))

    def _merged(self, chat: dict, store: str, current: dict, override: dict) -> dict:
        """The store's override with `override` merged in, checked. None for method, k or
        bar means the store's own; a search is merged into the one already edited."""
        if store not in chat["stores"]:
            raise UnknownCollection(f"'{store}' is not on this chat")
        spec = self.retriever.spec(store)
        unknown = [f for f in override if f not in OVERRIDE_FIELDS]
        if unknown:
            raise ValueError(f"unknown override fields {unknown}; known: {list(OVERRIDE_FIELDS)}")
        out = dict(current)
        for field, value in override.items():
            if field == "search":
                if value is None:
                    out.pop("search", None)
                    continue
                if not isinstance(value, dict):
                    raise ValueError("search must be an object")
                old = out.get("search", {})
                value = {**old, **value}
                if isinstance(old.get("weights"), dict) and isinstance(value.get("weights"), dict):
                    value["weights"] = {**old["weights"], **value["weights"]}
                search_settings(spec.search or SearchSettings(), value)
            elif field == "method" and value is not None:
                if value not in METHOD_NAMES:
                    raise ValueError(f"unknown method {value!r}; known: {', '.join(METHOD_NAMES)}")
                why = self._reasons(spec)[value]
                if why:
                    raise MethodUnavailable(f"{value} {why} on '{store}'")
            elif field == "k" and value is not None and not whole(value, 1, MAX_EVIDENCE):
                raise ValueError(f"k must be a whole number from 1 to {MAX_EVIDENCE}")
            elif field == "bar" and value is not None:
                number = isinstance(value, int | float) and not isinstance(value, bool)
                if not number or not math.isfinite(value):
                    raise ValueError("a bar must be a number")
            elif field == "keep" and not isinstance(value, bool):
                raise ValueError("keep must be true or false")
            out[field] = value
        return out

    def _reasons(self, spec: CollectionSpec) -> dict[str, str | None]:
        return available_methods(spec, db.get_keyword_lists(self.conn, spec.name) is not None)

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
