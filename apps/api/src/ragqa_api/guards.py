"""The checks the route modules share before they queue or spend, and the constants
more than one of them uses."""

from contextlib import contextmanager
from pathlib import Path

from anyio import from_thread
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State

from aiconnect.ai import NoConnection
from ragqa_core import db, jobs
from ragqa_core.evaluator import EvalSetMismatch
from ragqa_core.imports import FIELDS
from ragqa_core.retriever import METHOD_NAMES
from ragqa_core.types import CollectionSpec

STATIC = Path(__file__).parent / "static"
NO_KEYWORDS_MODEL = "connect a model for keywords first"
PROBLEM = {
    "failed": "failed",
    "interrupted": "was interrupted by a restart",
    "stopped_budget": "stopped at the monthly cap",
    "stopped_no_key": "stopped: no AI model is set up for this job (see the Settings page)",
    "stopped_replies": "stopped: its prompts wait for replies by hand",
}


def type_clash(eval_set: str, set_type: str, store: str, store_type: str) -> EvalSetMismatch:
    return EvalSetMismatch(f"eval set '{eval_set}' is {set_type}; store '{store}' is {store_type}")


def check_mapping(kind: str, mapping: dict | None) -> None:
    unknown = [k for k in mapping or {} if k not in FIELDS[kind]]
    if unknown:
        raise HTTPException(422, f"unknown fields {unknown}; known: {list(FIELDS[kind])}")


PRICE_CHANGED = "the price changed since you saw it: check it again"


def price_changed(estimate: dict, calls: int | None, est_usd: float | None) -> bool:
    """Whether the estimate differs from what the caller was shown: other calls, or a price
    that rose or can no longer be priced. A lower price is no reason to refuse. A caller
    that sent calls but no price saw none: a price above nothing now is a change. One that
    sent neither asked for no check."""
    if calls is not None and estimate["calls"] != calls:
        return True
    if est_usd is None:
        return calls is not None and (estimate["est_usd"] or 0) > 0
    if estimate["est_usd"] is None:
        return estimate["calls"] > 0
    return estimate["est_usd"] > est_usd + 1e-9


def check_shown_price(estimate: dict, calls: int | None, est_usd: float | None) -> None:
    if price_changed(estimate, calls, est_usd):
        raise HTTPException(409, PRICE_CHANGED)


def estimate_reply(estimate: dict | None, **extra) -> JSONResponse:
    """An unconfirmed spending request's reply: what it would spend, beside what else it
    shows. None only while it can't be priced yet: an import whose mapping is incomplete."""
    assert estimate is None or {"calls", "est_usd"} <= set(estimate), estimate
    return JSONResponse({**extra, "estimate": estimate}, 200)


def body_chunks(request: Request):
    """The request body, chunk by chunk, for a plain (threaded) route."""
    stream = request.stream().__aiter__()

    async def next_chunk() -> bytes:
        return await stream.__anext__()

    while True:
        try:
            chunk = from_thread.run(next_chunk)
        except StopAsyncIteration:
            return
        yield chunk


def check_methods(methods: list[str]) -> None:
    unknown = [m for m in methods if m not in METHOD_NAMES]
    if unknown:
        raise HTTPException(422, f"unknown methods {unknown}; known: {list(METHOD_NAMES)}")


class Guards:
    """The checks routes share before they queue or spend, read from the app's state."""

    def __init__(self, state: State):
        self.state = state

    @contextmanager
    def not_adding(self, name: str):
        """Held while a job on the collection is queued; refused while items are added."""
        with self.state.adding_lock:
            if name in self.state.adding:
                raise HTTPException(409, f"items are being added to '{name}'; try again shortly")
            yield

    def refuse_busy(self, name: str, stopped_switch_ok: bool = False) -> None:
        """409 while any job is running or queued on the collection, or while a rebuild's
        switch stopped part-way: the store is then emptied or partly refilled, and the
        resume refills it from the rebuilt copy. Only resuming that switch may pass it."""
        for status in reversed(jobs.BUSY):  # a running job is named before a queued one
            busy = db.list_jobs(self.state.conn, collection=name, status=status, limit=1)
            if busy:
                raise HTTPException(409, f"a {busy[0]['kind']} job is {status} on '{name}'; wait")
        # switching_model reads the newest rebuild: a new one would hide the stopped switch.
        switching = db.switching_model(self.state.conn, name)
        if switching and not stopped_switch_ok:
            raise HTTPException(409, f"the switch to {switching} stopped part-way; resume it first")

    def need_ai(self, role: str, model: str | None = None) -> None:
        """Refuse at once a request that would spend on a job with no usable model.
        A model the request names is checked itself, not the job's."""
        if not self.state.ai.ready(role, model):
            self.state.ai.need_price(role, model)  # a set-up model with no price: 422, names it
            raise NoConnection(role)

    def need_models(self, estimate: dict, model: str | None) -> None:
        """Every job a run spends on, with its answers on `model` when one is named. A
        named model is checked even when it is free: priced 0, it can still be off."""
        for role in estimate["roles"]:
            named = model if role == "answer" else None
            if estimate["est_usd"] != 0 or named:  # with no model each would stop at once
                self.need_ai(role, named)

    def proposes(self, spec: CollectionSpec) -> bool:
        """Whether an import or load queued now has a keywords proposal in its price, as the
        loader's estimate counts it. Recorded in the job's setup: the job proposes only then."""
        return self.state.ai.ready("keywords") and self.state.loader.makes_lists(spec)

    def store_questions(self, name: str) -> tuple[str | None, list]:
        """The store's default eval set and its questions (none, with no default)."""
        eval_set = db.default_eval_set(self.state.conn, name)
        return eval_set, db.get_questions(self.state.conn, eval_set) if eval_set else []

    def check_default_set(self, spec: CollectionSpec) -> None:
        """Refuses a store whose default eval set is of another type, before anything is
        estimated or queued. Reads only: no set is made and no pointer written."""
        name = db.default_eval_set(self.state.conn, spec.name)
        found = db.get_eval_set(self.state.conn, name) if name else None
        if name and found and found["qa_type"] != spec.qa_type:
            raise type_clash(name, found["qa_type"], spec.name, spec.qa_type)

    def leaked_ids(self, spec: CollectionSpec, eval_set: str | None) -> set[str]:
        """The eval set's questions whose text is inside the store's items."""
        return self.state.evaluator.leaked_ids(spec, eval_set) if eval_set else set()

    def known_set(self, name: str) -> dict:
        row = db.get_eval_set(self.state.conn, name)
        if row is None:
            raise HTTPException(404, f"no eval set {name!r}")
        return row

    def store_is_loading(self, store: str) -> bool:
        """Whether a load is queued or running on the store: it writes to the store's
        default eval set."""
        return any(
            job["kind"] == "load"
            for status in jobs.BUSY
            for job in db.list_jobs(self.state.conn, collection=store, status=status, limit=db.ALL)
        )

    def set_is_loading(self, name: str) -> bool:
        """Whether a load is queued or running on a store whose default
        is this set: it would write to the set while it is saved."""
        stores = [c.name for c in db.list_collections(self.state.conn)]
        return any(
            self.store_is_loading(s)
            for s in stores
            if db.default_eval_set(self.state.conn, s) == name
        )

    def not_loading(self, name: str) -> None:
        if self.set_is_loading(name):
            raise HTTPException(409, f"a load is under way on a store of '{name}'")
