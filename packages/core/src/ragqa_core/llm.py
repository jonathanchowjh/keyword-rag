"""Every Claude call goes through here: priced, cached, capped and ledgered.

The cap is the service's own guard against its own bugs. It only sees calls
made by this service; the Console workspace spend limit is the real ceiling.
"""

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import anthropic

from ragqa_core.connection import Connection
from ragqa_core.db import first, now
from ragqa_core.types import Completion

# USD per million tokens, (input, output), real-time. The Batch API is half.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
BATCH_THRESHOLD = 50
BATCH_POLL_SECONDS = 30
CURRENT_JOB = object()  # _record default: the calling thread's job


class BudgetExceeded(Exception):
    pass


class Cancelled(Exception):
    pass


class NoApiKey(Exception):
    def __init__(self) -> None:
        super().__init__("needs ANTHROPIC_API_KEY")


class UnknownModel(Exception):
    pass


def estimate_tokens(chars: int) -> int:
    # ~4 characters a token, +30% for the newer tokenizer.
    return math.ceil(chars / 4 * 1.3)


class Claude:
    def __init__(
        self, conn: Connection, api_key: str | None, cap_usd: float, default_model: str
    ) -> None:
        self.conn = conn
        self.api_key = api_key
        self.cap_usd = cap_usd
        self.default_model = default_model
        self._client: anthropic.Anthropic | None = None
        self._lock = threading.Lock()
        self._reserved = 0.0  # estimated spend of calls in flight, counted against the cap
        self._waiting: set[str] = set()  # batches this process is polling; spend in _reserved
        self._collecting = threading.Lock()
        self._next_check = 0.0
        self._job = threading.local()  # this thread's job_id, check, log
        self._tally = threading.local()  # this thread's running total, inside tally()

    @contextmanager
    def for_job(
        self, job_id: str, check: Callable[[], None], log: Callable[[str], None]
    ) -> Iterator[None]:
        """Calls on this thread belong to job_id; check() raises Cancelled to stop a wait."""
        self._job.job_id, self._job.check, self._job.log = job_id, check, log
        try:
            yield
        finally:
            self._job.job_id = self._job.check = self._job.log = None

    @contextmanager
    def tally(self) -> Iterator[dict[str, float]]:
        """Adds up what this thread's calls spend inside the block, as {"usd": total}.
        Per thread: the worker spending at the same time is not counted."""
        total = {"usd": 0.0}
        self._tally.total = total
        try:
            yield total
        finally:
            self._tally.total = None

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        purpose: str,
    ) -> Completion:
        return self.complete_many(
            [prompt], system=system, model=model, max_tokens=max_tokens, purpose=purpose
        )[0]

    def complete_many(
        self,
        prompts: list[str],
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        purpose: str,
    ) -> list[Completion]:
        model = self._model(model)
        keys = [self._key(model, system, p, max_tokens) for p in prompts]
        results: dict[str, Completion] = {}
        for key in dict.fromkeys(keys):
            row = first(self.conn.execute("select * from cache where key = ?", (key,)))
            if row:
                results[key] = Completion(
                    row["text"], 0.0, True, row["input_tokens"], row["output_tokens"]
                )
        misses = {k: p for k, p in zip(keys, prompts, strict=True) if k not in results}
        if misses and self.api_key:
            # A batch submitted before a crash may already hold these answers.
            self.collect_pending()
            for key in list(misses):
                row = first(self.conn.execute("select * from cache where key = ?", (key,)))
                if row:
                    results[key] = Completion(
                        row["text"], 0.0, True, row["input_tokens"], row["output_tokens"]
                    )
                    del misses[key]
        if misses:
            if not self.api_key:
                raise NoApiKey()
            batch = len(misses) > BATCH_THRESHOLD
            estimate = sum(
                self.estimate_usd(len(system) + len(p), max_tokens, model=model, batch=batch)
                for p in misses.values()
            )
            with self._lock:
                committed = self.month_spend() + self._reserved + self._pending_usd()
                if committed + estimate > self.cap_usd:
                    raise BudgetExceeded(
                        f"monthly cap reached: ${committed:.4f} spent or in flight"
                        f" + ${estimate:.4f} needed > ${self.cap_usd:.2f} cap"
                    )
                self._reserved += estimate
            try:
                if batch:
                    results.update(
                        self._run_batch(model, system, max_tokens, purpose, misses, estimate)
                    )
                else:
                    # Recorded one by one, so a failure part-way keeps what was paid for.
                    for key, p in misses.items():
                        text, tokens_in, tokens_out = self._send(model, system, p, max_tokens)
                        results[key] = self._record(
                            key, model, text, tokens_in, tokens_out, False, purpose
                        )
            finally:
                with self._lock:
                    self._reserved -= estimate
                total = getattr(self._tally, "total", None)
                if total is not None:  # what was paid for, even when a later send failed
                    total["usd"] += sum(results[k].usd for k in misses if k in results)
        return [results[k] for k in keys]

    def _record(
        self,
        key: str,
        model: str,
        text: str,
        tokens_in: int,
        tokens_out: int,
        batch: bool,
        purpose: str,
        job_id: str | None | object = CURRENT_JOB,
    ) -> Completion:
        if job_id is CURRENT_JOB:
            job_id = getattr(self._job, "job_id", None)
        usd = self._price(model, tokens_in, tokens_out, batch)
        self.conn.execute(
            "insert or replace into cache values (?, ?, ?, ?)", (key, text, tokens_in, tokens_out)
        )
        self.conn.execute(
            "insert into ledger"
            " (at, model, input_tokens, output_tokens, usd, batch, purpose, job_id)"
            " values (?, ?, ?, ?, ?, ?, ?, ?)",
            (now(), model, tokens_in, tokens_out, usd, int(batch), purpose, job_id),
        )
        return Completion(text, usd, False, tokens_in, tokens_out)

    def _run_batch(
        self,
        model: str,
        system: str,
        max_tokens: int,
        purpose: str,
        misses: dict[str, str],
        usd: float,
    ) -> dict[str, Completion]:
        requests = [
            {
                "custom_id": key,
                "model": model,
                "system": system,
                "prompt": p,
                "max_tokens": max_tokens,
            }
            for key, p in misses.items()
        ]
        batch_id = self._submit_batch(requests)
        with self._lock:
            self._waiting.add(batch_id)
        # Recorded before waiting: a batch is billed once submitted, so a
        # restart must be able to find it, count it and collect what it paid for.
        self.conn.execute(
            "insert into pending_batches (id, model, purpose, created_at, usd, job_id)"
            " values (?, ?, ?, ?, ?, ?)",
            (batch_id, model, purpose, now(), usd, getattr(self._job, "job_id", None)),
        )
        log = getattr(self._job, "log", None)
        if log:
            log(f"Claude batch {batch_id} submitted, {len(misses)} requests")
        try:
            while not self._batch_done(batch_id):
                self._wait(batch_id)
            return self._collect(batch_id, model, purpose, expected=len(misses))
        finally:
            with self._lock:
                self._waiting.discard(batch_id)

    def _wait(self, batch_id: str) -> None:
        """Sleep one poll interval in 1-second steps, letting the job's check() stop it."""
        check = getattr(self._job, "check", None)
        for _ in range(BATCH_POLL_SECONDS):
            time.sleep(1)
            if check:
                try:
                    check()
                except Cancelled:
                    try:
                        self._cancel_batch(batch_id)
                    except Exception as e:
                        log = getattr(self._job, "log", None)
                        if log:
                            log(
                                f"could not cancel Claude batch {batch_id}: {e!r};"
                                " it will be collected later"
                            )
                    raise

    def _collect(
        self, batch_id: str, model: str, purpose: str, expected: int | None = None
    ) -> dict[str, Completion]:
        row = first(
            self.conn.execute("select job_id from pending_batches where id = ?", (batch_id,))
        )
        job_id = row["job_id"] if row else None
        out: dict[str, Completion] = {}
        failed = 0
        for key, result in self._batch_results(batch_id).items():
            if result is None:
                failed += 1
                continue
            out[key] = self._record(key, model, *result, True, purpose, job_id=job_id)
        self.conn.execute("delete from pending_batches where id = ?", (batch_id,))
        if failed:
            total = expected if expected is not None else failed + len(out)
            raise RuntimeError(f"{failed} of {total} batch requests failed in {batch_id}")
        return out

    def collect_pending(self) -> None:
        """Collect batches submitted by an earlier process that stopped waiting.

        One collector at a time, at most once per poll interval: a second one
        would ledger the same batch twice, and every check is an API request.
        Batches this process is polling are left to the call that is waiting.
        """
        if not self._collecting.acquire(blocking=False):
            return
        try:
            if time.monotonic() < self._next_check:
                return
            self._next_check = time.monotonic() + BATCH_POLL_SECONDS
            for row in self.conn.execute("select * from pending_batches"):
                if row["id"] in self._waiting:
                    continue
                try:
                    if self._batch_done(row["id"]):
                        self._collect(row["id"], row["model"], row["purpose"])
                except Exception:
                    # Failed entries are simply not cached and will be re-sent; a
                    # batch that cannot be read is tried again next interval, and
                    # its estimate keeps counting against the cap until then.
                    continue
        finally:
            self._collecting.release()

    def _pending_usd(self) -> float:
        """Estimated spend of submitted batches not yet in the ledger, except the
        ones this process is polling: their spend is already in _reserved."""
        rows = self.conn.execute("select id, usd from pending_batches")
        return sum(row["usd"] for row in rows if row["id"] not in self._waiting)

    def estimate_usd(
        self,
        prompt_chars: int,
        output_tokens: int,
        *,
        model: str | None = None,
        batch: bool = False,
    ) -> float:
        model = self._model(model)
        return self._price(
            model, estimate_tokens(prompt_chars), math.ceil(output_tokens * 1.3), batch
        )

    def month_spend(self) -> float:
        month = datetime.now(UTC).strftime("%Y-%m")
        rows = self.conn.execute(
            "select coalesce(sum(usd), 0) from ledger where substr(at, 1, 7) = ?", (month,)
        )
        return float(rows[0][0])  # an aggregate always has one row

    def _model(self, model: str | None) -> str:
        model = model or self.default_model
        if model not in PRICES:
            raise UnknownModel(f"unknown model {model!r}; known: {', '.join(PRICES)}")
        return model

    @staticmethod
    def _key(model: str, system: str, prompt: str, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([model, system, prompt, max_tokens]).encode()).hexdigest()

    @staticmethod
    def _price(model: str, tokens_in: int, tokens_out: int, batch: bool) -> float:
        price_in, price_out = PRICES[model]
        usd = (tokens_in * price_in + tokens_out * price_out) / 1e6
        return usd / 2 if batch else usd

    def _client_or_create(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _send(self, model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        # No sampling parameters: newer models restrict them, and the cache is
        # what makes re-runs identical.
        message = self._client_or_create().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, message.usage.input_tokens, message.usage.output_tokens

    def _submit_batch(self, requests: list[dict]) -> str:
        batch = self._client_or_create().messages.batches.create(
            requests=[
                {
                    "custom_id": r["custom_id"],
                    "params": {
                        "model": r["model"],
                        "max_tokens": r["max_tokens"],
                        "system": r["system"],
                        "messages": [{"role": "user", "content": r["prompt"]}],
                    },
                }
                for r in requests
            ]
        )
        return batch.id

    def _cancel_batch(self, batch_id: str) -> None:
        self._client_or_create().messages.batches.cancel(batch_id)

    def _batch_done(self, batch_id: str) -> bool:
        status = self._client_or_create().messages.batches.retrieve(batch_id).processing_status
        return status == "ended"

    def _batch_results(self, batch_id: str) -> dict[str, tuple[str, int, int] | None]:
        """custom_id -> (text, input tokens, output tokens), or None if not succeeded."""
        out: dict[str, tuple[str, int, int] | None] = {}
        for entry in self._client_or_create().messages.batches.results(batch_id):
            if entry.result.type != "succeeded":
                out[entry.custom_id] = None
                continue
            message = entry.result.message
            text = "".join(block.text for block in message.content if block.type == "text")
            out[entry.custom_id] = (text, message.usage.input_tokens, message.usage.output_tokens)
        return out
