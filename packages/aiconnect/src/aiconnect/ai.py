"""Stored AI connections, models, roles and a monthly spend cap, in one SQLite file.

AI(path) opens its own connection to the file: WAL, a busy timeout, and every
statement under one lock. Model refs are "<connection id>/<model id>", split on
the first slash, so ids like "meta-llama/llama-3.1-8b" survive. Environment
values win: apply_env sets them at start and they are read-only afterwards.

Every call goes through complete / complete_many: priced, cached, capped and
ledgered. The cap is the service's own guard against its own bugs. It only sees
calls made by this service; the provider's own spend limit is the real ceiling.
"""

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiconnect.presets import KINDS, PRESETS
from aiconnect.types import Completion
from aiconnect.wire import BATCH, CLIENT, SEND

KEEP: Any = object()  # update_*: leave this field as it is
DEFAULT_CAP_USD = 20.0
BATCH_THRESHOLD = 50
BATCH_POLL_SECONDS = 30
CURRENT_JOB = object()  # _record default: the calling thread's job
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")

SCHEMA = """
create table if not exists ai_connections (
  id text primary key, label text not null, kind text not null, base_url text,
  api_key text, local integer not null default 0
);
create table if not exists ai_models (
  connection_id text not null, model_id text not null, label text not null,
  price_in real, price_out real, enabled integer not null default 1,
  primary key (connection_id, model_id)
);
create table if not exists ai_roles (role text primary key, model text not null);
create table if not exists ai_settings (key text primary key, value text not null);
create table if not exists ai_cache (
  key text primary key, text text not null, input_tokens integer not null,
  output_tokens integer not null
);
create table if not exists ai_ledger (
  at text not null, model text not null, input_tokens integer not null,
  output_tokens integer not null, usd real not null, batch integer not null,
  purpose text not null, job_id text, connection_id text
);
create table if not exists ai_batches (
  id text primary key, model text not null, purpose text not null, created_at text not null,
  usd real not null default 0, job_id text, connection_id text
);
"""


class Conflict(Exception):
    pass


class NoConnection(Exception):
    def __init__(self, role: str) -> None:
        self.role = role
        super().__init__(f"no model is set for {role}")


class UnknownModel(Exception):
    pass


class UnknownPrice(Exception):
    pass


class ProviderError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


class Cancelled(Exception):
    pass


def mask(key: str | None) -> str | None:
    if not key:
        return None
    return key[:3] + "…" + key[-4:] if len(key) >= 12 else "set"


def check_price(value: float | None) -> None:
    if value is not None and value < 0:
        raise ValueError("a price cannot be negative")


def estimate_tokens(chars: int) -> int:
    # ~4 characters a token, +30% for the newer tokenizer.
    return math.ceil(chars / 4 * 1.3)


def cache_key(model_id: str, system: str, prompt: str, max_tokens: int) -> str:
    return hashlib.sha256(json.dumps([model_id, system, prompt, max_tokens]).encode()).hexdigest()


def usd_for(rates: tuple[float, float], tokens_in: int, tokens_out: int, batch: bool) -> float:
    """USD at (input, output) prices per million tokens. The Batch API is half."""
    usd = (tokens_in * rates[0] + tokens_out * rates[1]) / 1e6
    return usd / 2 if batch else usd


@contextmanager
def provider_errors(connection: dict) -> Iterator[None]:
    """A provider failure becomes a ProviderError naming the connection, minus its key."""
    try:
        yield
    except Exception as e:
        message = str(e)
        if connection["api_key"]:
            message = message.replace(connection["api_key"], mask(connection["api_key"]) or "")
        raise ProviderError(f"{connection['id']}: {message}") from None


class AI:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        os.chmod(path, 0o600)
        self._db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.sql("pragma journal_mode=wal")
        self.sql("pragma busy_timeout=5000")
        with self.lock:
            self._db.executescript(SCHEMA)
        self.env: dict[str, set[str]] = {}  # connection id -> fields set by the environment
        self.env_cap = False
        self.env_default = False
        self._clients: dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self._reserved = 0.0  # estimated spend of calls in flight, counted against the cap
        self._waiting: set[str] = set()  # batches this process is polling; spend in _reserved
        self._collecting = threading.Lock()
        self._next_check = 0.0
        self._job = threading.local()  # this thread's job_id, check, log
        self._tally = threading.local()  # this thread's running total, inside tally()

    def sql(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self._db.execute(query, params).fetchall()

    # connections

    def add_connection(
        self,
        id: str,
        kind: str,
        label: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        local: bool = False,
    ) -> None:
        if not ID.fullmatch(id):
            raise ValueError("a connection id is lowercase letters, digits and dashes")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        with self.lock:
            if self.sql("select 1 from ai_connections where id = ?", (id,)):
                raise Conflict(f"connection {id} already exists")
            self.sql(
                "insert into ai_connections values (?, ?, ?, ?, ?, ?)",
                (id, label, kind, base_url or None, api_key or None, int(local)),
            )

    def update_connection(self, id: str, *, label=KEEP, base_url=KEEP, api_key=KEEP) -> None:
        self.need_connection(id)
        for field, value in (("label", label), ("base_url", base_url), ("api_key", api_key)):
            if value is KEEP:
                continue
            if field in self.env.get(id, set()):
                raise Conflict(f"{field} is set by the environment")
            self.sql(f"update ai_connections set {field} = ? where id = ?", (value or None, id))

    def remove_connection(self, id: str) -> None:
        self.need_connection(id)
        with self.lock:
            for role, ref in self.roles().items():
                if ref and self.split(ref)[0] == id:
                    raise Conflict(f"connection {id} is used by {role}")
            self.sql("delete from ai_models where connection_id = ?", (id,))
            self.sql("delete from ai_connections where id = ?", (id,))

    def connections(self) -> list[dict]:
        return [
            {
                "id": r["id"],
                "label": r["label"],
                "kind": r["kind"],
                "base_url": r["base_url"],
                "local": bool(r["local"]),
                "key": mask(r["api_key"]),
                "from_env": sorted(self.env.get(r["id"], set())),
            }
            for r in self.sql("select * from ai_connections order by rowid")
        ]

    def need_connection(self, id: str) -> sqlite3.Row:
        rows = self.sql("select * from ai_connections where id = ?", (id,))
        if not rows:
            raise ValueError(f"no connection {id}")
        return rows[0]

    # models

    def add_model(
        self,
        connection_id: str,
        model_id: str,
        *,
        label: str | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
    ) -> None:
        self.need_connection(connection_id)
        check_price(price_in)
        check_price(price_out)
        if not model_id:
            raise ValueError("a model needs an id")
        with self.lock:
            if self.sql(
                "select 1 from ai_models where connection_id = ? and model_id = ?",
                (connection_id, model_id),
            ):
                raise Conflict(f"model {connection_id}/{model_id} already exists")
            self.sql(
                "insert into ai_models values (?, ?, ?, ?, ?, 1)",
                (connection_id, model_id, label or model_id, price_in, price_out),
            )

    def update_model(
        self, connection_id, model_id, *, label=KEEP, price_in=KEEP, price_out=KEEP, enabled=KEEP
    ) -> None:
        if not self.sql(
            "select 1 from ai_models where connection_id = ? and model_id = ?",
            (connection_id, model_id),
        ):
            raise UnknownModel(f"no model {connection_id}/{model_id}")
        for value in (price_in, price_out):
            if value is not KEEP:
                check_price(value)
        for field, value in (
            ("label", label),
            ("price_in", price_in),
            ("price_out", price_out),
            ("enabled", enabled if enabled is KEEP else int(enabled)),
        ):
            if value is not KEEP:
                self.sql(
                    f"update ai_models set {field} = ? where connection_id = ? and model_id = ?",
                    (value, connection_id, model_id),
                )

    def models(self) -> list[dict]:
        return [
            {
                "ref": f"{r['connection_id']}/{r['model_id']}",
                "connection_id": r["connection_id"],
                "model_id": r["model_id"],
                "label": r["label"],
                "price_in": r["price_in"],
                "price_out": r["price_out"],
                "enabled": bool(r["enabled"]),
            }
            for r in self.sql("select * from ai_models order by rowid")
        ]

    def split(self, ref: str) -> tuple[str, str]:
        connection_id, _, model_id = ref.partition("/")
        return connection_id, model_id

    def resolve(self, model: str) -> tuple[str, str]:
        """(connection id, model id) for a ref, or for a bare id that one enabled model has."""
        connection_id, model_id = self.split(model)
        if self.sql(
            "select 1 from ai_models where connection_id = ? and model_id = ?",
            (connection_id, model_id),
        ):
            return connection_id, model_id
        rows = self.sql(
            "select connection_id from ai_models where model_id = ? and enabled", (model,)
        )
        if len(rows) == 1:
            return rows[0]["connection_id"], model
        if rows:
            refs = ", ".join(f"{r['connection_id']}/{model}" for r in rows)
            raise UnknownModel(f"{model} is on several connections; use one of {refs}")
        raise UnknownModel(f"no model {model}")

    # roles

    def set_role(self, role: str, model: str | None) -> None:
        if not role:
            raise ValueError("a role needs a name")
        if role == "default" and self.env_default:
            raise Conflict("the default model is set by the environment")
        if model is None:
            self.sql("delete from ai_roles where role = ?", (role,))
            return
        connection_id, model_id = self.resolve(model)
        self.sql(
            "insert or replace into ai_roles values (?, ?)",
            (role, f"{connection_id}/{model_id}"),
        )

    def role_model(self, role: str) -> str | None:
        rows = self.sql(
            "select model from ai_roles where role in (?, 'default') "
            "order by role = 'default' limit 1",
            (role,),
        )
        return rows[0]["model"] if rows else None

    def roles(self) -> dict[str, str | None]:
        return {r["role"]: r["model"] for r in self.sql("select * from ai_roles order by rowid")}

    # budget

    @property
    def cap_usd(self) -> float:
        rows = self.sql("select value from ai_settings where key = 'cap_usd'")
        return float(rows[0]["value"]) if rows else DEFAULT_CAP_USD

    def set_cap(self, usd: float) -> None:
        if self.env_cap:
            raise Conflict("the spend cap is set by the environment")
        self.store_cap(usd)

    def store_cap(self, usd: float) -> None:
        if usd < 0:
            raise ValueError("the cap cannot be negative")
        self.sql("insert or replace into ai_settings values ('cap_usd', ?)", (str(float(usd)),))

    # environment

    def apply_env(self, *, api_key: str | None, model: str | None, cap_usd: float | None) -> None:
        """Apply the environment's values; they win over stored ones and become read-only."""
        preset = PRESETS["anthropic"]
        if api_key:
            if not self.sql("select 1 from ai_connections where id = 'anthropic'"):
                self.add_connection("anthropic", preset["kind"], preset["label"])
            self.sql("update ai_connections set api_key = ? where id = 'anthropic'", (api_key,))
            self.env["anthropic"] = {"api_key"}
            for model_id, (label, price_in, price_out) in preset["models"].items():
                if not self.sql(
                    "select 1 from ai_models where connection_id = 'anthropic' and model_id = ?",
                    (model_id,),
                ):
                    self.add_model(
                        "anthropic", model_id, label=label, price_in=price_in, price_out=price_out
                    )
        chosen = model or (preset["default_model"] if api_key else None)
        if chosen and (model or not self.role_model("default")):
            if api_key and not self.sql(
                "select 1 from ai_models where connection_id = 'anthropic' and model_id = ?",
                (chosen,),
            ):
                self.add_model("anthropic", chosen)
            self.sql(
                "insert or replace into ai_roles values ('default', ?)", (f"anthropic/{chosen}",)
            )
            self.env_default = bool(model)
        if cap_usd is not None:
            self.store_cap(cap_usd)
            self.env_cap = True

    # calls

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

    def model_for(self, role: str) -> str:
        ref = self.role_model(role)
        if not ref:
            raise NoConnection(role)
        return ref

    def ready(self, role: str = "default") -> bool:
        try:
            connection_id, model_id = self.resolve(self.model_for(role))
        except (NoConnection, UnknownModel):
            return False
        return (
            self.usable(connection_id, model_id) and self.rates(connection_id, model_id) is not None
        )

    def usable(self, connection_id: str, model_id: str) -> bool:
        """The model is enabled and its connection can be called: it has a key, or needs none."""
        rows = self.sql(
            "select enabled from ai_models where connection_id = ? and model_id = ?",
            (connection_id, model_id),
        )
        connection = self.need_connection(connection_id)
        return bool(rows and rows[0]["enabled"]) and (
            bool(connection["api_key"]) or connection["kind"] == "openai"
        )

    def rates(self, connection_id: str, model_id: str) -> tuple[float, float] | None:
        row = self.sql(
            "select price_in, price_out from ai_models where connection_id = ? and model_id = ?",
            (connection_id, model_id),
        )[0]
        if row["price_in"] is None or row["price_out"] is None:
            return None
        return row["price_in"], row["price_out"]

    def need_rates(self, connection_id: str, model_id: str) -> tuple[float, float]:
        rates = self.rates(connection_id, model_id)
        if rates is None:
            raise UnknownPrice(f"{connection_id}/{model_id} has no price; enter one first")
        return rates

    def complete(
        self,
        prompt: str,
        *,
        role: str = "default",
        model: str | None = None,
        system: str = "",
        max_tokens: int = 1024,
        purpose: str,
        cache: bool = True,
    ) -> Completion:
        return self.complete_many(
            [prompt],
            role=role,
            model=model,
            system=system,
            max_tokens=max_tokens,
            purpose=purpose,
            cache=cache,
        )[0]

    def complete_many(
        self,
        prompts: list[str],
        *,
        role: str = "default",
        model: str | None = None,
        system: str = "",
        max_tokens: int = 1024,
        purpose: str,
        cache: bool = True,
    ) -> list[Completion]:
        connection_id, model_id = self.resolve(model or self.model_for(role))
        keys = [cache_key(model_id, system, p, max_tokens) for p in prompts]
        results: dict[str, Completion] = {}
        if cache:
            self.load_cached(keys, results)
        misses = {k: p for k, p in zip(keys, prompts, strict=True) if k not in results}
        if not misses:
            return [results[k] for k in keys]
        if not self.usable(connection_id, model_id):
            raise NoConnection(role)
        rates = self.need_rates(connection_id, model_id)
        connection = dict(self.need_connection(connection_id))
        if cache:
            # A batch submitted before a crash may already hold these answers.
            self.collect_pending()
            self.load_cached(list(misses), results)
            misses = {k: p for k, p in misses.items() if k not in results}
        if misses:
            batch = len(misses) > BATCH_THRESHOLD and connection["kind"] in BATCH
            estimate = sum(
                usd_for(
                    rates, estimate_tokens(len(system) + len(p)), math.ceil(max_tokens * 1.3), batch
                )
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
                        self._run_batch(
                            connection, model_id, system, max_tokens, purpose, misses, estimate
                        )
                    )
                else:
                    # Recorded one by one, so a failure part-way keeps what was paid for.
                    for key, p in misses.items():
                        text, tokens_in, tokens_out = self._send(
                            connection, model_id, system, p, max_tokens
                        )
                        results[key] = self._record(
                            key,
                            connection_id,
                            model_id,
                            text,
                            tokens_in,
                            tokens_out,
                            False,
                            purpose,
                        )
            finally:
                with self._lock:
                    self._reserved -= estimate
                total = getattr(self._tally, "total", None)
                if total is not None:  # what was paid for, even when a later send failed
                    total["usd"] += sum(results[k].usd for k in misses if k in results)
        return [results[k] for k in keys]

    def load_cached(self, keys: list[str], results: dict[str, Completion]) -> None:
        for key in dict.fromkeys(keys):
            rows = self.sql("select * from ai_cache where key = ?", (key,))
            if rows:
                row = rows[0]
                results[key] = Completion(
                    row["text"], 0.0, True, row["input_tokens"], row["output_tokens"]
                )

    def _record(
        self,
        key: str,
        connection_id: str,
        model_id: str,
        text: str,
        tokens_in: int,
        tokens_out: int,
        batch: bool,
        purpose: str,
        job_id: str | None | object = CURRENT_JOB,
    ) -> Completion:
        if job_id is CURRENT_JOB:
            job_id = getattr(self._job, "job_id", None)
        usd = usd_for(self.need_rates(connection_id, model_id), tokens_in, tokens_out, batch)
        self.sql(
            "insert or replace into ai_cache values (?, ?, ?, ?)",
            (key, text, tokens_in, tokens_out),
        )
        self.sql(
            "insert into ai_ledger"
            " (at, model, input_tokens, output_tokens, usd, batch, purpose, job_id, connection_id)"
            " values (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                model_id,
                tokens_in,
                tokens_out,
                usd,
                int(batch),
                purpose,
                job_id,
                connection_id,
            ),
        )
        return Completion(text, usd, False, tokens_in, tokens_out)

    def _run_batch(
        self,
        connection: dict,
        model_id: str,
        system: str,
        max_tokens: int,
        purpose: str,
        misses: dict[str, str],
        usd: float,
    ) -> dict[str, Completion]:
        requests = [
            {
                "custom_id": key,
                "model": model_id,
                "system": system,
                "prompt": p,
                "max_tokens": max_tokens,
            }
            for key, p in misses.items()
        ]
        batch_id = self._submit_batch(connection, requests)
        with self._lock:
            self._waiting.add(batch_id)
        # Recorded before waiting: a batch is billed once submitted, so a
        # restart must be able to find it, count it and collect what it paid for.
        self.sql(
            "insert into ai_batches (id, model, purpose, created_at, usd, job_id, connection_id)"
            " values (?, ?, ?, ?, ?, ?, ?)",
            (
                batch_id,
                model_id,
                purpose,
                datetime.now(UTC).isoformat(),
                usd,
                getattr(self._job, "job_id", None),
                connection["id"],
            ),
        )
        log = getattr(self._job, "log", None)
        if log:
            log(f"AI batch {batch_id} submitted, {len(misses)} requests")
        try:
            while not self._batch_done(connection, batch_id):
                self._wait(connection, batch_id)
            return self._collect(
                batch_id, connection["id"], model_id, purpose, expected=len(misses)
            )
        finally:
            with self._lock:
                self._waiting.discard(batch_id)

    def _wait(self, connection: dict, batch_id: str) -> None:
        """Sleep one poll interval in 1-second steps, letting the job's check() stop it."""
        check = getattr(self._job, "check", None)
        for _ in range(BATCH_POLL_SECONDS):
            time.sleep(1)
            if check:
                try:
                    check()
                except Cancelled:
                    try:
                        self._cancel_batch(connection, batch_id)
                    except Exception as e:
                        log = getattr(self._job, "log", None)
                        if log:
                            log(
                                f"could not cancel AI batch {batch_id}: {e!r};"
                                " it will be collected later"
                            )
                    raise

    def _collect(
        self,
        batch_id: str,
        connection_id: str,
        model_id: str,
        purpose: str,
        expected: int | None = None,
    ) -> dict[str, Completion]:
        rows = self.sql("select job_id from ai_batches where id = ?", (batch_id,))
        job_id = rows[0]["job_id"] if rows else None
        out: dict[str, Completion] = {}
        failed = 0
        results = self._batch_results(self._connection(connection_id), batch_id)
        for key, result in results.items():
            if result is None:
                failed += 1
                continue
            out[key] = self._record(
                key, connection_id, model_id, *result, True, purpose, job_id=job_id
            )
        self.sql("delete from ai_batches where id = ?", (batch_id,))
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
            for row in self.pending_batches():
                if row["id"] in self._waiting:
                    continue
                try:
                    # Rows from before connections have no connection id.
                    connection_id = row["connection_id"] or self.resolve(row["model"])[0]
                    if self._batch_done(self._connection(connection_id), row["id"]):
                        self._collect(row["id"], connection_id, row["model"], row["purpose"])
                except Exception:
                    # Failed entries are simply not cached and will be re-sent; a
                    # batch that cannot be read is tried again next interval, and
                    # its estimate keeps counting against the cap until then.
                    continue
        finally:
            self._collecting.release()

    def pending_batches(self) -> list[dict]:
        return [dict(r) for r in self.sql("select * from ai_batches order by rowid")]

    def _pending_usd(self) -> float:
        """Estimated spend of submitted batches not yet in the ledger, except the
        ones this process is polling: their spend is already in _reserved."""
        return sum(r["usd"] for r in self.pending_batches() if r["id"] not in self._waiting)

    def estimate_usd(
        self,
        prompt_chars: int,
        output_tokens: int,
        *,
        role: str = "default",
        model: str | None = None,
        batch: bool = False,
    ) -> float:
        connection_id, model_id = self.resolve(model or self.model_for(role))
        rates = self.need_rates(connection_id, model_id)
        return usd_for(rates, estimate_tokens(prompt_chars), math.ceil(output_tokens * 1.3), batch)

    def month_spend(self) -> float:
        month = datetime.now(UTC).strftime("%Y-%m")
        rows = self.sql(
            "select coalesce(sum(usd), 0) from ai_ledger where substr(at, 1, 7) = ?", (month,)
        )
        return float(rows[0][0])  # an aggregate always has one row

    def job_spend(self, job_id: str) -> float:
        rows = self.sql("select coalesce(sum(usd), 0) from ai_ledger where job_id = ?", (job_id,))
        return float(rows[0][0])

    # provider seams: tests override these on the instance

    def _connection(self, connection_id: str) -> dict:
        return dict(self.need_connection(connection_id))

    def _client(self, connection: dict) -> Any:
        key = (connection["kind"], connection["base_url"], connection["api_key"])
        if key not in self._clients:
            self._clients[key] = CLIENT[connection["kind"]](
                connection["base_url"], connection["api_key"]
            )
        return self._clients[key]

    def _send(
        self, connection: dict, model_id: str, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        with provider_errors(connection):
            send = SEND[connection["kind"]]
            return send(self._client(connection), model_id, system, prompt, max_tokens)

    def _submit_batch(self, connection: dict, requests: list[dict]) -> str:
        with provider_errors(connection):
            return BATCH[connection["kind"]]["submit"](self._client(connection), requests)

    def _batch_done(self, connection: dict, batch_id: str) -> bool:
        with provider_errors(connection):
            return BATCH[connection["kind"]]["done"](self._client(connection), batch_id)

    def _batch_results(
        self, connection: dict, batch_id: str
    ) -> dict[str, tuple[str, int, int] | None]:
        """custom_id -> (text, input tokens, output tokens), or None if not succeeded."""
        with provider_errors(connection):
            return BATCH[connection["kind"]]["results"](self._client(connection), batch_id)

    def _cancel_batch(self, connection: dict, batch_id: str) -> None:
        with provider_errors(connection):
            BATCH[connection["kind"]]["cancel"](self._client(connection), batch_id)
