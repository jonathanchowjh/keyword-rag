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
from aiconnect.wire import BATCH, CLIENT, LIST_MODELS, SEND, estimate_tokens

KEEP: Any = object()  # update_*: leave this field as it is
DEFAULT_CAP_USD = 20.0
BATCH_THRESHOLD = 50
BATCH_POLL_SECONDS = 30
CURRENT_JOB = object()  # _record default: the calling thread's job
ID = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")

SCHEMA = """
create table if not exists ai_connections (
  id text primary key, label text not null, kind text not null, base_url text,
  api_key text
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
create table if not exists ai_waiting (
  key text not null, model text not null, system text not null, prompt text not null,
  max_tokens integer not null, purpose text not null, job_id text not null default '',
  at text not null, primary key (key, job_id)
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


class AwaitingReplies(Exception):
    """A connection answered by hand was asked what its cache does not hold. The prompts
    wait (AI.waiting) until their replies are imported (AI.answer); then the call is cached."""

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(f"{count} prompt{'' if count == 1 else 's'} wait for replies by hand")


def mask(key: str | None) -> str | None:
    if not key:
        return None
    return key[:3] + "…" + key[-4:] if len(key) >= 12 else "set"


def check_price(value: float | None) -> None:
    if value is not None and not (math.isfinite(value) and value >= 0):
        raise ValueError("a price must be a number that is not negative")


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


def split_ref(ref: str) -> tuple[str, str]:
    connection_id, _, model_id = ref.partition("/")
    return connection_id, model_id


class AI:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.sql("pragma journal_mode=wal")
        self.sql("pragma busy_timeout=5000")
        with self.lock:
            self._db.executescript(SCHEMA)
        # Keys live in the file and, until a checkpoint, in its WAL. A connection opened
        # before this one made the WAL files with its own mode; later ones copy the file's.
        for file in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            if file.exists():
                os.chmod(file, 0o600)
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
    ) -> None:
        if not ID.fullmatch(id):
            raise ValueError("a connection id is lowercase letters, digits and dashes")
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        with self.lock:
            if self.sql("select 1 from ai_connections where id = ?", (id,)):
                raise Conflict(f"connection {id} already exists")
            self.sql(
                "insert into ai_connections (id, label, kind, base_url, api_key)"
                " values (?, ?, ?, ?, ?)",
                (id, label, kind, base_url or None, api_key or None),
            )

    def update_connection(self, id: str, *, label=KEEP, base_url=KEEP, api_key=KEEP) -> None:
        """A new base_url without a new api_key drops the stored key, given for the old one."""
        row = self.need_connection(id)
        if label is not KEEP and not label:
            raise ValueError("a connection needs a label")
        changes = {
            field: value or None
            for field, value in (("label", label), ("base_url", base_url), ("api_key", api_key))
            if value is not KEEP
        }
        for field in changes:
            if field in self.env.get(id, set()):
                raise Conflict(f"{field} is set by the environment")
        if "base_url" in changes and changes["base_url"] != row["base_url"]:
            changes.setdefault("api_key", None)
        with self.lock:
            for field, value in changes.items():
                self.sql(f"update ai_connections set {field} = ? where id = ?", (value, id))

    def remove_connection(self, id: str) -> None:
        self.need_connection(id)
        if self.env.get(id):  # it would come back at the next start
            raise Conflict(f"connection {id} is set by the environment")
        with self.lock:
            for role, ref in self.roles().items():
                if ref and split_ref(ref)[0] == id:
                    raise Conflict(f"connection {id} is used by {role}")
            if self.batch_pending(id):
                raise Conflict(f"connection {id} has a batch still pending")
            self.sql("delete from ai_models where connection_id = ?", (id,))
            self.sql("delete from ai_connections where id = ?", (id,))
            # Its waiting prompts could never be answered now.
            self.sql(
                "delete from ai_waiting where substr(model, 1, ?) = ?", (len(id) + 1, id + "/")
            )

    def connections(self) -> list[dict]:
        return [
            {
                "id": r["id"],
                "label": r["label"],
                "kind": r["kind"],
                "base_url": r["base_url"],
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
        price_in: float | None = 0.0,
        price_out: float | None = 0.0,
    ) -> None:
        """Priced $0 unless prices are given (None leaves it unpriced, refused until priced).
        At $0 its calls are free to the budget: the cap does not stop them."""
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
        if None in (price_in, price_out) and self.batch_pending(connection_id, model_id):
            raise Conflict(f"{connection_id}/{model_id} has a batch still pending")
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

    def batch_pending(self, connection_id: str, model_id: str | None = None) -> bool:
        """A submitted batch still needs its connection and prices to be collected.
        Rows from before connections have no connection id; they match by model."""
        for row in self.pending_batches():
            if row["connection_id"] == connection_id or (
                row["connection_id"] is None and self.resolve(row["model"])[0] == connection_id
            ):
                if model_id is None or row["model"] == model_id:
                    return True
        return False

    def fetch_models(self, connection_id: str) -> list[str]:
        """Adds the connection's models that are not stored yet, priced $0 until a price is
        entered, and returns their ids."""
        connection = self._connection(connection_id)
        if connection["kind"] == "manual":
            raise Conflict(f"{connection_id} is answered by hand: add its models by hand")
        known = {m["model_id"] for m in self.models() if m["connection_id"] == connection_id}
        added = [m for m in dict.fromkeys(self._list_models(connection)) if m not in known]
        for model_id in added:
            self.add_model(connection_id, model_id)
        return added

    def test_connection(self, connection_id: str) -> dict:
        """One tiny paid call, ledgered under purpose "test". Failures are reported, not raised."""
        if self.need_connection(connection_id)["kind"] == "manual":
            error = "answered by hand: nothing is sent, so there is nothing to test"
            return {"ok": False, "model": None, "usd": 0.0, "error": error}
        model_id = next(
            (
                m["model_id"]
                for m in self.models()
                if m["connection_id"] == connection_id
                and m["enabled"]
                and m["price_in"] is not None
                and m["price_out"] is not None
            ),
            None,
        )
        if model_id is None:
            has_models = any(m["connection_id"] == connection_id for m in self.models())
            error = (
                "no enabled model has a price; add a price to one"
                if has_models
                else "no model yet; fetch the list or add one"
            )
            return {"ok": False, "model": None, "usd": 0.0, "error": error}
        if not self.usable(connection_id, model_id):
            return {
                "ok": False,
                "model": model_id,
                "usd": 0.0,
                "error": "the connection has no key",
            }
        try:
            done = self.complete(
                "Reply with OK.",
                model=f"{connection_id}/{model_id}",
                max_tokens=8,
                purpose="test",
                cache=False,
            )
        except (ProviderError, BudgetExceeded) as e:
            return {"ok": False, "model": model_id, "usd": 0.0, "error": str(e)}
        return {"ok": True, "model": model_id, "usd": done.usd, "error": None}

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

    def resolve(self, model: str) -> tuple[str, str]:
        """(connection id, model id) for a ref, or for a bare id that one enabled model has."""
        connection_id, model_id = split_ref(model)
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
        if not (math.isfinite(usd) and usd >= 0):
            raise ValueError("the cap must be a number that is not negative")
        self.sql("insert or replace into ai_settings values ('cap_usd', ?)", (str(float(usd)),))

    # environment

    def apply_env(self, *, api_key: str | None, model: str | None, cap_usd: float | None) -> None:
        """Apply the environment's values; they win over stored ones and become read-only.

        model names a stored model (a ref, or a bare id one enabled model has) and is used
        as it resolves. Any other value is a model id on the "anthropic" connection (that
        prefix may lead): added, priced $0, when the connection exists, and the default
        either way, so jobs refuse until that model exists.
        """
        preset = PRESETS["anthropic"]
        if api_key:
            if not self.sql("select 1 from ai_connections where id = 'anthropic'"):
                self.add_connection("anthropic", preset["kind"], preset["label"])
            # The env key is for the preset's address only, so that is locked too.
            self.sql(
                "update ai_connections set api_key = ?, base_url = ? where id = 'anthropic'",
                (api_key, preset["base_url"]),
            )
            self.env["anthropic"] = {"api_key", "base_url"}
            for model_id, (label, price_in, price_out) in preset["models"].items():
                if not self.sql(
                    "select 1 from ai_models where connection_id = 'anthropic' and model_id = ?",
                    (model_id,),
                ):
                    self.add_model(
                        "anthropic", model_id, label=label, price_in=price_in, price_out=price_out
                    )
        if model:
            try:
                ref = "/".join(self.resolve(model))
            except UnknownModel:
                model_id = model.removeprefix("anthropic/")
                ref = f"anthropic/{model_id}"
                has_connection = self.sql("select 1 from ai_connections where id = 'anthropic'")
                if has_connection and ref not in {m["ref"] for m in self.models()}:
                    self.add_model("anthropic", model_id)
            self.sql("insert or replace into ai_roles values ('default', ?)", (ref,))
            self.env_default = True
        elif api_key and not self.role_model("default"):
            self.sql(
                "insert or replace into ai_roles values ('default', ?)",
                (f"anthropic/{preset['default_model']}",),
            )
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

    def ready(self, role: str = "default", model: str | None = None) -> bool:
        """The model a call would use (model when given, else the role's) is usable and priced."""
        try:
            connection_id, model_id = self.resolve(model or self.model_for(role))
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
            bool(connection["api_key"]) or connection["kind"] in ("openai", "manual")
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
        keys = self._keys(connection_id, model_id, system, prompts, max_tokens)
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
        if connection["kind"] == "manual":
            self._hold(connection_id, model_id, system, max_tokens, purpose, misses)
        if cache:
            # A batch submitted before a crash may already hold these answers.
            self.collect_pending()
            self.load_cached(list(misses), results)
            misses = {k: p for k, p in misses.items() if k not in results}
        if misses:
            batch = len(misses) > BATCH_THRESHOLD and connection["kind"] in BATCH
            estimate = self._usd(rates, system, list(misses.values()), max_tokens, batch)
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
                    # THE JOB'S CHECK RUNS BEFORE EACH SEND: a connection without a Batch
                    # API answers one at a time, and a cancel must not wait for every prompt left.
                    check = getattr(self._job, "check", None)
                    for key, p in misses.items():
                        if check:
                            check()
                        with provider_errors(connection):
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
                            rates=rates,
                        )
            finally:
                with self._lock:
                    self._reserved -= estimate
                total = getattr(self._tally, "total", None)
                if total is not None:  # what was paid for, even when a later send failed
                    total["usd"] += sum(results[k].usd for k in misses if k in results)
        return [results[k] for k in keys]

    def _keys(
        self, connection_id: str, model_id: str, system: str, prompts: list[str], max_tokens: int
    ) -> list[str]:
        # A model reached through two connections shares its answers, but one answered by
        # hand is keyed by its ref: a typed reply is never served as a provider's answer.
        by_hand = self.need_connection(connection_id)["kind"] == "manual"
        keyed = f"{connection_id}/{model_id}" if by_hand else model_id
        return [cache_key(keyed, system, p, max_tokens) for p in prompts]

    @staticmethod
    def _usd(rates, system: str, prompts: list[str], max_tokens: int, batch: bool) -> float:
        out = math.ceil(max_tokens * 1.3)
        return sum(
            usd_for(rates, estimate_tokens(len(system) + len(p)), out, batch) for p in prompts
        )

    def load_cached(self, keys: list[str], results: dict[str, Completion]) -> None:
        for key in dict.fromkeys(keys):
            rows = self.sql("select * from ai_cache where key = ?", (key,))
            if rows:
                row = rows[0]
                results[key] = Completion(
                    row["text"], 0.0, True, row["input_tokens"], row["output_tokens"]
                )

    # answered by hand

    def _hold(
        self,
        connection_id: str,
        model_id: str,
        system: str,
        max_tokens: int,
        purpose: str,
        misses: dict[str, str],
    ) -> None:
        """Keeps the prompts a connection answered by hand was asked, and raises: nothing is
        sent, and a call asked again after the replies are imported finds them cached."""
        job_id = getattr(self._job, "job_id", None) or ""  # "": asked by no job
        at = datetime.now(UTC).isoformat()
        for key, prompt in misses.items():
            self.sql(
                "insert or ignore into ai_waiting values (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    f"{connection_id}/{model_id}",
                    system,
                    prompt,
                    max_tokens,
                    purpose,
                    job_id,
                    at,
                ),
            )
        raise AwaitingReplies(len(misses))

    def waiting(self, job_id: str | None = None) -> list[dict]:
        """The prompts waiting for replies by hand, oldest first: those job_id asked when
        given, else each prompt once, under the first job that asked it. Two jobs can ask
        the same prompt; one reply answers both. A prompt cached by now no longer waits."""
        self.sql("delete from ai_waiting where key in (select key from ai_cache)")
        first: dict[str, dict] = {}
        for r in self.sql("select * from ai_waiting order by at, rowid"):
            row = {**dict(r), "job_id": r["job_id"] or None}
            if job_id is None:
                first.setdefault(row["key"], row)
            elif row["job_id"] == job_id:
                first[row["key"]] = row
        return list(first.values())

    def answer(self, replies: dict[str, str]) -> int:
        """Records replies to waiting prompts as if their model had sent them: cached, and
        ledgered to the first job that asked, at the model's price, on estimated tokens.
        Refuses them all, recording none, when one is empty or answers no waiting prompt, or
        its model has no price now."""
        waiting = {r["key"]: r for r in self.waiting()}
        unknown = [k for k in replies if k not in waiting]
        if unknown:
            raise ValueError(f"{len(unknown)} replies answer no waiting prompt: {unknown[:3]}")
        empty = [k for k, text in replies.items() if not text.strip()]
        if empty:
            raise ValueError(f"{len(empty)} replies are empty: {empty[:3]}")
        for model in {waiting[k]["model"] for k in replies}:
            self.need_rates(*split_ref(model))
        for key, text in replies.items():
            row = waiting[key]
            connection_id, model_id = split_ref(row["model"])
            self._record(
                key,
                connection_id,
                model_id,
                text,
                estimate_tokens(len(row["system"]) + len(row["prompt"])),
                estimate_tokens(len(text)),
                False,
                row["purpose"],
                job_id=row["job_id"],
            )
            self.sql("delete from ai_waiting where key = ?", (key,))
        return len(replies)

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
        rates: tuple[float, float] | None = None,
    ) -> Completion:
        """rates: the price the call was checked at, so a sent call is always ledgered."""
        if job_id is CURRENT_JOB:
            job_id = getattr(self._job, "job_id", None)
        usd = usd_for(
            rates or self.need_rates(connection_id, model_id), tokens_in, tokens_out, batch
        )
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
        # Only a kind with a Batch API sends batches; the others pay full price.
        batch = batch and self.need_connection(connection_id)["kind"] in BATCH
        return usd_for(rates, estimate_tokens(prompt_chars), math.ceil(output_tokens * 1.3), batch)

    def estimate_many(
        self,
        prompts: list[str],
        max_tokens: int,
        *,
        role: str = "default",
        model: str | None = None,
        system: str = "",
    ) -> float:
        """What complete_many would spend on these prompts; see plan_many."""
        return self.plan_many(prompts, max_tokens, role=role, model=model, system=system)[0]

    def plan_many(
        self,
        prompts: list[str],
        max_tokens: int,
        *,
        role: str = "default",
        model: str | None = None,
        system: str = "",
    ) -> tuple[float, int]:
        """What complete_many would spend on these prompts, and how many calls it would send:
        the cache misses, a repeated prompt once. A cached one is free, and the rest are
        priced batched exactly when it would send them as a batch (more than
        BATCH_THRESHOLD misses, on a kind with a Batch API). Raises as estimate_usd does."""
        connection_id, model_id = self.resolve(model or self.model_for(role))
        rates = self.need_rates(connection_id, model_id)
        keys = self._keys(connection_id, model_id, system, prompts, max_tokens)
        cached: dict[str, Completion] = {}
        self.load_cached(keys, cached)
        misses = {k: p for k, p in zip(keys, prompts, strict=True) if k not in cached}
        batch = (
            len(misses) > BATCH_THRESHOLD and self.need_connection(connection_id)["kind"] in BATCH
        )
        return self._usd(rates, system, list(misses.values()), max_tokens, batch), len(misses)

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

    def _list_models(self, connection: dict) -> list[str]:
        with provider_errors(connection):
            return LIST_MODELS[connection["kind"]](self._client(connection))

    def _send(
        self, connection: dict, model_id: str, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        return SEND[connection["kind"]](
            self._client(connection), model_id, system, prompt, max_tokens
        )

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
