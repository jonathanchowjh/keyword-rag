"""Stored AI connections, models, roles and a monthly spend cap, in one SQLite file.

AI(path) opens its own connection to the file: WAL, a busy timeout, and every
statement under one lock. Model refs are "<connection id>/<model id>", split on
the first slash, so ids like "meta-llama/llama-3.1-8b" survive. Environment
values win: apply_env sets them at start and they are read-only afterwards.
"""

import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any

from aiconnect.presets import KINDS, PRESETS

KEEP: Any = object()  # update_*: leave this field as it is
DEFAULT_CAP_USD = 20.0
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


class AI:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        os.chmod(path, 0o600)
        self.raw.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.sql("pragma journal_mode=wal")
        self.sql("pragma busy_timeout=5000")
        with self.lock:
            self.raw.executescript(SCHEMA)
        self.env: dict[str, set[str]] = {}  # connection id -> fields set by the environment
        self.env_cap = False
        self.env_default = False

    def sql(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.raw.execute(query, params).fetchall()

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
