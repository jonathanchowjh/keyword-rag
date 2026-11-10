"""The AI settings routes: connections, models, roles, the budget and search defaults."""

from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict

from aiconnect.presets import PRESETS
from ragqa_api.guards import Guards
from ragqa_core import search_defaults
from ragqa_core.catalog import ROLE_LABELS

KNOWN_ROLES = ["default", "answer", "summary", "keywords", "query", "rewrite"]


class NewConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str | None = None
    id: str
    label: str | None = None
    kind: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class ConnectionEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    base_url: str | None = None
    api_key: str | None = None


class NewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str
    label: str | None = None
    price_in: float | None = 0.0  # $0 unless given; null leaves it unpriced
    price_out: float | None = 0.0


class ModelEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    price_in: float | None = None
    price_out: float | None = None
    enabled: bool | None = None


class RoleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None


class BudgetSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cap_usd: float


class SearchDefaultsIn(BaseModel):
    """A partial change: a field left out keeps its value, null puts it back to recommended.
    Strict, so "yes" is not read as true: search_defaults.check says what each field takes."""

    model_config = ConfigDict(extra="forbid", strict=True)

    method: str | None = None
    candidates: int | None = None
    evidence_size: int | None = None
    rerank: bool | None = None
    rewrite: bool | None = None
    follow_up_turns: int | None = None
    rrf_k: int | None = None
    rerank_model: str | None = None
    rerank_pool: int | None = None
    dense_model: str | None = None


def add_routes(app: FastAPI, guards: Guards) -> None:
    state = app.state

    @app.get("/budget")
    def budget() -> dict:
        return {
            "month": datetime.now(UTC).strftime("%Y-%m"),
            "spent_usd": state.ai.month_spend(),
            "cap_usd": state.ai.cap_usd,
            "note": "This service's own spend; your provider's dashboard shows the rest.",
        }

    # -- AI settings ------------------------------------------------------------

    @app.get("/ai")
    def get_ai() -> dict:
        return {
            "connections": state.ai.connections(),
            "models": state.ai.models(),
            "roles": state.ai.roles(),
            "default_from_env": state.ai.env_default,
            "known_roles": KNOWN_ROLES,
            "resolved": {role: state.ai.ready(role) for role in ROLE_LABELS},
            "budget": {
                "cap_usd": state.ai.cap_usd,
                "spent_usd": state.ai.month_spend(),
                "from_env": state.ai.env_cap,
            },
            "presets": [
                {k: p[k] for k in ("label", "kind", "base_url")} | {"id": id}
                for id, p in PRESETS.items()
            ],
            # The list prices a model can be given on the Settings page, besides $0 and typed.
            "prices": [
                {"label": label, "price_in": price_in, "price_out": price_out}
                for p in PRESETS.values()
                for label, price_in, price_out in p["models"].values()
            ],
        }

    @app.post("/ai/connections", status_code=201)
    def add_connection(body: NewConnection) -> dict:
        if body.preset is not None and body.preset not in PRESETS:
            raise HTTPException(422, f"unknown preset {body.preset!r}; known: {list(PRESETS)}")
        preset = PRESETS.get(body.preset or "", {})
        kind = body.kind or preset.get("kind")
        if not kind:
            raise HTTPException(422, "give a preset or a kind")
        state.ai.add_connection(
            body.id,
            kind,
            body.label or preset.get("label") or body.id,
            base_url=body.base_url or preset.get("base_url"),
            api_key=body.api_key,
        )
        for model_id, (label, price_in, price_out) in preset.get("models", {}).items():
            state.ai.add_model(
                body.id, model_id, label=label, price_in=price_in, price_out=price_out
            )
        return {"id": body.id}

    @app.patch("/ai/connections/{id}")
    def edit_connection(id: str, body: ConnectionEdit) -> dict:
        state.ai.update_connection(id, **body.model_dump(exclude_unset=True))
        return {"id": id}

    @app.delete("/ai/connections/{id}", status_code=204)
    def remove_connection(id: str) -> Response:
        state.ai.remove_connection(id)
        return Response(status_code=204)

    @app.post("/ai/connections/{id}/test")
    def test_connection(id: str) -> dict:
        return state.ai.test_connection(id)

    @app.post("/ai/connections/{id}/models/fetch")
    def fetch_models(id: str) -> dict:
        return {"added": state.ai.fetch_models(id)}

    @app.post("/ai/connections/{id}/models")
    def add_model(id: str, body: NewModel) -> dict:
        state.ai.add_model(
            id, body.model_id, label=body.label, price_in=body.price_in, price_out=body.price_out
        )
        return {"ref": f"{id}/{body.model_id}"}

    @app.put("/ai/models/{connection_id}/{model_id:path}")
    def edit_model(connection_id: str, model_id: str, body: ModelEdit) -> dict:
        changes = body.model_dump(exclude_unset=True)
        if any(changes.get(f, 1) is None for f in ("label", "enabled")):
            raise HTTPException(422, "label and enabled cannot be null")
        state.ai.update_model(connection_id, model_id, **changes)
        return {"ref": f"{connection_id}/{model_id}"}

    @app.put("/ai/roles/{role}")
    def set_role(role: str, body: RoleSet) -> dict:
        if role not in KNOWN_ROLES:
            raise HTTPException(422, f"unknown role {role!r}; known: {KNOWN_ROLES}")
        state.ai.set_role(role, body.model)
        return {"role": role, "model": state.ai.roles().get(role)}

    @app.put("/ai/budget")
    def set_budget(body: BudgetSet) -> dict:
        state.ai.set_cap(body.cap_usd)
        return {"cap_usd": state.ai.cap_usd}

    # -- search defaults (Settings -> Search defaults) ----------------------------

    def search_defaults_reply() -> dict:
        models = state.models

        def named(values: dict) -> dict:  # None is the service's model: say which
            return {
                **values,
                "rerank_model": search_defaults.reranker_in_force(values, models.rerank_model),
                "dense_model": values["dense_model"] or models.dense_model,
            }

        saved = search_defaults.saved(state.conn)
        in_force = named(search_defaults.in_force(state.conn))
        reranker = str(in_force["rerank_model"])
        status = models.rerank_status(reranker)
        # The short list, then the service's model and the saved one if not on it.
        offered = [*search_defaults.EMBEDDERS, models.dense_model, saved["dense_model"]]
        embedders = [str(name) for name in dict.fromkeys(offered) if name is not None]
        return {
            "saved": saved,
            "recommended": named(search_defaults.RECOMMENDED),
            "in_force": in_force,
            # The in-force reranker's warm-up: idle, loading, ready or failed, and why it failed.
            "rerank_status": status,
            "rerank_error": models.rerank_errors.get(reranker) if status == "failed" else None,
            "limits": {field: list(bounds) for field, bounds in search_defaults.LIMITS.items()},
            "methods": list(search_defaults.CHAT_METHODS),
            "rerankers": [
                {"name": name, "size_gb": size}
                for name, size in search_defaults.supported_rerankers().items()
            ],
            "embedders": [
                {
                    "name": name,
                    "dim": models.dense_dim(name),
                    "size_gb": search_defaults.supported_embedders().get(name, {}).get("size_gb"),
                    "note": search_defaults.EMBEDDER_NOTES.get(name),
                    "seconds_per_1000": search_defaults.EMBED_SECONDS.get(name),
                }
                for name in embedders
            ],
            "embed_machine": search_defaults.EMBED_MACHINE,
            "notes": {"rerank": search_defaults.RERANK_OFF_WHY},
        }

    @app.get("/search-defaults")
    def get_search_defaults() -> dict:
        return search_defaults_reply()

    @app.put("/search-defaults")
    def put_search_defaults(body: SearchDefaultsIn) -> dict:
        changes = body.model_dump(exclude_unset=True)
        models = state.models
        before = search_defaults.reranker_in_force(
            search_defaults.in_force(state.conn), models.rerank_model
        )
        search_defaults.save(state.conn, changes)  # a bad value is a ValueError: 422
        picked = changes.get("rerank_model")
        # Loaded before the next message needs it. Saving the one in force again retries only
        # a failed load (say, the Hub was down), so a Save of other fields starts nothing.
        if "rerank_model" in changes and (
            (picked or models.rerank_model) != before or models.rerank_status(picked) == "failed"
        ):
            models.retry_rerank(picked)
        return search_defaults_reply()
