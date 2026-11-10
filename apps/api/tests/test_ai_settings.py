"""The /ai, /budget and /search-defaults routes."""

import time

from api_helpers import SEARCH_FIELDS, TURBO, settings
from fastapi.testclient import TestClient

from ragqa_api.app import create_app
from ragqa_core import search_defaults


def test_env_key_becomes_anthropic_connection(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456"))
    assert app.state.ai.connections()[0]["id"] == "anthropic"
    assert app.state.ai.connections()[0]["from_env"] == ["api_key", "base_url"]
    assert app.state.ai.ready("answer")


def test_budget_note(client):
    body = client.get("/budget").json()
    assert "own spend" in body["note"] and body["cap_usd"] == 20.0 and body["spent_usd"] == 0.0


def test_ai_reports_resolved_roles(client):
    ai = client.app.state.ai
    body = client.get("/ai").json()
    assert set(body["resolved"]) == {"default", "answer", "summary", "keywords", "query", "rewrite"}
    assert body["resolved"] == {
        "default": True,
        "answer": True,
        "summary": True,
        "keywords": True,
        "query": True,
        "rewrite": True,
    }
    ai.set_role("default", None)
    assert client.get("/ai").json()["resolved"]["rewrite"] is False


def test_ai_settings_round_trip(client):
    ai = client.app.state.ai
    ai._list_models = lambda connection: ["llama3.1", "qwen2"]  # type: ignore[method-assign]
    r = client.post("/ai/connections", json={"preset": "openrouter", "id": "router"})
    assert r.status_code == 201, r.text
    assert client.post("/ai/connections/router/models/fetch").json() == {
        "added": ["llama3.1", "qwen2"]
    }
    assert client.put("/ai/roles/answer", json={"model": "router/llama3.1"}).status_code == 200
    body = client.get("/ai").json()
    router = next(c for c in body["connections"] if c["id"] == "router")
    assert router["kind"] == "openai" and router["label"] == "OpenRouter" and "local" not in router
    # Fetched models start at $0 until a price is entered.
    assert {m["ref"]: m["price_in"] for m in body["models"] if m["connection_id"] == "router"} == {
        "router/llama3.1": 0,
        "router/qwen2": 0,
    }
    assert {"label": "Haiku 5.5", "price_in": 0.10, "price_out": 0.50} in body["prices"]
    assert body["roles"]["answer"] == "router/llama3.1"
    assert body["known_roles"] == [
        "default",
        "answer",
        "summary",
        "keywords",
        "query",
        "rewrite",
    ]
    assert body["budget"] == {"cap_usd": 20.0, "spent_usd": 0.0, "from_env": False}
    assert "ollama" not in {p["id"] for p in body["presets"]}
    assert {"id": "openrouter", "label": "OpenRouter", "kind": "openai"}.items() <= next(
        p for p in body["presets"] if p["id"] == "openrouter"
    ).items()
    assert client.put("/ai/budget", json={"cap_usd": 7.5}).status_code == 200
    assert client.get("/ai").json()["budget"]["cap_usd"] == 7.5
    r = client.post("/ai/connections/router/models", json={"model_id": "x", "price_in": 1})
    assert r.status_code == 200, r.text
    r = client.patch("/ai/connections/router", json={"label": "Mine", "api_key": "abc"})
    assert r.status_code == 200 and client.get("/ai").json()["connections"][1]["label"] == "Mine"
    assert client.put("/ai/roles/answer", json={"model": None}).status_code == 200
    assert client.delete("/ai/connections/router").status_code == 204


def test_ai_validation_errors_are_422(client):
    r = client.post("/ai/connections", json={"id": "x"})  # no preset, no kind
    assert r.status_code == 422
    assert client.post("/ai/connections", json={"preset": "nope", "id": "x"}).status_code == 422
    client.post("/ai/connections", json={"preset": "openai", "id": "oa"})
    assert client.patch("/ai/connections/oa", json={"label": ""}).status_code == 422
    r = client.post("/ai/connections/oa/models", json={"model_id": "m", "price_in": -1})
    assert r.status_code == 422
    assert client.put("/ai/models/oa/ghost", json={"enabled": False}).status_code == 422
    client.post("/ai/connections/oa/models", json={"model_id": "m"})
    assert client.put("/ai/models/oa/m", json={"enabled": None}).status_code == 422
    assert client.post("/ai/connections", json={"preset": "openai", "id": "oa"}).status_code == 409


def test_key_never_leaves_the_server(client):
    client.post(
        "/ai/connections", json={"preset": "openai", "id": "oa", "api_key": "sk-proj-SECRET123456"}
    )
    for path in ("/ai", "/status", "/health", "/budget"):
        assert "SECRET" not in client.get(path).text
    assert "SECRET" not in client.post("/ai/connections/oa/test").text


def test_price_route_accepts_slash_in_model_id(client):
    client.post("/ai/connections", json={"preset": "openrouter", "id": "or"})
    client.post("/ai/connections/or/models", json={"model_id": "meta-llama/llama-3.1-8b"})
    r = client.put(
        "/ai/models/or/meta-llama/llama-3.1-8b", json={"price_in": 0.02, "price_out": 0.05}
    )
    assert r.status_code == 200, r.text
    model = next(m for m in client.get("/ai").json()["models"] if m["ref"].startswith("or/"))
    assert model["model_id"] == "meta-llama/llama-3.1-8b"
    assert (model["price_in"], model["price_out"]) == (0.02, 0.05)


def test_delete_connection_in_use_is_409_naming_role(client):
    r = client.delete("/ai/connections/anthropic")
    assert r.status_code == 409 and "default" in r.json()["detail"]


def test_env_locked_key_is_409(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456", cap_usd=5.0))
    client = TestClient(app)
    r = client.patch("/ai/connections/anthropic", json={"api_key": "other"})
    assert r.status_code == 409 and "environment" in r.json()["detail"]
    assert client.put("/ai/budget", json={"cap_usd": 9}).status_code == 409
    assert client.get("/ai").json()["budget"]["from_env"] is True


def test_unknown_role_is_422(client):
    r = client.put("/ai/roles/nonsense", json={"model": None})
    assert r.status_code == 422 and "known" in r.json()["detail"]
    r = client.put("/ai/roles/answer", json={"model": "anthropic/ghost"})
    assert r.status_code == 422


def test_env_model_locks_the_default_job(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456", model="claude-sonnet-5-5"))
    client = TestClient(app)
    assert client.get("/ai").json()["default_from_env"] is True
    r = client.put("/ai/roles/default", json={"model": "anthropic/claude-haiku-5-5"})
    assert r.status_code == 409 and "environment" in r.json()["detail"]
    assert client.put("/ai/roles/answer", json={"model": "claude-haiku-5-5"}).status_code == 200


def test_env_connection_keeps_its_address_and_cannot_be_removed(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456"))
    client = TestClient(app)
    assert client.get("/ai").json()["default_from_env"] is False
    r = client.patch("/ai/connections/anthropic", json={"base_url": "http://elsewhere"})
    assert r.status_code == 409 and "environment" in r.json()["detail"]
    assert client.put("/ai/roles/default", json={"model": None}).status_code == 200
    r = client.delete("/ai/connections/anthropic")
    assert r.status_code == 409 and "environment" in r.json()["detail"]


def test_new_address_without_a_new_key_drops_the_stored_key(client):
    client.post(
        "/ai/connections", json={"preset": "openai", "id": "oa", "api_key": "sk-proj-SECRET123456"}
    )
    r = client.patch("/ai/connections/oa", json={"base_url": "http://other/v1"})
    assert r.status_code == 200, r.text
    oa = next(c for c in client.get("/ai").json()["connections"] if c["id"] == "oa")
    assert oa["key"] is None and oa["base_url"] == "http://other/v1"


def test_rewrite_is_a_known_role(client):
    assert "rewrite" in client.get("/ai").json()["known_roles"]


def test_search_defaults_round_trip_and_reset(client):
    service = client.app.state.models.rerank_model
    r = client.put("/search-defaults", json=dict.fromkeys(SEARCH_FIELDS))
    assert r.status_code == 200, r.text
    got = client.get("/search-defaults").json()
    assert got["in_force"] == {
        **search_defaults.RECOMMENDED,
        "rerank_model": service,
        "dense_model": client.app.state.models.dense_model,
    }
    assert got["recommended"]["rerank_model"] == service
    assert got["limits"] == {
        "evidence_size": [1, 20],
        "candidates": [1, 60],
        "follow_up_turns": [0, 10],
        "rrf_k": [0, 1000],
        "rerank_pool": [1, 60],
    }
    assert got["methods"] == list(search_defaults.CHAT_METHODS)
    tiny = [r for r in got["rerankers"] if r["name"] == "jinaai/jina-reranker-v1-tiny-en"]
    assert len(tiny) == 1 and tiny[0]["size_gb"] > 0
    assert got["notes"] == {"rerank": None}

    r = client.put("/search-defaults", json={"candidates": 30})
    assert r.status_code == 200 and r.json()["in_force"]["candidates"] == 30
    assert r.json()["saved"] == {**dict.fromkeys(SEARCH_FIELDS), "candidates": 30}
    r = client.put("/search-defaults", json={"candidates": None})
    assert r.json()["in_force"]["candidates"] == 20
    assert client.put("/search-defaults", json={"nope": 1}).status_code == 422
    assert client.put("/search-defaults", json={"method": "hybrid_rerank"}).status_code == 422
    assert client.get("/search-defaults").json()["in_force"]["rerank_model"] == service


def test_search_defaults_lists_embedders_and_names_the_service_model(client):
    service = client.app.state.models.dense_model
    got = client.get("/search-defaults").json()
    assert got["recommended"]["dense_model"] == service
    assert got["in_force"]["dense_model"] == service and got["saved"]["dense_model"] is None
    names = [e["name"] for e in got["embedders"]]
    assert names[: len(search_defaults.EMBEDDERS)] == list(search_defaults.EMBEDDERS)
    assert service in names and len(names) == len(set(names))
    small = next(e for e in got["embedders"] if e["name"] == "BAAI/bge-small-en-v1.5")
    assert small == {
        "name": "BAAI/bge-small-en-v1.5",
        "dim": 256,  # the fake's
        "size_gb": 0.067,
        "note": "fast",
        "seconds_per_1000": search_defaults.EMBED_SECONDS.get("BAAI/bge-small-en-v1.5"),
    }
    own = next(e for e in got["embedders"] if e["name"] == service)
    assert (own["size_gb"], own["note"]) == (None, None)  # not in fastembed's list
    r = client.put("/search-defaults", json={"dense_model": "BAAI/bge-base-en-v1.5"})
    assert r.json()["in_force"]["dense_model"] == "BAAI/bge-base-en-v1.5"
    r = client.put("/search-defaults", json={"dense_model": None})
    assert r.json()["in_force"]["dense_model"] == service
    assert client.put("/search-defaults", json={"dense_model": "nope/y"}).status_code == 422


def test_changing_the_reranker_warms_it_in_the_background(client):
    called = []
    client.app.state.models.warm_rerank_in_background = called.append  # type: ignore[attr-defined]
    r = client.put("/search-defaults", json={"rerank_model": TURBO})
    assert r.status_code == 200 and r.json()["in_force"]["rerank_model"] == TURBO
    assert called == [TURBO]
    client.put("/search-defaults", json={"candidates": 30})
    assert called == [TURBO]


def test_put_search_defaults_warms_only_a_changed_or_failed_reranker(client):
    models = client.app.state.models
    search_defaults.save(client.app.state.conn, {"rerank_model": TURBO})  # in force already
    warmed = []
    models.retry_rerank = warmed.append  # type: ignore[method-assign]
    assert models.rerank_status(TURBO) != "failed"
    assert client.put("/search-defaults", json={"rerank_model": TURBO}).status_code == 200
    assert warmed == []
    other = "jinaai/jina-reranker-v1-tiny-en"
    client.put("/search-defaults", json={"rerank_model": other})
    assert warmed == [other]
    models.rerank_status = lambda model=None: "failed"  # type: ignore[method-assign]
    client.put("/search-defaults", json={"rerank_model": other})
    assert warmed == [other, other]


def test_search_defaults_say_how_the_in_force_reranker_is_loading(client):
    models = client.app.state.models
    got = client.get("/search-defaults").json()
    assert (got["rerank_status"], got["rerank_error"]) == ("ready", None)  # the fake's
    asked = []
    status = {TURBO: "failed"}

    def rerank_status(model=None):
        asked.append(model)
        return status.get(model or "", "loading")

    models.rerank_status = rerank_status  # type: ignore[attr-defined]
    models.warm_rerank_in_background = lambda model=None: None  # type: ignore[attr-defined]
    got = client.get("/search-defaults").json()
    assert (got["rerank_status"], got["rerank_error"]) == ("loading", None)
    assert asked[-1] == models.rerank_model  # the service's, when Settings names none
    models.rerank_errors[TURBO] = "hub down"
    got = client.put("/search-defaults", json={"rerank_model": TURBO}).json()
    assert (got["rerank_status"], got["rerank_error"]) == ("failed", "hub down")
    status[TURBO] = "ready"
    got = client.get("/search-defaults").json()
    assert (got["rerank_status"], got["rerank_error"]) == ("ready", None)


def test_picking_a_reranker_that_just_failed_retries_it_at_once(client):
    models = client.app.state.models
    called = []
    models.warm_rerank_in_background = called.append  # type: ignore[attr-defined]
    models.rerank_failed_at[TURBO] = time.monotonic()  # failed a moment ago
    models.rerank_failed_at["other"] = time.monotonic()
    assert client.put("/search-defaults", json={"rerank_model": TURBO}).status_code == 200
    assert called == [TURBO] and TURBO not in models.rerank_failed_at
    assert "other" in models.rerank_failed_at  # only the model picked


def test_an_unknown_reranker_is_422(client):
    called = []
    client.app.state.models.warm_rerank_in_background = called.append  # type: ignore[attr-defined]
    r = client.put("/search-defaults", json={"rerank_model": "nope/y"})
    assert r.status_code == 422 and "unknown reranker" in r.text
    assert called == []
