"""The app-wide search defaults: Settings -> Search defaults, over the recommended values.

A field Settings holds none of (null) is the recommended value. Stores and chats inherit
from here the same way: null means inherit, highest level first. Saved under
app_settings keys `search.<field>`.
"""

import functools
from typing import Any, Literal

from ragqa_core import db
from ragqa_core.connection import Connection

RECOMMENDED: dict[str, object] = {
    "method": "hybrid",
    "candidates": 20,  # per store
    "evidence_size": 8,
    "rerank": True,
    "rewrite": True,
    "follow_up_turns": 2,
    "rrf_k": 60,
    # None is the service's reranker, Models.rerank_model: RAGQA_RERANK_MODEL when set.
    "rerank_model": None,
    "rerank_pool": 20,  # the most pooled passages reranked per message
    # The embedding model a new store is made with; a store keeps its own. None is the
    # service's, Models.dense_model: RAGQA_DENSE_MODEL when set, else bge-small.
    "dense_model": None,
}
# Retrieval by the question only: chat reranks as its own step (so not hybrid_rerank),
# iterative always needs AI, and question_only and the controls do not search.
CHAT_METHODS = (
    "hybrid",
    "vector",
    "keyword",
    "keyword_rag",
    "kw_all",
    "kw_individual",
    "kw_rrf",
    "kw_hybrid",
)
STORE_FIELDS = ("method", "candidates")  # per store; null: the store's own
MAX_EVIDENCE = 20
MAX_CANDIDATES = 60  # the pool cap is never above 60
MAX_FOLLOW_UP = 10
MAX_RRF_K = 1000
MAX_RERANK_POOL = 60
LIMITS = {
    "evidence_size": (1, MAX_EVIDENCE),
    "candidates": (1, MAX_CANDIDATES),
    "follow_up_turns": (0, MAX_FOLLOW_UP),
    "rrf_k": (0, MAX_RRF_K),
    "rerank_pool": (1, MAX_RERANK_POOL),
}
SWITCHES = ("rerank", "rewrite")
# Why reranking is off by default, shown beside the setting; None while it is on.
RERANK_OFF_WHY: str | None = None
PREFIX = "search."
# The embedding models the pages offer for new stores, fastest first; the API takes any
# in supported_embedders().
EMBEDDERS = ("BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5", "Qwen/Qwen3-Embedding-0.6B-Q")
EMBEDDER_NOTES = {
    "BAAI/bge-small-en-v1.5": "fast",
    "BAAI/bge-base-en-v1.5": "between the two",
    "Qwen/Qwen3-Embedding-0.6B-Q": "accurate, slower",
}
# Seconds to embed 1,000 passages, by model, on EMBED_MACHINE (scripts/time_embed.py,
# scripts/experiments.md); empty, and the machine None, until that timing is run.
EMBED_SECONDS: dict[str, int] = {}
EMBED_MACHINE: str | None = None


@functools.cache
def supported_rerankers() -> dict[str, float]:
    """fastembed's cross-encoders and their size in GB: its own list, read without the network."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder  # slow to import: not at start

    return {m["model"]: float(m["size_in_GB"]) for m in TextCrossEncoder.list_supported_models()}


@functools.cache
def supported_embedders() -> dict[str, dict]:
    """fastembed's embedding models, each with its dimension and size in GB: its own list,
    read without the network."""
    from fastembed import TextEmbedding  # slow to import: not at start

    return {
        m["model"]: {"dim": int(m["dim"]), "size_gb": float(m["size_in_GB"])}
        for m in TextEmbedding.list_supported_models()
    }


def reranker_in_force(settings: dict, service: str) -> str:
    """The reranker a search uses: the saved one, else the service's (Models.rerank_model)."""
    return str(settings["rerank_model"] or service)


def search_settings(settings: dict, service_reranker: str) -> dict[str, Any]:
    """The three plain fields every chat search reads the same way."""
    return {
        "rrf_k": int(settings["rrf_k"]),
        "rerank_pool": int(settings["rerank_pool"]),
        "rerank_model": reranker_in_force(settings, service_reranker),
    }


def check(field: str, value: object) -> None:
    """Raises ValueError unless `value` may be saved for `field`; None always may."""
    if field not in RECOMMENDED:
        raise ValueError(f"unknown search default {field!r}; known: {', '.join(RECOMMENDED)}")
    if value is None:
        return
    if field == "method":
        if value not in CHAT_METHODS:
            raise ValueError(
                f"method {value!r} is not a chat method; chat methods: {', '.join(CHAT_METHODS)}"
            )
    elif field == "rerank_model":
        supported = supported_rerankers()
        if value not in supported:
            raise ValueError(f"unknown reranker {value}; supported: {', '.join(supported)}")
    elif field == "dense_model":
        embedders = supported_embedders()
        if value not in embedders:
            raise ValueError(f"unknown embedding model {value}; supported: {', '.join(embedders)}")
    elif field in SWITCHES:
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be true or false")
    else:
        low, high = LIMITS[field]
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{field} must be a whole number from {low} to {high}")


def chat_method(value: str | None) -> str | None:
    """A method saved before it stopped being a chat method (hybrid_rerank) reads as inherit."""
    return value if value in CHAT_METHODS else None


def saved(conn: Connection) -> dict[str, object | None]:
    """Each field's stored value, or None when none is stored or it no longer passes `check`
    (a reranker fastembed dropped, a method that stopped being a chat method)."""
    stored = db.get_app_settings(conn, PREFIX)
    values: dict[str, object | None] = {}
    for field in RECOMMENDED:
        value = stored.get(field)
        try:
            check(field, value)
        except ValueError:
            value = None
        values[field] = value
    return values


def in_force(conn: Connection) -> dict[str, object]:
    return {f: RECOMMENDED[f] if v is None else v for f, v in saved(conn).items()}


def source(conn: Connection, field: str) -> Literal["settings", "recommended"]:
    return "recommended" if saved(conn)[field] is None else "settings"


def save(conn: Connection, changes: dict[str, object | None]) -> None:
    """Checks every change before writing any; None puts a field back to recommended."""
    for field, value in changes.items():
        check(field, value)
    with conn.lock:
        for field, value in changes.items():
            db.set_app_setting(conn, PREFIX + field, value)


def inherited(levels: dict[str, object]) -> dict[str, Any]:
    """The first level, highest first, that holds a value, as {"value", "from"}. The last
    level, the recommended value, always holds one."""
    source = next((s for s, v in levels.items() if v is not None), "recommended")
    return {"value": levels[source], "from": source}


def store_in_force(conn: Connection, store: str, chosen: dict | None = None) -> dict[str, Any]:
    """The store's method and candidates, each with the level it comes from. `chosen`
    holds the chat's values for the store."""
    chosen = chosen or {}
    values = saved(conn)
    own = {
        "method": chat_method(db.get_chat_method(conn, store)),
        "candidates": db.get_chat_k(conn, store),
    }
    return {
        f: inherited(
            {
                "chat": chosen.get(f),
                "store": own[f],
                "settings": values[f],
                "recommended": RECOMMENDED[f],
            }
        )
        for f in STORE_FIELDS
    }


def run_method(method: str, rerank: bool) -> str:
    """The method as a run names it: Hybrid with rerank on is hybrid_rerank."""
    return "hybrid_rerank" if method == "hybrid" and rerank else method


def store_search(conn: Connection, store: str) -> dict[str, Any]:
    """The search a chat on the store uses, as a run names it: Hybrid reranked is the run's
    hybrid_rerank; any other method with rerank on is scored before its rerank step.
    A chat's own overrides are not reflected."""
    method = store_in_force(conn, store)["method"]["value"]
    rerank = bool(in_force(conn)["rerank"])
    run = run_method(method, rerank)
    return {
        "method": method,
        "rerank": rerank,
        "run_method": run,
        "before_rerank": rerank and run != "hybrid_rerank",
    }
