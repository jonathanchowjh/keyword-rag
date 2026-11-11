"""Every environment variable the service reads, in one place."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# The Hub repo of published builds a load applies (ragqa_core.hub); RAGQA_BUILD_HUB names
# another, and set to empty turns the Hub off.
HUB_REPO = "jonathanmiro97/keyword-rag-builds"
# The recommended reranker, which Settings -> Search defaults can change: small and quick
# on a laptop CPU. BAAI/bge-reranker-v2-m3-int8 suits a faster machine.
DEFAULT_RERANK_MODEL = "jinaai/jina-reranker-v1-tiny-en"
# The recommended embedding model for new stores, which Settings -> Search defaults can
# change: fast, and already the judge model, so an install has it cached. A store keeps
# the model it was made with.
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_JUDGE_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class Settings:
    # None when unset: the values stored through the Settings page stand.
    api_key: str | None
    cap_usd: float | None
    model: str | None
    qdrant_url: str | None
    home: Path
    username: str | None
    password: str | None
    dense_model: str
    judge_model: str
    rerank_model: str
    build_hub: str


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    def get(name: str, default: str | None = None) -> str | None:
        value = env.get(name, "")
        return value if value else default

    raw_cap = get("RAGQA_MONTHLY_CAP_USD")
    cap = None
    if raw_cap is not None:
        try:
            cap = float(raw_cap)
        except ValueError:
            raise ValueError(f"RAGQA_MONTHLY_CAP_USD must be a number, got {raw_cap!r}") from None
        if cap < 0:
            raise ValueError(f"RAGQA_MONTHLY_CAP_USD must not be negative, got {cap}")
    return Settings(
        api_key=get("ANTHROPIC_API_KEY"),
        cap_usd=cap,
        model=get("RAGQA_MODEL"),
        qdrant_url=get("RAGQA_QDRANT_URL"),
        home=Path(get("RAGQA_HOME", "~/.ragqa") or "~/.ragqa").expanduser(),
        username=get("RAGQA_USERNAME"),
        password=get("RAGQA_PASSWORD"),
        dense_model=get("RAGQA_DENSE_MODEL", DEFAULT_DENSE_MODEL) or "",
        judge_model=get("RAGQA_JUDGE_MODEL", DEFAULT_JUDGE_MODEL) or "",
        rerank_model=get("RAGQA_RERANK_MODEL", DEFAULT_RERANK_MODEL) or "",
        build_hub=env.get("RAGQA_BUILD_HUB", HUB_REPO).strip(),  # empty is off, not unset
    )
