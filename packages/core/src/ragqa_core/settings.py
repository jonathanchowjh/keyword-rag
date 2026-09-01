"""Every environment variable the service reads, in one place."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


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
    port: int


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
        dense_model=get("RAGQA_DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B-Q") or "",
        judge_model=get("RAGQA_JUDGE_MODEL", "BAAI/bge-small-en-v1.5") or "",
        rerank_model=get("RAGQA_RERANK_MODEL", "BAAI/bge-reranker-v2-m3-int8") or "",
        port=int(get("PORT", "8080") or "8080"),
    )
