"""Test doubles: real objects with their outward-facing methods overridden.

Only tests import this module (tests/test_boundaries.py enforces it). Nothing
here calls Claude or downloads a model.
"""

import math
import sqlite3
from collections.abc import Callable

from ragqa_core.llm import Claude


def fake_claude(
    conn: sqlite3.Connection,
    reply: Callable[[str], str],
    api_key: str | None = "test",
    cap_usd: float = 20.0,
) -> Claude:
    claude = Claude(conn, api_key, cap_usd, "claude-haiku-5-5")

    def send(model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        text = reply(prompt)
        return text, math.ceil(len(prompt) / 4), math.ceil(len(text) / 4)

    def send_batch(requests: list[dict]) -> dict[str, tuple[str, int, int]]:
        return {
            r["custom_id"]: send(r["model"], r["system"], r["prompt"], r["max_tokens"])
            for r in requests
        }

    claude._send = send  # type: ignore[method-assign]
    claude._send_batch = send_batch  # type: ignore[method-assign]
    return claude
