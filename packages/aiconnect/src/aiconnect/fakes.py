"""Test doubles: a real AI with its provider seams overridden. Nothing here calls a provider."""

import math
from collections.abc import Callable
from pathlib import Path

from aiconnect.ai import AI
from aiconnect.presets import PRESETS

# The fake Batch API's server side: batches outlive the fake that submitted
# them, as real ones outlive a restarted process.
BATCHES: dict[str, dict[str, tuple[str, int, int]]] = {}
CANCELLED: set[str] = set()


def fake_ai(
    path: Path,
    reply: Callable[[str], str],
    *,
    api_key: str | None = "test",
    cap_usd: float = 20.0,
) -> AI:
    ai = AI(path)
    preset = PRESETS["anthropic"]
    if ai.sql("select 1 from ai_connections where id = 'anthropic'"):
        ai.update_connection("anthropic", api_key=api_key)
    else:
        ai.add_connection("anthropic", preset["kind"], preset["label"], api_key=api_key)
        for model_id, (label, price_in, price_out) in preset["models"].items():
            ai.add_model("anthropic", model_id, label=label, price_in=price_in, price_out=price_out)
    ai.set_role("default", f"anthropic/{preset['default_model']}")
    ai.set_cap(cap_usd)
    return fake_replies(ai, reply)


def fake_replies(ai: AI, reply: Callable[[str], str]) -> AI:
    """Patches `ai` in place to answer every prompt with reply(prompt)."""

    def send(
        connection: dict, model_id: str, system: str, prompt: str, max_tokens: int
    ) -> tuple[str, int, int]:
        text = reply(prompt)
        return text, math.ceil(len(prompt) / 4), math.ceil(len(text) / 4)

    def submit_batch(connection: dict, requests: list[dict]) -> str:
        batch_id = f"fake-batch-{len(BATCHES)}"
        BATCHES[batch_id] = {
            r["custom_id"]: send(connection, r["model"], r["system"], r["prompt"], 0)
            for r in requests
        }
        return batch_id

    def batch_results(connection: dict, batch_id: str) -> dict[str, tuple[str, int, int] | None]:
        # A reply of "FAIL" stands for an errored or expired batch entry.
        return {k: None if v[0] == "FAIL" else v for k, v in BATCHES[batch_id].items()}

    ai._send = send  # type: ignore[method-assign]
    ai._submit_batch = submit_batch  # type: ignore[method-assign]
    ai._batch_done = lambda connection, batch_id: True  # type: ignore[method-assign]
    ai._batch_results = batch_results  # type: ignore[method-assign]
    ai._cancel_batch = lambda connection, batch_id: CANCELLED.add(batch_id)  # type: ignore[method-assign]
    return ai
