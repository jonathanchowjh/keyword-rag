"""Every Claude call goes through here: priced, cached, capped and ledgered.

The cap is the service's own guard against its own bugs. It only sees calls
made by this service; the Console workspace spend limit is the real ceiling.
"""

import hashlib
import json
import math
import sqlite3
import time
from datetime import UTC, datetime

import anthropic

from ragqa_core.db import now
from ragqa_core.types import Completion

# USD per million tokens, (input, output), real-time. The Batch API is half.
PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
}
BATCH_THRESHOLD = 50
BATCH_POLL_SECONDS = 30


class BudgetExceeded(Exception):
    pass


class NoApiKey(Exception):
    def __init__(self) -> None:
        super().__init__("needs ANTHROPIC_API_KEY")


class UnknownModel(Exception):
    pass


def estimate_tokens(chars: int) -> int:
    # ~4 characters a token, +30% for the newer tokenizer.
    return math.ceil(chars / 4 * 1.3)


class Claude:
    def __init__(
        self, conn: sqlite3.Connection, api_key: str | None, cap_usd: float, default_model: str
    ) -> None:
        self.conn = conn
        self.api_key = api_key
        self.cap_usd = cap_usd
        self.default_model = default_model
        self._client: anthropic.Anthropic | None = None

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        purpose: str,
    ) -> Completion:
        return self.complete_many(
            [prompt], system=system, model=model, max_tokens=max_tokens, purpose=purpose
        )[0]

    def complete_many(
        self,
        prompts: list[str],
        *,
        system: str = "",
        model: str | None = None,
        max_tokens: int = 1024,
        purpose: str,
    ) -> list[Completion]:
        model = self._model(model)
        keys = [self._key(model, system, p, max_tokens) for p in prompts]
        results: dict[str, Completion] = {}
        for key in dict.fromkeys(keys):
            row = self.conn.execute("select * from cache where key = ?", (key,)).fetchone()
            if row:
                results[key] = Completion(
                    row["text"], 0.0, True, row["input_tokens"], row["output_tokens"]
                )
        misses = {k: p for k, p in zip(keys, prompts, strict=True) if k not in results}
        if misses:
            if not self.api_key:
                raise NoApiKey()
            batch = len(misses) > BATCH_THRESHOLD
            estimate = sum(
                self.estimate_usd(len(system) + len(p), max_tokens, model=model, batch=batch)
                for p in misses.values()
            )
            spent = self.month_spend()
            if spent + estimate > self.cap_usd:
                raise BudgetExceeded(
                    f"monthly cap reached: ${spent:.4f} spent + ${estimate:.4f} needed"
                    f" > ${self.cap_usd:.2f} cap"
                )
            if batch:
                requests = [
                    {
                        "custom_id": key[:64],
                        "model": model,
                        "system": system,
                        "prompt": p,
                        "max_tokens": max_tokens,
                    }
                    for key, p in misses.items()
                ]
                by_id = self._send_batch(requests)
                sent = {key: by_id[key[:64]] for key in misses}
            else:
                sent = {key: self._send(model, system, p, max_tokens) for key, p in misses.items()}
            for key, (text, tokens_in, tokens_out) in sent.items():
                usd = self._price(model, tokens_in, tokens_out, batch)
                self.conn.execute(
                    "insert or replace into cache values (?, ?, ?, ?)",
                    (key, text, tokens_in, tokens_out),
                )
                self.conn.execute(
                    "insert into ledger values (?, ?, ?, ?, ?, ?, ?)",
                    (now(), model, tokens_in, tokens_out, usd, int(batch), purpose),
                )
                results[key] = Completion(text, usd, False, tokens_in, tokens_out)
        return [results[k] for k in keys]

    def estimate_usd(
        self,
        prompt_chars: int,
        output_tokens: int,
        *,
        model: str | None = None,
        batch: bool = False,
    ) -> float:
        model = self._model(model)
        return self._price(
            model, estimate_tokens(prompt_chars), math.ceil(output_tokens * 1.3), batch
        )

    def month_spend(self) -> float:
        month = datetime.now(UTC).strftime("%Y-%m")
        row = self.conn.execute(
            "select coalesce(sum(usd), 0) from ledger where substr(at, 1, 7) = ?", (month,)
        ).fetchone()
        return float(row[0])

    def _model(self, model: str | None) -> str:
        model = model or self.default_model
        if model not in PRICES:
            raise UnknownModel(f"unknown model {model!r}; known: {', '.join(PRICES)}")
        return model

    @staticmethod
    def _key(model: str, system: str, prompt: str, max_tokens: int) -> str:
        return hashlib.sha256(json.dumps([model, system, prompt, max_tokens]).encode()).hexdigest()

    @staticmethod
    def _price(model: str, tokens_in: int, tokens_out: int, batch: bool) -> float:
        price_in, price_out = PRICES[model]
        usd = (tokens_in * price_in + tokens_out * price_out) / 1e6
        return usd / 2 if batch else usd

    def _client_or_create(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _send(self, model: str, system: str, prompt: str, max_tokens: int) -> tuple[str, int, int]:
        # No sampling parameters: newer models restrict them, and the cache is
        # what makes re-runs identical.
        message = self._client_or_create().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return text, message.usage.input_tokens, message.usage.output_tokens

    def _send_batch(self, requests: list[dict]) -> dict[str, tuple[str, int, int]]:
        client = self._client_or_create()
        batch = client.messages.batches.create(
            requests=[
                {
                    "custom_id": r["custom_id"],
                    "params": {
                        "model": r["model"],
                        "max_tokens": r["max_tokens"],
                        "system": r["system"],
                        "messages": [{"role": "user", "content": r["prompt"]}],
                    },
                }
                for r in requests
            ]
        )
        while client.messages.batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(BATCH_POLL_SECONDS)
        out: dict[str, tuple[str, int, int]] = {}
        for entry in client.messages.batches.results(batch.id):
            if entry.result.type != "succeeded":
                raise RuntimeError(f"batch request {entry.custom_id} {entry.result.type}")
            message = entry.result.message
            text = "".join(block.text for block in message.content if block.type == "text")
            out[entry.custom_id] = (text, message.usage.input_tokens, message.usage.output_tokens)
        return out
