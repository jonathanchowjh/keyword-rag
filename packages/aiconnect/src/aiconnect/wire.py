"""Provider request code, as bare functions. Provider names live here and in presets.py.

CLIENT, SEND and BATCH are looked up by connection kind. A kind with no BATCH
entry sends one request at a time.
"""

import math
from typing import Any
from urllib.parse import urlparse

import anthropic
import openai


def estimate_tokens(chars: int) -> int:
    # ~4 characters a token, +30% for the newer tokenizer.
    return math.ceil(chars / 4 * 1.3)


def anthropic_client(base_url: str | None, api_key: str | None) -> anthropic.Anthropic:
    # With no key the SDK would send the process's own key to this connection's address.
    return anthropic.Anthropic(api_key=api_key or "unused", base_url=base_url)


def anthropic_send(
    client: anthropic.Anthropic, model_id: str, system: str, prompt: str, max_tokens: int
) -> tuple[str, int, int]:
    # No sampling parameters: newer models restrict them, and the cache is
    # what makes re-runs identical.
    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    return text, message.usage.input_tokens, message.usage.output_tokens


def anthropic_submit_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
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
    return batch.id


def anthropic_batch_done(client: anthropic.Anthropic, batch_id: str) -> bool:
    return client.messages.batches.retrieve(batch_id).processing_status == "ended"


def anthropic_batch_results(
    client: anthropic.Anthropic, batch_id: str
) -> dict[str, tuple[str, int, int] | None]:
    """custom_id -> (text, input tokens, output tokens), or None if not succeeded."""
    out: dict[str, tuple[str, int, int] | None] = {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type != "succeeded":
            out[entry.custom_id] = None
            continue
        message = entry.result.message
        text = "".join(block.text for block in message.content if block.type == "text")
        out[entry.custom_id] = (text, message.usage.input_tokens, message.usage.output_tokens)
    return out


def anthropic_cancel_batch(client: anthropic.Anthropic, batch_id: str) -> None:
    client.messages.batches.cancel(batch_id)


def anthropic_list_models(client: anthropic.Anthropic) -> list[str]:
    return [m.id for m in client.models.list(limit=1000)]


def openai_client(base_url: str | None, api_key: str | None) -> openai.OpenAI:
    # A local server needs no key, but the SDK refuses to start without one.
    return openai.OpenAI(base_url=base_url, api_key=api_key or "unused")


def openai_send(
    client: openai.OpenAI, model_id: str, system: str, prompt: str, max_tokens: int
) -> tuple[str, int, int]:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    # Hosted OpenAI models reject max_tokens; other servers know only that one.
    # An unset base_url is the SDK default, which is OpenAI's host.
    hosted = urlparse(str(client.base_url)).hostname == "api.openai.com"
    limit: dict[str, Any] = {"max_completion_tokens" if hosted else "max_tokens": max_tokens}
    response = client.chat.completions.create(
        model=model_id,
        **limit,
        messages=messages,  # type: ignore[arg-type]
    )
    text = response.choices[0].message.content or ""
    if response.usage is None:  # some servers report no usage
        return text, estimate_tokens(len(system) + len(prompt)), estimate_tokens(len(text))
    return text, response.usage.prompt_tokens, response.usage.completion_tokens


def openai_list_models(client: openai.OpenAI) -> list[str]:
    return [m.id for m in client.models.list()]


CLIENT = {"anthropic": anthropic_client, "openai": openai_client}
SEND = {"anthropic": anthropic_send, "openai": openai_send}
LIST_MODELS = {"anthropic": anthropic_list_models, "openai": openai_list_models}
BATCH = {
    "anthropic": {
        "submit": anthropic_submit_batch,
        "done": anthropic_batch_done,
        "results": anthropic_batch_results,
        "cancel": anthropic_cancel_batch,
    }
}
