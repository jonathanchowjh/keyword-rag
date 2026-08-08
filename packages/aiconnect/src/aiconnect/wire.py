"""Provider request code, as bare functions. Provider names live here and in presets.py.

CLIENT, SEND and BATCH are looked up by connection kind. A kind with no BATCH
entry sends one request at a time.
"""

import anthropic


def anthropic_client(base_url: str | None, api_key: str | None) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key, base_url=base_url)


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


CLIENT = {"anthropic": anthropic_client}
SEND = {"anthropic": anthropic_send}
BATCH = {
    "anthropic": {
        "submit": anthropic_submit_batch,
        "done": anthropic_batch_done,
        "results": anthropic_batch_results,
        "cancel": anthropic_cancel_batch,
    }
}
