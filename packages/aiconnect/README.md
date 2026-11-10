# aiconnect

One `AI` class over stored provider connections, with prices, a monthly spend
cap, a response cache, a spend ledger and batches. Everything lives in one
SQLite file you choose. It depends on nothing else in this repo, so another
project can use it on its own.

## Installing it

It is a member of this repo's uv workspace. In the workspace, `uv sync` installs
it. In another project, add it as a path dependency:

```sh
uv add --editable /path/to/this-repo/packages/aiconnect
```

It needs Python 3.12 and brings the `anthropic` and `openai` clients.

## Using it

```python
from pathlib import Path
from aiconnect.ai import AI

ai = AI(Path("my-app.db"))
ai.add_connection("anthropic", "anthropic", "Anthropic", api_key="sk-ant-...")
ai.add_model("anthropic", "claude-haiku-5-5", price_in=0.10, price_out=0.50)
ai.set_role("default", "anthropic/claude-haiku-5-5")

reply = ai.complete("Name three noble gases.", purpose="demo")
print(reply.text, reply.usd, reply.cached)

replies = ai.complete_many(["Name a noble gas.", "Name a halogen."], purpose="demo")
print(ai.estimate_many(["Name a metal.", "Name a gas."], 200))  # USD, cached prompts free
```

- `complete(prompt, *, role="default", model=None, system="", max_tokens=1024,
  purpose, cache=True)` returns a `Completion` (`text`, `usd`, `cached`,
  `input_tokens`, `output_tokens`). `purpose` is required; it is written to the
  ledger.
- `complete_many(prompts, ...)` takes the same options and returns one
  `Completion` per prompt. Past 50 uncached prompts, a connection whose provider
  has a Batch API sends them as a batch at half price.
- `estimate_many(prompts, max_tokens, *, role, model, system)` prices what
  `complete_many` would spend; `estimate_usd(prompt_chars, output_tokens)` prices
  one call.
- A model ref is `"<connection id>/<model id>"`, split on the first slash.
  Connection kinds are `anthropic`, `openai` (any OpenAI-compatible endpoint, with
  a `base_url`) and `manual`.

## Pricing and the cap

Prices are USD per million tokens, set per model with `add_model` or
`update_model`. A model added without prices is $0, so its calls are free to the
budget; pass `price_in=None` to leave it unpriced, and calls are refused with
`UnknownPrice` until it is priced.

`set_cap(usd)` sets the monthly cap (`DEFAULT_CAP_USD`, 20, until you do). Before
each uncached call the layer adds what the month has spent, what is in flight and
what this call would cost, and raises `BudgetExceeded` if the sum passes the cap.
`month_spend()` reports the ledger. The cap only sees calls made through this
layer; the provider's own spend limit is the real ceiling.

## The manual connection

A `manual` connection sends nothing. A prompt its cache does not hold raises
`AwaitingReplies` and waits; `ai.waiting()` lists the waiting prompts, and
`ai.answer({key: reply})` records replies as if the model had sent them, cached
and ledgered at the model's price. This lets a person, or an agent session, be the
model.

Other errors are the package's own: `NoConnection`, `UnknownModel`,
`ProviderError`, `Conflict`, `Cancelled`.
