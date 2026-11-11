# Security

keyword-rag is a local test bench, not a hosted service.

Report a vulnerability privately to the repository owner rather than in a
public issue.

- **No login unless you set one.** Set `RAGQA_USERNAME` and `RAGQA_PASSWORD`
  together to put HTTP basic auth over every route. Setting only one makes
  every request a 503, so a half-configured deployment never runs open.
- **Localhost only by default.** `compose.yaml` publishes the API on
  `127.0.0.1:8080` and does not publish Qdrant at all. Anyone who can reach the
  port can spend on your AI provider's account up to the monthly cap, so widen
  the binding only behind the basic auth above.
- **API keys** are kept one per connection, for any provider, stored as
  described under [What is stored where](#what-is-stored-where). They are never
  logged or returned by the API. Give each key its own spend limit at the provider.
- **Program answers run code.** With the `program` search setting, a question
  without choices is answered by a short Python program the model writes. The
  program runs in a subprocess with time, memory, file and output limits, no
  child processes or threads, and no secrets in its environment. It is not
  network-isolated, it can read what the service can read, the stored API
  keys included, and it can write or delete what the service can write. Turn it on only for stores whose documents you trust.

## What leaves the machine

- **Prompts.** Prompts, with the passages they quote, go to the AI provider of the model each job uses: Anthropic, OpenAI, Gemini, xAI, OpenRouter, or a custom URL. An "Answered by hand" connection sends nothing.
- **Hugging Face.** Downloads come from Hugging Face: embedding and rerank models, published builds, and the benchmark datasets. `HF_TOKEN` is sent when set.
- **TAT-QA.** Its files are fetched from their published URL, on raw.githubusercontent.com.
- **Nothing else.** Nothing else calls out.

## What is stored where

- **Plaintext keys.** API keys entered on the Settings page are stored in plaintext in `ragqa.db` (`ai_connections.api_key`) in the `rag-qa_ragqa` volume (compose's `ragqa`). Anyone who can read the volume can read them. A key set through the environment is written there too.
- **The rest.** Uploads, stores' text, chats and runs are in that volume too, and vectors are in `rag-qa_qdrant` (compose's `qdrant`). See [Back up, reset, upgrade](REFERENCE.md#back-up-reset-upgrade).
