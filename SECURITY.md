# Security

keyword-rag is a local test bench, not a hosted service.

- **No login unless you set one.** Set `RAGQA_USERNAME` and `RAGQA_PASSWORD`
  together to put HTTP basic auth over every route. Setting only one makes
  every request a 503, so a half-configured deployment never runs open.
- **Localhost only by default.** `compose.yaml` publishes the API on
  `127.0.0.1:8080` and does not publish Qdrant at all. Anyone who can reach the
  port can spend on your AI provider's account up to the monthly cap, so widen
  the binding only behind the basic auth above.
- **API keys are stored in the database** (`ragqa.db` in the `ragqa` volume),
  one per connection, for any provider. They are never logged or returned by
  the API. Give each key its own spend limit at the provider.
- **Program answers run code.** With the `program` search setting, a question
  without choices is answered by a short Python program the model writes. The
  program runs in a subprocess with time, memory, file and output limits, no
  child processes or threads, and no secrets in its environment. It is not
  network-isolated, it can read what the service can read, the stored API
  keys included, and it can write or delete what the service can write. Turn it on only for stores whose documents you trust.

Report a vulnerability privately to the repository owner rather than in a
public issue.
