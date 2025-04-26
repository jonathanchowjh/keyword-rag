# Security

rag-qa is a local test bench, not a hosted service.

- **No login unless you set one.** Set `RAGQA_USERNAME` and `RAGQA_PASSWORD`
  together to put HTTP basic auth over every route. Setting only one makes
  every request a 503, so a half-configured deployment never runs open.
- **Localhost only by default.** `compose.yaml` publishes the API on
  `127.0.0.1:8080` and does not publish Qdrant at all. Anyone who can reach the
  port can spend Claude credit up to the monthly cap, so widen the binding only
  behind the basic auth above.
- **The API key** is read from `ANTHROPIC_API_KEY` and never logged or
  returned. Use a key from a Console workspace with its own spend limit.

Report a vulnerability privately to the repository owner rather than in a
public issue.
