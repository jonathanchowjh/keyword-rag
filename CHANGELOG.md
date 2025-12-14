# Changelog

## Unreleased

- The console replaces the Eval Board: `/` lists collections with their answer,
  `/c/{name}` and `/c/{name}/runs/{id}` show one collection and one run, and
  `/worker` shows the queue and what needs a person. `/lab` stays. `/board`
  redirects to `/`.
- A worker thread inside the service runs loads, re-keys and evaluations one at
  a time. Jobs can be cancelled and stopped runs resumed (the estimate comes
  first); `GET /jobs`, `/jobs/{id}`, `/status` and `/catalog` are new.
- `PUT /collections/{name}/keyword-lists` now returns 202 with a `job_id` and
  queues a `rekey` job instead of re-keying in the request. `DELETE
  /collections/{name}` and `DELETE /eval/{id}` are new.
- Breaking: `POST /collections/{name}/keyword-lists/propose` now needs
  `confirm: true` to call Claude; without it, it returns the estimate
  `{est_usd, calls}`.
- `POST /collections/{name}/load` now returns `200 {estimate: {est_usd, items,
  calls}}` first whenever the load would spend (summaries, or a Claude keyword
  proposal because a key is set and there are no lists yet), and queues only
  with `confirm: true`. A load that spends nothing still returns 202 `{job_id}`.
- Known gap: The Lab still spends on answers, hops and query summaries without
  showing an estimate first; spend stays capped and ledgered.
- Every run has a plain-language answer, paired gaps against Hybrid and a log.
- The gate runs a browser smoke test of the pages (`pytest -m pages`).

- Two control methods, `random_example` and `fixed_example`, so a method can
  be checked against an example chosen without looking at the question.
- Dense queries on Qwen3-Embedding carry the query-side task instruction the
  model was trained with; fastembed does not add it. Eval runs record it.

## 0.1.0 — 2026-10-23

- The service: collections, ingestion, ten retrieval methods (the 2024 paper's
  seven and three modern ones), query, background evaluation with bootstrap
  intervals, and a monthly cost cap with a ledger and a response cache.
- Three QA types: ScienceQA (exam multiple choice), MuSiQue (multi-hop,
  HippoRAG-2 protocol) and TAT-QA (table + text).
- Docker: one image for the API plus Qdrant, via `docker compose up`.
- Helper pages: the Retrieval Lab and the Eval Board.
