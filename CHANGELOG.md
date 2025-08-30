# Changelog

## Unreleased

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
