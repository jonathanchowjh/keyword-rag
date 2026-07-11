# Changelog

## Unreleased

- Your own corpus: upload JSONL, JSON, CSV, TSV, text, Markdown or a zip of
  them as items or questions, with a preview, a field mapping and an estimate
  before anything is queued. Each import is a source (`upload:<name>`) that can
  be switched off for retrieval or deleted with its items, chunks and
  questions. Re-importing updates by content hash instead of duplicating.
- A leakage check flags imported questions whose text is in the items; runs
  leave them out unless included, and the Run page says how many.
- Partial gold: recall is measured only on questions that name their evidence,
  and the Run page reads "on n of N questions". It also lists the sources
  switched off and the evidence switched off. "What changed since" reports
  sources.
- A load that died resumes by content hash: "Finish loading" embeds only what
  is missing. Stale chunks of a changed item are removed.
- Existing collections are labelled with their sources in place on first start.
- In an answering run, the Questions view judges each method only on its answer:
  a method not answered yet (a run stopped at the cap) is unmeasured, not a
  disagreement. "What changed since" notes keyword lists only for runs that used
  a keyword method, and also when the lists were set after a run that had none.
- A load that makes the keyword lists reports progress while it re-keys (adding
  is the first half) and can be cancelled then; the old lists stay in force.
- The Lab's cost per method counts every Claude call that method made: the
  query summary and iterative hops as well as the answer.
- A confirmed eval or resume that would spend returns 503 `needs
  ANTHROPIC_API_KEY` at once when no key is set, as a load does, instead of
  queueing a run that stops straight away.
- `POST /collections/{name}/documents` is also refused while a load on the
  collection is queued or running. Saving keyword lists is refused with 409
  when the service's dense model differs from the collection's.
- The worker's error clears after its next clean cycle instead of staying until
  a restart. The stall warning says how long the job has been quiet ("for N
  minutes") instead of a server clock time.
- Runs left `running` by a stopped service from before version tracking are
  marked untracked when the database opens, so resuming them is refused.
- Tables name their answer metrics "Exact answer" and "Answer overlap"; both
  were "Correct answer".
- `GET /collections/{name}/keyword-lists` (every version, newest first, with
  counts and which is active) and `GET .../keyword-lists/{version}` are new. The
  Collection page lists them and can load an old version into the editor. A
  version is a hash of the lists' content, so saving an old version unchanged
  reuses its version, moves it to the top of the history and re-keys as usual.
- Paid methods and the "Have Claude answer" box carry a muted "uses credit" mark
  in the run form and the Lab. A keyword method carries it only on a collection
  with summaries.
- A load job's log shows its progress at each 10% step, as a re-key's does, and a
  job that loads a model logs "loading model X" and "model X ready". The sidebar
  shows the first worker problem under the dot, with "and N more", linking to
  /worker. The Lab ignores a method list that arrives after the collection
  choice has moved on.
- `GET /eval/{id}` and the `answer` of `GET /collections/{name}` carry `since`
  (`{untracked, changes}`): what changed after the run (keyword lists, dense
  model), or untracked for a migrated run. The Run page head says so, and the
  Collection page reads it instead of fetching the job.
- `GET /eval/{id}/questions?disagree=` is new: each question with every
  method's row and outcome. The Run page's Questions section opens a question
  to every method's hits side by side and can show only where methods disagree.
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
- `POST /collections/{name}/documents` is refused with 409 while a re-key or an
  eval on the collection is queued or running, so the items cannot change under it.
- Runs from before version tracking cannot be resumed; the resume estimate
  says so instead of guessing.
- Question only rows carry no retrieval scores, since it retrieves nothing:
  absent means unmeasured, never 0. Older rows that stored 0.0 are cleaned
  when the database is opened after upgrading.
- Reasoning match is shown on the rescaled scale in each method's own score and
  in every pair, as in the gaps; counts have thousands separators, the Lab shows
  cost to the cent, and the verdict names the metric the collection's type uses.
- Two control methods, `random_example` and `fixed_example`, so a method can
  be checked against an example chosen without looking at the question.
- Dense queries on Qwen3-Embedding carry the query-side task instruction the
  model was trained with; fastembed does not add it. Eval runs record it.
- Science problems are a QA type and the console's default: worked examples are
  retrieved, and an answer is right when the choice matches, a number is within
  5% of the gold, or a short text matches exactly.
- SciBench (university physics and chemistry) loads into a science collection as 273
  examples (76 worked solutions) and 219 questions, split by a hash of the id,
  and "Set up the four benchmark collections" creates it as `sb`.
- The dev slice ranks each question by a hash of its id, so adding a question
  displaces at most one. New dev runs on an existing collection may use a
  different 200 than before; runs already recorded keep their question ids.
- GPQA as a second, harder exam dataset (`"dataset": "gpqa"`): Diamond's
  questions over the rest of Extended, which carries expert explanations.
  Gated: needs `HF_TOKEN` after accepting its terms.
- A run's `dataset_filter` names the datasets its collection actually loaded,
  not a note picked by QA type.

## 0.1.0 — 2026-10-23

- The service: collections, ingestion, ten retrieval methods (the 2024 paper's
  seven and three modern ones), query, background evaluation with bootstrap
  intervals, and a monthly cost cap with a ledger and a response cache.
- Three QA types: ScienceQA (exam multiple choice), MuSiQue (multi-hop,
  HippoRAG-2 protocol) and TAT-QA (table + text).
- Docker: one image for the API plus Qdrant, via `docker compose up`.
- Helper pages: the Retrieval Lab and the Eval Board.
