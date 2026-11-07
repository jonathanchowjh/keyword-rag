# Changelog

## Unreleased

- Store routing: a store has a routing profile (`route_profile`, set with
  `PATCH /collections/{name}`; `POST /collections/{name}/route-profile/draft`
  drafts one: a Store line, the store's top 20 terms, a Sources line and 5 sample
  passages; `GET` returns
  `route_profile` and `route_profile_drafted`). A chat's new Routing setting
  (`all` or `profiles`, `PATCH /chats/{id}`) gives each store a route score, the
  higher of the cosine to its profile and to its average item (centroid of up to
  2,000 items), over the message and its search text as the gate's best match is,
  against a route bar calibrated like the gate's bar and recomputed when the
  profile changes. Reasons: "no profile" (no saved profile, an empty draft and no
  items) and "no route bar" (always routed: the gate still decides, so such a
  store can be "below its bar"), "not routed" (not searched), then the gate;
  sticky and "search anyway" still apply. Decisions gain `route` and `route_bar`.
  On the store page, Use the drafted profile fills the box with the draft; Save
  keeps it.
- Routing eval: `POST /routing/eval` and a Check routing panel score the gate and
  the profile router on the stores' own eval set questions (held out by default,
  dev and full selectable; up to 200 per set): recall, precision, none rate, mean
  searched and per-store recall. The panel checks the chat's stores (at least two)
  and says why a store adds no questions (no eval set, or a set of 200 or fewer,
  all dev slice); the API takes any list and names each store's `eval_set`. It
  uses embeddings only and runs in the request.
- Chat API: `PATCH /chats/{id}` refuses an explicit null for title, stores,
  evidence_size, rewrite or routing (422). The attempts body takes an optional
  `drop: {store: [field, ...]}` that removes held edits before the new ones merge;
  a store not on the chat is a 404.
- Roadmap: profiles and routing are built; a learned router is next.
- Chat across stores: chats are saved on the server (`/chats`) and search several
  stores. Each store has a bar, the 5th percentile of the best matches of its own
  default eval set's questions (dev slice, up to 200; BRIGHT exclusions and leaked
  questions left out), shown by `GET /collections/{name}/bar`; with fewer than 20
  usable questions or no eval set the store is always kept, and a bar you set wins.
  The gate scores the message alone and the follow-up search text and takes the
  higher. A follow-up is searched with the message plus the previous 2 user
  messages, or, with Rewrite follow-ups with AI on, a standalone question from the
  new `rewrite` job (one AI call per follow-up, never repeated by a re-run). A store
  the previous message kept on its own merit is kept again (one message back). Kept
  stores are merged by RRF (k = 60) by store and id, cut to the evidence size
  (default 8, 1 to 20).
- Chat: a retrieval step under each answer shows every store's decision and
  evidence. Edit a store's method, k, search settings or bar, or search it anyway,
  then re-run (adds an attempt without answering) and Answer from this attempt.
  Edits last until the service restarts; Set as default saves method, k, search
  settings and bar for the store ("search anyway" is not saved). No estimate before
  a message; each reply and attempt shows its cost and the budget cap holds. The old
  `POST /chat` still works; conversations kept only in a browser tab are not migrated.
- Imports: a `.txt` or `.md` upload (also inside a zip) longer than one passage is
  split into passages of up to 1,500 characters (`imports.chunk_text`): Markdown
  headings first, then paragraphs and sentences; code blocks, tables and lists are
  kept whole where they fit; each passage starts with a breadcrumb such as
  `notes.md › Setup › Install`, and one under 200 characters merges into a neighbour.
  Ids are `<file>#<i>` with `{file, part, parts}` in meta; a re-import deletes the
  file's old passages it no longer yields from the same source (the job result has
  `removed`). `/chat` citations carry `part` and `parts`, and the chat's Sources say
  `part i of n`. A file that fits in one passage is one item, as before.
- Imports: a long text file imported before passages were split is one item, `name`.
  Re-importing it now adds `name#1..n` and removes `name`, so eval-set evidence
  (`gold_ids`) that names `name` stops counting.
- Chat: a reply shows a safe subset of Markdown. The text is escaped first, then code
  blocks, `code`, bold, italic, `-` and `1.` lists, `#` headings (as bold lines) and
  paragraphs are drawn; links, images, tables and HTML stay as typed text, and `[n]`
  links work everywhere but in code.
- Chat: a page at `/chat` (side nav, under Stores) and `POST /chat` answer from one
  store's passages and cite them as `[n]` (or `[n, m]`, `[n-m]`), with sources under
  each reply, the method, the query tags and the cost. It retrieves on the latest
  question with `keyword_rag` when the store has a vocabulary (keyword lists are not
  needed), else `hybrid`, or the chosen method (422 for an unknown name, 409 when the
  store cannot run it); oracle tags become classifier tags. The last 6 turns are sent.
  It spends like the Lab: no estimate, the cost on each reply, the budget cap enforced.
  The conversation stays in the browser tab. Add document adds `.txt` and `.md` files to
  the store (uploads take `accept=text`); a source name is cleaned the same way in the
  store and eval set uploads.
- Stores and eval sets: a store (the API still says collection) is what is retrieved
  on; an eval set is a named, typed, versioned list of questions, with pages at
  `/eval-sets` and `/e/{name}` and endpoints under `/eval-sets`. A store's default
  eval set is its `eval_set`, else the set named like it. Loads write their questions
  into the default set; a copy points at the source's set and copies none. Every write
  saves a snapshot (12 hex digits of a SHA-256); a run records the version it first
  ran and always reads it, and the migration pinned existing runs. `/eval` takes
  `eval_set`. Evidence counts only where the store holds it (none held: recall is
  unmeasured, not 0), leaked questions are worked out per store and eval set at run
  setup, deleting a store keeps its eval sets, and deleting an eval set deletes the
  runs that used it after a confirm. Reloading a dataset upserts its questions by id,
  so it overwrites hand edits and restores deleted questions.
  `PATCH /collections/{name}` `{eval_set}` picks a store's default eval set (null clears
  it), also from a select on the store page. A store and an eval set of one name have
  one type: creating either is refused with 409 otherwise, and a load or question
  import on a store whose default set has another type is refused with 409 before
  anything is estimated or queued. A question added by hand is saved with
  `source: "hand"` and checked for leaks; a question needs an id. The eval set page
  waits for unsaved edits to be saved before it imports, and after a stale save offers
  to reload.
- Method v2: collections get a recipe (`worked_examples`, `reference`, `paper_2024`),
  chosen at creation or by copying a collection's items under another recipe. Build
  steps (induced vocabulary, AI tagging, abstracted examples) are estimated before
  anything is spent; a vocabulary is exported, edited and re-imported through the
  API.
- New method `keyword_rag`: meaning, BM25 and the question's predicted tags
  (`classifier`, `similarity`, `model` — the query job's AI picks them, so the run
  forms mark it "costs money" — or, in evals, `oracle`) fused by weighted RRF, with
  per-collection search settings (`PUT /collections/{name}/search`) that `/eval` and
  `/query` override in part. Answers can use abstracted or full examples, numbered
  best-first-and-last passages with `[n]` citations, and a locally run program for
  numeric questions (10 s, 512 MB, no new processes, not network-isolated; on by
  default for new science and table collections, off for older ones).
- Published builds: a load that has something to build, of a dataset, limit and
  recipe with an entry on the Hub (`RAGQA_BUILD_HUB`, default
  `jonathanmiro97/keyword-rag-builds`), applies the entry's vocabulary, tags and
  abstracted examples with no AI call. That happens only when the entry was built with
  the collection's build settings and covers exactly its items, each usable, which is
  checked before anything is added, and never replaces a vocabulary the collection already has. Otherwise the load builds locally. Either way its
  estimate and result say which (`hub`). A load that the entry made free fails,
  adding nothing, if the entry is no longer usable when it runs. Only the owner
  publishes, with `python -m ragqa_core.publish` (a dry run unless `--publish`).
  Authored outputs go into a collection with `python -m ragqa_core.apply_outputs`.
- Connections answered by hand (kind `manual`, preset "Answered by hand"): nothing is
  sent. A job that asks such a model something it has no reply for stops with the new
  resumable status `stopped_replies`; `python -m ragqa_core.replies export` writes the
  waiting prompts out and `import` records the replies, ledgered at the model's price.
  The job then resumes from them. A query or Lab method reports the wait as its error.
- Fixes from a full review: a question with no gold answer (BRIGHT's) is unmeasured in
  an answering run, not wrong; a resume is refused for changed keyword lists only when the
  run used a keyword method; the Run page shows a run's search settings and Re-run repeats
  them; a provider error in `/query` is that method's error, with what the others spent;
  a reply typed for a model answered by hand is never served as another connection's
  answer; `experiments.py report` marks a run that has not finished.
- A copy that stopped or failed resumes in place (`POST /jobs/{id}/resume`, or Resume on
  the Worker page), priced for what it left. While a sync add (20 items or fewer) runs,
  a job queued on that collection is refused (409), so a retag or rekey can't miss the
  new items.
- The Run page marks the answer of a run that stopped, failed or was cancelled as
  partial: it is from the questions the run reached.
- Build and summary estimates price only the calls not already cached, with the batch
  discount only when the uncached calls are enough to be sent as a batch, so a
  re-run or resumed build is priced at what it will spend (`AI.estimate_many`).
- The Lab shows Keyword-RAG's tags for the question, each hit's tags (lit where the
  question shares them) and the channels that found it.
- A model fetched or added by id starts at $0 instead of unpriced. Settings gives
  each model a price list ($0, the preset rates, no price, or typed by hand) and
  marks a $0 model, which the budget does not limit.
- BRIGHT (`bright_aops`, `bright_theoremqa`) loads, with nDCG@10 and recall@1 for
  questions that name gold documents.
- `scripts/experiments.py` and `scripts/experiments.md`: queue and compare
  experiments X1-X4. Nothing is queued without `--confirm`, nor any paid arm
  without `--paid`.
- The TF-IDF keyword lists are removed: with no AI model a paper-2024 load saves
  no lists, and the AI proposal returns all three.
- No local-model setup: the Ollama preset and the `local` flag that priced a
  connection's fetched models at $0 are removed, with the Settings text and the
  `host.docker.internal` mapping. Indexing your own corpus goes through a
  priced provider; fetched models start unpriced. A database made before this
  keeps its unused `local` column.
- Embedding runs 8 texts at a time, and a text that is its own only chunk is
  embedded once: a BRIGHT load no longer runs out of memory, and takes half as
  long.
- A cancel stops a run of one-at-a-time AI calls between sends.

- AI connections: a Settings page (`/settings`) and `/ai` routes set up any
  number of providers (Anthropic, OpenAI, Gemini, xAI, OpenRouter or any
  OpenAI-compatible server), their models and prices, the model each job
  uses (`answer`, `summary`, `keywords`, `query`) and the monthly cap. Keys are
  stored in the database file and never returned. A model with no price is
  refused before anything is sent. The reusable package is `aiconnect`.
- `ANTHROPIC_API_KEY`, `RAGQA_MODEL` and `RAGQA_MONTHLY_CAP_USD` still work,
  are applied at every start and are read-only on the page. An existing
  database keeps its spend, cached answers and pending batches; batches from
  before the upgrade are collected through the Anthropic connection.
- Provider-neutral wording: pages say "costs money" and "AI call" instead of
  naming a provider or "credit". `/health` returns `ai` instead of `api_key`.
  The 503 for a job with no model names the job and the Settings page.
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
- The Lab's cost per method counts every AI call that method made: the
  query summary and iterative hops as well as the answer.
- A confirmed eval or resume that would spend returns 503 at once when no AI
  model is set up for its job, as a load does, instead of
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
- Paid methods and the "Have the AI answer" box carry a muted "costs money" mark
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
  `confirm: true` to make the AI call; without it, it returns the estimate
  `{est_usd, calls}`.
- `POST /collections/{name}/load` now returns `200 {estimate: {est_usd, items,
  calls}}` first whenever the load would spend (summaries, or an AI keyword
  proposal because a model is set up and there are no lists yet), and queues only
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
