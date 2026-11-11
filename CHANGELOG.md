# Changelog

Newest first. "(breaking)" marks a change a client of the API may need to act on; each release's "Upgrading from" section lists what to change.

## 0.3.0 — 2026-11-12

### Added

- REFERENCE.md says how to back up, reset and upgrade, naming the two volumes; SECURITY.md says what leaves the machine and that API keys are stored in plaintext.
- REFERENCE.md has Statuses and errors (every status the API answers, with an example) and Job statuses (which can resume), a Documents import by curl that works, and the request fields' real names.
- GET /ai lists each model's `usable` and `ready`.
- GET /jobs, /jobs/{id} and /status rows carry label, resumable and href.
- GET /collections/{name} returns busy: true while a job is queued or running on the store, or its rebuild's switch has stopped part-way.

### Changed

- The eval-set page strikes a removed question through with Undo, counts the questions Save would keep, and asks before you leave with unsaved changes. Run against its store waits for a first question.
- Every page fits a 400 px screen: long selects and the New store name box shrink, wide tables scroll, and the narrow nav leaves out the child links.
- Passages keep their line breaks and show headings in bold, and an opened passage shows its text once.
- Changed (breaking): an AI connection or model that doesn't exist answers 404 on the routes that name it in the path (it answered 422); POST /eval with an eval_set that doesn't exist answers 404.
- Changed (breaking): POST /ai/connections/{id}/models answers 201.
- Changed (breaking): adding documents is refused with 409 while a rebuild's switch has stopped part-way, as REFERENCE.md said.
- Changed (breaking): a job in the way reads "a <kind> job is queued or running on '<name>'; wait", and a running eval "an eval is running on '<name>'; wait".
- Changed (breaking): GET /status lists each problem as {text, href}, not a string, and names the job as the Worker page does ("Evaluating wl (ab12cd34) failed").
- Changed (breaking): resuming a load, import, re-key, re-tag or labelling job says "this <job> cannot be resumed; start it again" instead of "only evals can be resumed".
- The Worker page names every kind of job ("Importing into docs", "Re-tagging w") and links every problem to its page.
- Changed (breaking): PATCH /chats/{id} with a null title or stores answers 422 with a plain-text `detail`, as the other routes do, instead of a list.
- Chat's Add document is now Add file…: it names the store the file is added to, lists Documents stores first, and says what it added beside the button.
- Changed (breaking): a request whose job's model is set up and usable but unpriced is refused with 422 `<connection>/<model> has no price; enter one first` where it would spend (proposing keyword lists, saving a vocabulary and requests checked for a model first answered 503; a disabled or keyless unpriced model is 503 everywhere, adding keyword tags included); a load that would only propose keyword lists skips the proposal, and chat's attempt-answer route puts it in the attempt's `error`.
- Changed (breaking): an unconfirmed spending request answers 200 with its estimate when no model is usable, where load, copy, resuming a copy or tag build, Add keyword tags, proposing keyword lists and saving a vocabulary answered 503; every estimate lists the roles it would call (`roles`) and those with no usable model (`missing`); only a confirm is refused with 503, and the keywords refusal is the usual "no AI model is set up for the 'keywords' job" text. A load that summarises, on a store that doesn't build and with no calls or est_usd sent, still answers 503 before it downloads.
- Changed (breaking): an unconfirmed load that would queue because it is free is refused with 409 when the `calls` or `est_usd` it sends are stale, as adding documents already was.
- OpenAPI documents the estimate a spending route answers without confirm.
- scripts/experiments.py confirms with the calls and price it showed, and stops (exit 3) when the price changed.
- The Hub dataset card says keyword-rag; the link still points at the rag-qa repository.
- Changed (breaking): a Documents run that sends no rerank_pool reranks up to Settings' Rerank up to (at most 50), where it took 50; the run form no longer sends k, rrf_k or rerank_pool.
- GET /eval/{id} names the reranker and pool of a hybrid_rerank run recorded before they were stored.
- The Lab's Items per method accepts up to 50, as /query does, and Settings' RRF k says it applies to chat and /query.
- A job stopped for replies names the export command with its path.
- An empty Documents store says to add documents, and one with no labelled questions says to label some, instead of asking for a run that can't start; Add keyword tags waits for passages; Save question waits for Find passages and asks before saving a question with no right passage.
- Chat lists a reply's evidence best match first; each passage keeps the [n] the answer cites.
- Documents stores no longer offer benchmark-only options (controls, multiple-choice choices, keyword-list methods); a Documents run lists each question's right passages and puts unscored questions last.
- Changed (breaking): a chat whose answer model is missing, deleted, disabled or on a connection with no key (an answer role pointing at a deleted model included) answers with passages only and no error; with such a rewrite or query model, follow-ups are joined, not rewritten, and AI methods say they need AI. A usable but unpriced model is still refused.
- Start reads Connect AI first, linking to Settings, when what it would run needs a model that isn't usable; prices read the same everywhere (chat no longer shows "≈ < $0.01"); a refused Start or Resume shows why beside it, and only a changed price takes it away.
- Changed (breaking): importing into a store whose rebuild's switch stopped part-way is refused with 409 "the switch to <model> stopped part-way; resume it first", as adding documents is.
- A stopped copy's problem line says it can be resumed.
- The pages name the retrieval metrics plainly, with the code beside them: Ranking quality (nDCG@10), Found in top 10 (Recall@10), First right passage (MRR@10). A verdict reads "Can't tell yet: Keywords per list vs Hybrid, +0 pts (range −3 to 3)", and the store page's runs column is Result.

### Fixed

- The eval-set page's versions read "1 question · View".
- The store page no longer polls every 3 seconds while a rebuild's switch is stopped.
- Settings' Test, Fetch models, Save key and Save prices stay disabled until their request answers, so a double click can't test (and pay) twice.
- A refused request's reason shows on the page instead of "Unprocessable Entity".
- A store, eval set, run or chat that doesn't exist shows "No store named x." and a way back, instead of an empty page with live buttons; an unknown page is an HTML page for a browser.
- .env.example sets nothing, so copying it no longer pins the monthly cap on Settings; the commands for answering by hand write where the service can, and say how to copy the files in and out.
- On a store made from gpqa, bright_aops or bright_theoremqa, Finish loading loads that dataset, not the first one of its type; a new store's load card lets you pick the dataset.
- The run page's question picker now defaults to the first method that retrieves, not the no-retrieval baseline.
- After a refusal because the price changed, the page takes away the old Start, so a fresh click prices it again instead of resending the old price.
- GET /collections/{n} on a Documents store whose scored run's eval-set version is gone shows the next run's score, or none, instead of 404; the store page loads again.
- The Worker page no longer says a vocabulary is partly re-tagged when its retag stopped before writing any tags.
- A model on a connection with no key is no longer offered in the pickers, and Settings marks it `(no key)`.
- Importing several files when no summary model is usable disables Import and says Connect AI first, as one file does.

### Upgrading from 0.2.0

- Expect 404, not 422, for an AI connection or model that doesn't exist on the routes that name it in the path, and for `POST /eval` with an unknown `eval_set`.
- Expect 201 from `POST /ai/connections/{id}/models`.
- Expect 409 when adding documents or importing into a store whose rebuild's switch stopped part-way; resume the switch first.
- If you match job-in-the-way messages by text, match "a <kind> job is queued or running on '<name>'; wait" and "an eval is running on '<name>'; wait".
- Read `GET /status` problems as `problems[].text` (with `href`), not as strings, and expect the job named as the Worker page names it.
- If you match the resume refusal, expect "this <job> cannot be resumed; start it again" for loads, imports, re-keys, re-tags and labelling.
- Read `detail` as plain text on `PATCH /chats/{id}` 422 (null title or stores), not as a list.
- Expect 422 `<connection>/<model> has no price; enter one first` for a usable but unpriced model where a request would spend (it was 503 for proposing keyword lists, saving a vocabulary and checks for a model); a disabled or keyless unpriced model is still 503, and chat's attempt-answer route puts the refusal in the attempt's `error`.
- Expect 200 with an estimate from an unconfirmed spending request when no model is usable (it was 503); read `roles` and `missing` on the estimate, and expect 503 only on a confirm.
- Send current `calls` and `est_usd` with an unconfirmed load that would queue because it is free; stale ones are refused with 409.
- Send `rerank_pool` on a Documents run if you need 50: without it the run reranks up to Settings' Rerank up to (at most 50); the run form no longer sends `k`, `rrf_k` or `rerank_pool`.
- Expect a chat with a missing, deleted, disabled or keyless answer model to answer with passages only and no error; with such a rewrite or query model, follow-ups are joined, not rewritten, and AI methods say they need AI.

## 0.2.0 — 2026-11-11

Lines marked breaking are changes a client of 0.1.0 can notice. Routes, fields and settings are described in [REFERENCE.md](REFERENCE.md).

### Upgrading from 0.1.0

- Back up the database file first: the first start on an old database migrates it once, one way.
- compose.yaml now pins `name: rag-qa`, so the folder name no longer picks the volumes. If you ran a plain `docker compose up` from a folder named keyword-rag, your data is in the `keyword-rag_ragqa` and `keyword-rag_qdrant` volumes: run `docker compose -p keyword-rag up` to reach it, or copy it into `rag-qa_ragqa` and `rag-qa_qdrant`. A clone into `rag-qa` (the README's `cd rag-qa`) is unaffected.
- Chats: the chat bar, sticky stores, routing profiles, `store_bars` and the routing chat setting are dropped; a chat's saved evidence size of 8 and rewrite off become blank (inherit); each store's `chat_k` is cleared, because it now means candidates per store. The migration writes the marker `schema.routing_dropped`; a later start only drops leftovers and never resets settings again. The `chat_bar` and `route_profile` columns stay in old databases, unused.
- Old chats and attempts show their decisions and evidence as recorded, and re-running an old message keeps its search text, with no paid rewrite.
- `PATCH /collections/{n}` and `PATCH /chats/{id}` now refuse the removed fields with 422 (`route_profile`, `routing`).
- Eval sets: the migration made an eval set for every collection that had questions and pinned every existing run to that snapshot. A migrated run that recorded no versions is `untracked: true` in `since`.
- Batches submitted before the upgrade are collected through the Anthropic connection: keep the Anthropic key (the env key, or one added on the Settings page) until they finish. The ledger, cached answers and this month's spend carry over.
- The old TF-IDF keyword lists are gone: a paper-2024 load with no AI model saves no lists, and the store reads `no_lists`.
- New stores embed with `BAAI/bge-small-en-v1.5`; existing stores keep their model. To go back, set Settings → Search defaults → Embedding model for new stores, or `RAGQA_DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B-Q`.
- The default reranker is `jinaai/jina-reranker-v1-tiny-en`; to go back, pick `BAAI/bge-reranker-v2-m3-int8` in Settings → Search defaults → Reranker model, or set `RAGQA_RERANK_MODEL`.
- Follow-ups are rewritten by default whenever an AI is connected (one paid call per follow-up); Settings → Search defaults → Rewrite follow-ups → Off turns it off.
- A long `.txt` or `.md` imported before passages were split is one item, `name`; re-importing it adds `name#1..n` and removes `name`, so eval-set `gold_ids` naming `name` stop counting. Stores loaded before content ids keep their old ids until each file is added again.
- The Ollama preset and the `local` connection flag are removed; a database made before keeps its unused `local` column.

### Added

- Stores and eval sets: an eval set is a named, typed, versioned list of questions (pages `/eval-sets` and `/e/{name}`, endpoints under `/eval-sets`); a store's default is its `eval_set`, else the set named like it; every write saves a version (12 hex digits of a SHA-256), a run records the version it first ran and always reads it, and `/eval` takes `eval_set`.
- `PATCH /collections/{name}` sets a store's default `eval_set`, and its chat `chat_method` and `chat_k` (null inherits).
- Eval sets can be linked to a store (`store` on `POST` and `PATCH /eval-sets`), and `GET /eval-sets/{name}` returns `store`, `missing`, `questions_version` and `answer_available`; **Create one** on a store page opens the form with the store and type filled in.
- Per-question writes `POST`, `PUT` and `DELETE /eval-sets/{name}/questions[/{id}]` (a stale `PUT` is 409, a deleted question 404), `GET /eval-sets/{name}?version=`, and a versions list with **View**, **Show all versions** and **Restore this version**.
- Documents stores: the New store form defaults to Documents (`qa_type` `documents`, recipe `documents`, no AI), and `POST /collections` with no `recipe` takes the type's default.
- Documents eval sets are built by labelling: questions have `judged_ids` (always a superset of `gold_ids`), **Find passages** (`POST /eval-sets/{name}/pool`) pools the top 10 of each free method reranked with the Settings reranker, and **Run against its store** opens the run form with the set chosen.
- Re-importing into a Documents set: a file carrying `judged_ids` replaces a stored question whole; one that does not keeps the stored judged passages and answers and joins the gold ids. `gold_answers` reads as an `answer` synonym.
- **Get the AI's answer** on the labelling panel (`POST /eval-sets/{name}/answer`) prices first, then makes one cached call from the passages marked right; nothing is stored until **Save question**.
- Documents runs score Recall@10, MRR@10 and nDCG@10 at depth 10, headed by nDCG@10, and flag a method whose top 10 averaged more than 3 never-judged passages (`score unreliable`); `GET /eval/{id}` returns `labels` and `answers`.
- A Documents run may have the AI answer (**Have the AI answer the questions**) over each method's top evidence-size passages, fixed in `setup.evidence_size` at queue time, and **Judge answers** (`judge: true` on `POST /eval`) marks each answer right or wrong against the reference answer, recorded as `setup.answer_judge_model` with the raw reply in `judge_raw`.
- A Documents resume prices the answers and verdicts it can rebuild from the AI cache at nothing and the rest in full; the run form says how many questions have a reference answer, and the estimate returns `references` and `evidence_size`.
- Add keyword tags (`POST /collections/{name}/keyword-tags`, a button on the store page): priced first, then a `tag_build` job (state `tagging`) switches a Documents store to `documents_tagged`; Resume finishes a stopped one, and adding passages to a tagged store is free. `GET /collections/{name}` returns `keyword_tags`.
- Content ids: a `.md` or `.txt` passage in every store is named `<file>#<12 hex>` from its text, with `meta` `{file, part, parts}`; re-adding a file embeds only new or changed passages and deletes the old ones.
- Imports split a `.txt` or `.md` upload longer than one passage into passages of up to 1,500 characters (headings first, then paragraphs and sentences; code, tables and lists kept whole), each with a breadcrumb such as `notes.md › Setup › Install`; `/chat` citations carry `part` and `parts`.
- Rebuild with another model: `POST /collections/{name}/reembed` queues a free `reembed` job that embeds the store aside (`<name>__rebuild`) and switches; `GET /collections/{name}` returns `rebuild`, and `POST /jobs/{id}/resume` resumes a stopped one.
- `POST /collections` takes `dense_model`; `GET` and `PUT /search-defaults` carry `dense_model` and `embedders`; a copy keeps its source's model, and each store embeds questions with its own model.
- Chats are saved on the server (`GET` and `POST /chats`, `GET`, `PATCH` and `DELETE /chats/{id}`) and search several stores; the chat page lists them on the left.
- Chat pipeline: each store returns its top candidates by its own method, the lists are pooled round-robin by rank, reranked against the standalone question or the message alone, and cut to the evidence size; with the reranker loading or failed the pool is merged by RRF and the retrieval step says so.
- Chat: a retrieval step under each answer shows the search text, the rerank model, the pool and every store's candidates and evidence; Re-run adds an attempt without answering, and Answer from this attempt answers it (once, even when asked twice).
- Chat from your own app: `POST /chats` takes `{from}`, `POST /chats/{id}/messages` takes `?brief=true` for a short reply and `"answer": false` for the reranked passages with no answer paid for (`pipeline.answer_skipped`); the chat page has **Use from your app** with curl lines, each with a Copy button that says where the login comes from.
- Chat works without AI: Send returns the passages, marked `Passages only: no answer model`.
- A chat reply shows a safe subset of Markdown (code, bold, italic, lists, headings, paragraphs); links, images, tables and HTML stay as typed text.
- A chat page at `/chat` and `POST /chat` answer from one store's passages with `[n]` citations, sources under each reply, the method, the query tags and the cost.
- Settings → Search defaults (`GET` and `PUT /search-defaults`): the default method, candidates per store, evidence size, rerank, rewrite, previous messages joined, RRF k, reranker model and Rerank up to, each with its recommended value and Reset to recommended; the store page shows its chat settings with Reset to Settings.
- Chat settings are saved with the chat and inherit: chat, then store, then Settings, then the recommended value; `POST /chat` and `POST /query` take their defaults from Settings.
- The reranker warms in the background at start and when Settings picks another; `GET /search-defaults` returns `rerank_status` and `rerank_error`.
- `POST /eval` takes `rerank_pool` (1 to 50) and `rerank_model`; "Hybrid + reranker" runs record `setup.rerank_model`, and Re-run repeats the model and pool.
- Connections answered by hand (kind `manual`): a job with no reply stops with the resumable status `stopped_replies`, and `python -m ragqa_core.replies export` and `import` write the waiting prompts out and record the replies.
- Method `keyword_rag`: meaning, BM25 and the question's predicted tags fused by weighted RRF, with per-collection search settings (`PUT /collections/{name}/search`); answers can use abstracted examples, numbered passages with `[n]` citations and a locally run program for numeric questions.
- Method v2: collections get a recipe (`worked_examples`, `reference`, `paper_2024`), and build steps (induced vocabulary, AI tagging, abstracted examples) are estimated before anything is spent.
- Published builds: a load whose dataset, limit and recipe have a Hub entry (`RAGQA_BUILD_HUB`) applies its vocabulary, tags and examples with no AI call, and the estimate and result say so (`hub`); only the owner publishes, with `python -m ragqa_core.publish`.
- Your own corpus: upload JSONL, JSON, CSV, TSV, text, Markdown or a zip as items or questions, with a preview, a field mapping and an estimate; each import is a source (`upload:<name>`) that can be switched off or deleted, and re-importing updates by content hash.
- A leakage check flags imported questions whose text is in the items, and recall is measured only on questions that name their evidence (the Run page reads "on n of N questions").
- AI connections: a Settings page (`/settings`) and `/ai` routes set up any number of providers, their models and prices, the model each job uses (`answer`, `summary`, `keywords`, `query`) and the monthly cap; the reusable package is `aiconnect`.
- Settings gives each model a price list ($0, the preset rates, no price, or typed by hand) and marks a $0 model; a model fetched or added by id starts at $0.
- `ANTHROPIC_API_KEY`, `RAGQA_MODEL` and `RAGQA_MONTHLY_CAP_USD` still work, are applied at every start and are read-only on the page.
- A worker thread runs loads, re-keys and evaluations one at a time; jobs can be cancelled and stopped runs resumed (the estimate comes first); `GET /jobs`, `/jobs/{id}`, `/status` and `/catalog` are new.
- The console replaces the Eval Board: `/` lists collections with their answer, `/c/{name}` and `/c/{name}/runs/{id}` show one collection and one run, and `/worker` shows the queue; `/board` redirects to `/`.
- `GET /collections/{name}/keyword-lists` (every version, with counts) and `GET .../keyword-lists/{version}`; the Collection page lists them and can load an old version into the editor.
- `GET /eval/{id}` and the `answer` of `GET /collections/{name}` carry `since` (`{untracked, changes}`); `GET /eval/{id}/questions?disagree=` shows every method's row and outcome per question.
- `DELETE /collections/{name}` and `DELETE /eval/{id}`.
- Every run has a plain-language answer, paired gaps against Hybrid and a log; two control methods, `random_example` and `fixed_example`.
- The Lab shows Keyword-RAG's tags for the question, each hit's tags and the channels that found it.
- Datasets: BRIGHT (`bright_aops`, `bright_theoremqa`), SciBench (273 examples, 219 questions, created as `sb`) and GPQA (`"dataset": "gpqa"`, gated: needs `HF_TOKEN`); science problems are a QA type and the console's default.
- Optional `calls` and `est_usd` on `POST /collections/{n}/load`, `/documents`, `/copy`, `/keyword-lists/propose`, `/keyword-tags`, `PUT /collections/{n}/vocabulary`, `POST /eval-sets/{n}/answer`, `/jobs/{id}/resume` and `/eval`: a price that changed since it was shown is refused with 409, and the pages send the price they showed.
- `resolved` in `GET /ai` (whether each job has a usable model), `setup_labels` and `roles` in `GET /catalog`, `controls` in `GET /catalog`, `chat_search` and `load_error` in `GET /collections/{n}`, and `file` on each passage and citation of a chat reply.
- A **Start here** strip on the Stores page (`Connect AI`, `Pick or make a store`, `Chat`, `Test with an eval set`), a **Theme** button in the sidebar (System, Light, Dark), **Chat with this store**, and **+ New store** → **Create** opens the new store.
- A store page's items import queues on one click of Import, which is the confirm, and its line follows the job (`Waiting…`, `Adding… <pct>%`, `Added <n> <items>.`).
- A chat answer from the AI cache reads `Cost $0.00 · answer cached` (`pipeline.answer_cached`); a chat's own notes are kept in this browser across a reload.
- A load job's log shows progress at each 10% step; the sidebar shows the first worker problem under the dot, linking to `/worker`.
- `scripts/experiments.py` and `scripts/experiments.md` (X1 to X5) queue and compare experiments; nothing is queued without `--confirm`, nor any paid arm without `--paid`. `scripts/time_embed.py` times the offered embedding models.
- The gate runs a browser smoke test of the pages (`pytest -m pages`).

### Changed

- Changed (breaking): these request bodies refuse unknown fields with 422: POST /collections, /eval-sets, /eval-sets/{n}/imports, /collections/{n}/documents, /collections/{n}/imports, /collections/{n}/load, /collections/{n}/copy, /collections/{n}/keyword-lists/propose, /query, /chat, /eval, /jobs/{id}/resume, /ai/connections and /ai/connections/{id}/models, and /chats/{id}/messages/{n}/attempts/{a}/answer; PUT /eval-sets/{n}/questions, /eval-sets/{n}/questions/{id} (inside question and base), /collections/{n}/keyword-lists, /collections/{n}/vocabulary, /ai/models/{c}/{m}, /ai/roles/{role} and /ai/budget; PATCH /collections/{n}/sources/{s} and /ai/connections/{id}. A misspelt field no longer passes silently.
- Changed (breaking): POST /collections/{n}/keyword-tags and resuming a tag build answer 503, not 409, when no keywords model can price it (same text).
- Changed (breaking): POST /chat answers 503, not 409, when the method needs an AI model that isn't set up (same text).
- Changed (breaking): POST /chats/{id}/messages/{n}/attempts answers 503, not 409, when a store's method it sets needs an AI model that isn't set up (same text).
- Changed (breaking): POST /eval-sets/{n}/pool and /answer on an eval set that isn't Documents answer 409, not 422 (same text).
- Changed (breaking): aiconnect's AI.split(ref) is now the module function aiconnect.ai.split_ref(ref); it never read the instance.
- Changed (breaking): an unexpected KeyError inside a route is now a 500 instead of a 404; not-found and already-exists replies keep their status and text.
- Changed (breaking): on a run whose eval-set version is gone, GET /eval/{id} (a Documents run), POST /jobs/{id}/resume (a resumable run) and GET /collections/{n} (a Documents store scored by that run) answer 404 instead of 500, and /eval/{id}/items and /eval/{id}/questions say "That version of '<set>' doesn't exist" instead of "no job <id>".
- Changed (breaking): POST /query with answer on a Documents store retrieves at least Settings' evidence size of passages (8 by default, not k's 5), as a run does; results[m].hits grows to match.
- Changed (breaking): Add keyword tags and resuming one refuse with 503 a $0 build that still makes AI calls (a disabled $0 model); PUT /collections/{n}/vocabulary needs the keywords model only when the retag makes AI calls, so a fully cached one, or one of a store with nothing to tag, no longer needs it.
- Changed (breaking): a confirm that saw no price is refused (409) once the step can be priced.
- Changed (breaking): a step priced at $0 that still makes AI calls (a disabled or keyless $0 model) is now refused with 503 instead of queued to fail; a sync add of 20 documents or fewer now checks for a summary model.
- Changed (breaking): a Documents store's answer, and its runs' answer_summary, reference and gaps, are scored on the search its chats use: answer is {kind: "score", …}, frozen in the run's setup as chat_search. Pair answers gain kind: "pair".
- Changed (breaking): `POST /collections/{name}/keyword-lists/propose` needs `confirm: true` to make the AI call; without it, it returns the estimate `{est_usd, calls}`.
- Changed: `POST /collections/{name}/load` returns `200 {estimate: {est_usd, items, calls}}` first whenever the load would spend, and queues only with `confirm: true`; a load that spends nothing still returns 202 `{job_id}`. A stale `calls` or `est_usd` returns the fresh estimate instead of 409 (503 if a needed model is switched off).
- Changed: `PUT /collections/{name}/keyword-lists` returns 202 with a `job_id` and queues a `rekey` job instead of re-keying in the request.
- Changed: the default chat method is hybrid, also for stores with a vocabulary; `iterative` and `hybrid_rerank` are no longer chat methods (422), and a paid method with no AI falls back to hybrid with a note.
- Changed: the default reranker is `jinaai/jina-reranker-v1-tiny-en` (the bge one took 11 to 18 s for 20 passages on CPU); `RAGQA_RERANK_MODEL` still replaces it.
- Changed: new stores embed with `BAAI/bge-small-en-v1.5` instead of `Qwen/Qwen3-Embedding-0.6B-Q`, which ran hot and slow on a laptop CPU; `ModelMismatch` (409) now means the model can't be loaded, or the store is switching.
- Changed: follow-ups are rewritten by default whenever an AI is connected; a follow-up is otherwise searched with the message plus the previous messages joined (2 by default).
- Changed: per-store search overrides are dropped from chat: a re-run edits the chat's settings, and Set as default saves the store's own.
- Changed: stored Search defaults are re-checked each time they are read; **Set as default** is refused (409) when the store cannot run the chat's method; `/eval` checks a `rerank_model` it is sent.
- Changed: `POST /query` on a Documents store uses the run's answer prompt and reply reading, over at most Settings' evidence size of the hits shown.
- Changed: a Documents store's answer is its newest run with the pair, judged or not, and reads retrieval only.
- Changed: Documents runs tick the chat's search and three free methods, not the controls, and rerank the pool chat reranks; the Stores, store and run pages show a Documents store's score, not the keyword verdict.
- Changed: a job whose store was deleted fails with UnknownCollection("no store named 'x'"), and one whose upload or eval-set version is gone fails with NotFound("…"), instead of ValueError or KeyError.
- Changed: a queued or stopped question-import job fails pointing to its eval set's page, and resuming one is refused (409).
- Changed: a copy, import or load whose summaries are all cached no longer needs a summary model.
- Changed: `python -m ragqa_core.apply_outputs` refuses a store built with another embedding model with "refused: …" and exit code 2, instead of a traceback.
- Changed: an items import whose price changed since it was shown is refused with `the price changed since you saw it: check it again`.
- Changed: a run's estimate prices a batched part at its worst case, `max(n × batched, 50 × full)`, so it is never below what the run spends; build and summary estimates price only the calls not already cached.
- Changed: imports and loads decide whether to propose keyword lists when they are queued, so a keywords model set up afterwards doesn't make them spend.
- Changed: the Stores page leads with your own documents, and the benchmarks are under Try a built-in benchmark; **New run** is disabled until an eval set of the store's type exists.
- Changed: Upload is now Preview, and a Documents store takes several files at once; `.md` and `.txt` files must be UTF-8 (422 otherwise), and a Documents store refuses anything else.
- Changed: provider-neutral wording: pages say "costs money" and "AI call", `/health` returns `ai` instead of `api_key`, and error text says "store" instead of "collection".
- Changed: the leak check is off for a Documents store; a refused store or eval-set name says what is allowed and suggests one; new store names may not end in `__rebuild` or `__chunks` (422).
- Changed: Settings' Remove is Delete connection…, and asks first; Start opens the new run; Cancel always means don't (Stop run, Stop job); Chat with this store reopens its chat; labelling buttons read Right/Wrong; a run's question list opens on the reference method.
- Changed: the sidebar asks for the month's spend and the worker's state once on page load; pages stop polling while their tab is hidden; a missing value shows as –; a Create button is disabled while it saves; Copy is announced to screen readers.
- Changed: tables name their answer metrics "Exact answer" and "Answer overlap"; reasoning match is shown on the rescaled scale; the Lab's cost per method counts every AI call that method made.
- Changed: dense queries on Qwen3-Embedding carry the query-side task instruction; the dev slice ranks each question by a hash of its id, so new dev runs on an existing collection may use a different 200.
- Changed: embedding runs 8 texts at a time, the reranker scores in ONNX batches of 16, and a text that is its own only chunk is embedded once.
- Changed: a confirmed eval or resume that would spend returns 503 at once when no AI model is set up for its job.
- Changed: `POST /collections/{name}/documents` is refused with 409 while a load, re-key, eval or tag build on the collection is queued or running, and saving keyword lists is refused when the dense model differs.
- Known gap: the Lab still spends on answers, hops and query summaries without showing an estimate first; spend stays capped and ledgered.

### Removed

- Removed (breaking): `GET /collections/{n}/questions`, `PATCH /collections/{n}/questions/leaked`, and question imports into a store (`kind=questions` is now 422); import questions on the eval set.
- Removed (breaking): `profile` from the store replies (`GET /collections`, `GET` and `POST /collections/{n}`).
- Removed from chat: the store bar, sticky stores, routing profiles, Check routing and Search anyway, with their endpoints and fields (`GET /collections/{name}/bar`, `POST /collections/{name}/route-profile/draft`, `POST /routing/eval`, `route_profile`, the `routing` chat setting, and the decision fields `bar`, `route`, `kept` and `reason`); none of these were released.
- Removed: the 422 `Documents runs score retrieval only for now; judging answers is not available yet` (`ANSWERS_LATER`).
- Removed: the TF-IDF keyword lists, and the Ollama preset with the `local` flag, the Settings text and the `host.docker.internal` mapping.

### Fixed

- A Documents run's answers are no longer scored by token overlap (`PAIR_ANSWER_METRIC` for Documents is accuracy), and the evidence size is fixed at queue time.
- The store serializes its calls to embedded Qdrant (`Store.lock`), waits out a rebuild's switch, and reads a store whose search data is gone as empty, not an error; an eval set whose store lost its search data lists every labelled passage as missing.
- A question id holding a `/` (an imported `hr/leave-1`) saves and deletes one at a time; it used to answer a bare `Not Found`.
- `PATCH /eval-sets/{name}` with `include_leaked: null` says `include_leaked cannot be null: leave it out to keep it`.
- Find passages searches with the store settings it checked, so a change mid-search can't make it spend; an import's merge and a per-question save no longer overwrite each other; it reranks up to 40 passages whatever **Rerank up to** says.
- Labelling keeps unsaved marks (`This question has unsaved marks.`, with `Discard them` or `Keep editing`); **Restore this version** holds the panel still while it saves; the versions list always shows the current version; the eval set page's table could read "NaN questions".
- A Documents run on a store holding none of the set's right passages is refused with `…; choose another store`, and its run page no longer says the answers are measured.
- A mark made while a question's Delete confirm is up withdraws it, and the AI's answer Start can't be clicked twice or during a save or Restore.
- **New run** is disabled until the store page has loaded, and `?eval_set=` opens the run form once with the set picked; an Estimate that returns after the form changed draws no Start.
- An items import preview says `N old passages will be removed, since the text changed.`; a store whose last load stopped shows `Last load stopped: <error>`.
- A question with no gold answer (BRIGHT's) is unmeasured in an answering run, not wrong; the Questions view judges each method only on its answer; a provider error in `/query` is that method's error.
- A copy that stopped or failed resumes in place (`POST /jobs/{id}/resume`, or Resume on the Worker page), and a load that died resumes by content hash.
- A load that makes keyword lists reports progress while it re-keys and can be cancelled then; a cancel stops a run of one-at-a-time AI calls between sends.
- The worker's error clears after its next clean cycle; runs left `running` by a stopped service from before version tracking are marked untracked; older rows that stored 0.0 retrieval scores are cleaned when the database opens.
- A BRIGHT load no longer runs out of memory and takes half as long; a reply typed for a model answered by hand is never served as another connection's answer.
- Three flaky tests no longer race the worker or the run page's poll.

### Security

- Changing a connection's base URL without giving a new key clears its stored key, so a key is never sent to an address it was not given for; keys are stored in the database file and never returned.
- A locally run numeric program is limited to 10 s and 512 MB with no new processes (it is not network-isolated).

## 0.1.0 — 2026-10-23

- The service: collections, ingestion, ten retrieval methods (the 2024 paper's
  seven and three modern ones), query, background evaluation with bootstrap
  intervals, and a monthly cost cap with a ledger and a response cache.
- Three QA types: ScienceQA (exam multiple choice), MuSiQue (multi-hop,
  HippoRAG-2 protocol) and TAT-QA (table + text).
- Docker: one image for the API plus Qdrant, via `docker compose up`.
- Helper pages: the Retrieval Lab and the Eval Board.
