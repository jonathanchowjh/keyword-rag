# Changelog

## Unreleased

- The Stores page leads with **Start here**: `Connect AI`, `Pick or make a store`,
  `Chat` and `Test with an eval set`, each worked out from what exists (nothing is
  stored). `Connect AI` reads `Optional for Documents: without it, chat gives
  passages only.` and does not keep the strip up; it hides once the store, chat and run
  steps are done. A step whose check fails shows as not done.
- **+ New store** → **Create** opens the new store.
- A store page has **Chat with this store**; an empty store disables it with `Add items
  first: an empty store has nothing to search.`
- A store page's items import queues on one click of Import, which is the confirm: its
  price and AI-call count sit beside it, and the server prices the same upload again on
  confirm. The
  line follows the job: `Waiting for the store's other job…`, `Adding… <pct>%`, then
  `Added <n> <items>.` with `<k> already in the store.` when some were.
- `GET /collections/{name}` returns `load_error`, and a store whose last load stopped
  shows `Last load stopped: <error>` on its Stores row and its page.
- A chat answer from the AI cache reads `Cost $0.00 · answer cached`
  (`pipeline.answer_cached` on the attempt); a paid rewrite is not marked.
- A chat's own notes (an added file, a failed send) are kept in this browser across a
  reload, and dropped for chats that no longer exist.
- Small walkthrough fixes: the Lab says store, the chat delete prompt has one `?`, the
  chat's New chat picker and Add store list show store type labels, the conversation
  comes first on a phone, and the store-type sentences read `worked examples whose
  reasoning carries over` and `the table and text that hold the figures for the answer`.

- A Documents run may have the AI answer. Each method's top evidence-size passages
  (Settings' `evidence_size`, fixed in `setup.evidence_size` when the run is queued, so
  the run sends what its estimate priced) go to chat's answer prompt, `[n]` markers
  stripped; a method that finds nothing still asks, with no passages. Without Judge
  answers the answers are shown unscored.
- Judge answers (`judge: true` on `POST /eval`, Documents only): the run's answer model
  marks each answer right or wrong against the question's reference answer, recorded as
  `setup.answer_judge_model`, and the share right is `Answers judged right`. The judge
  is asked for one word, its verdict is read from the last non-empty line, and its raw
  reply is kept as `judge_raw`. A question with no reference answer, or a reply that
  can't be read, is answered but not judged, and the run counts both. Refused: not a
  Documents store, answers not ticked, no question with a reference answer. A judged
  run is allowed on a store holding none of the set's right passages; an unjudged one
  is refused, offering Judge answers when a question has a reference answer.
- A Documents resume's estimate checks the AI cache for the answers and verdicts it can
  rebuild from the stored rows, and prices the rest in full.
- A Documents store's answer is its newest run with the pair, judged or not, and reads
  retrieval only: a judged run no longer hides a newer free one, and one that measured
  no nDCG@10 is passed over.
- The Documents run form has **Have the AI answer the questions** (off by default) and
  **Judge answers** (enabled when answers are ticked), and says `<r> of <n> questions
  have a reference answer.`, or with Judge answers ticked `<r> of <n> questions have a
  reference answer; the rest are not judged.` The estimate returns `references` and,
  when answering, `evidence_size`. An Estimate that returns after the form changed
  draws no Start.
- The Documents run page says `How often are its answers right?`: per method, `<c> of
  <j> answers judged right`, then the questions with no reference answer and the
  answers whose verdict couldn't be read, with the partial note when the run stopped;
  an unjudged answer run says `The AI's answers are not scored.` and to start a new run
  from the store page with Judge answers ticked. Each question shows the AI's answer
  and its verdict, and an unreadable verdict the judge's reply. `GET /eval/{id}` returns
  `answers`.
- **Get the AI's answer** on the labelling panel: `POST /eval-sets/{name}/answer`
  prices first, then (confirmed) makes one cached call from the passages marked `y` with
  chat's prompt; **y** puts the answer in the reference-answer field and nothing is
  stored until **Save question**. It is refused with 422 `mark at least one passage y
  that is in <store> first` when none of the passages is in the store. `GET
  /eval-sets/{name}` returns `answer_available`.
- Removed: the 422 `Documents runs score retrieval only for now; judging answers is
  not available yet` (`ANSWERS_LATER`).
- Fixed: a Documents run's answers are no longer scored by token overlap,
  `PAIR_ANSWER_METRIC` for Documents is accuracy, and the evidence size is fixed at
  queue time.
- Documents eval sets are built by labelling. A question has `judged_ids`, the
  passages marked right or wrong (null: old, imported or unknown; `[]`: labelled with
  nothing judged), and every write keeps judged a superset of `gold_ids`. A set
  from before it, saved unchanged, keeps its version ids, so old runs are not
  flagged "eval set changed".
- Re-importing into a Documents set: a file that carries `judged_ids` replaces a
  stored question whole; one that does not keeps the stored judged passages and
  typed answers, joins the gold ids, and takes the file's text and `meta`. Other
  types replace the question. `gold_answers` is read as an `answer` synonym, so
  `GET /eval-sets/{name}` saved as `.json` imports back with its labels.
- An eval set can be linked to a store: `POST /eval-sets` takes `store` (and makes
  the set that store's default when it has none), `PATCH /eval-sets/{name}` takes
  `store` (only links it), and **Create one** on a store page opens the new-set form
  with the store and its type filled in. `GET /eval-sets/{name}` returns `store`,
  `missing` (right passages the store no longer holds) and `questions_version`.
- Per-question writes: `POST`, `PUT` and `DELETE` on
  `/eval-sets/{name}/questions[/{id}]`. A `PUT` sends the question as read and is
  refused (409) when it changed since; a deleted question is 404. Each makes a
  version. The whole-set save keeps its `base_version` check.
- Versions: `GET /eval-sets/{name}?version=` reads an old one, and the set page
  lists the newest 20 with **View**, **Show all versions** and **Restore this
  version**, which is disabled while there are unsaved edits or marks.
- **Find passages** (`POST /eval-sets/{name}/pool`) and the labelling page: the top
  10 of each free method, deduplicated and reranked with the Settings reranker, to
  mark `y` or `n`; **Run against its store** opens the store's run form with the
  set chosen. Free, with no AI.
- The run form warns when most of the chosen set's questions have no right passage in
  the store. The free estimate now also runs when the form opens with a set already
  chosen.
- A Documents run page leads with `How well does this store find the right
  passages?` and flags a method whose top 10 averaged more than 3 never-judged
  passages (`score unreliable`), once the run is done. It counts the questions it
  could not measure (as scored, so later changes to the store do not alter it) and
  those with no right passage. `GET /eval/{id}` returns `labels`.
- Fixed: the eval set page's table could read "NaN questions".
- Labelling keeps unsaved marks: while the open question has any, **Label** on a
  row and **Delete** first say `This question has unsaved marks.` and offer
  `Discard them` or `Keep editing`. Delete is disabled while a question is being
  saved.
- **Restore this version** holds the labelling panel still while its save is under
  way (marks, Find, Save, Label and Delete are disabled), then clears it only when
  nothing was typed meanwhile. A Documents set's restore brings back `judged_ids`.
- The versions list always shows the current version, marked `current`, even after
  a Restore made an older one current again.
- Linking an older Documents set starts the store picker on the store whose default
  set it is, else the store of the same name. `GET /collections` returns each
  store's default `eval_set`.
- A question id holding a `/` (an imported `hr/leave-1`) saves and deletes one at a
  time; it used to answer a bare `Not Found`, and the page then saved the marks as a
  new question.
- A Documents run on a store holding none of the set's right passages is refused
  with `…; choose another store`, without offering an AI answer it would also
  refuse. Its run page no longer says the answers are measured, shows no
  Best-keyword line, and its note under the gaps reads `Leading is not winning: the
  order above is this run's alone.`
- `PATCH /eval-sets/{name}` with `include_leaked: null` now says `include_leaked
  cannot be null: leave it out to keep it`.
- Find passages searches with the store settings it checked, so a change to them
  mid-search can't make it spend. An import's merge and a per-question save no
  longer overwrite each other: each waits for the other.
- Find passages reranks up to 40 passages, whatever **Rerank up to** says, and the
  first Find after the reranker changes waits while it loads.
- The refusal of a Documents run that asks for answers now reads `Documents runs
  score retrieval only for now; judging answers is not available yet`; it no longer
  promises answers "with Documents eval sets".
- The New store form defaults to a Documents store (`qa_type` `documents`, recipe
  `documents`; `POST /collections` still needs a `qa_type`): it keeps your passages
  as uploaded and searches them by meaning and by words, with no AI. `POST /collections` with no `recipe` takes the type's default
  (`documents` for Documents, paper 2024 for the others). The New store form leads
  with Documents and offers each type's own recipes; a Documents store takes only
  `documents` or `documents_tagged`.
- Content ids: a `.md` or `.txt` passage, in every store, is named
  `<file>#<12 hex>` from its text, and `meta` holds `{file, part, parts}` (a short
  file is part 1 of 1). An edit keeps the ids of the other sections, and an edited
  passage is a new id. Re-adding a file embeds only new or changed passages and
  deletes the old ones, including the old `<file>` and `<file>#<digits>` ids. A store
  loaded before keeps its old ids until each file is added again; an eval set's gold
  ids that name an old or edited passage go unmeasured.
- `.md` and `.txt` files must be UTF-8: any other is refused (422, `<file> is not
  UTF-8 text; save it as UTF-8 and upload it again`) instead of read with characters
  replaced. JSON, CSV and TSV are read as before. A Documents store refuses anything
  but `.md` and `.txt` (422, `A Documents store takes .md and .txt files`).
- Add keyword tags: `POST /collections/{name}/keyword-tags` (and a button on the
  store page) is priced first; `confirm: true` queues a `tag_build` job (state
  `tagging`, shown as Adding keyword tags), which switches the Documents store to
  `documents_tagged` in place, induces a vocabulary and tags each passage by
  similarity. Resume finishes a stopped one (only the newest), and is free with no
  model once the vocabulary is saved. An import is refused (409) while a `tag_build`
  is queued or running on the store. Once the vocabulary is in force, adding
  passages to a tagged store is free and needs no model.
  `GET /collections/{name}` returns `keyword_tags`. Until every passage is tagged,
  `keyword_rag` is unavailable. `PUT /vocabulary` on an untagged Documents store is
  409.
- Documents runs score Recall@10, MRR@10 and nDCG@10 at depth 10, headed by nDCG@10.
  A Documents run that asks for answers is refused (422) until answers are judged.
- The leak check is off for a Documents store: a question whose words sit in your
  passages is expected.
- Plan 1 cleanups that change behaviour: stored Search defaults are re-checked each
  time they are read, so a reranker fastembed dropped or a method that is not a chat
  method reads as unset; **Set as default** is refused (409) when the store cannot
  run the chat's method; Save starts loading a reranker only when it is a new pick or
  the one in force failed; `/eval` checks a `rerank_model` it is sent.
- Chat from your own app: `POST /chats` takes `{from}` to start a conversation set up
  like a chat (its stores and saved settings, no messages); `POST /chats/{id}/messages`
  takes `?brief=true` for a short reply (`n`, `answer`, `citations`, `passages`,
  `passages_only`, `error`, `usd`) and `"answer": false` for the reranked passages
  with no answer written and nothing paid for one (`pipeline.answer_skipped`). The
  chat page has **Use from your app** with the curl lines.
- New stores are embedded with `BAAI/bge-small-en-v1.5` by default instead of
  `Qwen/Qwen3-Embedding-0.6B-Q`, which ran hot and slow on a laptop CPU. Existing
  stores keep theirs. To go back: Settings → Search defaults → Embedding model for
  new stores, or `RAGQA_DENSE_MODEL=Qwen/Qwen3-Embedding-0.6B-Q`. X1–X4 were
  measured with Qwen3 0.6B.
- Each store embeds questions with its own model, so a service whose
  `RAGQA_DENSE_MODEL` differs no longer refuses older stores. `ModelMismatch` (409)
  now means the model can't be loaded, or the store is switching. Chat no longer
  leaves out a store built with another model.
- Rebuild with another model: `POST /collections/{name}/reembed` queues a free
  `reembed` job that embeds the store aside (`<name>__rebuild`) and then switches.
  The store's state is `rebuilding`, `GET /collections/{name}` returns `rebuild`, and
  `POST /jobs/{id}/resume` resumes a stopped one. A stopped switch refuses search,
  deleting a source and copying (409).
- New store names (a create or a copy's target) may not end in `__rebuild` or
  `__chunks`: those are a store's search data, and a store named so would lose them
  when `<name>` is rebuilt or deleted (422). Stores already so named are unchanged.
  Publishing is refused while a switch is stopped, and the store reads `rebuilding`.
- `POST /collections` takes `dense_model`; `GET`/`PUT /search-defaults` carry
  `dense_model` and `embedders`. A copy keeps its source's model.
- `scripts/time_embed.py` times the offered embedding models (results pending).
- Removed from chat: the store bar, sticky stores, routing profiles, Check
  routing and Search anyway, with their endpoints and fields
  (`GET /collections/{name}/bar`, `POST /collections/{name}/route-profile/draft`,
  `POST /routing/eval`, `route_profile`, the `routing` chat setting, and the
  decision fields `bar`, `route`, `kept` and `reason`). `ChatPatch` and
  `CollectionPatch` now refuse the removed fields (422). The `store_bars` table is
  dropped; the `chat_bar` and `route_profile` columns stay in old databases,
  unused. None of these were released; a database that has them is migrated
  once (see the chat settings entry below).
- Chat pipeline: a chat searches every store on it. Each store returns its top
  candidates by its own method, the lists are pooled round-robin by rank
  (deduplicated by text, capped at Rerank up to, never below the evidence size),
  reranked against the standalone question or the message alone, and cut to the
  evidence size. With rerank off, or the reranker loading or failed, the pool is
  merged by RRF with the notes `reranker still loading: merged by rank` or
  `reranker unavailable: merged by rank`. A method a store cannot run, or a paid
  one with no AI, is replaced by hybrid with `<method> <why>, so hybrid was used`.
  The retrieval step shows the search text, what it was reranked against and by
  which model, how many passages were pooled (`Pooled 20 of 60 (Rerank up to
  20)`), and each passage's relevance.
- The reranker warms in the background at start and when Settings picks another,
  and scores in ONNX batches of 16 (the embedder in batches of 8).
- The reranker is a Setting (Reranker model, Rerank up to), and the default model
  is now `jinaai/jina-reranker-v1-tiny-en` instead of
  `BAAI/bge-reranker-v2-m3-int8`, which was too slow on CPU (11 to 18 s for 20
  passages against about 1.5 s on an M4 Pro). `RAGQA_RERANK_MODEL` still replaces
  the default and is then the recommended model. To go back to bge, pick it in
  Settings → Search defaults → Reranker model, or set
  `RAGQA_RERANK_MODEL=BAAI/bge-reranker-v2-m3-int8`. Settings says while the
  reranker downloads and, if it failed, why (`GET /search-defaults` returns
  `rerank_status` and `rerank_error`); saving it again retries at once.
- "Hybrid + reranker" runs record the model they used in `setup.rerank_model`
  and use it (Settings' at queue time, unless the body names one). Earlier runs
  used `bge-reranker-v2-m3-int8` at a pool of 50, and a resumed one keeps it. A
  run page names a `hybrid_rerank` run's reranker and pool, and Re-run repeats
  both.
- Chat settings are saved with the chat (evidence size, rerank, rewrite, and per
  store the method and candidates) and inherit: chat, then store, then Settings,
  then the recommended value. The migration turns a saved evidence size of 8 and
  rewrite off into inherit, drops `routing`, and resets each store's saved
  `chat_k`, which now means candidates per store. It writes the marker
  `schema.routing_dropped`; a later start only drops leftovers and never resets
  settings again.
- Per-store search overrides are dropped from chat: a re-run edits the chat's
  settings, each store's method and candidates, and Set as default saves the
  store's own.
- Follow-ups are now rewritten by default whenever any AI is connected: one paid
  call per follow-up, in chats from before too. Settings → Search defaults →
  Rewrite follow-ups → Off turns it off, or switch it off in a chat's panel.
  Re-running an old message (from before this change) reuses its search text,
  with no paid rewrite.
- `iterative` and `hybrid_rerank` are no longer chat methods: `POST /chat` and the
  chat settings refuse them (422). A paid method with no AI falls back to hybrid
  with a note in a chat; an explicit one on `POST /chat` is 409 `needs AI`, and an
  unknown or unpriced `model` is the reply's `error`.
- The default chat method is now hybrid, also for stores with a vocabulary
  (it was `keyword_rag`).
- Chat works without AI: Send returns the passages, marked `Passages only: no
  answer model`, with `Connect AI in Settings to get written answers`.
- Settings → Search defaults: the default method, candidates per store, evidence
  size, rerank, rewrite, previous messages joined, RRF k, reranker model and
  Rerank up to, each with its recommended value and Reset to recommended
  (`GET`/`PUT /search-defaults`). The store page shows its chat settings with
  Reset to Settings.
- `POST /chat` and `POST /query` take their defaults from Settings: `/chat`
  with no `method` or `k`, `/query` with no `methods` or `rrf_k`. `POST /chat`
  runs the chat pipeline for one store and returns `notes`.
- `POST /eval` takes `rerank_pool` (1 to 50) and `rerank_model` for
  `hybrid_rerank`; `scripts/experiments.py` gains X5, hybrid against hybrid
  reranked at 20.
- Experiments: `scripts/experiments.md` has the X1 to X4 results (hybrid beat the
  label channel with classifier tags; Haiku query tags separated from hybrid,
  Opus ones did not; nothing in X3 beat the defaults; examples did not beat
  zero-shot on SciBench), the reranker timing for the smaller models, and X5,
  pending.
- Chats are saved on the server (`GET`/`POST /chats`, `GET`/`PATCH`/`DELETE
  /chats/{id}`) and search several stores; the chat page lists them on the left.
  `PATCH /chats/{id}` refuses a null title or stores (422); a null evidence size,
  rerank or rewrite inherits from Settings. A follow-up is searched with the
  message plus the previous messages joined (2 by default), or, with Rewrite
  follow-ups on and AI connected, a standalone question from the new `rewrite`
  role.
- Chat: a retrieval step under each answer shows the search and every store's
  candidates and evidence. Re-run adds an attempt without answering, and Answer
  from this attempt answers it. An attempt answered by two requests at once is
  answered and paid once (the other gets 409), and answering keeps a failed
  rewrite's note in the attempt's error. A store taken off a chat, then deleted,
  still shows as deleted on the chat's old rows. No estimate before a message;
  each reply and attempt shows its cost and the budget cap holds. The old
  `POST /chat` still works; conversations kept only in a browser tab are not
  migrated.
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
  question with the store's chat method (hybrid by default) or the chosen method (422
  for an unknown name, 409 when the store cannot run it); oracle tags become classifier
  tags. The last 6 turns are sent. It spends like the Lab: no estimate, the cost on each
  reply, the budget cap enforced. Add document adds `.txt` and `.md` files to
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
