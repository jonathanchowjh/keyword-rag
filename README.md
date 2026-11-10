# rag-qa

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![Runs in Docker](https://img.shields.io/badge/runs%20in-Docker-2496ED.svg)
[![Paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/keyword-vectorization-rag-science-qa.pdf)

**A retrieval-augmented question-answering service and test bench for keyword
vectorization: does describing every document with keywords find better
evidence than searching its text?**

It indexes four common kinds of QA, runs the retrieval methods from the paper
[*Keyword Vectorization for Retrieval Augmented Generation on Science Question
Answering*](paper/keyword-vectorization-rag-science-qa.pdf) next to modern
ones and two controls, and evaluates them over an HTTP API and a small web
console. Every result says how sure it is. Nothing spends money without
showing the estimate first, except the Lab, which reports each method's cost
after the call.

Any AI provider can answer: Anthropic, OpenAI, Gemini, xAI, OpenRouter or any
OpenAI-compatible service, priced per call. Retrieval itself is always free and
local.

- **Status:** v0.1.0, in active development (see [Roadmap](#roadmap)). Runs
  locally in Docker; no published image.
- **License:** Apache-2.0.

## Contents

- [The paper](#the-paper)
- [Quick start](#quick-start)
- [First run](#first-run)
- [Your own documents](#your-own-documents)
- [Your own corpus](#your-own-corpus)
- [Recipes](#recipes)
- [Retrieval methods](#retrieval-methods)
- [Answering](#answering)
- [Chat](#chat)
- [Search defaults](#search-defaults)
- [Datasets](#datasets)
- [AI connections](#ai-connections)
- [Configuration](#configuration)
- [Cost safety](#cost-safety)
- [Pages](#pages)
- [Endpoints](#endpoints)
- [Reading a run](#reading-a-run)
- [Experiments](#experiments)
- [Using the AI layer in another project](#using-the-ai-layer-in-another-project)
- [Roadmap](#roadmap)
- [References and resources](#references-and-resources)

How it is built and how to work on it is in [CONTRIBUTING.md](CONTRIBUTING.md).

## The paper

> **Keyword Vectorization for Retrieval Augmented Generation on Science
> Question Answering**
> Jonathan Chow (Hartono Tjakrawinata), Singapore Management University, 2024.
> [Read the paper (PDF)](paper/keyword-vectorization-rag-science-qa.pdf)

The paper asks how to pick the one solved example that best helps a language
model answer a long, table-heavy science question. Plain vector search over
whole questions does badly: a 361-word question with tables and units embeds
as noise. Instead, each stored question is described by keywords chosen from
three hand-built lists (science topics, experiment types and common terms)
plus a short AI summary, and the new question is described the same way. The
example whose keywords match best is shown to the model as a one-shot
chain-of-thought demonstration.

It was tested on ACT Science: 446 questions from 12 past papers as the
example store, 76 questions from 2 further papers as the test, gpt-3.5-turbo
answering, `text-embedding-3-large` embeddings, Faiss, BM25 and reciprocal
rank fusion (k = 1). One-shot results:

| Retrieval | Method here | Correct |
|---|---|---|
| No retrieval (question only) | `question_only` | 53 / 76 (69.7%) |
| Vector search (parent-document retriever) | `vector` | 48 / 76 (63.1%) |
| Keyword search (BM25) | `keyword` | 56 / 76 (73.6%) |
| Hybrid: BM25 + vector, fused with RRF | `hybrid` | 58 / 76 (76.3%) |
| Keyword vectors: all keywords in one string | `kw_all` | 51 / 76 (67.1%) |
| **Keyword vectors: top 3 from each list** | **`kw_individual`** | **60 / 76 (78.9%)** |
| Keyword vectors: one store per list, fused with RRF | `kw_rrf` | 58 / 76 (76.3%) |

Two-shot prompts (about 8,000 characters) did worse across the board. With 76
questions, the gap between the best two rows is two questions, so the finding
is a direction, not a settled size. This service exists to measure it
properly: every row above is a method here, run on larger question sets with
paired confidence intervals, and beside controls the paper did not have (see
[Retrieval methods](#retrieval-methods)).

To cite it:

```bibtex
@misc{chow2024keywordvectorization,
  title  = {Keyword Vectorization for Retrieval Augmented Generation on
            Science Question Answering},
  author = {Chow, Jonathan},
  year   = {2024},
  note   = {Singapore Management University. Author also known as Hartono Tjakrawinata},
  howpublished = {\url{https://github.com/jonathanchowjh/rag-qa/blob/main/paper/keyword-vectorization-rag-science-qa.pdf}}
}
```

## Quick start

Requires Docker.

```sh
cp .env.example .env
docker compose up --build
```

Open http://localhost:8080/ for the stores and their evaluations,
http://localhost:8080/lab to compare methods on one question, and
http://localhost:8080/docs for every endpoint (see [Pages](#pages)).

Then connect an AI on **Settings** (http://localhost:8080/settings), for
example:

**Anthropic, with a key that cannot overspend.** Claude Max 5x, Max 20x and
Team plans include monthly API credit ($100 a month on Max 5x).
1. On claude.ai, open **Settings → Billing → API credits** and link a Console
   organisation.
2. In the Console, create a workspace for rag-qa and give it its own spend
   limit.
3. Add no payment method and leave auto-reload off. When the credit runs
   out, requests stop; nothing is charged to your plan.
4. Create an API key in that workspace and paste it into `ANTHROPIC_API_KEY`
   in `.env` (then restart), or add it on the Settings page.

The claude.ai login itself must not be used: Anthropic's Consumer Terms
allow scripted access only through an API key.

**Anything else that speaks the OpenAI API.** OpenAI, Gemini, xAI and
OpenRouter have presets; LM Studio, vLLM and other servers use **Custom**
with their base URL.

With no AI model set up, the service still starts. Every free retrieval method
works, and calls that would spend return 503 `no AI model is set up for the
'<job>' job; choose one on the Settings page`.

## First run

A *store* is what is retrieved on: items, their chunks and keyword lists. An
*eval set* is what is asked: a named, versioned list of questions of one QA type.
They are separate, so one eval set can be asked against several stores. (The API
paths still say `collections` for stores; the pages and this text say store.)

The default eval set of a store `X` is the one its `eval_set` names, else the eval
set called `X`, else none. A run uses the default unless you pick another.

The Stores page leads with **Start here**, four steps worked out from what exists:
`Connect AI` (done when a model is enabled, priced and has a connection with a key,
or a local or manual one; its note reads `Optional for Documents: without it, chat
gives passages only.`), `Pick or make a store` (a store holds items), `Chat` (a chat
has a message) and `Test with an eval set` (a run has finished). The strip hides once
the store, chat and run steps are done; `Connect AI` is optional and does not keep it
up. Nothing about it is stored, and a step whose check fails shows as not done.
**+ New store** → **Create** opens the new store, and a store page has **Chat with this
store**, which an empty store disables with `Add items first: an empty store has nothing
to search.`

A store whose last load stopped shows `Last load stopped: <error>` on its Stores row
and its page, until a later load succeeds.

On the home page, **Add a default store** loads any built-in dataset in one click,
and **Set up the four benchmark stores** creates and loads ScienceQA, MuSiQue,
TAT-QA and SciBench, asking before anything that would spend. Or do it over the API,
one store per QA type. Loading downloads the data, embeds it locally and saves
keyword lists, and writes the dataset's questions into the store's default eval set,
creating one named after the store when there is none. A default set of another QA
type stops the load. It runs as a background job; poll `GET /jobs/<job_id>`.

```sh
api=http://localhost:8080
curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"sb","qa_type":"science"}'
curl -X POST $api/collections/sb/load -H 'content-type: application/json' \
  -d '{"dataset":"scibench"}'

curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"sqa","qa_type":"mcq","summarize":true}'
curl -X POST $api/collections/sqa/load -H 'content-type: application/json' \
  -d '{"dataset":"scienceqa"}'                  # returns the cost estimate, if it would spend
curl -X POST $api/collections/sqa/load -H 'content-type: application/json' \
  -d '{"dataset":"scienceqa","confirm":true}'   # starts the job

curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"gpqa","qa_type":"mcq"}'          # needs HF_TOKEN, see Datasets
curl -X POST $api/collections/gpqa/load -H 'content-type: application/json' \
  -d '{"dataset":"gpqa"}'

curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"mq","qa_type":"multihop"}'
curl -X POST $api/collections/mq/load -H 'content-type: application/json' \
  -d '{"dataset":"musique"}'

curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"tq","qa_type":"table"}'
curl -X POST $api/collections/tq/load -H 'content-type: application/json' \
  -d '{"dataset":"tatqa"}'
```

Then start a run from a store's page, choosing the eval set, or with `POST /eval`
(`eval_set` defaults to the store's default; 422 `has no eval set; choose one` when
there is none, 409 when its QA type is not the store's). A run that would measure
nothing, for example retrieval only on questions that name no evidence the store
holds, is refused.

Reloading a dataset upserts its questions by id. It overwrites any hand edit to
those ids and brings back questions you deleted.

Docker on macOS cannot use the Apple GPU, so embedding runs on the CPU. The
first load of MuSiQue's 11,656 passages takes roughly 10–20 minutes. After
that, everything is kept in the `ragqa` and `qdrant` volumes.

## Your own documents

The New store form defaults to a Documents store (`POST /collections` still needs
a `qa_type`). It keeps your passages as you upload them and finds them with no AI.
Create one on the Stores page, then add files on the store page.

- **Files:** `.md` and `.txt` only, and they must be UTF-8. A `.zip`, a table or
  JSON lines is refused with `A Documents store takes .md and .txt files`, and a
  file that is not UTF-8 with `<file> is not UTF-8 text; save it as UTF-8 and upload
  it again`. Nothing is imported and no character is replaced.
- **Passages:** each file is split into passages of up to 1,500 characters, at
  headings first, each starting with a breadcrumb such as `notes.md › Setup ›
  Install` (the splitting rules are under [Your own corpus](#your-own-corpus)). A
  file with no text is refused (`no usable rows`).
- **Ids:** a passage is named `<file>#<12 hex>`, from its text. Add a file again
  after editing it: a passage that did not change keeps its id and is not embedded
  again, a changed or new passage is written, and a passage that left the file is
  deleted. An edit keeps the ids of the other *sections*. Passages are packed
  greedily within a section, so an edit changes the passages after it in its own
  section, and a `.txt` is one section. A file that grows past 1,500 characters in
  all gains breadcrumbs, and its passages take new ids. An eval set's old
  positional gold ids (such as `x.md#2`), and the gold id of an edited passage, go
  unmeasured once the file is added again.
- **Search:** vector, keyword (BM25) and hybrid, with no AI call. `keyword_rag` is
  off until you **Add keyword tags**.
- **Add keyword tags** is on the store page. It is paid: an AI proposes a
  vocabulary from your passages, then each passage is tagged by similarity. You see
  the estimate first, and nothing is spent until you confirm. Its job switches the
  store to the `documents_tagged` recipe in place when it starts. If the job stops,
  **Resume** finishes it (free once the vocabulary is saved); until every passage
  is tagged, `keyword_rag` stays unavailable with `needs keyword tags: Add keyword
  tags on the store page`. With no model for `keywords`, the button is refused and
  the store stays as it was. Once the vocabulary is in force, adding passages to a
  store that has tags is $0 and needs no AI model: they are tagged by similarity.
  An import is refused while the job is queued or running.
- **Runs:** a run scores Recall@10, MRR@10 and nDCG@10 at depth 10, and the head
  metric is nDCG@10. The leak check is off for a Documents store: a question
  whose words sit in your passages is expected. A run may also have the AI answer
  (**Have the AI answer the questions**, unchecked by default; with it ticked the run
  form says `<r> of <n> questions have a reference answer.`, and with Judge answers
  ticked too, `<r> of <n> questions have a reference answer; the rest are not
  judged.`). Each method's top evidence-size
  passages (Settings' `evidence_size`, recorded in `setup.evidence_size` when the run
  is queued) go to chat's answer prompt, and the `[n]` markers are stripped from the
  reply. A method whose search finds nothing for a question still asks, with no
  passages. Without **Judge answers** the answers are shown unscored.
  With **Judge answers** (it costs money, and needs the answers ticked) the run's
  answer model marks each answer right or wrong against the question's reference
  answer (`gold_answers`); the run records it as `setup.answer_judge_model`, always
  the run's answer model. The share marked right is `Answers judged right`. The judge
  is asked to reply with exactly one word; its verdict is read from the last non-empty
  line of the reply, allowing a leading `Verdict:` and a trailing `.` or `!`. A
  question with no reference answer is answered, and paid for, but not judged, and
  the run counts it. A reply that cannot be read is not judged either, and is counted;
  the raw reply is kept with the row. One model answers and judges, which may flatter
  its own answers. nDCG@10 stays the headline. Judge answers is refused on a store
  that is not a Documents store, without the answers ticked, and when no question in
  the run has a reference answer. A run that is not judged is refused when the store
  holds none of the set's right passages: `nothing to measure: this store holds none
  of the eval set's right passages; tick Judge answers to score the answers, or
  choose another store` when some question in the run has a reference answer, else
  `nothing to measure: this store holds none of the eval set's evidence; choose
  another store`. A judged run is allowed there. A resume's estimate checks the AI
  cache for the answers and verdicts it can rebuild from the stored rows, and prices
  the rest in full. The store's answer is the nDCG@10 of the search its chats use
  (the store's chat method in force, as a run names it: `hybrid_rerank` for Hybrid with
  rerank on), from its newest run with that method, judged or not. Each run freezes
  that search when it is queued and its own page is scored on it, so a later change in
  Settings does not change an older run's score. A store with passages and no keyword lists reads Ready.
- **Labelling:** a Documents eval set is built by labelling. Link the set to one
  Documents store (**Create one** on the store page makes a set already linked;
  on `/e/{name}` pick a store under `Pick a store to label against:` and press
  `Link`; it starts on the store whose default set this is, else the store of the
  same name, else the first). Type a question under `Label a question` and press
  **Find passages**.
  Find passages takes the top 10 of each free method the store can run (vector,
  keyword and hybrid, plus `keyword_rag` once its tags are built), drops repeats
  and sorts the rest with the Settings reranker, best first. It shows no method
  names. While the reranker loads, or if it failed, the passages stay in search
  order and a note says so. Mark each passage **Right** or **Wrong**; press a
  mark again to clear it. `None of these? Search the store` runs a hybrid search
  and lists its hits under `From your search`. Passages you already marked stay in
  view when you find again. **Save question** stores it (the `Reference answer`
  is a typed field, optional, and only a right answer is kept as `gold_answers`);
  **Save and write the next** clears the panel. The table below lists each
  question with its right and judged counts; **Label** reopens one and **Delete**
  asks first. Find, Save, Label and Delete use no AI and spend nothing. Find passages reranks
  up to 40 passages (10 from each of four methods) and ignores **Rerank up to**
  (Settings `rerank_pool`, default 20), which caps chat only. A reranker not loaded
  yet (the first Find after it changes, say) loads then, so that Find can take a few
  seconds. While a save is under way, Find, Save, Label and Delete are disabled.
  While the open question has unsaved marks or typing, **Label** and
  **Delete** first say `This question has unsaved marks.` and offer `Discard them`
  or `Cancel`. If someone else deleted the question, saving shows the 404 and
  keeps your marks, and they save as a new question. A question id may hold a `/`
  (an imported `hr/leave-1`, say); it saves and deletes like any other.
- **Get the AI's answer:** under the `Reference answer` field, enabled once the
  question has text and at least one passage is marked Right. It shows its price first;
  **Start** pays, and the AI answers from the passages marked Right with chat's prompt.
  Asking again with the same question and passages is free (the AI cache). Then press
  **Right** to put the answer in the reference-answer field, replacing what was typed
  there, or **Wrong** to drop it. Nothing is stored until **Save question**. The button
  is hidden when no answer model is connected.
- **Labels:** a question has `gold_ids` (the right passages) and `judged_ids`
  (every passage you marked, right or wrong). Every write keeps judged a superset
  of gold. `judged_ids` null means the question is old, imported or unknown;
  `[]` means labelled with nothing judged. A passage a right answer names that the
  store no longer holds is unmeasured, not a miss. The set page says `<n> labelled
  passages are no longer in <store>` and lists them by id per question, and
  **Label** runs Find passages again. A question with no right passage is asked
  but not scored. The linked store is separate from the store's default eval set:
  linking an existing set (`PATCH`) does not change which set a run uses by
  default. A set created with a store becomes that store's default only when the
  store has none. Any set type can be linked, but Find passages is for Documents sets only (422 otherwise, and 409
  with no store linked).
- **Two people labelling:** saving one question sends the question as it was read.
  If someone saved it since, the save is refused (409, `This question was changed
  by someone else since you opened it`) and the page offers `Reload it (discard my
  marks)`. A question deleted meanwhile is 404 (`This question was deleted since
  you opened it`). Saving or deleting a question is also refused (409, `a load
  is under way on a store of '<set>'`) while a load writes to the set; the page offers Reload on any 409. Deleting has no
  stale-base check. The whole-set save still checks `base_version`.
- **Run warnings:** a Documents run page leads with `How well does this store find
  the right passages?`. For each method it counts how many of a question's top 10
  were never in that question's `judged_ids`, takes the mean over the questions the
  method scored whose `judged_ids` are known, and flags the method when the mean
  is over 3: `score unreliable: on average <n> of its top 10 were never judged`,
  with `<n>` rounded up. The controls and `question_only` are left out, and the
  flags show once the run is done (while it runs the page says `The scores appear
  when the run finishes.`). A run page also counts questions it could not measure,
  because the store held none of their right passages (`<n> questions unmeasured:
  their right passages aren't in this store`; the count is the one the run was
  scored with, so removing passages later does not change an old run), and
  questions with no right passage. A set with no `judged_ids` at all says
  `This eval set doesn't record which passages were judged, so score reliability
  can't be checked`. Unjudged hits in a question's rows read `(never judged)`.
  `hybrid_rerank` reranks hybrid's top 50, and chat may keep passages from hybrid's
  candidates 11-20; labelling pooled neither, so either can raise the flag. That is
  what the flag is for. The run form also warns, for
  any store, when most of the chosen set's questions have no right passage in the
  store (`only <held> of <named> do`); it is worked out from the free estimate,
  which runs whenever a set is chosen, including when the form opens with one.

## Your own corpus

Any store takes your own items (what is retrieved). Your own questions (what is
asked) go into an eval set. On the store page (items) or the eval set page
(`/e/{name}`, questions) choose a file, check the preview, fix the field mapping if
the guess is wrong, and import. On the store page, an items import queues a job on one
click of **Import**, which is the confirm: its price and AI-call count sit beside it,
nothing is spent before the click, and the server prices the same upload and mapping
again when it confirms; an import whose price changed between the preview and the
click (other AI calls or a higher price) is refused with `the price changed since you
saw it: check it again`, and the page shows the new price. Whether an import proposes
keyword lists is decided when it is queued, so a keywords model set up after that
doesn't make it spend. When the text of a file changed, the preview says `N old passages will
be removed, since the text changed.` (`1 old passage …`), and the done line adds
` N old passages removed.`; an eval set's questions that marked a removed passage
list it as missing. The line follows the job:
`Waiting for the store's other job…`, `Adding… <pct>%`, then `Added <n> <items>.`
with `<k> already in the store.` when some were; a stopped job reads
`The import <status>: <error>`. A question import on the eval set page
(`/eval-sets/{name}/imports`) runs in the request; a store takes no question
import (`kind=questions` is 422).

- **Formats:** JSONL, a JSON array, CSV, TSV, plain text (`.txt`), Markdown
  (`.md`), or a `.zip` of `.txt` and `.md` files.
- **Text uploads are split:** a `.txt` or `.md` file (also inside a zip) longer
  than one passage is split into passages of up to 1,500 characters, so the chat
  sees all of it (each passage it reads is cut at 1,500). Markdown is cut at
  headings first, then paragraphs and sentences. Code blocks, tables and lists are
  kept whole where they fit; a table cut in two repeats its header row. Each
  passage starts with a breadcrumb such as `notes.md › Setup › Install`, and one
  under 200 characters is merged into a neighbour. A file that fits in one passage
  has no breadcrumb. Text files must be UTF-8 (JSON, CSV and TSV are read leniently,
  as before). A section that is only a heading makes no passage: its children's
  breadcrumbs name it. Splitting means more items: on a store that spends per item
  (summaries), the add's estimate shows the extra cost.
- **Passage ids name the text:** in every store, a passage is
  `<file>#<12 hex>`, the first 12 hex digits of the SHA-256 of the passage as stored
  (breadcrumb included). A text that repeats in one file takes `-2`, `-3`, and so
  on. `meta` holds `{file, part, parts}` for every text passage, and a file that
  fits in one passage is `part 1 of 1`. Re-adding a file updates its passages by id,
  embeds only the new or changed ones, and deletes, from the same source, the old
  passages it no longer yields: the ids `<file>`, `<file>#<digits>` (the old
  numbered ids) and `<file>#<12 hex>`, but not a file named `a.md#b.md`, which is
  not a passage of `a.md`. A store loaded before this change keeps its old ids
  (`<file>#1`, …, or `<file>` for a short file) until each file is added again, and
  an eval set's gold ids naming them go unmeasured after that. A passage whose text
  did not change keeps its id when the file gains or loses a passage; its `parts` is
  rewritten.
- **Mapping:** items need `text`; `id`, `answer`, `explanation`, `skill` and
  `topic` are optional. Questions need `text` and `answer` (and `choices` for
  multiple choice), and may carry `gold_ids` (the items that hold the
  evidence), `solution`, `skill` and `scale`. Common column names are matched
  for you; the preview lists what is still unmapped.
- **Sources:** every import is a source named `upload:<name>`. Switch a source
  off and retrieval ignores it (the last item source must stay on); a run
  records which were off, and resuming a run after a switch is refused.
  Deleting a source removes its items and chunks. The Sources table counts items
  only, and deleting a source never deletes questions.
- **Re-import updates:** items are matched by id (the hash of their content
  only decides whether one changed), so importing the same file again reports
  "N new, N updated, N unchanged" and adds no duplicates. A load that died is
  finished with "Finish loading", which embeds only what is missing.
- **Leaked questions:** an `upload:` or hand-written question whose text is inside
  the store's items is leaked: the answer is in what is retrieved. A question new to
  a set and saved with no `meta.source` (added on the eval set page, or by
  `PUT /eval-sets/{name}/questions`) is saved as hand-written, `source: "hand"`.
  Leaks are worked out per store and eval set when a run is set up, before the dev
  slice, so the same
  eval set can have leaks against one store and none against another. Leaked
  questions are left out of runs unless the eval set includes them (a setting of the
  eval set, `PATCH /eval-sets/{name}`). The Run page says how many were left out.
  Questions saved before this change may still carry an old `meta.leaked` mark; the
  first re-import of the same file counts them as updated and saves one new version.
- **Partial gold:** recall is measured only on questions that name their
  evidence (`gold_ids`). A recall cell reads "on n of N questions" when some
  questions have none. A question with no evidence is still asked, but it
  does not count toward recall.
- **Evidence across stores:** a question's evidence ids count only where the store
  holds them. If the store holds none of a question's evidence, its recall is
  unmeasured, not 0. If it holds some, the metric reads "on n of N". The run's
  estimate returns `evidence_named` and `evidence_held`, and the Run page says so
  when a store holds none.
- **Eval sets:** `/eval-sets` lists and creates them (a name and a QA type, and
  optionally the store they are labelled against). On `/e/{name}` edit questions in
  place, add or delete them, and Save; import a file; switch "include leaked"; read
  the versions; or delete the set. A Documents set's page has no Leaked questions
  section. A set with a linked store (any type) offers **Run against its store**,
  which opens the store's run form with that set chosen.
- **Re-importing into a Documents set:** for a question whose id is already there,
  a file that carries `judged_ids` (a `judged_ids` or `judged` column; an export
  does) replaces the stored question whole. A file without them keeps the stored
  judged passages and typed answers (unless the file has answers of its own), joins
  its `gold_ids` to the stored ones, and takes the file's text and `meta`. Either
  way judged still holds gold. Other types replace the whole question. The
  `gold_answers` column is read as an `answer` synonym, so the `questions` of
  `GET /eval-sets/{name}`, saved as `.json`, imports back with its labels.
- **Versions:** every write that changes the questions saves a snapshot. Its id is
  the first 12 hex digits of the SHA-256 of the questions' canonical JSON, sorted by
  id. Save sends the `base_version` it started from and is refused with 409 when the
  set has moved on. A run records its eval set when queued and the version at its
  first run, and always reads that version, so later edits never change a run and a
  resume asks the same questions. The migration made a set for every existing
  collection that had questions, and pinned every existing run to that snapshot.
  A set saved unchanged keeps its version id, including a set from before
  `judged_ids`, so old runs are not flagged "eval set changed". Per-question saves
  and deletes each make a version too.
- **View and Restore:** `/e/{name}` lists the newest 20 versions, and **Show all
  versions** lists the rest. **View** (`GET /eval-sets/{name}?version=`) shows an old
  version's questions (`Version <v>: <n> questions`), with `Back to the current
  version`. **Restore this version** saves those questions as a new version, the
  whole-set save with the current `base_version`; it says `Restored version <v>.`
  Ids the set once held are not marked hand-written again. It is disabled while the
  page has unsaved edits or unsaved marks, or a question is being saved. While its
  save is under way the labelling panel holds still: marks, Find, Save, Label and
  Delete are disabled. Then the panel clears, unless you typed in it meanwhile, when
  it is left as it is (its next Save is checked against the question as you read
  it). The current version is always listed, marked `current`, even when a Restore
  has made an older one current again.
- **Deletes:** deleting a store keeps its eval sets. Deleting an eval set deletes the
  runs that used it, after a confirm that gives their count; it is refused (409) while
  an eval or a load uses the set.
- **Copies:** a copy points at the source's default eval set and copies no questions.
- **Existing collections** are labelled with their sources in place on first
  start (the `backfill` job, shown as "labelling"); nothing is re-embedded.
```sh
curl -X POST "http://localhost:8080/collections/notes/uploads?filename=notes.jsonl&kind=items" \
  --data-binary @notes.jsonl
# {"upload_id": "...", "preview": {...}, "estimate": {...}}
curl -X POST http://localhost:8080/collections/notes/imports \
  -H 'content-type: application/json' \
  -d '{"upload_id": "...", "kind": "items", "mapping": {"text": "body"}, "source_name": "notes", "confirm": true}'

# questions go to an eval set
curl -X POST http://localhost:8080/eval-sets -H 'content-type: application/json' \
  -d '{"name": "notes", "qa_type": "science"}'
curl -X POST "http://localhost:8080/eval-sets/notes/uploads?filename=qs.jsonl" \
  --data-binary @qs.jsonl
curl -X POST http://localhost:8080/eval-sets/notes/imports \
  -H 'content-type: application/json' \
  -d '{"upload_id": "...", "mapping": {"text": "q", "answer": "a"}, "source_name": "qs"}'
```

### Rebuild with another model

The store page's **Embedding model** section shows what the store was built with and
offers another. Rebuilding is a free background job (`reembed`): it embeds every item
and chunk again on this machine, with progress and Stop job. Search keeps the old model
until the switch, and the switch itself, a few seconds, refuses search, deleting a
source and copying. If the job stops, **Resume** carries on: before the switch it embeds again from the start; a stop during the switch resumes the switch. If the copy made for the switch is gone, delete the store and load it again. Lists are re-keyed, but
tags and the vocabulary are kept, so if the build tags by similarity, re-tag by PUTting
the vocabulary. A run from before the change is marked `dense model changed` and can't
be resumed. Until the switch the store holds two copies in Qdrant (`<name>__rebuild`),
about twice its disk.

## Recipes

A recipe is a set of build choices, picked when a collection is created
(`recipe` on `POST /collections`) or applied to a copy of another collection's
items (`POST /collections/{name}/copy`). Each stores its choices with the
collection, and `PUT /collections/{name}/search` sets its default search settings.

| Recipe | For | Build |
|---|---|---|
| `worked_examples` | Solved problems (science, multiple choice) | One unit per item; an induced vocabulary; items tagged by the AI from their solution; abstracted examples; the classifier |
| `reference` | Passages (multi-hop, tables) | 100-word chunks with 10 words of overlap; the title put before the text; an induced vocabulary; tags by similarity; the classifier |
| `paper_2024` | The paper's method (the default for the types above) | 100-word chunks with 25 words of overlap; the paper's three keyword lists (yours, or AI-proposed) and optional summaries |
| `documents` | Passages as uploaded (Documents stores; the default) | The uploaded passages kept whole; no AI; searched by meaning and by words |
| `documents_tagged` | The same, with keyword tags (Documents stores) | An AI-made vocabulary; passages tagged by similarity; reached by Add keyword tags, not chosen |

A Documents store takes only `documents` or `documents_tagged`, and the other types
can take neither. The New store form offers each type's own.

Build steps that call the AI (inducing a vocabulary, AI tagging, abstracted
examples) are priced with the `keywords` job's model and shown as an estimate
first; with no model set up for `keywords` they return 503 rather than spend. The
question's `meta` (ScienceQA's skill, topic and lecture) never reaches those
prompts. A recipe's vocabulary is reviewed through the API: `GET
/collections/{name}/vocabulary` exports the terms, and `PUT` with the edited list
re-tags every item (see [Endpoints](#endpoints)). A re-tag that stops after it
has begun writing leaves `keyword_rag` refused until the vocabulary is saved
again, so no search mixes two versions' tags. Without an AI model a
paper-2024 load saves no keyword lists and the collection reads `no_lists`; the
old TF-IDF lists are gone.

### Published builds

A build can be published to a Hugging Face dataset repo, so that anyone loading the
same dataset, `limit` and recipe gets the same vocabulary, tags and abstracted
examples with no AI call and no model set up. The repo is
[`jonathanmiro97/keyword-rag-builds`](https://huggingface.co/datasets/jonathanmiro97/keyword-rag-builds)
unless `RAGQA_BUILD_HUB` names another; set to empty, loads never read it.

- **A load applies a matching entry.** When a load has something to build, the
  estimate and the load read the repo's `manifest.json` at one commit, then that
  entry's files at the same commit. Each file must match its sha256. The entry is
  applied only when it was built with the collection's build settings, and covers
  exactly the items the collection will hold, each one usable, and only while the
  collection has no vocabulary in force: an entry never replaces a vocabulary built,
  authored or edited here. This is checked before anything is added. Otherwise, and
  whenever the Hub cannot be reached, the load builds locally as usual, and its
  estimate and result say why (`hub: {skipped: ...}`). A load that applies one
  reports `hub: {entry, items}`.
  There is one exception. A load the entry made free is queued without `confirm`, so
  if the entry is no longer usable when that load runs, it fails and adds nothing,
  rather than spend what nobody agreed to.
- **Any model loads it.** A Hub build holds text only (no embeddings), so it loads into a
  store of any embedding model.
- **Only the owner publishes.** You run the publish script inside the api
  container. It is a dry run unless you pass `--publish`, and it refuses a token
  that does not belong to the repo's owner:

  ```sh
  docker compose exec -e HF_TOKEN api python -m ragqa_core.publish aops \
      --authored-by "who made the build" [--vocabulary vocabulary.json] [--publish]
  ```

  It sends a collection's vocabulary (`vocabulary.parquet`, with the definitions from
  `--vocabulary` if given), each item's tags and abstracted form (`outputs.parquet`,
  keyed by the dataset's own document ids), the manifest (with the build settings),
  and a dataset card crediting the source dataset, all in one commit. Nothing else
  leaves: no embeddings, keys, connections or ledger rows. It refuses a build a load
  would not apply, and a collection with a job running on it.
- **Authored outputs.** Outputs written by hand go into a collection with
  `python -m ragqa_core.apply_outputs COLLECTION VOCABULARY.json OUTPUTS.jsonl`.

## Retrieval methods

| Method | Origin | What it does | Cost |
|---|---|---|---|
| `question_only` | paper | No retrieval; zero-shot baseline | free |
| `vector` | paper | Parent-document retriever over 100-word chunks | free |
| `keyword` | paper | BM25 on the full text | free |
| `hybrid` | paper | BM25 + dense, fused with reciprocal rank fusion | free |
| `kw_all` | paper | Top 9 keywords from the pooled lists (+ summary), one vector | free* |
| `kw_individual` | paper | Top 3 from each list (+ summary), one vector; the paper's best | free* |
| `kw_rrf` | paper | One vector per list (+ summary), fused with RRF | free* |
| `hybrid_rerank` | modern | `hybrid` top 50, or `rerank_pool` (1 to 50), reranked by the run's `rerank_model` (runs: the Settings reranker at queue time, which the run records; the Lab and `/query`: the service's reranker) | free, slower |
| `kw_hybrid` | new | `kw_individual` fused with `hybrid`: a dual index like Doc2Query++ | free* |
| `keyword_rag` | new | Meaning (item vectors for worked examples, chunks for reference), BM25, and the question's predicted tags against each item's `kw_tags`, fused by weighted RRF with the collection's channel weights; needs a vocabulary | free* |
| `iterative` | modern | Up to 3 retrieval hops; the AI decides the next query | paid per hop |
| `random_example` | control | Random items, seeded by the question: the same draw on every run | free |
| `fixed_example` | control | The same random items for every question | free |

\*Searching is free. The summaries, an AI keyword-list proposal and a recipe's
AI build steps cost once, at load time, and only when asked for.

The two controls answer a question the paper could not: in 2024 plain vector
search (63.1%) scored below no retrieval (69.7%), which is either a badly
chosen example or any example at all getting in the way. A method that does
not separate from `random_example` has not shown that choosing the example
helped. For the exam QA type they also give the chance level of the free
metrics: the solution similarity and same-skill rate of an unchosen example.

Dense queries carry a task instruction per QA type when the dense model was
trained with one (Qwen3-Embedding only): `Instruct: <task>\nQuery:`
before the query, nothing before documents. A bge store gets none. Each eval run records the exact
prefix in its setup.

The RRF constant is a request parameter `rrf_k`: the paper used 1, today's
default is 60. Fusion is computed in this service, not by Qdrant, so both
settings behave the same in every mode.

`keyword_rag` predicts the question's tags from the collection's vocabulary in
one of four ways (`query_tags`): `classifier` lets the 20 nearest items vote
with their tags, weighted by similarity; `similarity` takes the terms nearest
the question, as the paper did; `model` asks the query job's AI to pick up to
`tag_k` terms (one paid call per question, priced in an eval's estimate, falling
back to `similarity` when the reply names none); `oracle`, in evals only, takes
the tags of the question's gold items (or of its worked solution), a ceiling for
the others.
A channel weighted 0 is not searched.

## Answering

Worked examples are shown to the answering model in the form the search
settings' `example` names: `abstracted` (principle, method, answer), `full`
(the paper's question, answer and explanation) or `off`. Reference passages
are numbered best first and last (ranks 1, 3, 5, …, 4, 2) and the model is
asked to cite them as `[n]`.

With `program` on, a question without choices is answered by a program: the
model writes a short Python block that prints the answer, and the service runs
it locally and takes the last line it printed. A program that fails, times out
or prints nothing falls back to the model's `Answer:` line. The run starts in a
fresh temp directory with no secrets in its environment, and gets 10 s, 512 MB of
memory (on Linux), files of at most 1 MB each, 2,000 characters of output, and no new
processes or threads. A limit the platform refuses (any of them when the service
runs as root, the memory limit off Linux) is skipped.

`program` is on by default for new science and table collections; collections
made before recipes keep it off. Turn it off with `PUT
/collections/{name}/search` `{"program": false}` for any corpus you did not
write. The program is not network-isolated. It can read any file the service
can read, including the database that holds stored API keys, and write, truncate or
delete any file the service can write, such as that database or Qdrant's store: the
temp directory is where it starts, not a boundary. Its code comes
from the AI model set up for the `answer` job, but that model reads the
question and the passages, so text planted in a corpus can steer what it
writes.

## Chat

A chat searches every store it lists at once, pools their best passages, reranks
the pool against the question and answers from the top. The evidence is shown
under every answer. Chats are saved on the server, with their own settings. The
single-store `POST /chat` runs the same pipeline for one store.

For each message the service does this:

1. **Search text.** The first message is searched as typed. A follow-up is
   searched as the message plus the previous 2 user messages (Settings'
   *Previous messages joined, without rewrite*), one per line, and reranked
   against the message alone. With **Rewrite follow-ups** on and any AI
   connected, the `rewrite` job turns the follow-up into one standalone
   question, which is both searched and reranked against: one paid call per
   follow-up, never for the first message. A failed rewrite falls back to the
   joined text and says so.
2. **Candidates per store.** Each store is searched with its own method and
   returns its top *candidates* (default 20). A method the store cannot run (a
   keyword method before it has a vocabulary, say) is replaced by hybrid with a
   note: `keyword_rag needs a vocabulary, so hybrid was used`. A method that
   would call AI when none is connected (a paid method, or `keyword_rag` with
   model tags) is replaced the same way: `<method> needs AI, so hybrid was used`.
3. **Pool.** The stores' lists are pooled round-robin by rank (every store's
   rank 1, then rank 2, and so on, in the chat's store order), skipping a text
   already taken, so the same passage in two stores counts once. The pool is
   capped at **Rerank up to** (Settings `rerank_pool`, default 20), but never
   below the evidence size.
4. **Rerank.** The pool is scored against the question by the Settings
   reranker model (a local cross-encoder, so free) and sorted by score. Each
   passage shows its relevance, the logistic of the score.
5. **Evidence.** The top *evidence size* passages (default 8, 1 to 20) go to the
   answer model, which cites them as `[n]`. When no store finds anything, the
   model answers without passages, and the page says so.

With **Rerank** off, the pool keeps its round-robin order and is scored
`1 / (rrf_k + rank)`, which is reciprocal rank fusion over the stores' lists.
The same happens when the reranker cannot be used, with a note:
`reranker still loading: merged by rank` (it is never waited for) or
`reranker unavailable: merged by rank` (the load failed, and a retry starts in
the background).

**Warm-up.** The reranker loads in the background when the service starts, and
again when Settings picks another one, so a chat does not wait for it on its
first message. It scores in ONNX batches of 16 and the embedder in batches of 8.
The first start downloads the model into the `ragqa` volume (about 0.13 GB for
the default).

**Cached answers and notes.** An answer served from the AI cache shows
`Cost $0.00 · answer cached` (the answer only: a paid rewrite is not marked). The
chat's own notes (an added file, a failed send) are kept in this browser across a
reload, and dropped when the chat is deleted. On a phone the conversation comes
first. The header keeps the store chips, Add store, Model, Add document and Use from
your app; the retrieval settings (Evidence size, Rerank, Rewrite) are in *Settings
for this chat*, under the latest message.

**Passages only, without AI.** With no answer model, Send still works: the
reply is the retrieval step and the evidence, marked `Passages only: no answer
model`, and the page says `Connect AI in Settings to get written answers`. Nothing
is paid, and Answer from this attempt answers it once a model is connected.

**The retrieval step.** Under each answer, open the step to see the search text
and where it came from (message, joined or rewrite), what the pool was reranked
against and by which model, how many passages were pooled of all the stores'
candidates (`Pooled 20 of 60 (Rerank up to 20)`: Rerank up to caps the pool, so
more candidates may change nothing), each store's candidates and how many were kept, its
method (for `keyword_rag`, also the query tags it picked), the pipeline's notes
and the evidence with each passage's store, rank and relevance.

**The settings panel.** Under the latest message, *Settings for this chat*
holds the chat's own *Evidence size*, *Rerank* and *Rewrite follow-ups*, and per
store its *Search method* and *Candidates*. **Re-run** searches that message
again with the edits and adds an attempt without answering; **Answer from this
attempt** answers it (an answer cannot be repeated). A re-run saves the edits on
the chat, so its later messages use them. A store's **Set as default** saves
that store's method and candidates on the store (after which the chat inherits
them); the store page's *Chat settings* shows them, and **Reset to Settings** clears a value the store saved. Set as default is refused (409) when
the store cannot run the chat's method, and saves nothing. A panel edit that changes
nothing is dropped.

**Which value wins.** A blank field (null) inherits, highest level first:

- a store's *method* and *candidates*: the chat, then the store, then Settings,
  then the recommended value;
- the chat's *evidence size*, *rerank* and *rewrite*: the chat, then Settings,
  then the recommended value;
- *previous messages joined*, *RRF k*, *Reranker model* and *Rerank up to*
  are Settings-only.

The panel shows what is inherited, and from where.

**Old chats.** A chat or attempt from before this pipeline reads as recorded: its
decisions and evidence are shown as they were. Re-running an old message keeps
its search text, with no paid rewrite. The first start on an old database
migrates it once (one way; back it up first): the chat bar, sticky stores,
routing profiles, `store_bars` and the *routing* chat setting are dropped; a
chat's saved evidence size of 8 and rewrite off become blank (inherit); each
store's `chat_k` is cleared, because it now means candidates per store. The
migration writes the marker `schema.routing_dropped`; a later start only drops
leftovers and never resets settings again. The `chat_bar` and `route_profile`
columns stay in old databases, unused.

**Spending.** There is no estimate before a message. Each reply and attempt
shows its cost: the rewrite (one paid call for each follow-up when Rewrite
follow-ups is on and any AI is connected, **including in chats made before**),
any model-tag calls of `keyword_rag`, and the answer. Reranking is free. The
monthly cap holds. To stop paying for rewrites, set Settings → Search defaults →
*Rewrite follow-ups* → Off, or switch it off in one chat's panel. A re-run that
would rewrite again says so before it runs.

### Chat from your own app

Your own app can chat over the API with the same pipeline. Set up a chat in the
UI first (its stores, evidence size, rerank and so on), then open **Use from your
app** on the chat page and copy its lines. The example below is a chat over a
Documents store. Two calls are enough:

```sh
# start a conversation set up like chat <id>
curl -s -u user:password http://localhost:8080/chats \
  -H 'content-type: application/json' -d '{"from": "<id>"}'

# send each next message; the URL is quoted because zsh reads ? as a glob
curl -s -u user:password 'http://localhost:8080/chats/<id>/messages?brief=true' \
  -H 'content-type: application/json' -d '{"content":"How do I reset my password?"}'
```

`{"from": "<id>"}` copies the stores and the chat's saved settings, nulls
included, so the copy keeps following each store and Settings; it copies no
messages, and a store deleted since is left out (422 `a chat needs at least one
store` if none is left). Use the new chat's `id` in the second call. The server
keeps the history, so your app sends only the new message.

The short reply (`?brief=true`) has these keys: `n`, `answer`, `citations`,
`passages`, `passages_only`, `error` and `usd`. Each passage has `n`, `store`,
`id`, `text`, `file`, `source`, `part`, `parts` and `relevance`. `relevance` is the
reranker's score as a logistic, 0 to 1, or null when the pool was merged by rank.
`citations` are the passages the answer cites, as saved on the attempt. The raw score is left out. Only this route has the short form; re-runs and
answers keep the full shape.

```json
{
  "n": 1,
  "answer": "Open Settings, choose Security, then Reset password [1].",
  "citations": [
    {"n": 1, "id": "handbook.md#3f9a1c07b2de", "text": "handbook.md › Account › Password\n\nOpen Settings, choose Security, …",
     "file": "handbook.md", "source": "upload:handbook", "part": 2, "parts": 5, "store": "handbook"}
  ],
  "passages": [
    {"n": 1, "store": "handbook", "id": "handbook.md#3f9a1c07b2de",
     "text": "handbook.md › Account › Password\n\nOpen Settings, choose Security, …",
     "source": "upload:handbook", "part": 2, "parts": 5, "relevance": 0.93}
  ],
  "passages_only": false,
  "error": null,
  "usd": 0.0004
}
```

- **Retrieval for your own model:** send `"answer": false` to get the reranked
  passages with no answer written and nothing paid for one. This is the mode for an
  agent harness with its own model, such as Hermes Agent, which curls the endpoint
  from its shell tool. `answer` null with `passages_only` true means no
  answer was written, either by request or because no answer model is connected.
  The attempt stays unanswered, so **Answer from this attempt** on the chat page can
  still answer it.
- **Check `error`:** a failed answer is still HTTP 200, and the message is saved.
  Retrying would save it twice.
- **See why:** every API conversation is a saved chat. Open it on the chat page to
  see the retrieval step behind each answer.
- **Auth:** the service login (`curl -u user:password`). Put HTTPS in front before
  exposing it beyond localhost. There are no API keys, CORS or streaming yet.
- **Reference:** the OpenAPI docs are at `/docs`.

## Search defaults

Settings → *Search defaults* holds app-wide values; each shows its recommended
value, and **Reset to recommended** puts them all back. A blank or reset field is
null and follows the recommendation. A stored value is checked each time it is
read: one that no longer passes (a reranker fastembed dropped, a method that is not
a chat method) reads as null. Save starts loading a reranker only when you picked a
new one, or the one in force had failed to load.

| Option | Recommended | Basis |
|---|---|---|
| Default search method | `hybrid` | X1 on BRIGHT AoPS: hybrid .376 against the label channel .314; hybrid is the standard default |
| Candidates per store | 20 | a common pool size before a reranker (20 to 100) |
| Evidence size | 8 | the earlier default |
| Rerank | on | the standard multi-source pipeline |
| Rewrite follow-ups | on (needs a rewrite model) | the standard "condense question" step |
| Previous messages joined, without rewrite | 2 | the earlier default |
| RRF k (rerank off, and inside hybrid) | 60 | the standard constant; X3 found no arm that beat it (best, k 10, was within noise) |
| Reranker model | `jinaai/jina-reranker-v1-tiny-en` | about 1.5 s for 20 passages on an M4 Pro; see [Reranker timing](scripts/experiments.md#reranker-timing) |
| Rerank up to | 20 (1 to 60) | the most pooled passages reranked per message |
| Embedding model for new stores | `BAAI/bge-small-en-v1.5` | fast; timing pending (`scripts/time_embed.py`, [Embedding timing](scripts/experiments.md#embedding-timing)); its retrieval quality on these benchmarks has not been measured |

The basis for Rerank itself is X5 (hybrid against hybrid reranked at 20,
[scripts/experiments.md](scripts/experiments.md)); its result is pending.

**The embedding model.** Each store keeps the model it was built with, and new stores
take this setting. `RAGQA_DENSE_MODEL` replaces the default and is then the recommended
one. The page offers `BAAI/bge-small-en-v1.5` (384 dimensions, 0.067 GB),
`BAAI/bge-base-en-v1.5` (768, 0.21 GB) and `Qwen/Qwen3-Embedding-0.6B-Q` (1024, 1.12 GB),
plus the service's model and any saved value; the API accepts any model fastembed
lists. Stores of different models can share a chat, because chat reranks or merges by
rank, never by score. The bge models read at most 512 tokens, so a long document is cut
short. **X1–X4 were measured on stores embedded with Qwen/Qwen3-Embedding-0.6B-Q**, so
their numbers say nothing yet about bge-small. The tag cut-off (0.5) and the term merge
(0.88) were tuned on Qwen's scores; bge's are likely higher (unmeasured), so a similarity build
or an induced vocabulary on a bge store may keep more tags or merge more terms.

**The reranker** is a Setting. The default is `jinaai/jina-reranker-v1-tiny-en`,
about 1.5 s for 20 passages on an M4 Pro, where `BAAI/bge-reranker-v2-m3-int8`,
the earlier default, took 11 to 18 s. Any cross-encoder fastembed supports can be
chosen; picking one downloads it in the background (chat merges by rank until it
is ready). The page says while it downloads, and if it failed, why; Save then tries
again at once. **Rerank up to** caps how many pooled passages are scored, so a
smaller number is faster; it is never below the evidence size.
`RAGQA_RERANK_MODEL` replaces the default, and it is then the recommended one.

**Going back to bge.** Pick `BAAI/bge-reranker-v2-m3-int8` under Settings →
Search defaults → Reranker model, or set `RAGQA_RERANK_MODEL=BAAI/bge-reranker-v2-m3-int8`,
which also makes it the recommended one. It is slower: about 11 to 18 s for 20
passages on an M4 Pro.

**Where they apply.** Chat, `POST /chat`, and `POST /query` when it sends no
`methods` or `rrf_k`. Runs (`/eval`) and the Lab choose their own, with one
exception: a `hybrid_rerank` run with no `rerank_model` takes the Settings
reranker when it is queued and records it in `setup.rerank_model`, so a resumed
or repeated run uses the same one. Runs from before recorded no model and used
`BAAI/bge-reranker-v2-m3-int8` at 50. A run page names a `hybrid_rerank` run's
reranker and pool in its head, and **Re-run** sends both, so the repeat is the same run.

## Datasets

All are downloaded at runtime into the `ragqa` volume and never committed.

| QA type | Dataset | Licence | Used as | Free retrieval metric | Paid answer metric |
|---|---|---|---|---|---|
| Science problems | SciBench (physics and chemistry textbook problems) | MIT | 273 examples (76 worked solutions) and 219 questions: solved problems are all examples, the 416 unsolved split by a hash of their id. Small, a stand-in until your own corpus is imported | same-skill@1 | accuracy |
| Exam multiple choice | ScienceQA | CC BY-SA 4.0 | 693 solved train examples; 291 test questions | solution similarity@1, same-skill@1 | accuracy |
| Exam multiple choice, hard | GPQA (gated) | CC BY 4.0 | GPQA Extended minus Diamond as examples; Diamond's 198 as questions | solution similarity@1, same-subdomain@1 | accuracy |
| Reasoning-intensive retrieval | BRIGHT `bright_aops`, `bright_theoremqa` (xlangai/BRIGHT) | CC BY 4.0 | Each task's documents as examples (the pool of both is 188,002 documents; use server Qdrant, or a load `limit`); its questions, with the documents to exclude removed from every search | nDCG@10, recall@1 | none: BRIGHT has no gold answers |
| Multi-hop factual | MuSiQue (HippoRAG-2 protocol) | CC BY 4.0 | 11,656 passages; 1,000 dev questions | recall@2, recall@5 | EM, F1 |
| Table + text numeric | TAT-QA | CC BY 4.0 | 2,757 contexts; 1,668 dev questions | recall@1, recall@3 | EM, F1 |

ScienceQA is filtered to text-only natural science, grades 7–12, keeping one
item per near-duplicate template. Its human-written solutions serve as the
paper's chain-of-thought examples. TAT-QA answers use simplified scoring:
numbers are compared at 2 decimal places after stripping `$`, `,`, `%` and
scale words, and text is scored like SQuAD.

ScienceQA turned out too easy to separate methods on: plain vector search
finds a same-skill example for 84% of test questions (a random one, 11%),
and its templated solutions put every content method near 0.91 solution
similarity. GPQA is the harder stand-in for the paper's ACT data: expert
questions with a written explanation each. Diamond is held out of the
example store by record, so no question retrieves itself; choices are
shuffled with a seed per record. "Same skill" there means same subdomain.

GPQA is gated. Accept its terms on
[its Hugging Face page](https://huggingface.co/datasets/Idavidrein/gpqa) and
set `HF_TOKEN` to a read token from that account. The terms ask that examples
are not shown online, so keep screenshots of it private.

MuSiQue recall can be set beside published HippoRAG-2 numbers, but only as
**gain over our own dense baseline**: their baseline is NV-Embed-v2, ours a
small local model. EM and F1 come from a different reader and are not
comparable to the papers.

BRIGHT's questions name their gold documents, so a worked-examples collection that
has them also reports nDCG@10 (binary relevance) and recall@1, measured after the
question's excluded ids are removed. Answering runs on BRIGHT score 0; use SciBench
or ScienceQA to compare answers. Copy a collection under another recipe to compare
recipes on the same items.

## AI connections

The **Settings** page (`/settings`) sets up which AI the service calls. Nothing
is needed in `.env`.

- **Connections.** A connection is a provider with an address and a key.
  Presets cover Anthropic, OpenAI, Gemini, xAI and OpenRouter;
  **Custom** takes any OpenAI-compatible server. Test it, then fetch its model
  list. Keys are stored in the database file (mode `0600`) and are never shown
  again or put in a log or response; the page shows a masked key.
- **Two kinds cover every provider:** Anthropic's Messages API, and the
  OpenAI-compatible Chat Completions API. A third, **Answered by hand**, sends
  nothing (see below).
- **Prices.** A model's price is per million input and output tokens. A model
  that is fetched or added by id starts at $0; on the Settings page choose $0,
  a listed price (the Anthropic presets' rates), "No price" or "Enter by hand…".
  At $0 a model's calls cost the budget nothing, so the cap does not stop them,
  and the page says so: price a paid model before you spend with it. A model
  with no price is refused before anything is sent.
- **Jobs.** Each job picks a model, with `default` for any job left alone.

  | Job | Used for |
  |---|---|
  | `default` | anything not assigned below |
  | `answer` | answers in the Lab and in evals (a run can pick its own model) |
  | `summary` | per-item summaries for the keyword methods |
  | `keywords` | AI keyword-list proposals |
  | `query` | paid query-time steps: query summaries, keywords and hops |
  | `rewrite` | the chat's optional follow-up rewrite: one AI call turns a follow-up into a standalone question |

- **Budget.** The monthly cap is set here too. This month's spend counts every
  connection together.
- **Environment.** `ANTHROPIC_API_KEY`, `RAGQA_MODEL` and
  `RAGQA_MONTHLY_CAP_USD`, when set, are applied at every start and cannot be
  edited on the page. The key is stored as the Anthropic connection, at the
  preset's address, and that connection cannot be removed while the variable is
  set. If you later remove the variable it stays as an ordinary stored key you
  can edit. Changing a connection's base URL without giving a new key clears its
  stored key, so a key is never sent to an address it was not given for.
- **Upgrading.** An existing database keeps its ledger, cached answers and
  pending batches; this month's spend still counts against the cap. A batch
  submitted before the upgrade is collected through the Anthropic connection,
  so keep the Anthropic key (the env key, or one added on the page) until
  those batches finish.

**Model refs** are `<connection id>/<model id>`, split on the first `/`, so
OpenRouter ids like `openrouter/meta-llama/llama-3.1-8b` work. A bare model id
is accepted when exactly one enabled model has it.

### Answered by hand

A connection of kind `manual` sends nothing: a person, or an assistant session,
writes the replies. Add one with the **Answered by hand** preset, add its models by
hand and give each a price (0 is allowed), then choose one for a job.

A job that asks such a model something its cache does not hold stops with
`stopped_replies`, and its prompts wait. Each AI step of a job asks all its prompts at
once, so a stop holds a whole step: an eval's tags, then each method's answers; a
load's vocabulary, then its tags and its abstracted examples. Inside the api
container:

```sh
python -m ragqa_core.replies export prompts.jsonl [--job JOB_ID]
# one {"key", "model", "purpose", "job_id", "max_tokens", "system", "prompt"} a line
python -m ragqa_core.replies import replies.jsonl
# one {"key", "reply"} a line, for any of the waiting prompts
```

Then resume the run (or load again): the job finds the replies cached and stops again
at its next step that asks for something new. A reply is recorded as the model's
answer, ledgered to its job at the model's price on estimated tokens. An import is
refused whole, recording nothing, if a reply is empty or answers no waiting prompt.
Removing the connection drops its waiting prompts.

## Configuration

Environment variables, read at start. `compose.yaml` passes `.env` through.
All are optional.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Becomes the Anthropic connection: stored, shown read-only on the Settings page, applied at every start |
| `RAGQA_MODEL` | The `default` job's model: `<connection>/<model id>`, or a bare model id that one enabled model has, e.g. `claude-haiku-5-5`. A value that names no stored model is taken as an Anthropic model id and added, at $0, to the Anthropic connection when there is one; price it on the Settings page. Read-only on the Settings page while set. With only a key set, the default starts as `claude-haiku-5-5` and stays editable |
| `RAGQA_MONTHLY_CAP_USD` | The monthly cap. Read-only on the Settings page while set; unset, the stored cap stands (default `20`) |
| `RAGQA_USERNAME`, `RAGQA_PASSWORD` | HTTP basic auth over every route. Set both or neither; exactly one set is a 503 on every request |
| `RAGQA_DENSE_MODEL` | Retrieval embedding model (fastembed). Default `BAAI/bge-small-en-v1.5`; it was `Qwen/Qwen3-Embedding-0.6B-Q`, which ran hot and slow on a laptop CPU. It is the recommendation for new stores, not a setting for old ones |
| `RAGQA_JUDGE_MODEL` | A different model, used only to score solution similarity. Default `BAAI/bge-small-en-v1.5` |
| `RAGQA_RERANK_MODEL` | The default reranker (fastembed cross-encoder) for chat and `hybrid_rerank`, and the one Settings recommends. Default `jinaai/jina-reranker-v1-tiny-en`; it was `BAAI/bge-reranker-v2-m3-int8`, which is slower on CPU. Settings → Search defaults can pick another |
| `RAGQA_QDRANT_URL` | Set by compose. Unset means embedded Qdrant in the API process (fine for a small store; BRIGHT needs server Qdrant, see `scripts/experiments.md`). The store serializes its calls to it, so pages can poll while a job writes; it waits out a rebuild's switch, and reads a store whose search data is gone as empty |
| `HF_TOKEN` | Hugging Face token: read, to load GPQA (gated); write, only for the owner's `ragqa_core.publish --publish` |
| `RAGQA_BUILD_HUB` | The Hub repo of [published builds](#published-builds) a load applies. Default `jonathanmiro97/keyword-rag-builds`; empty turns it off |
| `RAGQA_HOME` | State directory. `/home/app/.ragqa` in the image |
| `PORT` | Listen port. `8080` |

A store keeps the embedding model it was built with and embeds its questions with
it. Changing `RAGQA_DENSE_MODEL` changes only the recommendation for new stores.
A store whose model can't be loaded refuses search and new items with a 409.

## Cost safety

- **The provider's own spend limit is the real ceiling.** The service only
  sees its own calls; other keys on the same account draw on the same budget.
  For Anthropic that is the Console workspace limit from the quick start.
- **A monthly cap inside the service.** Each call is priced from the
  provider's reported token usage and written to a ledger. A call that would
  cross the cap is refused before it is sent, and an evaluation that hits the
  cap stops with `stopped_budget` and keeps its rows.
- **No price, no call.** A model with no price is refused before anything is
  sent. A model priced $0 is free to the budget, so price paid models.
- **Estimate before spend.** `/eval`, resumes, paid loads, paid `/documents`,
  imports, keyword-list proposals, Documents answers and Judge answers, and Get
  the AI's answer on the labelling panel return the estimate unless
  `confirm: true` is set. The Lab (`/query`) and the chat (`/chat`, `/chats`) are the
  exception: they spend at once and show each reply's and attempt's cost after
  (the follow-up rewrite, one AI call per follow-up, is part of it), and the
  monthly cap still holds.
- **An estimate may be above the cost, never below it**, apart from prompt sizes
  taken from a sample of the questions. Only more than 50 cache misses go to the
  Batch API, and a new run's answer prompts can't be checked against the cache
  beforehand, so a part of a run with more than 50 calls is priced as if up to 50
  of them missed and were sent real-time: `max(n × batched, 50 × full)`. A
  51–100 call part can therefore show up to about twice its likely cost.
- **Every response is cached.** A re-run with the same prompts is free, and its
  answers are identical.
- **Batches.** With the Anthropic connection, runs of more than 50 uncached calls
  use the Batch API at half price; other connections send one call at a time. A
  batch usually finishes in minutes but can take up to 24 hours.
- **Localhost only.** The port is published on `127.0.0.1`. Opening it wider
  without basic auth would let anyone on the network spend up to the cap.

Rough prices on Haiku 5.5: a one-shot multiple-choice answer (400 answer tokens
at most) is about $0.0005, and a 200-question × 10-method evaluation with
answers about $1. A science answer works the problem out and may use up to
1,500 answer tokens, so science runs cost more. Sonnet costs about 20 times as
much.

## Pages

The side nav reads Stores, Chat, Eval sets, Lab, Settings. The store pages live under `/c/` and the API under `/collections`; both mean a store. The side nav's **Theme** button cycles System, Light and Dark, remembered in this browser.

| Route | What it shows |
|---|---|
| `/` | **Start here** (four steps, hidden once a store, a chat and a run are done), then every store with its answer (does keyword retrieval help?), the run it came from, its state and, if its last load stopped, `Last load stopped: <error>`; add a default store (one that exists is opened instead), create a store (the form leads with Documents, offers each type's own recipes and says `Embedded with <model> (Settings → Search defaults)`) or set up the benchmarks; **+ New store** → **Create** opens the new store |
| `/c/{name}` | One store: **Chat with this store** (disabled while it is empty), the answer with its gap bar, its runs and a form to start one (pick the eval set; **+ New run** stays disabled until the store has loaded, and `?eval_set=` opens the form once with that set picked), its items and sources, imports, its default eval set (pick another of its type, or none) with how many of its questions leaked, the three keyword lists (edit, propose, save), its **Embedding model** (what it was built with; rebuild with another, with progress, cancel and resume), for a Documents store its **Keyword tags** (`Not built` with an **Add keyword tags** button that shows the estimate first, `Adding keyword tags: <pct>`, a stopped build with Resume, or `Built: <n> terms`; a Documents store has no keyword lists section and its imports take `.md` and `.txt` only), its **Chat settings** (the method and candidates a chat uses for it, each with where the value comes from, and Reset to Settings for one saved on the store), delete |
| `/c/{name}/runs/{id}` | One run, with the eval set and version it asked |
| `/chat` | Chat. The left lists your saved chats (chats are saved on the server; conversations that lived only in a browser tab before are not migrated). A chat has a list of stores. Each reply has `[n]` links to its sources (id, store, source, `part i of n` for a passage of a split file, the first 300 characters, expandable) and its cost. A reply shows a safe subset of Markdown: code blocks, `code`, bold, italic, `-` and `1.` lists, `#` headings as bold lines, and paragraphs; links, images, tables and HTML stay as typed text. Enter sends, Shift+Enter adds a newline. On a phone the conversation comes first. A cached answer reads `Cost $0.00 · answer cached`, and the chat's own notes survive a reload in this browser. Add document adds `.txt` and `.md` files to a store for good (an add that would spend shows its estimate with Add and Cancel). Send works without an answer model: the reply is the passages, with a note linking to Settings. Under each answer a **retrieval step** shows the search text, what it was reranked against, each store's candidates and the evidence, and under the latest message one **Settings for this chat** panel holds the chat's evidence size, rerank and rewrite, each store's method and candidates, Re-run and Set as default; **Use from your app** holds the curl lines to send the next message and to start a copy of the chat, see [Chat from your own app](#chat-from-your-own-app); see [Chat](#chat) |
| `/eval-sets` | Every eval set; create one, with a store to label against |
| `/e/{name}` | One eval set: edit questions in place, add, delete, Save, import, include leaked, versions (View, Restore this version, Show all versions), its linked store, and for a Documents set the labelling panel (Find passages) and a Run against its store link; delete the set |
| `/lab` | One question, methods side by side |
| `/worker` | The worker, its queue and what needs a person; a `tag_build` job is named Adding keyword tags |
| `/settings` | AI connections, models and prices, the model each job uses (each with a note on what uses it, under an intro saying AI writes chat answers, rewrites follow-ups and builds keyword tags, and that search and eval runs work without it), the monthly cap, and [Search defaults](#search-defaults), including Embedding model for new stores |
| `/docs` | Every endpoint |

## Endpoints

Every endpoint is listed with its schema at http://localhost:8080/docs. A JSON body
with a field its endpoint doesn't know is refused with 422 (`PUT
/collections/{name}/search` excepted, which says which search fields it knows).

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, and whether the `answer` job has an AI model (`{ok, ai}`) |
| `POST /collections` | `{name, qa_type: documents\|science\|mcq\|multihop\|table, summarize, recipe?: documents\|documents_tagged\|worked_examples\|reference\|paper_2024, dense_model?}`; no `recipe` takes the type's default: `documents` for a Documents store and paper 2024 for the others, as before. No `dense_model` takes the Settings one (422 for one fastembed does not list). 409 when the name is taken, or an eval set of that name has another type; 422 for a name ending in `__rebuild` or `__chunks`, which are kept for a store's search data |
| `GET /eval-sets` | Every eval set (the page, for a browser that asks for `text/html`; JSON otherwise) |
| `POST /eval-sets` | `{name, qa_type, store?}`; 409 when the name is taken, or a store of that name has another type. With `store` the set is linked to it (404 for an unknown store, 409 for another type), and becomes that store's default set when the store has none (409 while a load is under way on it) |
| `GET /eval-sets/{name}` | The set, its `store`, `answer_available` (true when an answer model is connected), its questions (each with `judged_ids`, null when unknown), its versions, `missing` (per question, the right passage ids the linked store no longer holds; every labelled passage when the store lost its search data, still a 200) and `questions_version`. `?version=` reads an old version (404 `That version of '<name>' doesn't exist`). Saved as `.json`, its `questions` import back |
| `POST /eval-sets/{name}/questions` | `{question}`: adds one question, with the id made from its text (`-2`, `-3` if taken), `source: "hand"`; returns `{id, version}`. 409 while a load writes to the set; 422 for a bad question |
| `PUT /eval-sets/{name}/questions/{id}` | `{question, base}`: saves one question; `base` is the question as read. 409 when the stored question differs, or while a load writes to the set; 404 when it is gone, 422 when an id does not match the address. Returns `{version}`; `{id}` may hold a `/` |
| `DELETE /eval-sets/{name}/questions/{id}` | Deletes one question, with no stale-base check; 409 while a load writes to the set; 404 when it is gone. Returns `{version}`; `{id}` may hold a `/` |
| `POST /eval-sets/{name}/answer` | `{question, passages, confirm}`: the AI's answer for Get the AI's answer; `passages` are 1 to 20 passage ids, and other fields are refused. Without `confirm`: `{estimate: {est_usd, calls}}` (`est_usd` null with no answer model). With it: `{answer, usd}`, one capped call, cached. Nothing is stored. 404 for an unknown set; 409 `The AI's answer is for Documents eval sets`; 409 `pick a store for this eval set first`; 422 `a question needs some text`; 422 `mark at least one passage Right that is in <store> first` when none of the passages sent is in the store; 503 when confirming with no answer model |
| `POST /eval-sets/{name}/pool` | `{question, labelled?}`: Find passages. Returns `passages` (each `id`, `text`, `file`, `part`, `parts`, `relevance`, which is null when ordered by search rank), `merged_by` (`rerank` or `rank`), `notes`, and `labelled` and `missing` for the ids sent that the pool did not hold. 409 `Find passages is for Documents eval sets` for a set that is not Documents, 409 with no store linked. Free |
| `PUT /eval-sets/{name}/questions` | `{questions, base_version}`: replace the set's questions, all or nothing; returns `{version}`. 409 when `base_version` is stale, or while a load writes to the set; 422 for a bad or repeated id |
| `PATCH /eval-sets/{name}` | `{include_leaked?, store?}`: whether leaked questions are used by runs, and the store its labels are made against (`null` unlinks). `store` only links: it does not change any store's default set. `include_leaked` cannot be null (422) |
| `DELETE /eval-sets/{name}` | Without `?confirm=true`, `{runs}`: how many runs would go. With it, deletes the set and those runs. 409 while an eval or load uses the set |
| `POST /eval-sets/{name}/uploads?filename=` | The raw file as the body; returns `{upload_id, preview}` |
| `POST /eval-sets/{name}/imports` | `{upload_id, mapping, source_name}`: imports in the request, upserting by id; returns the counts (new, updated, unchanged), `skipped` and the new `version` |
| `POST /collections/{name}/reembed` | `{model}`: queue a `reembed` job that re-embeds the store with another model into `<name>__rebuild`, then switches; free, no `confirm`. 404 for an unknown store; 422 for a model fastembed does not list; 409 while any job is queued or running on the store, for the model it already has, and `resume it first` while a switch has stopped part-way |
| `GET /collections`, `GET /collections/{name}` | Specs, each with `eval_set`, the store's default eval set (null for none); one store's `load_error` (why its last load stopped; null when the newest load did not fail), count, keyword lists, available methods, its saved chat settings `chat_method` and `chat_k` (candidates per store; null when none is saved) and `chat_settings`, what chat uses for it with the level each value comes from (`store`, `settings` or `recommended`), `chat_search` (the search a chat uses, as a run names it: `{method, rerank, run_method, before_rerank}`; a Documents store is scored on `run_method`), `state` (a running job's `loading`, `rekeying`, `retagging`, `tagging` (Add keyword tags), `copying`, `evaluating`, `importing`, `labelling` or `rebuilding`; else `queued`, `empty`, `no_lists` or `ready`; a Documents store with passages and no keyword lists is `ready`) with its job and progress, `keyword_tags` (for a Documents store; null for any other: `{state: none}`, `{state: building, job_id, progress}`, `{state: stopped, job_id}` or `{state: built, terms}`), question counts, its answer and number of runs; `rebuild` (`{job_id, status, model, from, progress, switching}`) while the newest re-embed is unfinished |
| `PATCH /collections/{name}` | `{eval_set, chat_method, chat_k}`, each optional: a field left out is kept. `eval_set` makes that eval set the store's default; `null` clears it, so the set named after the store (if any) is the default again. `chat_method` is a chat method (not `hybrid_rerank` or `iterative`) and `chat_k` the candidates per store (1 to 60); `null` for either goes back to Settings. Any other field, such as the removed `route_profile`, is 422. Returns the store as `GET /collections/{name}` does. 404 for an unknown store or set; 409 when the set's type differs from the store's, or, for `eval_set` only, while a load is queued or running on the store, or when the store cannot run `chat_method` |
| `DELETE /collections/{name}` | Delete a store, its items, lists and runs. Its eval sets stay. Its queued jobs are cancelled; 409 while one of its jobs runs |
| `POST /collections/{name}/documents` | Add your own items `{items: [{id, text, answer?, explanation?, meta?}], confirm}`; a Documents store takes plain `{text}` items here too (the hand-add path for an API client; the `.md` and `.txt` rule is for file uploads and imports); 409 while a load, re-key, `retag`, `copy`, `import`, `backfill`, `reembed`, `tag_build` or eval on the collection is queued or running, and while a rebuild's switch has stopped part-way. Over 20 items it queues an import job instead and returns `{job_id}`. Recipes other than paper 2024 run their build steps on the new items in the request; their cost is in the estimate. The estimate is `{est_usd, items, calls, summary_calls, keyword_calls}`. 503 when the add would make AI calls (summaries, or build steps) and that job has no usable model, even when they are priced at $0. `calls` and `est_usd` (optional, what the estimate showed): 409 `the price changed since you saw it: check it again` when the estimate now has other `calls` or a higher `est_usd`; left out, no check. While such a request runs, any job queued on the collection is refused with 409 |
| `POST /collections/{name}/uploads?filename=&kind=items` | The raw file as the body (up to 200 MB). Returns `{upload_id, preview, estimate}`: the first rows under a guessed field mapping, and what importing would cost. `accept=text` takes only `.txt` and `.md` files (422 otherwise); the chat uses it. On a Documents store, `kind=items` takes only `.txt` and `.md` files too: 422 `A Documents store takes .md and .txt files` for a zip, a table or JSON lines, and `<file> is not UTF-8 text; save it as UTF-8 and upload it again` for a text file that is not UTF-8 (nothing is saved). A source name is cleaned to letters, digits, space, `_`, `.` and `-`, at most 64 characters |
| `POST /collections/{name}/imports` | `{upload_id, kind, mapping, source_name, confirm}`; `kind` is `items` (the default), and `questions` is 422. Without `confirm`: the preview and estimate. With it: queues an import job and returns `{job_id}`; 422 when a required field is unmapped or no row is usable (`no usable rows`, as for a `.md` with no text); `calls` and `est_usd` (optional, the AI-call count and price the estimate showed): with `confirm`, 409 `the price changed since you saw it: check it again` when the estimate now has other `calls` or a higher `est_usd`; left out, no check. 503 when the import would make AI calls (summaries, or the build steps) and that job has no usable model, even when they are priced at $0; the 409 for a changed price comes first. 409 while a `tag_build` is queued or running on the store (it was priced on the recipe the build switches). |
| `GET /collections/{name}/sources` | Each source with its item count and whether retrieval uses it |
| `PATCH /collections/{name}/sources/{source}` | `{enabled}`: switch a source on or off for retrieval. 409 while an eval runs, or to switch off the last item source |
| `DELETE /collections/{name}/sources/{source}` | Remove a source's items and chunks; questions stay; 409 while a job is queued or running on the collection |
| `PUT /collections/{name}/search` | Set the collection's default search settings, in part (`{weights: {meaning, words, keywords}, query_tags, example, program}`), and return all of them; `/query` and `/eval` override them per request. 422 for an unknown key or value |
| `POST /collections/{name}/load` | Load `scienceqa`, `gpqa`, `musique`, `tatqa` or `scibench` (`GET /catalog` lists them as `datasets`); its questions go to the store's default eval set, upserted by id. When it would spend (summaries; an AI keyword proposal because an AI model is set up and a recipe that uses keyword lists has none; or the build steps of the other recipes: inducing a vocabulary while there is none, AI tagging and abstracted examples), without `confirm` it returns the estimate `{est_usd, items, calls}`; with `{confirm: true}`, or when it is free, it queues a job and returns `{job_id}`. 503 when the load would make AI calls (summaries, or the build steps) and that job has no usable model, even when they are priced at $0; a summarising store with no summary model is refused before the dataset is read. `calls` and `est_usd` (optional, what the estimate showed): the load is then priced even when confirmed, and with `confirm`, 409 `the price changed since you saw it: check it again` when the estimate now has other `calls` or a higher `est_usd` (without it, the fresh estimate); left out, no check. Whether the job proposes keyword lists is decided when it is queued, so a keywords model set up after that doesn't make it spend. 409 before anything is estimated or queued when the store's default eval set has another type |
| `POST /collections/{name}/copy` | `{name, recipe, build, confirm}`: a new store of this one's items under another recipe, with its switched-off sources. It points at the source's default eval set and copies no questions. `build` overrides the recipe's build settings in part; the recipe keeps its vocabulary, `chunk_overlap` stays below `chunk_words`, and `added_text: "summary"` needs keyword lists. Without `confirm`, returns `{estimate}` (embedding is free; summaries, and the build steps of the recipes other than paper 2024); with `confirm: true`, makes the collection and queues a `copy` job, which adds every item under its source, then builds (a paper 2024 copy takes the source's keyword lists in force instead of proposing any); returns `{name, job_id}`. 409 when the name is taken or a job is queued or running on the source; 404 for an unknown source; 422 for a bad name, recipe or build setting; 503 when the copy would make AI calls (summaries, or the build steps) and that job has no usable model, even when they are priced at $0; a name ending in `__rebuild` or `__chunks` is 422 |
| `POST /collections/{name}/keyword-tags` | Add keyword tags to a Documents store using the `documents` recipe: `{confirm?}`. Without `confirm`: 200 `{estimate: {est_usd, calls}}` and nothing is queued. With `confirm: true`: queues a `tag_build` job (the store's state is `tagging`), which switches the store to `documents_tagged` in place, then induces a vocabulary and tags every passage by similarity; returns 202 `{job_id}`. Until the worker starts the job, `GET /collections/{name}` still shows `recipe: documents`. 409 `only a Documents store without keyword tags can add them` for any other store, `add passages first` for an empty one, and while any job is queued or running on the store; 503 `connect a model for keywords first` when there is no keywords model (the recipe stays `documents`), and 503 when it would spend and no model is set up. Until every passage is tagged, `keyword_rag` is unavailable with `needs keyword tags: Add keyword tags on the store page`: a 409 on `PATCH /collections/{name}` `chat_method`, `/eval` and chat, and a per-method `error` (still 200) on `/query` |
| `POST /collections/{name}/keyword-lists/propose` | The AI proposes the topic, method and common lists; nothing is saved. Without `confirm`: the estimate `{est_usd, calls}`; with `{confirm: true}`: the lists |
| `GET /collections/{name}/keyword-lists`, `GET …/keyword-lists/{version}` | Every saved version of the lists, newest first; one version with its lists |
| `PUT /collections/{name}/keyword-lists` | Save `{topic, method, common}` as a version and queue a `rekey` job; returns `{version, job_id}`. The collection switches to the new lists once every item is re-keyed. 409 while an eval runs on it, or when the store's model can't be loaded or it is switching |
| `GET /collections/{name}/vocabulary` | The induced vocabulary in force and every saved version: `{version, terms, versions: [{version, created_at, count, active}]}`; `version` and `terms` are null until one is in force. This body is the export. 404 for an unknown collection |
| `PUT /collections/{name}/vocabulary` | `{terms, confirm}` (the GET body, edited, is the import). Without `confirm`, returns `{estimate}` for tagging every item and saves nothing; with `confirm: true`, saves the terms as a version and queues a `retag` job, which tags every item and then switches to the version; returns `{version, job_id}`. 503 when the collection tags with the AI and no model is set up for `keywords`; 409 when the collection uses keyword lists, or while a load, import, retag, `reembed` or eval is queued or running on it, and `has no keyword tags yet; use Add keyword tags on the store page` for a Documents store using the `documents` recipe; 422 for an empty list |
| `GET /chats`, `POST /chats` | Saved chats, newest first; create one with `{stores, title?}` or `{from, title?}`: `from` is the id of a chat to copy (its stores and saved settings, no messages; a store since deleted is left out). 404 for an unknown store, or an unknown `from` chat; 422 for no stores, `a chat needs stores, or from: a chat to copy` with neither, `send stores or from, not both` with both, and `a chat needs at least one store` when none of the copied chat's stores is left |
| `GET /chats/{id}`, `PATCH /chats/{id}`, `DELETE /chats/{id}` | One chat with its messages, attempts and decisions, its own `settings`, and `in_force`, what the next search uses with the level each value comes from; change `{title, stores, evidence_size, rewrite, rerank}`; delete. A null `evidence_size`, `rewrite` or `rerank` inherits from Settings. 404 for an unknown chat, or a store being added that does not exist; 422 for a blank title, no stores, an evidence size outside 1 to 20, a null title or stores, or an unknown field (such as `routing`) |
| `POST /chats/{id}/messages` | `{content, model?, answer?}`: search every store on the chat, pool and rerank their passages, and answer (or, with no answer model, return the passages alone). `answer` defaults to true; with `"answer": false` the attempt is searched, pooled and reranked as usual (a follow-up rewrite still runs when it is on) but no answer call is made and nothing is paid for one. With `?brief=true` it returns the short reply instead (`n`, `answer`, `citations`, `passages`, `passages_only`, `error`, `usd`; see [Chat from your own app](#chat-from-your-own-app)), whose `passages_only` is true when no answer model was available or `answer` was false. Without it, returns the message with its first attempt: `search_text`, `decisions` (per store: `method`, `candidates`, `search`, `tags`, `found`, `used`, `note`, `usd`, `error`), `evidence` (each passage with its `file`, `store`, `rank`, `score`, `relevance` and `channels`), `answer`, `citations`, `usd`, `error` and `pipeline` (where the search text came from, what it was reranked against and by which model, the pool, the notes, `passages_only`: no answer model was available when this attempt was made, `answer_cached`: the answer came from the AI cache, and `answer_skipped`: this attempt was made without an answer call. It is true for a message sent with `"answer": false` and for every re-run, and stays true after **Answer from this attempt** fills the answer). A failed answer is still 200 with `error` set, and the message is saved. 404 for an unknown chat; 422 for empty text, a field other than `content`, `model` and `answer` (so a misspelt `answer` is refused, not answered and paid for), or an answer model that is unknown or has no price, before anything is saved |
| `POST /chats/{id}/messages/{n}/attempts` | `{settings: {evidence_size?, rerank?, rewrite?}, stores: {store: {method?, candidates?}}}`: save those values on the chat (`null` inherits, a field left out is kept) and re-run message `n`, adding an attempt without answering. The attempt reuses the latest attempt's search text unless the rewrite or follow-up turns now in force differ from how it was made; a message from before this pipeline always keeps its text, with no paid rewrite. 404 for an unknown chat, message or store not on the chat; 409 for a method the store cannot run, and 503 for one that needs AI when none is connected; 422 for a bad value, and then none is saved |
| `POST /chats/{id}/messages/{n}/attempts/{attempt}/answer` | `{model?}`: answer from that attempt. 404 for an unknown attempt; 409 when it is already answered or another request is answering it (nothing is paid twice). An answer model that is unknown or has no price becomes the attempt's `error`, after any note the attempt already had (a failed rewrite) |
| `POST /chats/{id}/stores/{store}/default` | Set as default: save the method and candidates the chat holds for the store as the store's own, then clear them from the chat so it inherits them. Returns `{chat_method, chat_k, in_force}`. 404 for an unknown chat, or a store not on it; 409 when the store cannot run the method the chat holds (nothing is saved, since every chat that inherits it would fail) |
| `POST /chat` | The single-store chat, through the same pipeline as a chat of one store. `{collection, messages: [{role, content}], method?, k?, model?}`: an answer from the store's passages for the last message, which must be the user's (422 otherwise). Only the last 6 turns are used, and retrieval uses the latest question only. No `method` means the store's chat method as a chat inherits it (store, then Settings); `k` is the evidence size (1 to 20), Settings' when left out. Candidates come from the store (then Settings); rerank, RRF k, the reranker and Rerank up to are Settings'. An unknown method is 422, and so are `iterative`, `hybrid_rerank` and the other methods that are not chat methods; a method the store cannot run is 409 with the reason, and an explicit paid method (or `keyword_rag` with model tags) when no AI is connected is 503 `needs AI`. An unknown or unpriced `model` is the reply's `error`. The model gets the history and numbered passages (each cut to 1,500 characters) and cites `[n]`; `[n, m]` and `[n-m]` cite each number. Returns `{answer, citations: [{n, id, text, file, source, part, parts}], hit_ids, method, query_tags, usd, error, notes}`; citations are in the order cited. Spends like `/query`: no estimate, and over the budget cap `error` says so and `answer` is null |
| `POST /query` | One question, several methods side by side, optionally answered; each method's `usd` is every AI call it made (query summary, hops, answer). With no `methods` it runs Settings' default method, and with no `rrf_k` Settings' RRF k (the Lab sends both). `search` overrides the collection's search settings in part (`{weights: {meaning, words, keywords}, query_tags, example, program}`); 422 for an unknown key or value, and for `query_tags: "oracle"` with `keyword_rag`. `keyword_rag` also returns its `query_tags`, and each hit its `tags` and the `channels` that found it. A method that cannot run reports its own `error` with status 200: `keyword_rag` on a Documents store reads `needs keyword tags: Add keyword tags on the store page`, also while Add keyword tags is unfinished |
| `POST /eval` | `eval_set` (default: the store's default set) picks the questions; 409 for a QA type mismatch, 422 when there is no eval set, and a retrieval-only run that would measure nothing is refused. Without `confirm`: the estimate, which includes `questions`, `dev_questions`, `evidence_named` and `evidence_held`, and for a Documents store `references` (questions in the chosen split with a reference answer) and, when answering, `evidence_size`. A resume's estimate checks the AI cache for the answers and verdicts it can rebuild from the stored rows, and prices the rest in full. With it: queues a run and returns `{job_id, estimate}`; 503 when the run would spend and no AI model is set up; 409 for `keyword_rag` without a usable vocabulary (on a Documents store, while Add keyword tags is unfinished); `judge: true` (Documents only) has the run's answer model judge each answer against its reference answer, recording `setup.answer_judge_model`; it is refused with 422 `Judge answers is for Documents stores`, `Judge answers needs the AI's answers: tick Have the AI answer the questions`, or `none of the questions in this run has a reference answer to judge against; add reference answers on the eval set page, or choose All`. A Documents run that is not judged on a store holding none of the set's right passages is refused with `nothing to measure: ...` (see [Your own documents](#your-own-documents)). `rerank_pool` (1 to 50, default 50) and `rerank_model` are read by `hybrid_rerank` alone: how many of hybrid's top it reranks, and with which model (none takes the Settings reranker when the run is queued; the run records it in `setup.rerank_model`). `search` as for `/query` (here `query_tags: "oracle"` is allowed); the run's setup records the effective settings and the vocabulary version, and each `keyword_rag` row its `query_tags` |
| `GET /eval?collection=` | Every run, newest first, one summary row each |
| `GET /eval/{job_id}`, `GET /eval/{job_id}/items` | A run's answer, gaps and table, with its log and `since` (see [Reading a run](#reading-a-run)); one method's per-question rows. `GET /eval/{job_id}` of a Documents run also has `answers` (null unless it answered; see [Reading a run](#reading-a-run)) and `labels` (`tracked`, `unjudged` per method as `{shown, questions, flag}`, `unmeasured`, `no_right_passage`) |
| `GET /eval/{job_id}/questions?disagree=` | Each question with every method's row and its `outcome`; `disagree=true` keeps the questions where the methods' outcomes differ, leaving out the controls and `question_only` |
| `DELETE /eval/{job_id}` | Delete a finished run and its rows; 409 while it is queued or running |
| `GET /jobs?collection=&kind=&status=`, `GET /jobs/{job_id}` | Jobs, newest first; one job's status, progress and log |
| `POST /jobs/{job_id}/cancel` | Cancel a queued job, or ask a running one to stop; 409 for any other status |
| `POST /jobs/{job_id}/resume` | Resume a stopped run, copy, `reembed` or `tag_build`. Without `confirm`: the estimate for what is left, or 409 with why it cannot resume. With it: queued again. A `reembed` is free: it queues at once, with no estimate and no `confirm`. A copy runs again in place: adding skips the items it added and the build finishes what it left unbuilt. A `tag_build` runs again in place too and finishes only what is left (the vocabulary if none is in force, the passages without tags); without `confirm` it returns `{estimate: {est_usd, calls}}`, with it the job is queued again. A `tag_build` resume that nothing can price is 503 `connect a model for keywords first`. Only a store's newest `tag_build` resumes: an older one is 409 `a newer tag build of '<name>' exists; resume that one` |
| `GET /status` | The worker, the queue, pending AI batches and the problems that need a person |
| `GET /catalog` | Plain names and descriptions of methods, metrics and QA types, the built-in `datasets`, `setup_labels` (each run setup field's label) and `roles` (each AI job's label) |
| `GET /budget` | This service's spend this month against its cap |
| `GET /ai` | Connections (keys masked), models with prices, jobs (`default_from_env` when `RAGQA_MODEL` sets the default), `resolved` (per job, whether a usable model is set up for it), budget and presets |
| `POST /ai/connections`, `PATCH`/`DELETE /ai/connections/{id}` | Add, edit, remove a connection; 409 when a job uses it, or the field or connection is set by the environment. A new `base_url` without a new `api_key` clears the stored key |
| `POST /ai/connections/{id}/test`, `…/models/fetch`, `…/models` | Test a connection (an unreachable server is reported, not raised), fetch its model list, add a model by hand |
| `PUT /ai/models/{connection}/{model}` | Price, label or enable a model |
| `PUT /ai/roles/{job}`, `PUT /ai/budget` | Choose the model a job uses; set the monthly cap |
| `GET /search-defaults`, `PUT /search-defaults` | The app-wide [Search defaults](#search-defaults): `{saved, recommended, in_force, limits, methods, rerankers, embedders, embed_machine, notes, rerank_status, rerank_error}`, where a null `saved` value follows the recommendation and `rerank_model` is shown as a name; `rerank_status` is the in-force reranker's load (`idle`, `loading`, `ready` or `failed`) and `rerank_error` why it failed. `PUT` takes any of `method`, `candidates`, `evidence_size`, `rerank`, `rewrite`, `follow_up_turns`, `rrf_k`, `rerank_model`, `rerank_pool`, `dense_model`, in part; `null` puts one back to recommended. A `PUT` with `rerank_model` starts loading it in the background, at once even if it failed minutes ago. 422 for an unknown field or a value out of range, a method that is not a chat method, a reranker or an embedding model fastembed does not list. `embedders` is `{name, dim, size_gb, note, seconds_per_1000}` for each offered model |

Errors: a bad value is 422; a state conflict is 409; a provider's error is
502, carrying its message with any key masked; no model for a job is 503.

Loads, re-keys and evaluations run one at a time, oldest first, on a worker
thread inside the service. A resumed run keeps its original place in that
order. A job that was running when the service stopped comes back
`interrupted`; queued jobs stay queued. A method that cannot run inside
`/query`, for example a keyword method before lists exist or one whose provider
fails, reports its own `error` (a provider's with any key masked) and what it
spent, and the other methods still return.

## Reading a run

`GET /eval/{job_id}` returns everything the Run page shows, computed in the
service:

- **`answer`** compares one pair declared in advance: `kw_individual`
  (Keywords per list) against `hybrid` (Hybrid), on the same questions. Its
  `kind` is `pair` and its
  `verdict` is `yes`, `no` or `cant_tell`, read from the 95% paired bootstrap
  interval of the gap. A run without both methods has no answer.
- **A Documents run's `answer`** is a score, not a pair:
  `{kind: "score", method, metric: "ndcg@10", mean, lo, hi, n, before_rerank, switch}`.
  It scores `setup.chat_search.run_method`, the search the store's chats used when the
  run was queued, and `switch` names a chat method clearly ahead of it, or is null.
  `reference` and `gaps` are against that method. A run without it has no answer; an
  older run with no `chat_search` is scored on the store's current one.
- **`headline`** is the metric the answer uses: accuracy (or F1 for multi-hop
  and tables) when the run answered, else a free retrieval metric.
- **`gaps`** puts every method against the reference (Hybrid), as a paired
  difference with its interval: `ahead`, `behind` or `cant_tell`. Two
  intervals that overlap say nothing about a paired gap. At 200 questions, a
  gap of a few points is usually noise: the paper's 78.9% vs 76.3% was two
  questions out of 76.
- **`best_of_each`** (best keyword method against best other method) is
  picked after the fact, so its gap is likely overstated. It is never the
  answer.
- **`eval_set`** and **`eval_set_version`**: which questions the run asked. The Run
  page shows "eval set <name> · version <v>". Edits to the set after the run never
  change it; `since` says so.
- **`since`** says what changed after the run: `{untracked, changes}`, where
  `changes` lists `keyword lists`, `vocabulary`, `dense model`, `sources` and `eval set`
  (shown as "eval set changed after this run"). A migrated run
  that recorded no versions is `untracked: true`.
- **`methods`** holds each method's own score as `mean [low, high]`, and
  **`pairs`** every pair compared.
- **`proxy`** (exam QA type, with answers): whether solution similarity, the
  free metric, ranks retrieval methods the way accuracy does. It needs 4 or
  more methods, leaves out No retrieval and the controls, and passes only if
  the rank correlation is at least 0.5 and the declared pair points the same
  way on both.
- Evidence: when the store holds none of the eval set's evidence, recall is unmeasured
  (never 0) and the Run page explains why; when it holds some, a metric reads "on n of N".
  Leaked questions are worked out for the run's store and eval set, and the page says how
  many were left out.
- **`answers`** (a Documents run that answered; else null): `{judged, no_reference,
  methods: {method: {answered, judged, correct, unreadable}}}`. The Run page leads
  with `How often are its answers right?` when the run was judged, then one line per
  method, `<method>: <c> of <j> answers judged right` (`1 answer` when one), and
  `<n> questions not judged: they have no reference answer` and `<n> answers not
  judged: the judge's reply couldn't be read` when there are any. A run that
  answered without Judge answers says `The AI's answers are not scored.` and how to
  score them: a new run from the store page with Judge answers ticked. A judged run
  that stopped or was cancelled adds the `Partial: …` line. Each question shows `The
  AI's answer: …` and, once judged, `(judged right)`, `(judged wrong)`, `(not judged:
  no reference answer)` or `(not judged: the judge's reply couldn't be read)`, the
  last followed by `Judge's reply: …`. **Re-run** keeps Judge answers.
- Every run records its full setup: models, keyword-list version, `k`,
  `rrf_k`, seed, dataset filter, sources and question ids. `log` holds the
  run's last 50 events.

## Experiments

`scripts/experiments.py` queues the method-v2 experiments (X1 retrieval, X2 query
tags, X3 fusion, X4 answer accuracy, and X5, rerank against hybrid at chat's pool
size) through the API: `estimate` prices every arm,
`run --confirm` queues them (`--paid` for the arms that cost money: X2's `model`
arm, one call per question, and every X4 arm, one answer call per question), and
`report` prints one table with confidence intervals. Setup,
commands, the X1 to X4 results and what each result proves or kills are in
[scripts/experiments.md](scripts/experiments.md).

## Using the AI layer in another project

The AI layer is its own workspace package, `aiconnect`
([packages/aiconnect](packages/aiconnect)), with no dependency on the rest of
rag-qa. It brings connections, prices, jobs, the monthly cap, the response
cache, the spend ledger and batches, all stored in one SQLite file you choose.

```python
from pathlib import Path
from aiconnect.ai import AI

ai = AI(Path("my-app.db"))
ai.add_connection("anthropic", "anthropic", "Anthropic", api_key="sk-ant-...")
ai.add_model("anthropic", "claude-haiku-5-5", price_in=0.10, price_out=0.50)
ai.set_role("default", "anthropic/claude-haiku-5-5")

reply = ai.complete("Name three noble gases.", role="default", purpose="demo")
print(reply.text, reply.usd, reply.cached)
```

`complete_many` batches where the provider supports it, `estimate_usd` prices
a call before it is made, `estimate_many` prices a set of prompts as
`complete_many` would send them (cached ones free, the rest batched only when
the misses are), and `month_spend` reports the ledger. Errors are the
package's own: `BudgetExceeded`, `NoConnection`, `UnknownModel`,
`UnknownPrice`, `ProviderError`, `Conflict`, `Cancelled`.

## Roadmap

The bench is a chat across your stores that pools every store's best passages,
reranks them against the question and shows its evidence, with the keyword
method brought up to date. The learned router is dropped: chat searches every
store. Where each part stands:

- **Keyword-RAG v2** (built, and its [experiments](#experiments) X1 to X4 have
  been run): build recipes per collection, a keyword vocabulary induced from the
  corpus and reviewed through the API instead of curated lists, keywords that
  describe the skill a solution uses rather than the question's topic, a
  classifier that predicts a question's keywords, and abstracted worked
  examples. New benchmarks: BRIGHT AoPS and TheoremQA-Q, scored by nDCG@10.
  The results, in [scripts/experiments.md](scripts/experiments.md):
  - hybrid beat the label channel with classifier tags (X1), so the channel adds
    nothing on that path;
  - query tags from a model separated from hybrid with Haiku (+.054, CI above
    zero) but not with Opus (+.030), and the oracle ceiling is far higher (.545
    against .430), so tag prediction is the bottleneck (X2);
  - nothing in the fusion experiment beat the defaults (X3);
  - abstracted or full retrieved examples did not beat zero-shot on SciBench (X4).
- **Chat** (built): saved chats that search several stores, pool their
  candidates and rerank them with a reranker chosen in Settings, work without AI
  as passages, and keep their own settings. Rerank's default is the open
  question of X5.
- **Library**: collections renamed to corpora, with PDF, Word and HTML
  imports.

## References and resources

### The paper this is built on

- Chow, J. (2024). [*Keyword Vectorization for Retrieval Augmented Generation
  on Science Question Answering*](paper/keyword-vectorization-rag-science-qa.pdf).
  Singapore Management University.

### What the paper builds on

- Lewis et al. (2020). [Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).
- Brown et al. (2020). [Language Models are Few-Shot
  Learners](https://arxiv.org/abs/2005.14165).
- Wei et al. (2022). [Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models](https://arxiv.org/abs/2201.11903).
- Li et al. (2023). [Guiding Large Language Models via Directional Stimulus
  Prompting](https://arxiv.org/abs/2302.11520).
- Cormack, Clarke & Büttcher (2009). [Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning
  Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).
- Robertson, Zaragoza & Taylor (2004). [Simple BM25 Extension to Multiple
  Weighted
  Fields](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=10b2752c9940e9c761a161a0fef7ae08cae66301).
- Çano & Bojar (2019). [Keyphrase Generation: A Multi-Aspect
  Survey](https://arxiv.org/abs/1910.05059).
- Gao et al. (2022). [Retrieval-Augmented Multilingual Keyphrase Generation
  with Retriever-Generator Iterative
  Training](https://arxiv.org/abs/2205.10471).
- OpenAI (2023). [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774).
- Tools: [Faiss](https://faiss.ai), LangChain's parent-document and ensemble
  (hybrid) retrievers,
  [reichenbch/RAG-examples](https://github.com/reichenbch/RAG-examples),
  [brexhq/prompt-engineering](https://github.com/brexhq/prompt-engineering).
- Data: ACT Science past papers ([ACT Test Prep](https://www.actexam.net/))
  with worked explanations from
  [PrepSharp](https://www.prepsharp.com/act-answer-keys/).

### Later work this project draws on

- Didolkar et al. (2024). [Metacognitive Capabilities of LLMs: An Exploration
  in Mathematical Problem Solving](https://arxiv.org/abs/2405.12205): pick
  examples by the skill the solution uses, not the question's topic.
- [Doc2Query++](https://arxiv.org/html/2510.09557v2) (2025): a separate
  keyword index fused with text search beats either alone; the idea behind
  `kw_hybrid`.
- Li et al. (2024). [mFAR: multi-field adaptive
  retrieval](https://arxiv.org/html/2410.20056v2): index each field and the
  whole document.
- Kang et al. (2024). [TaxoIndex](https://arxiv.org/abs/2410.19218): a
  classifier maps a question onto the corpus's concepts.
- Wan et al. (2024). [TnT-LLM](https://arxiv.org/abs/2403.12173): an AI
  proposes a taxonomy once, a light classifier labels the rest.
- Weller et al. (2024). [When do Generative Query and Document Expansions
  Fail?](https://aclanthology.org/2024.findings-eacl.134): expansion helps weak
  retrievers and hurts strong ones.
- Liu et al. (2023). [Lost in the Middle](https://arxiv.org/abs/2307.03172):
  where evidence sits in the prompt matters.
- Chen et al. (2025). [SA-ICL](https://arxiv.org/abs/2510.13905) and Wang et
  al. (2025). [I2S](https://arxiv.org/abs/2509.23196): show an example's
  method, not the full worked example.
- [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): the
  default embedding model.
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): an
  option, and the model of stores built before this change.

### Benchmarks

[ScienceQA](https://arxiv.org/abs/2209.09513) ·
[SciBench](https://arxiv.org/abs/2307.10635) ·
[GPQA](https://arxiv.org/abs/2311.12022) ·
[MuSiQue](https://arxiv.org/abs/2108.00573) ·
[TAT-QA](https://arxiv.org/abs/2105.07624) ·
[HippoRAG 2](https://arxiv.org/abs/2502.14802) (the MuSiQue protocol) ·
[BRIGHT](https://arxiv.org/abs/2407.12883) (next version)

## License

Apache-2.0. See [LICENSE](LICENSE). The paper is © its author and is included
for reference.
