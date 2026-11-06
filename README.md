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
- [Your own corpus](#your-own-corpus)
- [Recipes](#recipes)
- [Retrieval methods](#retrieval-methods)
- [Answering](#answering)
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

## Your own corpus

Any store takes your own items (what is retrieved). Your own questions (what is
asked) go into an eval set. On the store page (items) or the eval set page
(`/e/{name}`, questions) choose a file, check the preview, fix the field mapping if
the guess is wrong, and import. Item imports queue a job and spend nothing until you
confirm the estimate. A question import on the eval set page
(`/eval-sets/{name}/imports`) runs in the request; one through the store
(`/collections/{name}/imports` with `kind=questions`) still queues a job.

- **Formats:** JSONL, a JSON array, CSV, TSV, plain text (`.txt`), Markdown
  (`.md`), or a `.zip` of `.txt` and `.md` files. Text and Markdown files become
  one item each.
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
- **Eval sets:** `/eval-sets` lists and creates them (a name and a QA type). On
  `/e/{name}` edit questions in place, add or delete them, and Save; import a file;
  switch "include leaked"; read the versions; or delete the set.
- **Versions:** every write that changes the questions saves a snapshot. Its id is
  the first 12 hex digits of the SHA-256 of the questions' canonical JSON, sorted by
  id. Save sends the `base_version` it started from and is refused with 409 when the
  set has moved on. A run records its eval set when queued and the version at its
  first run, and always reads that version, so later edits never change a run and a
  resume asks the same questions. The migration made a set for every existing
  collection that had questions, and pinned every existing run to that snapshot.
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

## Recipes

A recipe is a set of build choices, picked when a collection is created
(`recipe` on `POST /collections`) or applied to a copy of another collection's
items (`POST /collections/{name}/copy`). Each stores its choices with the
collection, and `PUT /collections/{name}/search` sets its default search settings.

| Recipe | For | Build |
|---|---|---|
| `worked_examples` | Solved problems (science, multiple choice) | One unit per item; an induced vocabulary; items tagged by the AI from their solution; abstracted examples; the classifier |
| `reference` | Passages (multi-hop, tables) | 100-word chunks with 10 words of overlap; the title put before the text; an induced vocabulary; tags by similarity; the classifier |
| `paper_2024` | The paper's method (the default) | 100-word chunks with 25 words of overlap; the paper's three keyword lists (yours, or AI-proposed) and optional summaries |

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
  authored or edited here. This is checked before anything is added. Otherwise, and whenever the Hub cannot be reached, the
  load builds locally as usual, and its estimate and result say why
  (`hub: {skipped: ...}`). A load that applies one reports `hub: {entry, items}`.
  There is one exception. A load the entry made free is queued without `confirm`, so
  if the entry is no longer usable when that load runs, it fails and adds nothing,
  rather than spend what nobody agreed to.
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
| `hybrid_rerank` | modern | `hybrid` top 50, reranked by a local cross-encoder | free, slower |
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
trained with one (Qwen3-Embedding, the default): `Instruct: <task>\nQuery:`
before the query, nothing before documents. Each eval run records the exact
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
| `RAGQA_DENSE_MODEL` | Retrieval embedding model (fastembed). Default `Qwen/Qwen3-Embedding-0.6B-Q` |
| `RAGQA_JUDGE_MODEL` | A different model, used only to score solution similarity. Default `BAAI/bge-small-en-v1.5` |
| `RAGQA_RERANK_MODEL` | Cross-encoder for `hybrid_rerank`. Default `BAAI/bge-reranker-v2-m3-int8` |
| `RAGQA_QDRANT_URL` | Set by compose. Unset means in-process Qdrant (tests only) |
| `HF_TOKEN` | Hugging Face token: read, to load GPQA (gated); write, only for the owner's `ragqa_core.publish --publish` |
| `RAGQA_BUILD_HUB` | The Hub repo of [published builds](#published-builds) a load applies. Default `jonathanmiro97/keyword-rag-builds`; empty turns it off |
| `RAGQA_HOME` | State directory. `/home/app/.ragqa` in the image |
| `PORT` | Listen port. `8080` |

A collection remembers the dense model it was built with. Changing
`RAGQA_DENSE_MODEL` afterwards makes that collection refuse new items with a
409 that names both models.

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
  imports and keyword-list proposals return the estimate unless
  `confirm: true` is set. The Lab (`/query`) and the chat (`/chat`) are the exception: they
  spend at once and show each reply's cost after, and the monthly cap still
  holds.
- **Every response is cached.** A re-run with the same prompts is free, and its
  answers are identical.
- **Batches.** With the Anthropic connection, runs of more than 50 calls use
  the Batch API at half price; other connections send one call at a time. A
  batch usually finishes in minutes but can take up to 24 hours.
- **Localhost only.** The port is published on `127.0.0.1`. Opening it wider
  without basic auth would let anyone on the network spend up to the cap.

Rough prices on Haiku 5.5: a one-shot multiple-choice answer (400 answer tokens
at most) is about $0.0005, and a 200-question × 10-method evaluation with
answers about $1. A science answer works the problem out and may use up to
1,500 answer tokens, so science runs cost more. Sonnet costs about 20 times as
much.

## Pages

The side nav reads Stores, Chat, Eval sets, Lab, Settings. The store pages live under `/c/` and the API under `/collections`; both mean a store.

| Route | What it shows |
|---|---|
| `/` | Every store with its answer (does keyword retrieval help?), the run it came from and its state; add a default store (one that exists is opened instead), create a store or set up the benchmarks |
| `/c/{name}` | One store: the answer with its gap bar, its runs and a form to start one (pick the eval set), its items and sources, imports, its default eval set (pick another of its type, or none) with how many of its questions leaked, the three keyword lists (edit, propose, save), delete |
| `/c/{name}/runs/{id}` | One run, with the eval set and version it asked |
| `/chat` | Chat over a store: pick the store, method and model; each reply has `[n]` links to its sources (id, source, the first 300 characters, expandable), the method, the query tags and the cost. Enter sends, Shift+Enter adds a newline. Add document adds `.txt` and `.md` files to the store for good (an add that would spend shows its estimate with Add and Cancel). The conversation is kept per store in the browser tab, not on the server; New chat clears it. With no answer model Send is disabled and a note links to Settings |
| `/eval-sets` | Every eval set; create one |
| `/e/{name}` | One eval set: edit questions in place, add, delete, Save, import, include leaked, versions, delete the set |
| `/lab` | One question, methods side by side |
| `/worker` | The worker, its queue and what needs a person |
| `/settings` | AI connections, models and prices, the model each job uses, the monthly cap |
| `/docs` | Every endpoint |

## Endpoints

Every endpoint is listed with its schema at http://localhost:8080/docs.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, and whether the `answer` job has an AI model (`{ok, ai}`) |
| `POST /collections` | `{name, qa_type: science\|mcq\|multihop\|table, summarize, recipe?: worked_examples\|reference\|paper_2024}`; no `recipe` means paper 2024, as before. 409 when the name is taken, or an eval set of that name has another type |
| `GET /eval-sets` | Every eval set (the page, for a browser that asks for `text/html`; JSON otherwise) |
| `POST /eval-sets` | `{name, qa_type}`; 409 when the name is taken, or a store of that name has another type |
| `GET /eval-sets/{name}` | The set, its questions and its versions |
| `PUT /eval-sets/{name}/questions` | `{questions, base_version}`: replace the set's questions, all or nothing; returns `{version}`. 409 when `base_version` is stale, or while a load or question import writes to the set; 422 for a bad or repeated id |
| `PATCH /eval-sets/{name}` | `{include_leaked}`: whether leaked questions are used by runs |
| `DELETE /eval-sets/{name}` | Without `?confirm=true`, `{runs}`: how many runs would go. With it, deletes the set and those runs. 409 while an eval or load uses the set |
| `POST /eval-sets/{name}/uploads?filename=` | The raw file as the body; returns `{upload_id, preview}` |
| `POST /eval-sets/{name}/imports` | `{upload_id, mapping, source_name}`: imports in the request, upserting by id; returns the counts (new, updated, unchanged), `skipped` and the new `version` |
| `GET /collections`, `GET /collections/{name}` | Specs; one store's count, keyword lists, available methods, `state` (a running job's `loading`, `rekeying`, `retagging`, `copying`, `evaluating`, `importing` or `labelling`; else `queued`, `empty`, `no_lists` or `ready`) with its job and progress, question counts, its answer and number of runs |
| `PATCH /collections/{name}` | `{eval_set}`: make that eval set the store's default; `null` clears it, so the set named after the store (if any) is the default again. Returns the store as `GET /collections/{name}` does. 404 for an unknown store or set; 409 when the set's type differs from the store's, or while a load or question import is queued or running on the store |
| `DELETE /collections/{name}` | Delete a store, its items, lists and runs. Its eval sets stay. Its queued jobs are cancelled; 409 while one of its jobs runs |
| `POST /collections/{name}/documents` | Add your own items `{items: [{id, text, answer?, explanation?, meta?}], confirm}`; 409 while a load, re-key or eval on the collection is queued or running. Over 20 items it queues an import job instead and returns `{job_id}`. Recipes other than paper 2024 run their build steps on the new items in the request; their cost is in the estimate, and a confirmed request with no model for `keywords` is 503. While such a request runs, any job queued on the collection is refused with 409 |
| `POST /collections/{name}/uploads?filename=&kind=items\|questions` | The raw file as the body (up to 200 MB). Returns `{upload_id, preview, estimate}`: the first rows under a guessed field mapping, and what importing would cost. `accept=text` takes only `.txt` and `.md` files (422 otherwise); the chat uses it. A source name is cleaned to letters, digits, space, `_`, `.` and `-`, at most 64 characters |
| `POST /collections/{name}/imports` | `{upload_id, kind, mapping, source_name, confirm}`; `kind=questions` goes to the store's default eval set. Without `confirm`: the preview and estimate. With it: queues an import job and returns `{job_id}`; 422 when a required field is unmapped or no row is usable; 503 when it would spend and no AI model is set up. With `kind=questions`, 409 before anything else when the store's default eval set has another type |
| `GET /collections/{name}/sources` | Each source with its item count and whether retrieval uses it |
| `PATCH /collections/{name}/sources/{source}` | `{enabled}`: switch a source on or off for retrieval. 409 while an eval runs, or to switch off the last item source |
| `DELETE /collections/{name}/sources/{source}` | Remove a source's items and chunks; questions stay; 409 while a job is queued or running on the collection |
| `GET /collections/{name}/questions?leaked=` | The questions of the store's default eval set, with how many are leaked against this store; `leaked=true` or `false` filters |
| `PATCH /collections/{name}/questions/leaked` | `{include}`: the same setting as `PATCH /eval-sets/{name}`, on the store's default set |
| `PUT /collections/{name}/search` | Set the collection's default search settings, in part (`{weights: {meaning, words, keywords}, query_tags, example, program}`), and return all of them; `/query` and `/eval` override them per request. 422 for an unknown key or value |
| `POST /collections/{name}/load` | Load `scienceqa`, `gpqa`, `musique`, `tatqa` or `scibench` (`GET /catalog` lists them as `datasets`); its questions go to the store's default eval set, upserted by id. When it would spend (summaries; an AI keyword proposal because an AI model is set up and a recipe that uses keyword lists has none; or the build steps of the other recipes: inducing a vocabulary while there is none, AI tagging and abstracted examples), without `confirm` it returns the estimate `{est_usd, items, calls}`; with `{confirm: true}`, or when it is free, it queues a job and returns `{job_id}`. 503 when the build would spend and no model is set up for `keywords`. 409 before anything is estimated or queued when the store's default eval set has another type |
| `POST /collections/{name}/copy` | `{name, recipe, build, confirm}`: a new store of this one's items under another recipe, with its switched-off sources. It points at the source's default eval set and copies no questions. `build` overrides the recipe's build settings in part; the recipe keeps its vocabulary, `chunk_overlap` stays below `chunk_words`, and `added_text: "summary"` needs keyword lists. Without `confirm`, returns `{estimate}` (embedding is free; summaries, and the build steps of the recipes other than paper 2024); with `confirm: true`, makes the collection and queues a `copy` job, which adds every item under its source, then builds (a paper 2024 copy takes the source's keyword lists in force instead of proposing any); returns `{name, job_id}`. 409 when the name is taken or a job is queued or running on the source; 404 for an unknown source; 422 for a bad name, recipe or build setting; 503 when it would spend and no model is set up for the job |
| `POST /collections/{name}/keyword-lists/propose` | The AI proposes the topic, method and common lists; nothing is saved. Without `confirm`: the estimate `{est_usd, calls}`; with `{confirm: true}`: the lists |
| `GET /collections/{name}/keyword-lists`, `GET …/keyword-lists/{version}` | Every saved version of the lists, newest first; one version with its lists |
| `PUT /collections/{name}/keyword-lists` | Save `{topic, method, common}` as a version and queue a `rekey` job; returns `{version, job_id}`. The collection switches to the new lists once every item is re-keyed. 409 while an eval runs on it, or when the service's dense model differs from the collection's |
| `GET /collections/{name}/vocabulary` | The induced vocabulary in force and every saved version: `{version, terms, versions: [{version, created_at, count, active}]}`; `version` and `terms` are null until one is in force. This body is the export. 404 for an unknown collection |
| `PUT /collections/{name}/vocabulary` | `{terms, confirm}` (the GET body, edited, is the import). Without `confirm`, returns `{estimate}` for tagging every item and saves nothing; with `confirm: true`, saves the terms as a version and queues a `retag` job, which tags every item and then switches to the version; returns `{version, job_id}`. 503 when the collection tags with the AI and no model is set up for `keywords`; 409 when the collection uses keyword lists, or while a load, import, retag or eval is queued or running on it; 422 for an empty list |
| `POST /chat` | `{collection, messages: [{role, content}], method?, k = 5 (1 to 20), model?}`: an answer from the store's passages for the last message, which must be the user's (422 otherwise). Only the last 6 turns are used, and retrieval uses the latest question only. No `method` means `keyword_rag` when the store has a vocabulary (it needs no keyword lists), else `hybrid`; an unknown method name is 422, and a method the store cannot run is 409 with the reason. A store whose saved search uses oracle tags is chatted with classifier tags. The model gets the history and numbered passages (each cut to 1,500 characters) and cites `[n]`; `[n, m]` and `[n-m]` cite each number. Returns `{answer, citations: [{n, id, text, source}], hit_ids, method, query_tags, usd, error}`; citations are in the order cited. Spends like `/query`: no estimate, and over the budget cap `error` says so and `answer` is null |
| `POST /query` | One question, several methods side by side, optionally answered; each method's `usd` is every AI call it made (query summary, hops, answer). `search` overrides the collection's search settings in part (`{weights: {meaning, words, keywords}, query_tags, example, program}`); 422 for an unknown key or value, and for `query_tags: "oracle"` with `keyword_rag`. `keyword_rag` also returns its `query_tags`, and each hit its `tags` and the `channels` that found it |
| `POST /eval` | `eval_set` (default: the store's default set) picks the questions; 409 for a QA type mismatch, 422 when there is no eval set, and a retrieval-only run that would measure nothing is refused. Without `confirm`: the estimate, which includes `questions`, `dev_questions`, `evidence_named` and `evidence_held`. With it: queues a run and returns `{job_id, estimate}`; 503 when the run would spend and no AI model is set up; 409 for `keyword_rag` without a usable vocabulary. `search` as for `/query` (here `query_tags: "oracle"` is allowed); the run's setup records the effective settings and the vocabulary version, and each `keyword_rag` row its `query_tags` |
| `GET /eval?collection=` | Every run, newest first, one summary row each |
| `GET /eval/{job_id}`, `GET /eval/{job_id}/items` | A run's answer, gaps and table, with its log and `since` (see [Reading a run](#reading-a-run)); one method's per-question rows |
| `GET /eval/{job_id}/questions?disagree=` | Each question with every method's row and its `outcome`; `disagree=true` keeps the questions where the methods' outcomes differ, leaving out the controls and `question_only` |
| `DELETE /eval/{job_id}` | Delete a finished run and its rows; 409 while it is queued or running |
| `GET /jobs?collection=&kind=&status=`, `GET /jobs/{job_id}` | Jobs, newest first; one job's status, progress and log |
| `POST /jobs/{job_id}/cancel` | Cancel a queued job, or ask a running one to stop; 409 for any other status |
| `POST /jobs/{job_id}/resume` | Resume a stopped run or copy. Without `confirm`: the estimate for what is left, or 409 with why it cannot resume. With it: queued again. A copy runs again in place: adding skips the items it added and the build finishes what it left unbuilt |
| `GET /status` | The worker, the queue, pending AI batches and the problems that need a person |
| `GET /catalog` | Plain names and descriptions of methods, metrics and QA types, and the built-in `datasets` |
| `GET /budget` | This service's spend this month against its cap |
| `GET /ai` | Connections (keys masked), models with prices, jobs (`default_from_env` when `RAGQA_MODEL` sets the default), budget and presets |
| `POST /ai/connections`, `PATCH`/`DELETE /ai/connections/{id}` | Add, edit, remove a connection; 409 when a job uses it, or the field or connection is set by the environment. A new `base_url` without a new `api_key` clears the stored key |
| `POST /ai/connections/{id}/test`, `…/models/fetch`, `…/models` | Test a connection (an unreachable server is reported, not raised), fetch its model list, add a model by hand |
| `PUT /ai/models/{connection}/{model}` | Price, label or enable a model |
| `PUT /ai/roles/{job}`, `PUT /ai/budget` | Choose the model a job uses; set the monthly cap |

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
  `verdict` is `yes`, `no` or `cant_tell`, read from the 95% paired bootstrap
  interval of the gap. A run without both methods has no answer.
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
- Every run records its full setup: models, keyword-list version, `k`,
  `rrf_k`, seed, dataset filter, sources and question ids. `log` holds the
  run's last 50 events.

## Experiments

`scripts/experiments.py` queues the method-v2 experiments (X1 retrieval, X2 query
tags, X3 fusion, X4 answer accuracy) through the API: `estimate` prices every arm,
`run --confirm` queues them (`--paid` for the arms that cost money: X2's `model`
arm, one call per question, and every X4 arm, one answer call per question), and
`report` prints one table with confidence intervals. Setup,
commands and what each result proves or kills are in
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

The bench is now a chat over one store of your own documents that shows its
evidence, with the keyword method brought up to date; chat across stores is
next. Where each part stands:

- **Keyword-RAG v2** (built; its [experiments](#experiments) have not been run,
  so it is not yet known to beat the paper's method): build recipes per
  collection, a keyword vocabulary induced from the corpus and reviewed through
  the API instead of curated lists, keywords that describe the skill a solution
  uses rather than the question's topic, a classifier that predicts a question's
  keywords, and
  abstracted worked examples. New benchmarks: BRIGHT AoPS and TheoremQA-Q,
  scored by nDCG@10.
- **Chat across stores**: the chat (built) answers from one store with
  citations; asking across several stores, with routing, and an editable
  retrieval step under every answer are next.
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
- [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B): the
  default embedding model.

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
