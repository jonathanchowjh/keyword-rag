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

Any AI provider can answer: Anthropic, any OpenAI-compatible service, or a
local model through Ollama at $0. Retrieval itself is always free and local.

- **Status:** v0.1.0, in active development (see [Roadmap](#roadmap)). Runs
  locally in Docker; no published image.
- **License:** Apache-2.0.

## Contents

- [The paper](#the-paper)
- [Quick start](#quick-start)
- [First run](#first-run)
- [Your own corpus](#your-own-corpus)
- [Retrieval methods](#retrieval-methods)
- [Datasets](#datasets)
- [AI connections](#ai-connections)
- [Configuration](#configuration)
- [Cost safety](#cost-safety)
- [Pages](#pages)
- [Endpoints](#endpoints)
- [Reading a run](#reading-a-run)
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

Open http://localhost:8080/ for the collections and their evaluations,
http://localhost:8080/lab to compare methods on one question, and
http://localhost:8080/docs for every endpoint (see [Pages](#pages)).

Then connect an AI on **Settings** (http://localhost:8080/settings). Pick one:

**A local model, free.** Install [Ollama](https://ollama.com), pull a model
(`ollama pull llama3.1:8b`), and add a connection from the **Ollama (local)**
preset. It needs no key, its base URL is
`http://host.docker.internal:11434/v1`, and **Fetch models** lists what you
have pulled, priced at $0. Choose it for the jobs you want it to do.

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

On the home page, **Set up the four benchmark collections** creates and loads
ScienceQA, MuSiQue, TAT-QA and SciBench, asking before anything that would
spend. Or do it over the API, one collection per QA type. Loading
downloads the data, embeds it locally and saves keyword lists. It runs as a
background job; poll `GET /jobs/<job_id>`.

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

Then start a run from a collection's page, or with `POST /eval`.

Docker on macOS cannot use the Apple GPU, so embedding runs on the CPU. The
first load of MuSiQue's 11,656 passages takes roughly 10–20 minutes. After
that, everything is kept in the `ragqa` and `qdrant` volumes.

## Your own corpus

Any collection takes your own items (what is retrieved) and your own questions
(what is asked). On the collection page choose a file, check the preview, fix
the field mapping if the guess is wrong, and import. Nothing is queued, and
nothing is spent, until you confirm the estimate.

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
  Deleting a source removes its items, chunks and imported questions.
- **Re-import updates:** items are matched by id (the hash of their content
  only decides whether one changed), so importing the same file again reports
  "N new, N updated, N unchanged" and adds no duplicates. A load that died is
  finished with "Finish loading", which embeds only what is missing.
- **Leaked questions:** an imported question whose text also appears in the
  items is flagged as leaked, and is left out of runs unless you include it.
  The Run page says how many were left out.
- **Partial gold:** recall is measured only on questions that name their
  evidence (`gold_ids`). A recall cell reads "on n of N questions" when some
  questions have none. A question with no evidence is still asked, but it
  does not count toward recall.
- **Existing collections** are labelled with their sources in place on first
  start (the `backfill` job, shown as "labelling"); nothing is re-embedded.

```sh
curl -X POST "http://localhost:8080/collections/notes/uploads?filename=notes.jsonl&kind=items" \
  --data-binary @notes.jsonl
# {"upload_id": "...", "preview": {...}, "estimate": {...}}
curl -X POST http://localhost:8080/collections/notes/imports \
  -H 'content-type: application/json' \
  -d '{"upload_id": "...", "kind": "items", "mapping": {"text": "body"}, "source_name": "notes", "confirm": true}'
```

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
| `iterative` | modern | Up to 3 retrieval hops; the AI decides the next query | paid per hop |
| `random_example` | control | Random items, seeded by the question: the same draw on every run | free |
| `fixed_example` | control | The same random items for every question | free |

\*Searching is free. The summaries and an AI keyword-list proposal cost once,
at load time, and only when asked for.

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

## Datasets

All are downloaded at runtime into the `ragqa` volume and never committed.

| QA type | Dataset | Licence | Used as | Free retrieval metric | Paid answer metric |
|---|---|---|---|---|---|
| Science problems | SciBench (physics and chemistry textbook problems) | MIT | 273 examples (76 worked solutions) and 219 questions: solved problems are all examples, the 416 unsolved split by a hash of their id. Small, a stand-in until your own corpus is imported | same-skill@1 | accuracy |
| Exam multiple choice | ScienceQA | CC BY-SA 4.0 | 693 solved train examples; 291 test questions | solution similarity@1, same-skill@1 | accuracy |
| Exam multiple choice, hard | GPQA (gated) | CC BY 4.0 | GPQA Extended minus Diamond as examples; Diamond's 198 as questions | solution similarity@1, same-subdomain@1 | accuracy |
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

## AI connections

The **Settings** page (`/settings`) sets up which AI the service calls. Nothing
is needed in `.env`.

- **Connections.** A connection is a provider with an address and a key.
  Presets cover Anthropic, OpenAI, Gemini, xAI, OpenRouter and Ollama (local);
  **Custom** takes any OpenAI-compatible server. Test it, then fetch its model
  list. Keys are stored in the database file (mode `0600`) and are never shown
  again or put in a log or response; the page shows a masked key.
- **Two kinds cover every provider:** Anthropic's Messages API, and the
  OpenAI-compatible Chat Completions API.
- **Prices.** Every model needs a price per million input and output tokens
  before it can be used, and a call to an unpriced model is refused before
  anything is sent. A price of 0 is allowed. Models fetched from a local
  connection are priced 0.
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

### Running a local model with Ollama

Run Ollama on the host and add a connection from the Ollama preset. On Linux
hosts `compose.yaml` maps `host.docker.internal` to the host, so a host-run
Ollama is reachable from the container. If Ollama is stopped, **Test** reports
it and a job using it fails with a readable log line (the API returns 502)
instead of crashing.

## Configuration

Environment variables, read at start. `compose.yaml` passes `.env` through.
All are optional.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Becomes the Anthropic connection: stored, shown read-only on the Settings page, applied at every start |
| `RAGQA_MODEL` | The `default` job's model: `<connection>/<model id>`, or a bare model id that one enabled model has, e.g. `claude-haiku-5-5`. A value that names no stored model is taken as an Anthropic model id and added, unpriced, to the Anthropic connection when there is one; price it on the Settings page before use. Read-only on the Settings page while set. With only a key set, the default starts as `claude-haiku-5-5` and stays editable |
| `RAGQA_MONTHLY_CAP_USD` | The monthly cap. Read-only on the Settings page while set; unset, the stored cap stands (default `20`) |
| `RAGQA_USERNAME`, `RAGQA_PASSWORD` | HTTP basic auth over every route. Set both or neither; exactly one set is a 503 on every request |
| `RAGQA_DENSE_MODEL` | Retrieval embedding model (fastembed). Default `Qwen/Qwen3-Embedding-0.6B-Q` |
| `RAGQA_JUDGE_MODEL` | A different model, used only to score solution similarity. Default `BAAI/bge-small-en-v1.5` |
| `RAGQA_RERANK_MODEL` | Cross-encoder for `hybrid_rerank`. Default `BAAI/bge-reranker-v2-m3-int8` |
| `RAGQA_QDRANT_URL` | Set by compose. Unset means in-process Qdrant (tests only) |
| `HF_TOKEN` | Hugging Face read token, only to load GPQA (gated) |
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
  sent. Local models are priced $0.
- **Estimate before spend.** `/eval`, resumes, paid loads, paid `/documents`,
  imports and keyword-list proposals return the estimate unless
  `confirm: true` is set. The Lab (`/query`) is the exception: it spends at
  once and shows each method's cost after.
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
much. On a local model, all of it is $0.

## Pages

| Route | What it shows |
|---|---|
| `/` | Every collection with its answer (does keyword retrieval help?), the run it came from and its state; create a collection or set up the benchmarks |
| `/c/{name}` | One collection: the answer with its gap bar, its runs and a form to start one, its items, questions and sources, imports, the three keyword lists (edit, propose, save), delete |
| `/c/{name}/runs/{id}` | One run |
| `/lab` | One question, methods side by side |
| `/worker` | The worker, its queue and what needs a person |
| `/settings` | AI connections, models and prices, the model each job uses, the monthly cap |
| `/docs` | Every endpoint |

## Endpoints

Every endpoint is listed with its schema at http://localhost:8080/docs.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, and whether the `answer` job has an AI model (`{ok, ai}`) |
| `POST /collections` | `{name, qa_type: science\|mcq\|multihop\|table, summarize}` |
| `GET /collections`, `GET /collections/{name}` | Specs; one collection's count, keyword lists, available methods, `state` (`loading`, `rekeying`, `evaluating`, `queued`, `empty`, `no_lists` or `ready`) with its job and progress, question counts, its answer and number of runs |
| `DELETE /collections/{name}` | Delete a collection, its items, questions, lists and runs. Its queued jobs are cancelled; 409 while one of its jobs runs |
| `POST /collections/{name}/documents` | Add your own items `{items: [{id, text, answer?, explanation?, meta?}], confirm}`; 409 while a load, re-key or eval on the collection is queued or running. Over 20 items it queues an import job instead and returns `{job_id}` |
| `POST /collections/{name}/uploads?filename=&kind=items\|questions` | The raw file as the body (up to 200 MB). Returns `{upload_id, preview, estimate}`: the first rows under a guessed field mapping, and what importing would cost |
| `POST /collections/{name}/imports` | `{upload_id, kind, mapping, source_name, confirm}`. Without `confirm`: the preview and estimate. With it: queues an import job and returns `{job_id}`; 422 when a required field is unmapped or no row is usable; 503 when it would spend and no AI model is set up |
| `GET /collections/{name}/sources` | Each source with its item count and whether retrieval uses it |
| `PATCH /collections/{name}/sources/{source}` | `{enabled}`: switch a source on or off for retrieval. 409 while an eval runs, or to switch off the last item source |
| `DELETE /collections/{name}/sources/{source}` | Remove a source's items, chunks and imported questions; 409 while a job is queued or running on the collection |
| `GET /collections/{name}/questions?leaked=` | The collection's questions, with how many are leaked; `leaked=true` or `false` filters |
| `PATCH /collections/{name}/questions/leaked` | `{include}`: whether leaked imported questions are used by runs |
| `POST /collections/{name}/load` | Load `scienceqa`, `gpqa`, `musique`, `tatqa` or `scibench`. When it would spend (summaries, or an AI keyword proposal because an AI model is set up and there are no lists), without `confirm` it returns the estimate `{est_usd, items, calls}`; with `{confirm: true}`, or when it is free, it queues a job and returns `{job_id}` |
| `POST /collections/{name}/keyword-lists/propose` | The AI proposes topic and method lists; nothing is saved. Without `confirm`: the estimate `{est_usd, calls}`; with `{confirm: true}`: the lists |
| `GET /collections/{name}/keyword-lists`, `GET …/keyword-lists/{version}` | Every saved version of the lists, newest first; one version with its lists |
| `PUT /collections/{name}/keyword-lists` | Save `{topic, method, common}` as a version and queue a `rekey` job; returns `{version, job_id}`. The collection switches to the new lists once every item is re-keyed. 409 while an eval runs on it, or when the service's dense model differs from the collection's |
| `POST /query` | One question, several methods side by side, optionally answered; each method's `usd` is every AI call it made (query summary, hops, answer) |
| `POST /eval` | Without `confirm`: the estimate. With it: queues a run and returns `{job_id, estimate}`; 503 when the run would spend and no AI model is set up |
| `GET /eval?collection=` | Every run, newest first, one summary row each |
| `GET /eval/{job_id}`, `GET /eval/{job_id}/items` | A run's answer, gaps and table, with its log and `since` (see [Reading a run](#reading-a-run)); one method's per-question rows |
| `GET /eval/{job_id}/questions?disagree=` | Each question with every method's row and its `outcome`; `disagree=true` keeps the questions where the methods' outcomes differ, leaving out the controls and `question_only` |
| `DELETE /eval/{job_id}` | Delete a finished run and its rows; 409 while it is queued or running |
| `GET /jobs?collection=&kind=&status=`, `GET /jobs/{job_id}` | Jobs, newest first; one job's status, progress and log |
| `POST /jobs/{job_id}/cancel` | Cancel a queued job, or ask a running one to stop; 409 for any other status |
| `POST /jobs/{job_id}/resume` | Resume a stopped run. Without `confirm`: the estimate for what is left, or 409 with why it cannot resume. With it: queued again |
| `GET /status` | The worker, the queue, pending AI batches and the problems that need a person |
| `GET /catalog` | Plain names and descriptions of methods, metrics and QA types |
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
`/query`, for example a keyword method before lists exist, reports its own
`error`, and the other methods still return.

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
- **`since`** says what changed after the run: `{untracked, changes}`, where
  `changes` lists `keyword lists`, `dense model` and `sources`. A migrated run
  that recorded no versions is `untracked: true`.
- **`methods`** holds each method's own score as `mean [low, high]`, and
  **`pairs`** every pair compared.
- **`proxy`** (exam QA type, with answers): whether solution similarity, the
  free metric, ranks retrieval methods the way accuracy does. It needs 4 or
  more methods, leaves out No retrieval and the controls, and passes only if
  the rank correlation is at least 0.5 and the declared pair points the same
  way on both.
- Every run records its full setup: models, keyword-list version, `k`,
  `rrf_k`, seed, dataset filter, sources and question ids. `log` holds the
  run's last 50 events.

## Using the AI layer in another project

The AI layer is its own workspace package, `aiconnect`
([packages/aiconnect](packages/aiconnect)), with no dependency on the rest of
rag-qa. It brings connections, prices, jobs, the monthly cap, the response
cache, the spend ledger and batches, all stored in one SQLite file you choose.

```python
from pathlib import Path
from aiconnect.ai import AI

ai = AI(Path("my-app.db"))
ai.add_connection("ollama", "openai", "Ollama", base_url="http://localhost:11434/v1", local=True)
ai.add_model("ollama", "llama3.1:8b", price_in=0, price_out=0)
ai.set_role("default", "ollama/llama3.1:8b")

reply = ai.complete("Name three noble gases.", role="default", purpose="demo")
print(reply.text, reply.usd, reply.cached)
```

`complete_many` batches where the provider supports it, `estimate_usd` prices
a call before it is made, and `month_spend` reports the ledger. Errors are the
package's own: `BudgetExceeded`, `NoConnection`, `UnknownModel`,
`UnknownPrice`, `ProviderError`, `Conflict`, `Cancelled`.

## Roadmap

The next version turns the bench into a chat over your own documents that
shows its evidence, with the keyword method brought up to date:

- **Keyword-RAG v2** (in progress): build recipes per collection, a keyword
  vocabulary induced from the corpus and reviewed by hand instead of curated
  lists, keywords that describe the skill a solution uses rather than the
  question's topic, a classifier that predicts a question's keywords, and
  abstracted worked examples. New benchmarks: BRIGHT AoPS and TheoremQA-Q,
  scored by nDCG@10.
- **Chat**: ask across several corpora, with routing, inline citations and an
  editable retrieval step under every answer.
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
