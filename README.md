# rag-qa

A question-answering RAG microservice and test bench. It indexes three common
kinds of QA, runs the retrieval methods from *Keyword Vectorization for
Retrieval Augmented Generation on Science Question Answering* (Jonathan Chow,
2024) next to modern ones, and evaluates them over an HTTP API. Every result
says how sure it is, and nothing spends Claude credit without showing the
estimate first.

The paper's idea: long questions are noisy to embed, so describe each stored
item by a few keywords picked from curated lists (topic, method, common
terms) plus a short summary, and retrieve on that description. In 2024, on
ACT Science with gpt-3.5-turbo, the best keyword variant scored 78.9% against
63.1% for plain vector search. This service asks whether that still holds,
and where.

- **Status:** v0.1.0. Runs locally in Docker; no published image.
- **License:** Apache-2.0.

## Contents

- [Quick start](#quick-start)
- [First run](#first-run)
- [Configuration](#configuration)
- [Endpoints](#endpoints)
- [Retrieval methods](#retrieval-methods)
- [Datasets](#datasets)
- [Cost safety](#cost-safety)
- [Pages](#pages)
- [Reading a run](#reading-a-run)

How it is built and how to work on it is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

Requires Docker.

**1. A key that cannot overspend.** Claude Max 5x, Max 20x and Team plans
include monthly API credit ($100 a month on Max 5x).
1. On claude.ai, open **Settings → Billing → API credits** and link a Console
   organisation.
2. In the Console, create a workspace for rag-qa and give it its own spend
   limit.
3. Add no payment method and leave auto-reload off. When the credit runs
   out, requests stop; nothing is charged to your plan.
4. Create an API key in that workspace.

The claude.ai login itself must not be used: Anthropic's Consumer Terms
allow scripted access only through an API key.

**2. Run it.**

```sh
cp .env.example .env        # paste the key into ANTHROPIC_API_KEY
docker compose up --build
```

Open http://localhost:8080/docs for every endpoint, http://localhost:8080/lab
to compare methods on one question, and http://localhost:8080/ for the
collections and their evaluations (see [Pages](#pages)).

With no key, the service still starts. Every free retrieval method works, and
calls that would spend return 503 `needs ANTHROPIC_API_KEY`.

## First run

Create one collection per QA type and load its dataset. Loading downloads the
data, embeds it locally and saves keyword lists. It runs as a background job;
poll `GET /jobs/<job_id>`.

```sh
api=http://localhost:8080
curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"sqa","qa_type":"mcq","summarize":true}'
curl -X POST $api/collections/sqa/load -H 'content-type: application/json' \
  -d '{"dataset":"scienceqa"}'                  # returns the cost estimate
curl -X POST $api/collections/sqa/load -H 'content-type: application/json' \
  -d '{"dataset":"scienceqa","confirm":true}'   # starts the job

curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"mq","qa_type":"multihop"}'
curl -X POST $api/collections/mq/load -H 'content-type: application/json' \
  -d '{"dataset":"musique"}'

curl -X POST $api/collections -H 'content-type: application/json' \
  -d '{"name":"tq","qa_type":"table"}'
curl -X POST $api/collections/tq/load -H 'content-type: application/json' \
  -d '{"dataset":"tatqa"}'
```

Docker on macOS cannot use the Apple GPU, so embedding runs on the CPU. The
first load of MuSiQue's 11,656 passages takes roughly 10–20 minutes. After
that, everything is kept in the `ragqa` and `qdrant` volumes.

## Configuration

Environment variables, read at start. `compose.yaml` passes `.env` through.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Key from the Console workspace above. Empty: retrieval only. |
| `RAGQA_MONTHLY_CAP_USD` | This service's own monthly ceiling, counted from its own calls. Default `20`. |
| `RAGQA_MODEL` | Default Claude model: `claude-haiku-5-5` (default) or `claude-sonnet-5-5`. |
| `RAGQA_USERNAME`, `RAGQA_PASSWORD` | HTTP basic auth over every route. Set both or neither; exactly one set is a 503 on every request. |
| `RAGQA_DENSE_MODEL` | Retrieval embedding model (fastembed). Default `Qwen/Qwen3-Embedding-0.6B-Q`. |
| `RAGQA_JUDGE_MODEL` | A different model, used only to score solution similarity. Default `BAAI/bge-small-en-v1.5`. |
| `RAGQA_RERANK_MODEL` | Cross-encoder for `hybrid_rerank`. Default `BAAI/bge-reranker-v2-m3-int8`. |
| `RAGQA_QDRANT_URL` | Set by compose. Unset means in-process Qdrant (tests only). |
| `RAGQA_HOME` | State directory. `/home/app/.ragqa` in the image. |
| `PORT` | Listen port. `8080`. |

A collection remembers the dense model it was built with. Changing
`RAGQA_DENSE_MODEL` afterwards makes that collection refuse new items with a
409 that names both models.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness, and whether a key is set |
| `POST /collections` | `{name, qa_type: mcq\|multihop\|table, summarize}` |
| `GET /collections`, `GET /collections/{name}` | Specs; one collection's count, keyword lists, available methods, `state` (`loading`, `rekeying`, `evaluating`, `queued`, `empty`, `no_lists` or `ready`) with its job and progress, question counts, its answer and number of runs |
| `DELETE /collections/{name}` | Delete a collection, its items, questions, lists and runs. Its queued jobs are cancelled; 409 while one of its jobs runs |
| `POST /collections/{name}/documents` | Add your own items `{items: [{id, text, answer?, explanation?, meta?}], confirm}` |
| `POST /collections/{name}/load` | Load `scienceqa`, `musique` or `tatqa`: queues a job and returns `{job_id}` |
| `POST /collections/{name}/keyword-lists/propose` | Claude proposes topic and method lists; nothing is saved |
| `PUT /collections/{name}/keyword-lists` | Save `{topic, method, common}` as a version and queue a `rekey` job; returns `{version, job_id}`. The collection switches to the new lists once every item is re-keyed. 409 while an eval runs on it |
| `POST /query` | One question, several methods side by side, optionally answered |
| `POST /eval` | Without `confirm`: the estimate. With it: queues a run and returns `{job_id, estimate}` |
| `GET /eval?collection=` | Every run, newest first, one summary row each |
| `GET /eval/{job_id}`, `GET /eval/{job_id}/items` | A run's answer, gaps and table, with its log (see [Reading a run](#reading-a-run)); per-question rows |
| `DELETE /eval/{job_id}` | Delete a finished run and its rows; 409 while it is queued or running |
| `GET /jobs?collection=&kind=&status=`, `GET /jobs/{job_id}` | Jobs, newest first; one job's status, progress and log |
| `POST /jobs/{job_id}/cancel` | Cancel a queued job, or ask a running one to stop; 409 for any other status |
| `POST /jobs/{job_id}/resume` | Resume a stopped run. Without `confirm`: the estimate for what is left, or 409 with why it cannot resume. With it: queued again |
| `GET /status` | The worker, the queue, pending Claude batches and the problems that need a person |
| `GET /catalog` | Plain names and descriptions of methods, metrics and QA types |
| `GET /budget` | This service's spend this month against its cap |

Loads, re-keys and evaluations run one at a time, in the order they were
queued, on a worker thread inside the service. A job that was running when the
service stopped comes back `interrupted`; queued jobs stay queued.

A method that cannot run inside `/query`, for example a keyword method before
lists exist, reports its own `error`, and the other methods still return.

## Retrieval methods

| Method | Origin | What it does |
|---|---|---|
| `question_only` | paper | No retrieval; zero-shot baseline |
| `vector` | paper | Parent-document retriever over 100-word chunks |
| `keyword` | paper | BM25 on the full text |
| `hybrid` | paper | BM25 + dense, fused with reciprocal rank fusion |
| `kw_all` | paper | Top-9 keywords from the pooled lists (+ summary), one vector |
| `kw_individual` | paper | Top-3 from each list (+ summary), one vector; the paper's best |
| `kw_rrf` | paper | One vector per list (+ summary), fused with RRF |
| `hybrid_rerank` | modern | `hybrid` top-50 reranked by a local cross-encoder |
| `kw_hybrid` | new | `kw_individual` fused with `hybrid`: a dual index like Doc2Query++ |
| `iterative` | modern | Up to 3 retrieval hops; Claude decides the next query |
| `random_example` | control | Random items, seeded by the question: the same draw on every run |
| `fixed_example` | control | The same random items for every question |

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
| Exam multiple choice | ScienceQA | CC BY-SA 4.0 | 693 solved train examples; 291 test questions | solution similarity@1, same-skill@1 | accuracy |
| Multi-hop factual | MuSiQue (HippoRAG-2 protocol) | CC BY 4.0 | 11,656 passages; 1,000 dev questions | recall@2, recall@5 | EM, F1 |
| Table + text numeric | TAT-QA | CC BY 4.0 | 2,757 contexts; 1,668 dev questions | recall@1, recall@3 | EM, F1 |

ScienceQA is filtered to text-only natural science, grades 7–12, keeping one
item per near-duplicate template. Its human-written solutions serve as the
paper's chain-of-thought examples. TAT-QA answers use simplified scoring:
numbers are compared at 2 decimal places after stripping `$`, `,`, `%` and
scale words, and text is scored like SQuAD.

MuSiQue recall can be set beside published HippoRAG-2 numbers, but only as
**gain over our own dense baseline**: their baseline is NV-Embed-v2, ours a
small local model. EM and F1 come from a different reader and are not
comparable to the papers.

## Cost safety

- **The Console workspace spend limit is the real ceiling.** The service only
  sees its own calls; other keys in the organisation draw on the same credit.
- **A monthly cap inside the service** (`RAGQA_MONTHLY_CAP_USD`). Each call is
  priced from Claude's reported token usage and written to a ledger. A call
  that would cross the cap is refused before it is sent, and an evaluation
  that hits the cap stops with `stopped_budget` and keeps its rows.
- **Estimate before spend.** `/eval` and paid loads return the estimate unless
  `confirm: true` is set.
- **Every response is cached.** A re-run with the same prompts is free, and its
  answers are identical.
- **Haiku by default, and batches.** Runs of more than 50 calls use the Batch
  API at half price. A batch usually finishes in minutes but can take up to
  24 hours.
- **Localhost only.** The port is published on `127.0.0.1`. Opening it wider
  without basic auth would let anyone on the network spend up to the cap.

Rough prices on Haiku 5.5: a one-shot science answer is about $0.0005, and a
200-question × 10-method evaluation with answers about $1. Sonnet costs
about 20 times as much.

## Pages

| Route | What it shows |
|---|---|
| `/` | Every collection with its answer (does keyword retrieval help?), the run it came from and its state; create a collection or set up the three benchmarks |
| `/c/{name}` | One collection: the answer with its gap bar, its runs and a form to start one, its items and questions, the three keyword lists (edit, propose, save), delete |
| `/c/{name}/runs/{id}` | One run |
| `/lab` | One question, methods side by side |
| `/worker` | The worker, its queue and what needs a person |
| `/docs` | Every endpoint |

## Reading a run

`GET /eval/{job_id}` returns everything the run page shows, computed in the
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
- **`methods`** holds each method's own score as `mean [low, high]`, and
  **`pairs`** every pair compared.
- **`proxy`** (exam QA type, with answers): whether solution similarity, the
  free metric, ranks retrieval methods the way accuracy does. It needs 4 or
  more methods, leaves out No retrieval and the controls, and passes only if
  the rank correlation is at least 0.5 and the declared pair points the same
  way on both.
- Every run records its full setup: models, keyword-list version, `k`,
  `rrf_k`, seed, dataset filter and question ids. `log` holds the run's last
  50 events.
