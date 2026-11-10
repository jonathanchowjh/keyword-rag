# Contributing to keyword-rag

This is the guide for working on the code. To *run* keyword-rag, see the
[README](README.md).

## Setup

You need **Python 3.12** (uv installs it), **uv** and **Docker**. Docker is
only for Qdrant and the slow checks; the unit suite runs without it.

```sh
git clone <this repo> && cd rag-qa
uv sync
git config core.hooksPath .githooks       # one-time: the hooks below
./scripts/dev-qdrant.sh up                # Qdrant on 127.0.0.1:56333
```

To run the API outside Docker against that Qdrant:

```sh
RAGQA_QDRANT_URL=$(./scripts/dev-qdrant.sh url) RAGQA_HOME=.ragqa-dev \
  uv run uvicorn --factory ragqa_api.app:create_app --port 8080
```

## Everyday commands

| Command | What it runs | Needs Docker |
|---|---|---|
| `uv run pytest -m "not integration and not pages"` | Unit suite: pure logic, the API on fakes, source guards | No |
| `RAGQA_TEST_QDRANT_URL=$(./scripts/dev-qdrant.sh url) uv run pytest -m integration` | Real Qdrant, real small models, real dataset downloads | Yes |
| `uv run playwright install chromium && uv run pytest -m pages` | The console pages in a real browser, on the app with fakes; fails on any console error or failed request | No |
| `./scripts/image-smoke.sh` | Builds the image and drives it with no API key | Yes |
| `./scripts/gate.sh` | The commit checks (lint, format, types, unit tests), then the integration suite, the page tests and the image smoke test, then records the commit as pushable | Yes |
| `uv run ruff check . && uv run ruff format --check . && uv run pyright` | Lint, format, types | No |

## Hooks instead of CI

The git hooks are the whole of CI.

- **`pre-commit`** runs lint, format, types and the unit suite. It takes
  seconds and needs no Docker.
- **`scripts/gate.sh`** is the slow half. It runs the commit checks first, then records the commit SHA it passed
  on in `.git/ragqa-gate-receipt`.
- **`pre-push`** is instant. It refuses to push any commit without that
  receipt, so committing again after gating means gating again.
- **`commit-msg`** strips `Co-Authored-By` trailers.

Pre-push stays fast on purpose. `git push` opens the remote connection before
the hook runs, and a hook that works for minutes holds that connection idle
until it drops. `tests/test_boundaries.py` pins both halves of this.

**When the gate fails.** Read the failing test before assuming a regression. The
integration stage downloads datasets and models from Hugging Face, so a network
drop shows as `ConnectError`, a TLS timeout or a `CAS Client Error`, and makes
the run take about 30 minutes instead of 3. A page test that times out on
Create while the machine is busy (check `uptime`) is load, not code: rerun it
alone. Playwright teardown warnings about `Page._on_route` are noise. Anything
else is a real failure.

**No automated test calls an AI provider.** Unit tests use `ragqa_core.fakes`: real
objects with their outward methods overridden on the instance. Only tests may
import it, and a guard test enforces that.

## The shape of the repo

- `packages/aiconnect/`: the `AI` class, a separate package with no dependency
  on the rest (see its README)
  - `packages/aiconnect/src/aiconnect/`: `ai.py` (connections, models, roles,
    cap, cache, ledger, batches), `presets.py`, `types.py`, `wire.py`, `fakes.py`
- `packages/core/`: everything that knows retrieval. No FastAPI, no HTTP.
  - `packages/core/src/ragqa_core/types.py`: dataclasses shared by every module
  - `packages/core/src/ragqa_core/connection.py`: the one SQLite connection, one lock per statement
  - `packages/core/src/ragqa_core/db.py`: SQLite: catalog, eval sets, questions, jobs
  - `packages/core/src/ragqa_core/models.py`: lazy fastembed dense / judge / BM25 / rerank
  - `packages/core/src/ragqa_core/keywords.py`: keyword lists, the pooled pick and versions
  - `packages/core/src/ragqa_core/vocabulary.py`: an induced vocabulary: clean, merge, version
  - `packages/core/src/ragqa_core/recipes.py`: build recipes and build- and search-setting validation
  - `packages/core/src/ragqa_core/builder.py`: the build steps of the recipes
  - `packages/core/src/ragqa_core/store.py`: Qdrant collections, named vectors, searches
  - `packages/core/src/ragqa_core/prompts.py`: prompt builders and answer parsing
  - `packages/core/src/ragqa_core/indexer.py`: build and refresh representations
  - `packages/core/src/ragqa_core/retriever.py`: the 13 methods, and `rrf()`
  - `packages/core/src/ragqa_core/datasets.py`: ScienceQA, MuSiQue, TAT-QA, GPQA, SciBench, BRIGHT AoPS and BRIGHT TheoremQA loaders, dev slice
  - `packages/core/src/ragqa_core/loader.py`: the dataset-load job, and its keyword lists
  - `packages/core/src/ragqa_core/imports.py`: read an uploaded file into rows, items and questions
  - `packages/core/src/ragqa_core/text_split.py`: split a text file into named chunks
  - `packages/core/src/ragqa_core/chats.py`: saved chats searched across stores
  - `packages/core/src/ragqa_core/metrics.py`: recall, EM/F1, similarity, bootstrap, Spearman
  - `packages/core/src/ragqa_core/verdict.py`: what a run says: headline, gaps, answer, proxy check
  - `packages/core/src/ragqa_core/evaluator.py`: estimates, background runs, tables
  - `packages/core/src/ragqa_core/jobs.py`: what a job is: its kinds and endings
  - `packages/core/src/ragqa_core/program.py`: program-and-run answers in a limited subprocess
  - `packages/core/src/ragqa_core/catalog.py`: plain names of methods, metrics and QA types
  - `packages/core/src/ragqa_core/search_defaults.py`: the app-wide search defaults
  - `packages/core/src/ragqa_core/settings.py`: `Settings` and `load_settings()`: every environment variable
  - `packages/core/src/ragqa_core/hub.py`: authored build outputs on the Hugging Face Hub
  - `packages/core/src/ragqa_core/publish.py`: owner tool: publish a build to the Hub
  - `packages/core/src/ragqa_core/apply_outputs.py`: owner tool: apply authored build outputs with no AI call
  - `packages/core/src/ragqa_core/replies.py`: owner tool: answer a `manual` connection's prompts by hand
  - `packages/core/src/ragqa_core/fakes.py`: test doubles and shared test data (tests only)
  - `packages/core/tests/`: `test_*.py` unit; `test_*_int.py` integration
- `apps/api/`: the FastAPI layer
  - `apps/api/src/ragqa_api/app.py`: the wiring, the login, error mapping and the pages
  - `apps/api/src/ragqa_api/guards.py`: the checks the routes share before they queue or spend
  - `apps/api/src/ragqa_api/routes/`: one module per resource
  - `apps/api/src/ragqa_api/static/`: the console pages
  - `apps/api/tests/`: `test_app.py`, then one `test_*.py` per route module, with
    `api_helpers.py`, `conftest.py` and `api_snapshot.json`
- `apps/worker/`: `ragqa_worker`, the job-queue loop the API runs on a thread
- `scripts/`: `dev-qdrant.sh`, `gate.sh`, `image-smoke.sh`, the experiment tools
- `tests/`: repo-level guards (`tests/test_boundaries.py`), the experiment and
  timing script tests, and the smoke fixture in `tests/fixtures/`
  - `tests/pages/`: the page tests in a real browser, one file per page, with
    `page_helpers.py` and `conftest.py`

The worker loop runs on a thread the API starts. Loads, re-keys and
evaluations are queued as jobs for it. A few routes still work in the request:
`POST /collections/{name}/documents` indexes its items (and summarises them on
a summarising collection), and `/query`, `POST /chats/{id}/messages`,
`POST /eval-sets/{name}/answer` and a confirmed keyword-list proposal call the
AI provider.

`ragqa_core` must never import `fastapi`, `starlette`, `uvicorn`, `ragqa_api`
or `ragqa_worker`; `ragqa_worker` must never import the web layer. `tests/test_boundaries.py` fails if it does.

## Code style

Light OOP. A class exists only where several functions share state
(`AI`, `Connection`, `Models`, `Store`, `Indexer`, `Retriever`, `Loader`,
`Evaluator`, `Worker`); everything
else is a plain function. No interfaces, base classes, inheritance or
injection seams. A test that needs a seam overrides the method on the
instance. One class per file, and few files.
