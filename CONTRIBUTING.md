# Contributing to rag-qa

This is the guide for working on the code. To *run* rag-qa, see the
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
| `./scripts/gate.sh` | The integration suite, the page tests and the image smoke test, then records the commit as pushable | Yes |
| `uv run ruff check . && uv run ruff format --check . && uv run pyright` | Lint, format, types | No |

## Hooks instead of CI

As in pg-ops, the git hooks are the whole of CI.

- **`pre-commit`** runs lint, format, types and the unit suite. It takes
  seconds and needs no Docker.
- **`scripts/gate.sh`** is the slow half. It records the commit SHA it passed
  on in `.git/ragqa-gate-receipt`.
- **`pre-push`** is instant. It refuses to push any commit without that
  receipt, so committing again after gating means gating again.
- **`commit-msg`** strips `Co-Authored-By` trailers.

Pre-push stays fast on purpose. `git push` opens the remote connection before
the hook runs, and a hook that works for minutes holds that connection idle
until it drops. `tests/test_boundaries.py` pins both halves of this.

**No automated test calls Claude.** Unit tests use `ragqa_core.fakes`: real
objects with their outward methods overridden on the instance. Only tests may
import it, and a guard test enforces that.

## The shape of the repo

```
packages/core/            Everything that knows retrieval. No FastAPI, no HTTP.
  src/ragqa_core/
    types.py              Dataclasses shared by every module
    db.py                 SQLite: ledger, cache, catalog, questions, jobs
    llm.py                Claude: send, cache, ledger, cap, estimate, batch
    models.py             Models: lazy fastembed dense / judge / BM25 / rerank
    keywords.py           Keyword assignment, pooled pick, TF-IDF, versions
    store.py              Store: Qdrant collections, named vectors, searches
    prompts.py            Prompt builders and answer parsing
    indexer.py            Indexer: build and refresh representations
    retriever.py          Retriever + rrf(): the twelve methods
    datasets.py           ScienceQA / MuSiQue / TAT-QA loaders, dev slice
    metrics.py            Recall, EM/F1, similarity, bootstrap, Spearman
    evaluator.py          Evaluator: estimates, background runs, tables
    settings.py           Settings + load_settings(): every environment variable
    fakes.py              Test doubles (tests only)
  tests/                  test_*.py unit; test_*_int.py integration
apps/api/                 FastAPI layer: routes, errors, helper pages
apps/worker/              ragqa_worker: the job-queue loop the API runs on a thread
scripts/                  dev-qdrant.sh, gate.sh, image-smoke.sh
tests/                    Repo-level guards and the smoke fixture
```

The worker loop runs on a thread the API starts; routes only queue jobs.

`ragqa_core` must never import `fastapi`, `starlette`, `uvicorn`, `ragqa_api`
or `ragqa_worker`; `ragqa_worker` must never import the web layer. `tests/test_boundaries.py` fails if it does.

## Code style

Light OOP. A class exists only where several functions share state
(`Claude`, `Models`, `Store`, `Indexer`, `Retriever`, `Evaluator`); everything
else is a plain function. No interfaces, base classes, inheritance or
injection seams. A test that needs a seam overrides the method on the
instance. One class per file, and few files.
