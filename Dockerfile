# syntax=docker/dockerfile:1

# ---- build ----------------------------------------------------------------
# uv resolves from the committed lock, so the image gets exactly the versions
# the tests ran against. --no-editable installs the four workspace packages
# (aiconnect, ragqa-core, ragqa-worker, ragqa-api) into the venv itself, which is
# what lets the runtime stage carry only .venv and not the source tree.
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /repo
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock .python-version ./
COPY packages ./packages
COPY apps ./apps
RUN uv sync --frozen --no-dev --package ragqa-api --no-editable

# ---- runtime --------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app

# Everything the service writes lives under RAGQA_HOME: the SQLite file
# (ledger, cache, catalog, jobs), downloaded datasets and the fastembed model
# cache. One volume on that directory is the whole of its state; Qdrant keeps
# its own in its own container.
ENV RAGQA_HOME=/home/app/.ragqa \
    PORT=8080 \
    PATH=/repo/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1

COPY --from=build /repo/.venv /repo/.venv

# uid 1000, the usual first user. The directories are made here and owned by
# app because a named volume mounted on a path copies that path's owner from
# the image; on a path the image lacks, Docker makes it root's and the service
# cannot write. models/ is its own directory for the same reason: the smoke
# test mounts a persistent volume there alone.
RUN useradd --uid 1000 --create-home app \
 && mkdir -p /home/app/.ragqa/models \
 && chown -R app:app /home/app/.ragqa
USER app

EXPOSE 8080
# sh -c so ${PORT} is read at start, not frozen at build time.
CMD ["sh", "-c", "exec uvicorn --factory ragqa_api.app:create_app --host 0.0.0.0 --port ${PORT}"]
