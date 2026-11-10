#!/usr/bin/env bash
#
# Local dev Qdrant for keyword-rag: a throwaway container the integration tests and
# a locally-run API can point at. Not for production.
#
#   ./scripts/dev-qdrant.sh up      # start (idempotent)
#   ./scripts/dev-qdrant.sh down    # stop and remove
#   ./scripts/dev-qdrant.sh url     # print the URL
#   ./scripts/dev-qdrant.sh status  # exit 0 if running, 1 if not
#   ./scripts/dev-qdrant.sh image   # print the image (compose.yaml's pin)
#
# Override defaults via env: RAGQA_DEV_QDRANT_PORT, RAGQA_DEV_QDRANT_IMAGE.
set -euo pipefail

NAME="ragqa-dev-qdrant"
# The image is pinned once, in compose.yaml: the qdrant service under
# `services:` (the file also has a `qdrant:` volume, which has no image).
compose_image() {
  awk '
    /^[^ #]/ { in_services = ($0 == "services:"); in_qdrant = 0 }
    in_services && /^  [a-z]/ { in_qdrant = ($0 == "  qdrant:") }
    in_qdrant && /^    image:/ { print $2; exit }
  ' "$(dirname "$0")/../compose.yaml"
}
IMAGE="${RAGQA_DEV_QDRANT_IMAGE:-$(compose_image)}"
PORT="${RAGQA_DEV_QDRANT_PORT:-56333}"
URL="http://localhost:${PORT}"

need_docker() {
  command -v docker >/dev/null 2>&1 || { echo "docker not found — install Docker Desktop." >&2; exit 1; }
}

is_running() { [ -n "$(docker ps -q -f "name=^${NAME}$" 2>/dev/null)" ]; }
exists()     { [ -n "$(docker ps -aq -f "name=^${NAME}$" 2>/dev/null)" ]; }

up() {
  need_docker
  if is_running; then
    echo "✓ ${NAME} already running" >&2
  else
    if exists; then docker rm -f "$NAME" >/dev/null; fi
    echo "starting ${NAME} (${IMAGE}) on 127.0.0.1:${PORT}…" >&2
    # Loopback only: a dev store needs no network exposure.
    docker run -d --name "$NAME" -p "127.0.0.1:${PORT}:6333" "$IMAGE" >/dev/null
  fi
  for _ in $(seq 1 60); do
    if curl -sf "${URL}/readyz" >/dev/null; then echo "✓ ready at ${URL}" >&2; return; fi
    sleep 1
  done
  echo "qdrant did not become ready at ${URL}" >&2
  exit 1
}

down() {
  need_docker
  if exists; then docker rm -f "$NAME" >/dev/null && echo "✓ removed ${NAME}" >&2; fi
}

case "${1:-}" in
  up) up ;;
  down) down ;;
  url) echo "$URL" ;;
  image) echo "$IMAGE" ;;
  status) need_docker; is_running ;;
  *) echo "usage: $0 up|down|url|status|image" >&2; exit 2 ;;
esac
