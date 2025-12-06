#!/usr/bin/env bash
# Build the image and drive it the way a user would, with NO API key: the
# retrieval-only path must work without one, and a spending call must say
# what is missing rather than fail obscurely.
set -euo pipefail

PROJECT=ragqa-smoke
BASE="http://127.0.0.1:58080"
FIXTURE=tests/fixtures/smoke_items.json
compose() { docker compose -p "$PROJECT" -f compose.yaml -f scripts/smoke.compose.yaml "$@"; }
cleanup() { compose down -v --remove-orphans >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

docker build -t ragqa:smoke .
docker volume create ragqa-smoke-models >/dev/null
compose up -d

echo "smoke: waiting for /health"
for _ in $(seq 1 120); do
  if curl -sf "$BASE/health" >/dev/null; then break; fi
  sleep 1
done
curl -sf "$BASE/health" | python3 -I -c 'import json,sys; h=json.load(sys.stdin); assert h=={"ok":True,"api_key":False}, h'

post() { curl -sf -X "$1" "$BASE$2" -H 'content-type: application/json' -d "$3"; }

post POST /collections '{"name":"smoke","qa_type":"mcq"}' >/dev/null
post POST /collections/smoke/documents "{\"items\": $(cat "$FIXTURE")}" >/dev/null
# Saving lists queues a rekey job; queries use the new lists only once it is done.
job=$(post PUT /collections/smoke/keyword-lists \
  '{"topic":["photosynthesis","gravity","magnetism","evaporation"],"method":["experiment","observation"],"common":["energy","water","force","light"]}' \
  | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')
echo "smoke: waiting for rekey job $job"
status=""
for _ in $(seq 1 120); do
  status=$(curl -sf "$BASE/jobs/$job" | python3 -I -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  case "$status" in done | failed | cancelled) break ;; esac
  sleep 1
done
[ "$status" = "done" ] || { echo "smoke: rekey job ended as '$status'" >&2; exit 1; }

post POST /query '{"collection":"smoke","question":"Which gas do plants take in for photosynthesis?","methods":["keyword","hybrid","kw_individual"]}' \
  | python3 -I -c 'import json,sys; r=json.load(sys.stdin)["results"]; bad=[m for m,v in r.items() if not v["hits"] or v["error"]]; assert not bad, r; print("smoke: retrieval ok:", {m: v["hits"][0]["id"] for m,v in r.items()})'

post POST /query '{"collection":"smoke","question":"Which gas?","choices":["oxygen","carbon dioxide"],"methods":["keyword"],"answer":true}' \
  | python3 -I -c 'import json,sys; r=json.load(sys.stdin)["results"]["keyword"]; assert r["error"]=="needs ANTHROPIC_API_KEY", r'

echo "image smoke: OK"
