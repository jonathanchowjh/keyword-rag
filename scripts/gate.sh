#!/usr/bin/env bash
# The slow half of verification: the integration suite against a real Qdrant
# and real models, then the image smoke test. .githooks/pre-push refuses any
# commit this has not passed.
#
# WHY A SCRIPT AND NOT THE HOOK (learned in pg-ops). `git push` opens the
# connection to the remote BEFORE running pre-push, so a hook's work happens
# while an idle receive-pack session is held open. Long enough and the remote
# drops it, and git dies writing the pack: exit 141, no output. Doing the work
# here, then pushing, is the only fix.
#
# Run this yourself, then push. The hook checks the receipt this writes.
set -euo pipefail

# THE COMMIT IS READ BEFORE THE TESTS, and the tree must be clean: a sha read
# at the end could bless a commit nobody tested, and a dirty tree tests work no
# commit holds.
sha=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "gate: refusing — the tree has changes no commit holds. Commit or stash them, then gate." >&2
  exit 1
fi

echo "gate: integration tests against a real Qdrant"
./scripts/dev-qdrant.sh up
RAGQA_TEST_QDRANT_URL=$(./scripts/dev-qdrant.sh url) uv run pytest -m integration -q

echo "gate: image smoke test"
./scripts/image-smoke.sh

# THE RECEIPT IS A SHA, not a timestamp: commit again and it no longer covers
# what you are pushing, so the hook refuses and you re-gate. It lives in .git/
# because it is a fact about this clone's verification and must never be
# committed.
if [ "$(git rev-parse HEAD)" != "$sha" ] || [ -n "$(git status --porcelain)" ]; then
  echo "gate: refusing — HEAD or the tree moved during the run, so $sha is not what was tested. Gate again." >&2
  exit 1
fi
receipt="$(git rev-parse --git-dir)/ragqa-gate-receipt"
printf '%s\n' "$sha" >> "$receipt"
# Bounded: the file is append-only and nothing else prunes it.
tail -n 50 "$receipt" > "$receipt.tmp" && mv "$receipt.tmp" "$receipt"

echo "gate: green. $sha may be pushed."
