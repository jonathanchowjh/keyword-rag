"""Put authored build outputs (a vocabulary, and each item's tags and abstracted form) into
a collection with no AI call. The owner runs it inside the api container, where the
database and Qdrant are:

    docker compose -p rag-qa exec api python -m ragqa_core.apply_outputs \\
        COLLECTION VOCABULARY.json OUTPUTS.jsonl

VOCABULARY.json is {"terms": [{"term": ..., "definition": ...}, ...]}. OUTPUTS.jsonl has one
row per stored item: {"id", "tags", "principle", "method", "answer"}. The service never
imports this module.
"""

import argparse
import json
import sys
from pathlib import Path

from aiconnect.ai import AI
from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.indexer import Indexer
from ragqa_core.models import Models
from ragqa_core.settings import load_settings
from ragqa_core.store import Store

BUSY = ("load", "import", "retag", "copy", "eval")


def make_builder() -> Builder:
    """The same objects the service builds, from the same settings."""
    settings = load_settings()
    conn = db.connect(settings.home / "ragqa.db")
    store = Store(settings.qdrant_url, None if settings.qdrant_url else settings.home / "qdrant")
    models = Models(
        settings.dense_model, settings.judge_model, settings.rerank_model, settings.home / "models"
    )
    ai = AI(settings.home / "ragqa.db")
    return Builder(conn, store, models, ai, Indexer(conn, store, models, ai))


def read_outputs(path: Path) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if line.strip():
            row = json.loads(line)
            item_id = row.pop("id", None)
            if not isinstance(item_id, str) or item_id in outputs:
                raise ValueError(f"{path.name} line {n}: a missing or repeated id")
            outputs[item_id] = row
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Put authored build outputs into a collection.")
    parser.add_argument("collection")
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument("outputs", type=Path)
    args = parser.parse_args(argv)
    try:
        terms = [t["term"] for t in json.loads(args.vocabulary.read_text())["terms"]]
        outputs = read_outputs(args.outputs)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    builder = make_builder()
    spec = db.get_collection(builder.conn, args.collection)
    if spec is None:
        print(f"refused: unknown collection {args.collection!r}", file=sys.stderr)
        return 2
    busy = [k for k in BUSY if db.job_waiting_or_running(builder.conn, k, spec.name)]
    if busy:
        print(f"refused: a {busy[0]} job is queued or running on {spec.name}", file=sys.stderr)
        return 2
    try:
        result = builder.apply_outputs(spec, terms, outputs)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
