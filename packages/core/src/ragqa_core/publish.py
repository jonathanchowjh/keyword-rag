"""Publish a collection's build (its vocabulary, and each item's tags and abstracted form)
to the Hub, so a load of the same dataset, limit and recipe applies it with no AI call.
The owner runs it inside the api container, where the database and Qdrant are:

    docker compose -p rag-qa exec -e HF_TOKEN api python -m ragqa_core.publish \\
        COLLECTION --authored-by "WHO" [--definitions VOCABULARY.json] [--publish]

It is a dry run unless --publish: the files are written under --out and nothing is sent.
With --publish it refuses unless the token's owner owns the repo, and sends every file in
one commit on top of the manifest it read. Only the vocabulary, the tags and the
abstracted forms are published: never embeddings, settings, keys, connections or the
ledger. The service never imports this module.
"""

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError, RepositoryNotFoundError

from ragqa_core import db
from ragqa_core.apply_outputs import make_builder
from ragqa_core.builder import Builder
from ragqa_core.connection import Connection
from ragqa_core.datasets import ID_PREFIX
from ragqa_core.hub import (
    MANIFEST,
    OUTPUT_COLUMNS,
    SOURCES,
    Hub,
    dataset_card,
    entry_id,
    merge_manifest,
    sha256,
)
from ragqa_core.settings import load_settings
from ragqa_core.vocabulary import clean_terms

ABSTRACTED = re.compile(r"Principle: (.*?)\nMethod: (.*)\nAnswer: (.*)", re.S)


class Refused(Exception):
    pass


def abstracted_parts(text: str | None) -> dict[str, str | None]:
    """The principle, method and answer of a stored abstracted form (builder's
    abstracted_text read back); all None without one."""
    found = ABSTRACTED.fullmatch(text or "")
    if not found:
        return {"principle": None, "method": None, "answer": None}
    principle, method, answer = found.groups()
    return {"principle": principle, "method": method, "answer": answer}


def source_revision(home: Path, repo: str) -> str | None:
    """The commit of `repo` the loads read, from the Hub cache's refs/main; None if unknown."""
    ref = home / "datasets" / f"datasets--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def read_definitions(path: Path) -> dict[str, str]:
    """{"terms": [{term, definition}]} by each term as the collection stores it (cleaned,
    so lower case), since that is the form the vocabulary is published in."""
    out: dict[str, str] = {}
    for t in json.loads(path.read_text())["terms"]:
        cleaned = clean_terms([t.get("term")])
        if cleaned and t.get("definition"):
            out.setdefault(cleaned[0], t["definition"])
    return out


def write_build(
    out: Path, ident: str, terms: list[str], definitions: dict[str, str], rows: list[dict]
) -> dict[str, Path]:
    """The entry's two files under out/builds/<ident>/, by their path in the repo."""
    folder = out / "builds" / ident
    folder.mkdir(parents=True, exist_ok=True)
    vocabulary = {  # VOCABULARY_COLUMNS
        "position": list(range(len(terms))),
        "term": terms,
        "definition": [definitions.get(t) for t in terms],
    }
    pq.write_table(pa.table(vocabulary), folder / "vocabulary.parquet")
    schema = pa.schema(
        [
            ("doc_id", pa.string()),
            ("tags", pa.list_(pa.string())),
            *((name, pa.string()) for name in OUTPUT_COLUMNS[2:]),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / "outputs.parquet")
    return {f"builds/{ident}/{f.name}": f for f in sorted(folder.iterdir())}


def publish(
    builder: Builder,
    hub: Hub,
    conn: Connection,
    collection: str,
    *,
    authored_by: str,
    out: Path,
    home: Path,
    definitions: dict[str, str] | None = None,
    api: HfApi | None = None,
) -> dict:
    """Writes the entry's files, the merged manifest and the card under `out`; with an
    `api`, sends them all in one commit. Returns {entry, files, parent, uploaded}.
    Raises Refused, before anything is sent, for a collection that cannot be published."""
    spec = db.get_collection(conn, collection)
    if spec is None:
        raise Refused(f"unknown collection {collection!r}")
    loads = db.list_jobs(conn, kind="load", collection=collection, status="done", limit=1)
    if not loads:
        raise Refused(f"collection {collection!r} has no finished load to name its dataset")
    dataset, limit = loads[0]["setup"]["dataset"], loads[0]["setup"].get("limit")
    if dataset not in SOURCES:
        raise Refused(f"no source and licence on record for dataset {dataset!r}")
    terms = db.get_vocabulary(conn, collection)
    if not terms:
        raise Refused(f"collection {collection!r} has no vocabulary in force")
    if api is not None:
        owner = api.whoami().get("name")
        if owner != hub.repo.split("/")[0]:
            raise Refused(f"the token belongs to {owner!r}, not the owner of {hub.repo}")
    prefix = ID_PREFIX[dataset]
    stored = [h for batch in builder.store.scroll(collection) for h in batch]
    untagged = [h.id for h in stored if not h.tags]
    if untagged:
        raise Refused(f"{len(untagged)} items have no tags; build them first: {untagged[:5]}")
    unprefixed = [h.id for h in stored if not h.id.startswith(prefix)]
    if unprefixed:
        raise Refused(f"{len(unprefixed)} items are not from {dataset}: {unprefixed[:5]}")
    rows = sorted(
        (
            {"doc_id": h.id[len(prefix) :], "tags": h.tags, **abstracted_parts(h.abstracted)}
            for h in stored
        ),
        key=lambda r: r["doc_id"],
    )
    try:
        parent, remote = hub.manifest()
    except (RepositoryNotFoundError, RemoteEntryNotFoundError):
        parent, remote = None, None  # a new repo, or one with no manifest yet
    except Exception as e:  # noqa: BLE001 - its other entries must not be replaced
        raise Refused(f"the Hub's manifest could not be read ({e}); nothing sent") from e
    ident = entry_id(dataset, limit, spec.recipe)
    files = write_build(out, ident, terms, definitions or {}, rows)
    source = SOURCES[dataset]
    entry = {
        "id": ident,
        "dataset": dataset,
        "source": {"repo": source["repo"], "revision": source_revision(home, source["repo"])},
        "limit": limit,
        "recipe": spec.recipe,
        "authored_by": authored_by,
        "vocabulary_version": spec.vocabulary_version,
        "items": len(rows),
        "files": {path: sha256(local) for path, local in files.items()},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest = merge_manifest(remote, entry)
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "README.md").write_text(dataset_card(manifest, hub.repo))
    files = {**files, MANIFEST: out / MANIFEST, "README.md": out / "README.md"}
    if api is not None:
        api.create_repo(hub.repo, repo_type="dataset", exist_ok=True)
        api.create_commit(
            hub.repo,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
                for path, local in files.items()
            ],
            commit_message=f"Publish {ident}",
            parent_commit=parent,
        )
    return {"entry": entry, "files": files, "parent": parent, "uploaded": api is not None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a collection's build to the Hub.")
    parser.add_argument("collection")
    parser.add_argument("--authored-by", required=True, help="who or what made the build")
    parser.add_argument("--definitions", type=Path, help='{"terms": [{term, definition}]}')
    parser.add_argument("--repo", help="the Hub dataset repo (default: RAGQA_BUILD_HUB)")
    parser.add_argument("--out", type=Path, help="where the files are written")
    parser.add_argument("--publish", action="store_true", help="send them (needs HF_TOKEN)")
    args = parser.parse_args(argv)
    settings = load_settings()
    repo = args.repo or settings.build_hub
    if not repo:
        print("refused: no repo (--repo, or RAGQA_BUILD_HUB)", file=sys.stderr)
        return 2
    definitions = read_definitions(args.definitions) if args.definitions else {}
    builder = make_builder()
    out = args.out or settings.home / "publish" / args.collection
    try:
        done = publish(
            builder,
            Hub(repo, settings.home / "hub"),
            builder.conn,
            args.collection,
            authored_by=args.authored_by,
            out=out,
            home=settings.home,
            definitions=definitions,
            api=HfApi() if args.publish else None,
        )
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(done["entry"], indent=1))
    print(f"files under {out}; on top of commit {done['parent'] or '(a new manifest)'}")
    print(f"published to {repo}" if done["uploaded"] else "dry run: nothing sent (--publish)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
