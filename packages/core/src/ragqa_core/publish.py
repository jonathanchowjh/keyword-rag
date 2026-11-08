"""Publish a collection's build (its vocabulary, and each item's tags and abstracted form)
to the Hub, so a load of the same dataset, limit and recipe applies it with no AI call.
The owner runs it inside the api container, where the database and Qdrant are:

    docker compose -p rag-qa exec -e HF_TOKEN api python -m ragqa_core.publish \\
        COLLECTION --authored-by "WHO" [--vocabulary VOCABULARY.json] [--publish]

It is a dry run unless --publish: the files are written under --out and nothing is sent.
With --publish it refuses unless the token's owner owns the repo, and sends every file in
one commit on top of the manifest it read. Only the vocabulary, the tags and the
abstracted forms are published: never embeddings, settings, keys, connections or the
ledger. The service never imports this module.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ragqa_core import db
from ragqa_core.builder import Builder, abstracted_parts, as_item, build_of, make_builder
from ragqa_core.datasets import BRIGHT_REPO, ID_PREFIX
from ragqa_core.hub import MANIFEST, Hub, entry_id, sha256, write_entry
from ragqa_core.settings import load_settings
from ragqa_core.vocabulary import read_vocabulary

# The datasets a build may be published for: each names where its text comes from, so the
# card credits it. A dataset not here is refused, since its licence was never checked.
SOURCES: dict[str, dict[str, str]] = {
    "bright_aops": {
        "repo": BRIGHT_REPO,
        "task": "aops",
        "problems": "AoPS, MATH, AQuA-RAT and TheoremQA problems (as collected in BRIGHT)",
    },
    "bright_theoremqa": {
        "repo": BRIGHT_REPO,
        "task": "theoremqa_questions",
        "problems": "TheoremQA problems (as collected in BRIGHT)",
    },
}
BRIGHT_CREDIT = (
    "BRIGHT: Su et al. 2024, *BRIGHT: A Realistic and Challenging Benchmark for"
    " Reasoning-Intensive Retrieval* ([`xlangai/BRIGHT`](https://huggingface.co/datasets/"
    "xlangai/BRIGHT), CC BY 4.0)"
)


class Refused(Exception):
    pass


def source_revision(home: Path, repo: str) -> str | None:
    """The commit of `repo` the loads read, from the Hub cache's refs/main; None if unknown."""
    ref = home / "datasets" / f"datasets--{repo.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def merge_manifest(remote: dict | None, entry: dict) -> dict:
    """The Hub's manifest with `entry` in place of the one with its id, or added last."""
    entries = list((remote or {}).get("entries", []))
    ids = [e.get("id") for e in entries]
    if entry["id"] in ids:
        entries[ids.index(entry["id"])] = entry
    else:
        entries.append(entry)
    return {**(remote or {}), "entries": entries}


def dataset_card(manifest: dict, repo: str) -> str:
    """The repo's README: viewer configs, what each entry is, its columns, a pandas
    example, and credit to the data the outputs describe."""
    entries = manifest.get("entries", [])
    configs = []
    for e in entries:
        for path in sorted(e.get("files", {})):
            kind = Path(path).stem  # vocabulary or outputs
            configs.append(f"- config_name: {e['id']}-{kind}\n  data_files: {path}")
    rows = [
        f"| `{e['id']}` | {e.get('dataset')} | {e.get('limit') or 'all'} | {e.get('recipe')}"
        f" | {e.get('items')} | {e.get('authored_by')} |"
        for e in entries
    ]
    known = [e["dataset"] for e in entries if e.get("dataset") in SOURCES]
    credits = sorted({SOURCES[d]["problems"] for d in known})
    first = entries[0]["id"] if entries else "<entry id>"
    return "\n".join(
        [
            "---",
            "license: cc-by-4.0",
            "pretty_name: keyword-rag builds",
            "configs:",
            *configs,
            "---",
            "",
            "# keyword-rag builds",
            "",
            "Build outputs for [rag-qa](https://github.com/jonathanchowjh/rag-qa)"
            " collections: a vocabulary of technique terms, and each document's tags from it"
            " with its abstracted form (principle, method, answer). A rag-qa load of the"
            " same dataset, limit, recipe and build settings applies them with no AI call;"
            " anyone can read them without rag-qa.",
            "",
            "| Entry | Dataset | Limit | Recipe | Documents | Authored by |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "`manifest.json` lists every entry with its build settings and the sha256 of its"
            " files.",
            "",
            "## Columns",
            "",
            "`vocabulary.parquet`: `position` (the term's place in the list), `term`,"
            " `definition` (may be empty).",
            "",
            "`outputs.parquet`: `doc_id` (the source dataset's own document id), `tags`"
            " (1 to 3 terms of the vocabulary), `principle`, `method`, `answer` (empty for a"
            " document with no worked solution).",
            "",
            "```python",
            "import pandas as pd",
            "",
            f'base = "hf://datasets/{repo}/builds/{first}"',
            'outputs = pd.read_parquet(f"{base}/outputs.parquet")',
            'vocabulary = pd.read_parquet(f"{base}/vocabulary.parquet")',
            "```",
            "",
            "## Credit and licence",
            "",
            "The documents these outputs describe are not copied here; they come from"
            f" {BRIGHT_CREDIT}, which collects {'; '.join(credits) or 'its source problems'}."
            " The outputs are released under CC BY 4.0.",
            "",
        ]
    )


def publish(
    builder: Builder,
    hub: Hub,
    collection: str,
    *,
    authored_by: str,
    out: Path,
    home: Path,
    definitions: dict[str, str] | None = None,
    send: bool = False,
) -> dict:
    """Writes the entry's files, the merged manifest and the card under `out`; with
    `send`, sends them all in one commit. Returns {entry, files, parent, sent}. Raises
    Refused, before anything is sent, for a collection that cannot be published."""
    conn = builder.conn
    spec = db.get_collection(conn, collection)
    if spec is None:
        raise Refused(f"unknown collection {collection!r}")
    busy = db.busy_job(conn, collection)
    if busy:
        raise Refused(f"a {busy} job is queued or running on {collection!r}; wait")
    switching = db.switching_model(conn, collection)
    if switching:
        raise Refused(
            f"{collection!r} is switching to {switching}; resume it on the store page first"
        )
    loads = db.list_jobs(conn, kind="load", collection=collection, status="done", limit=1)
    if not loads:
        raise Refused(f"collection {collection!r} has no finished load to name its dataset")
    dataset, limit = loads[0]["setup"]["dataset"], loads[0]["setup"].get("limit")
    if dataset not in SOURCES:
        raise Refused(f"no source and licence on record for dataset {dataset!r}")
    terms = db.get_vocabulary(conn, collection)
    if not terms:
        raise Refused(f"collection {collection!r} has no vocabulary in force")
    if send:
        try:
            owner = hub.owner()
        except Exception as e:  # noqa: BLE001 - no token, a bad one, or no network
            raise Refused(
                f"the Hub did not say who the token belongs to ({e}); set HF_TOKEN"
            ) from e
        if owner != hub.repo.split("/")[0]:
            raise Refused(f"the token belongs to {owner!r}, not the owner of {hub.repo}")
    prefix = ID_PREFIX[dataset]
    stored = [h for batch in builder.store.scroll(collection) for h in batch]
    unprefixed = [h.id for h in stored if not h.id.startswith(prefix)]
    if unprefixed:
        raise Refused(f"{len(unprefixed)} items are not from {dataset}: {unprefixed[:5]}")
    outputs = {h.id: {"tags": h.tags, **abstracted_parts(h.abstracted)} for h in stored}
    try:  # the check a load makes before applying it: nothing is published that it would skip
        builder.checked_outputs(spec, [as_item(h) for h in stored], terms, outputs)
    except ValueError as e:
        raise Refused(f"a load could not apply this build: {e}") from e
    try:
        parent, remote = hub.published()
    except Exception as e:  # noqa: BLE001 - its other entries must not be replaced
        raise Refused(f"the Hub's manifest could not be read ({e}); nothing sent") from e
    ident = entry_id(dataset, limit, spec.recipe)
    rows = [{"doc_id": i[len(prefix) :], **outputs[i]} for i in sorted(outputs)]
    files = write_entry(out, ident, terms, definitions or {}, rows)
    source = SOURCES[dataset]
    entry = {
        "id": ident,
        "dataset": dataset,
        "source": {"repo": source["repo"], "revision": source_revision(home, source["repo"])},
        "limit": limit,
        "recipe": spec.recipe,
        "build": asdict(build_of(spec)),
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
    if send:
        hub.send(files, f"Publish {ident}", parent)
    return {"entry": entry, "files": files, "parent": parent, "sent": send}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish a collection's build to the Hub.")
    parser.add_argument("collection")
    parser.add_argument("--authored-by", required=True, help="who or what made the build")
    parser.add_argument(
        "--vocabulary", type=Path, help='the authored {"terms": [{term, definition}]}'
    )
    parser.add_argument("--repo", help="the Hub dataset repo (default: RAGQA_BUILD_HUB)")
    parser.add_argument("--out", type=Path, help="where the files are written")
    parser.add_argument("--publish", action="store_true", help="send them (needs HF_TOKEN)")
    args = parser.parse_args(argv)
    settings = load_settings()
    repo = args.repo or settings.build_hub
    if not repo:
        print("refused: no repo (--repo, or RAGQA_BUILD_HUB)", file=sys.stderr)
        return 2
    definitions = read_vocabulary(args.vocabulary)[1] if args.vocabulary else {}
    out = args.out or settings.home / "publish" / args.collection
    try:
        done = publish(
            make_builder(),
            Hub(repo, settings.home / "hub"),
            args.collection,
            authored_by=args.authored_by,
            out=out,
            home=settings.home,
            definitions=definitions,
            send=args.publish,
        )
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(json.dumps(done["entry"], indent=1))
    print(f"files under {out}; on top of commit {done['parent'] or '(a new manifest)'}")
    print(f"published to {repo}" if done["sent"] else "dry run: nothing sent (--publish)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
