"""Authored build outputs on the Hugging Face Hub, read by a load so a collection gets the
same vocabulary, tags and abstracted examples with no AI call.

The repo holds `manifest.json` (one entry per build: dataset, limit, recipe, build
settings, who authored it, and the sha256 of each file), a dataset card, and each entry's
`vocabulary.parquet` and `outputs.parquet` under builds/<entry id>/. A reader takes the
commit of the manifest on main first and reads everything at that commit, so a publish
made meanwhile cannot mix two versions. This module is the only one that talks to the Hub
or knows the files' format; only `ragqa_core.publish`, which the owner runs, writes.
"""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
)
from huggingface_hub.errors import RemoteEntryNotFoundError, RepositoryNotFoundError

from ragqa_core.datasets import ID_PREFIX

MANIFEST = "manifest.json"
VOCABULARY_COLUMNS = ("position", "term", "definition")
OUTPUT_COLUMNS = ("doc_id", "tags", "principle", "method", "answer")


def entry_id(dataset: str, limit: int | None, recipe: str) -> str:
    return f"{dataset}-{limit or 'all'}-{recipe}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_entry(
    root: Path, ident: str, terms: list[str], definitions: dict[str, str], rows: list[dict]
) -> dict[str, Path]:
    """An entry's two files under root/builds/<ident>/, by their path in the repo. `rows`
    hold OUTPUT_COLUMNS, with the dataset's own document ids."""
    folder = root / "builds" / ident
    folder.mkdir(parents=True, exist_ok=True)
    position, term, definition = VOCABULARY_COLUMNS
    vocabulary = {
        position: list(range(len(terms))),
        term: terms,
        definition: [definitions.get(t) for t in terms],
    }
    pq.write_table(pa.table(vocabulary), folder / "vocabulary.parquet")
    doc_id, tags, *text = OUTPUT_COLUMNS
    schema = pa.schema(
        [(doc_id, pa.string()), (tags, pa.list_(pa.string())), *((c, pa.string()) for c in text)]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), folder / "outputs.parquet")
    return {f"builds/{ident}/{f.name}": f for f in sorted(folder.iterdir())}


def read_entry(vocabulary: Path, outputs: Path, prefix: str) -> tuple[list[str], dict[str, dict]]:
    """(the terms in order, the outputs by the collection's item id: `prefix` put back on)."""
    position, term, _ = VOCABULARY_COLUMNS
    listed = sorted(pq.read_table(vocabulary).to_pylist(), key=lambda r: r[position])
    doc_id = OUTPUT_COLUMNS[0]
    rows = pq.read_table(outputs).to_pylist()
    return [r[term] for r in listed], {prefix + r.pop(doc_id): r for r in rows}


class Hub:
    def __init__(self, repo: str, cache: Path):
        self.repo = repo
        self.cache = cache

    def _commit(self) -> str:
        """The commit of the manifest on main, from the Hub's x-repo-commit header."""
        url = hf_hub_url(self.repo, MANIFEST, repo_type="dataset")
        commit = get_hf_file_metadata(url).commit_hash
        if not commit:
            raise ValueError("the Hub named no commit for the manifest")
        return commit

    def _download(self, path: str, revision: str) -> Path:
        return Path(
            hf_hub_download(
                self.repo, path, repo_type="dataset", revision=revision, cache_dir=self.cache
            )
        )

    def manifest(self) -> tuple[str, dict]:
        """(the commit, the manifest at it). Raises whatever the Hub raises."""
        commit = self._commit()
        return commit, json.loads(self._download(MANIFEST, commit).read_text())

    def published(self) -> tuple[str | None, dict | None]:
        """What a publish goes on top of: manifest() or, for a repo or manifest not made yet,
        (None, None). Any other failure raises, so no other entry is ever replaced."""
        try:
            return self.manifest()
        except (RepositoryNotFoundError, RemoteEntryNotFoundError):
            return None, None

    def owner(self) -> str | None:
        """Who HF_TOKEN belongs to. Raises without a usable token."""
        return HfApi().whoami().get("name")

    def send(self, files: dict[str, Path], message: str, parent: str | None) -> None:
        """Every file, by its path in the repo, in one commit on top of `parent`: the Hub
        refuses it if main moved meanwhile."""
        api = HfApi()
        api.create_repo(self.repo, repo_type="dataset", exist_ok=True)
        api.create_commit(
            self.repo,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(local))
                for path, local in files.items()
            ],
            commit_message=message,
            parent_commit=parent,
        )

    def fetch(self, dataset: str, limit: int | None, recipe: str) -> tuple[dict | None, str]:
        """The entry for this load, as ({entry, terms, outputs}, "") with outputs keyed by
        the collection's item ids; or (None, why not). NEVER RAISES: a load goes on
        without the Hub, whatever went wrong with it."""
        try:
            commit, manifest = self.manifest()
        except Exception as e:  # noqa: BLE001 - any failure of the Hub means a local build
            return None, f"unreachable ({type(e).__name__}: {e})"
        wanted = entry_id(dataset, limit, recipe)
        entry = next((e for e in manifest.get("entries", []) if e.get("id") == wanted), None)
        if entry is None:
            return None, f"no entry {wanted}"
        local: dict[str, Path] = {}
        for path, digest in entry.get("files", {}).items():
            try:
                local[Path(path).stem] = self._download(path, commit)
            except Exception as e:  # noqa: BLE001
                return None, f"{path} unreachable ({type(e).__name__}: {e})"
            if sha256(local[Path(path).stem]) != digest:
                return None, f"{path} does not match its sha256 in the manifest"
        if set(local) != {"vocabulary", "outputs"}:
            return None, f"entry {wanted} lacks its vocabulary or outputs"
        try:
            terms, outputs = read_entry(
                local["vocabulary"], local["outputs"], ID_PREFIX.get(dataset, "")
            )
        except Exception as e:  # noqa: BLE001
            return None, f"entry {wanted} is unreadable ({type(e).__name__}: {e})"
        return {"entry": entry, "terms": terms, "outputs": outputs}, ""
