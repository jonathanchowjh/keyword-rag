"""Authored build outputs on the Hugging Face Hub, read by a load so a collection gets the
same vocabulary, tags and abstracted examples with no AI call.

The repo holds `manifest.json` (one entry per build: dataset, limit, recipe, who authored
it, and the sha256 of each file), a dataset card, and each entry's `vocabulary.parquet`
and `outputs.parquet` under builds/<entry id>/. A reader takes the commit of the manifest
on main first and reads everything at that commit, so a publish made meanwhile cannot mix
two versions. Only `ragqa_core.publish`, which the owner runs, ever writes to it.
"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url

from ragqa_core.datasets import BRIGHT_REPO, ID_PREFIX

MANIFEST = "manifest.json"
VOCABULARY_COLUMNS = ("position", "term", "definition")
OUTPUT_COLUMNS = ("doc_id", "tags", "principle", "method", "answer")
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


def entry_id(dataset: str, limit: int | None, recipe: str) -> str:
    return f"{dataset}-{limit or 'all'}-{recipe}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_manifest(remote: dict | None, entry: dict) -> dict:
    """The Hub's manifest with `entry` in place of the one with its id, or added last."""
    entries = list((remote or {}).get("entries", []))
    ids = [e.get("id") for e in entries]
    if entry["id"] in ids:
        entries[ids.index(entry["id"])] = entry
    else:
        entries.append(entry)
    return {**(remote or {}), "entries": entries}


def dataset_card(manifest: dict) -> str:
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
            "Build outputs for [keyword-rag](https://github.com/jonathanchowjh/keyword-rag)"
            " collections: a vocabulary of technique terms, and each document's tags from it"
            " with its abstracted form (principle, method, answer). A keyword-rag load of the"
            " same dataset, limit and recipe applies them with no AI call; anyone can read"
            " them without keyword-rag.",
            "",
            "| Entry | Dataset | Limit | Recipe | Documents | Authored by |",
            "|---|---|---|---|---|---|",
            *rows,
            "",
            "`manifest.json` lists every entry with the sha256 of its files.",
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
            f'base = "hf://datasets/<repo>/builds/{first}"',
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
            listed = pq.read_table(local["vocabulary"]).to_pylist()
            terms = [r["term"] for r in sorted(listed, key=lambda r: r["position"])]
            prefix = ID_PREFIX.get(dataset, "")
            outputs = {
                prefix + r.pop("doc_id"): r for r in pq.read_table(local["outputs"]).to_pylist()
            }
        except Exception as e:  # noqa: BLE001
            return None, f"entry {wanted} is unreadable ({type(e).__name__}: {e})"
        return {"entry": entry, "terms": terms, "outputs": outputs}, ""
