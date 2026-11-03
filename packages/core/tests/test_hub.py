import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ragqa_core import db, loader
from ragqa_core.builder import Builder
from ragqa_core.db import connect, create_job, get_job
from ragqa_core.fakes import fake_ai, fake_hub, fake_models
from ragqa_core.hub import (
    MANIFEST,
    Hub,
    dataset_card,
    entry_id,
    merge_manifest,
    sha256,
)
from ragqa_core.indexer import Indexer
from ragqa_core.loader import Loader
from ragqa_core.store import Store
from ragqa_core.types import Item

TERMS = ["relative speed", "unit conversion"]
ROWS = [
    {"doc_id": "d1", "tags": ["relative speed"], "principle": "P", "method": "M", "answer": "4"},
    {
        "doc_id": "d2",
        "tags": ["unit conversion"],
        "principle": None,
        "method": None,
        "answer": None,
    },
]


def write_build(root: Path, ident: str, rows=ROWS) -> dict:
    """One entry's files under root/builds/<ident>/, and the entry naming them."""
    folder = root / "builds" / ident
    folder.mkdir(parents=True)
    vocabulary = {
        "position": list(range(len(TERMS))),
        "term": TERMS,
        "definition": [None] * len(TERMS),
    }
    pq.write_table(pa.table(vocabulary), folder / "vocabulary.parquet")
    pq.write_table(pa.Table.from_pylist(rows), folder / "outputs.parquet")
    files = {
        f"builds/{ident}/{name}": sha256(folder / name)
        for name in ("vocabulary.parquet", "outputs.parquet")
    }
    dataset, limit, recipe = "bright_aops", 2, "worked_examples"
    return {"id": ident, "dataset": dataset, "limit": limit, "recipe": recipe, "files": files}


def served(tmp_path, commits: dict[str, dict], newest: str | None) -> tuple[Hub, list]:
    """A Hub whose commits each hold a manifest, with files in tmp_path/<commit>/."""
    for commit, manifest in commits.items():
        (tmp_path / commit).mkdir(parents=True, exist_ok=True)
        (tmp_path / commit / MANIFEST).write_text(json.dumps(manifest))
    hub = Hub("owner/builds", tmp_path / "cache")
    return hub, fake_hub(hub, tmp_path, newest)


def test_entry_matches_dataset_limit_and_recipe():
    assert entry_id("bright_aops", 2000, "worked_examples") == "bright_aops-2000-worked_examples"
    assert entry_id("bright_aops", None, "reference") == "bright_aops-all-reference"


def test_manifest_and_files_are_read_at_one_commit(tmp_path):
    ident = entry_id("bright_aops", 2, "worked_examples")
    entry = write_build(tmp_path / "c2", ident)
    hub, asked = served(tmp_path, {"c2": {"entries": [entry]}}, "c2")
    build, reason = hub.fetch("bright_aops", 2, "worked_examples")
    assert reason == "" and build is not None
    assert {revision for _, revision in asked} == {"c2"}
    assert build["entry"]["id"] == ident and build["terms"] == TERMS
    # Published ids are the dataset's own; the collection's prefix is put back on.
    assert set(build["outputs"]) == {"ba-d1", "ba-d2"}
    assert build["outputs"]["ba-d1"] == {
        "tags": ["relative speed"],
        "principle": "P",
        "method": "M",
        "answer": "4",
    }


def test_a_file_whose_sha_differs_is_not_used(tmp_path):
    ident = entry_id("bright_aops", 2, "worked_examples")
    entry = write_build(tmp_path / "c1", ident)
    entry["files"][f"builds/{ident}/outputs.parquet"] = "0" * 64
    hub, _ = served(tmp_path, {"c1": {"entries": [entry]}}, "c1")
    build, reason = hub.fetch("bright_aops", 2, "worked_examples")
    assert build is None and "sha256" in reason


def test_an_unreachable_hub_gives_none_and_a_reason(tmp_path):
    hub, _ = served(tmp_path, {}, None)
    build, reason = hub.fetch("bright_aops", 2, "worked_examples")
    assert build is None and "unreachable" in reason and "no route to host" in reason
    # A manifest without the entry is a reason too, not an error.
    hub, _ = served(tmp_path / "other", {"c1": {"entries": []}}, "c1")
    build, reason = hub.fetch("bright_aops", 2, "worked_examples")
    assert build is None and "no entry bright_aops-2-worked_examples" in reason


def test_merge_replaces_one_entry_and_keeps_the_rest():
    old = {"entries": [{"id": "a", "items": 1}, {"id": "b", "items": 2}]}
    merged = merge_manifest(old, {"id": "a", "items": 9})
    assert merged == {"entries": [{"id": "a", "items": 9}, {"id": "b", "items": 2}]}
    assert old["entries"][0]["items"] == 1  # the Hub's copy is not changed in place
    assert merge_manifest(None, {"id": "c"}) == {"entries": [{"id": "c"}]}


def test_card_lists_entries_columns_and_attribution():
    entry = {
        "id": "bright_aops-2000-worked_examples",
        "dataset": "bright_aops",
        "limit": 2000,
        "recipe": "worked_examples",
        "authored_by": "Claude Opus 5.5, in-session; vocabulary reviewed",
        "items": 2000,
        "files": {"builds/bright_aops-2000-worked_examples/outputs.parquet": "x"},
    }
    card = dataset_card({"entries": [entry]})
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert "data_files: builds/bright_aops-2000-worked_examples/outputs.parquet" in card
    assert "Claude Opus 5.5, in-session" in card
    for column in ("position", "term", "definition", "doc_id", "tags", "principle", "method"):
        assert f"`{column}`" in card
    assert "pd.read_parquet" in card
    assert "Su et al." in card and "xlangai/BRIGHT" in card and "CC BY 4.0" in card
    assert "AQuA-RAT" in card and "TheoremQA" in card


# -- a load applies a matching entry -------------------------------------------------

LOAD_ITEMS = [
    Item("ba-d1", "two trains approach each other", "4", "add the two speeds"),
    Item("ba-d2", "convert kilometres to metres", "2000", None),  # no solution to abstract
]


def hub_load(tmp_path, monkeypatch, newest: str | None = "c1", stored=()):
    """A worked-examples collection, a load job of bright_aops limit 2, and a loader whose
    Hub serves one entry for it at commit c1."""
    conn = connect(tmp_path / "r.db")
    reply = '{"terms": ["relative speed"], "principle": "P", "method": "M", "answer": "4"}'
    ai = fake_ai(conn, lambda p: reply)
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("aops", "science", False, "worked_examples")
    if stored:
        ix.add(spec, list(stored), "hand")
    ident = entry_id("bright_aops", 2, "worked_examples")
    rows = [{**ROWS[0], "method": "M"}, ROWS[1]]
    entry = write_build(tmp_path / "hub" / "c1", ident, rows)
    served(tmp_path / "hub", {"c1": {"entries": [entry]}}, "c1")
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (LOAD_ITEMS, []))
    builder = Builder(conn, ix.store, ix.models, ai, ix)
    ld = Loader(conn, ix, builder, ai, tmp_path, hub_repo="owner/builds")
    fake_hub(ld.hub, tmp_path / "hub", newest)
    job = get_job(conn, create_job(conn, "load", "aops", {"dataset": "bright_aops", "limit": 2}))
    return conn, ix, ld, job


def test_a_load_with_an_entry_sends_no_ai_call(tmp_path, monkeypatch):
    conn, ix, ld, job = hub_load(tmp_path, monkeypatch)
    ld.ai.update_connection("anthropic", api_key=None)  # no model for keywords at all
    estimate = ld.estimate_load(ix.spec("aops"), "bright_aops", 2)
    assert estimate["est_usd"] == 0 and estimate["calls"] == 0
    assert estimate["hub"] == "bright_aops-2-worked_examples"
    result = ld.run(job, lambda p: None)
    assert result["hub"] == {"entry": "bright_aops-2-worked_examples", "items": 2}
    assert result["tagged"] == 2 and result["abstracted"] == 1
    assert conn.execute("select count(*) from ai_ledger")[0][0] == 0
    by_id = {h.id: h for h in ix.store.get("aops", ["ba-d1", "ba-d2"])}
    assert by_id["ba-d1"].tags == ["relative speed"] and by_id["ba-d2"].tags == ["unit conversion"]
    assert by_id["ba-d1"].abstracted == "Principle: P\nMethod: M\nAnswer: 4"
    assert db.get_vocabulary(conn, "aops") == TERMS


def test_a_load_builds_locally_when_the_hub_is_down(tmp_path, monkeypatch):
    conn, ix, ld, job = hub_load(tmp_path, monkeypatch, newest=None)
    result = ld.run(job, lambda p: None)
    assert "unreachable" in result["hub"]["skipped"]
    assert result["tagged"] == 2  # built by the keywords model instead
    assert conn.execute("select count(*) from ai_ledger")[0][0] > 0
    [line] = [e["message"] for e in get_job(conn, job["id"])["log"] if "Hub" in e["message"]]
    assert line.startswith("Hub: unreachable") and line.endswith("; building locally")


def test_an_entry_that_does_not_cover_every_item_is_not_applied(tmp_path, monkeypatch):
    # All or nothing: one author's vocabulary for every item, or a local build.
    other = Item("ba-d3", "a third problem", "1", "count them")
    conn, ix, ld, job = hub_load(tmp_path, monkeypatch, stored=[other])
    result = ld.run(job, lambda p: None)
    assert result["hub"] == {"skipped": "the entry covers 2 of 3 items"}
    assert conn.execute("select count(*) from ai_ledger")[0][0] > 0
