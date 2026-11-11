import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from ragqa_core import db, loader
from ragqa_core.builder import Builder
from ragqa_core.db import connect, create_job, get_job
from ragqa_core.fakes import fake_ai, fake_hub, fake_models
from ragqa_core.hub import MANIFEST, Hub, entry_id, sha256, write_entry
from ragqa_core.indexer import Indexer
from ragqa_core.loader import Loader
from ragqa_core.recipes import RECIPES, build_for
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
BUILD = asdict(build_for("worked_examples"))


def published(root: Path, ident: str, rows=ROWS, build=BUILD) -> dict:
    """One entry's files under root/builds/<ident>/, written as publish writes them, and
    the manifest entry naming them."""
    files = write_entry(root, ident, TERMS, {}, rows)
    return {
        "id": ident,
        "dataset": "bright_aops",
        "limit": 2,
        "recipe": "worked_examples",
        "build": build,
        "files": {path: sha256(local) for path, local in files.items()},
    }


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
    entry = published(tmp_path / "c2", ident)
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
    entry = published(tmp_path / "c1", ident)
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


# -- a load applies a matching entry -------------------------------------------------

LOAD_ITEMS = [
    Item("ba-d1", "two trains approach each other", "4", "add the two speeds"),
    Item("ba-d2", "convert kilometres to metres", "2000", None),  # no solution to abstract
]


def hub_load(
    tmp_path,
    monkeypatch,
    newest: str | None = "c1",
    stored=(),
    rows=ROWS,
    build=BUILD,
    held=False,
    dense_model: str | None = None,
):
    """A worked-examples collection, a load job of bright_aops limit 2, and a loader whose
    Hub serves one entry for it at commit c1; with what the Hub was asked for. `held`: the
    job was queued unconfirmed because the entry made it free. `dense_model`: the store's
    model (fake/small has 64 dimensions), else the service's."""
    conn = connect(tmp_path / "r.db")
    reply = '{"terms": ["relative speed"], "principle": "P", "method": "M", "answer": "4"}'
    ai = fake_ai(conn, lambda p: reply)
    ix = Indexer(conn, Store(None, None), fake_models(dims={"fake/small": 64}), ai)
    spec = ix.create("aops", "science", False, "worked_examples", dense_model=dense_model)
    if stored:
        ix.add(spec, list(stored), "hand")
    ident = entry_id("bright_aops", 2, "worked_examples")
    entry = published(tmp_path / "hub" / "c1", ident, rows, build)
    served(tmp_path / "hub", {"c1": {"entries": [entry]}}, "c1")
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (LOAD_ITEMS, []))
    builder = Builder(conn, ix.store, ix.models, ai, ix)
    ld = Loader(conn, ix, builder, ai, tmp_path, hub_repo="owner/builds")
    asked = fake_hub(ld.hub, tmp_path / "hub", newest)
    setup = {"dataset": "bright_aops", "limit": 2, **({"hub_entry": ident} if held else {})}
    job = get_job(conn, create_job(conn, "load", "aops", setup))
    return conn, ix, ld, job, asked


def test_a_load_with_an_entry_sends_no_ai_call(tmp_path, monkeypatch):
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch)
    ld.ai.update_connection("anthropic", api_key=None)  # no model for keywords at all
    estimate = ld.estimate_load(db.get_collection(ix.conn, "aops"), "bright_aops", 2)
    assert estimate["est_usd"] == 0 and estimate["calls"] == 0
    applied = {"entry": "bright_aops-2-worked_examples", "items": 2}
    assert estimate["hub"] == applied  # the estimate says what the result will
    result = ld.run(job, lambda p: None)
    assert result["hub"] == applied
    assert result["tagged"] == 2 and result["abstracted"] == 1
    assert conn.execute("select count(*) from ai_ledger")[0][0] == 0
    by_id = {h.id: h for h in ix.store.get("aops", ["ba-d1", "ba-d2"])}
    assert by_id["ba-d1"].tags == ["relative speed"] and by_id["ba-d2"].tags == ["unit conversion"]
    assert by_id["ba-d1"].abstracted == "Principle: P\nMethod: M\nAnswer: 4"
    assert db.get_vocabulary(conn, "aops") == TERMS


def test_a_hub_build_applies_to_a_store_of_any_embedding_model(tmp_path, monkeypatch):
    # The Hub publishes text only, so its build applies whatever the store embeds with.
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch, dense_model="fake/small")
    ld.ai.update_connection("anthropic", api_key=None)
    result = ld.run(job, lambda p: None)
    assert result["hub"] == {"entry": "bright_aops-2-worked_examples", "items": 2}
    assert conn.execute("select count(*) from ai_ledger")[0][0] == 0
    assert db.get_vocabulary(conn, "aops") == TERMS
    assert db.get_collection(ix.conn, "aops").dense_model == "fake/small"
    assert ix.store.vectors("aops", "kw_tags", 3).shape[1] == 64


def test_a_reload_with_nothing_to_build_never_reads_the_hub(tmp_path, monkeypatch):
    conn, ix, ld, job, asked = hub_load(tmp_path, monkeypatch)
    ld.run(job, lambda p: None)
    asked.clear()
    again = get_job(conn, create_job(conn, "load", "aops", {"dataset": "bright_aops", "limit": 2}))
    assert "hub" not in ld.run(again, lambda p: None) and asked == []
    assert ld.estimate_load(db.get_collection(ix.conn, "aops"), "bright_aops", 2)["hub"] == {}


def test_a_load_builds_locally_when_the_hub_is_down(tmp_path, monkeypatch):
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch, newest=None)
    result = ld.run(job, lambda p: None)
    assert "unreachable" in result["hub"]["skipped"]
    assert result["tagged"] == 2  # built by the keywords model instead
    assert conn.execute("select count(*) from ai_ledger")[0][0] > 0
    [line] = [e["message"] for e in get_job(conn, job["id"])["log"] if "Hub" in e["message"]]
    assert line.startswith("Hub: unreachable") and line.endswith("; building locally")


def test_an_entry_that_does_not_cover_every_item_is_not_applied(tmp_path, monkeypatch):
    # All or nothing: one author's vocabulary for every item, or a local build.
    other = Item("ba-d3", "a third problem", "1", "count them")
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch, stored=[other])
    result = ld.run(job, lambda p: None)
    assert result["hub"] == {"skipped": "the entry is unusable: no output for 1 items: ['ba-d3']"}
    assert conn.execute("select count(*) from ai_ledger")[0][0] > 0


def test_an_entry_built_with_other_settings_is_not_applied(tmp_path, monkeypatch):
    other = asdict(replace(build_for("worked_examples"), tag_k=1))
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch, build=other)
    result = ld.run(job, lambda p: None)
    assert result["hub"] == {"skipped": "the entry was built with other build settings"}


def test_an_entry_never_replaces_a_vocabulary_the_collection_has(tmp_path, monkeypatch):
    # A vocabulary built, authored or edited here stays: the new items are built under it.
    conn, ix, ld, job, asked = hub_load(tmp_path, monkeypatch)
    own = ld.builder.save_vocabulary(db.get_collection(ix.conn, "aops"), ["my own term"])
    db.set_vocabulary_version(conn, "aops", own)
    estimate = ld.estimate_load(db.get_collection(ix.conn, "aops"), "bright_aops", 2)
    assert estimate["hub"] == {"skipped": "the collection has a vocabulary of its own"}
    result = ld.run(job, lambda p: None)
    assert result["hub"] == estimate["hub"] and asked == []
    assert db.get_vocabulary(conn, "aops") == ["my own term"]


def test_an_apply_that_stops_part_way_leaves_no_vocabulary_switched_to(tmp_path, monkeypatch):
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch)

    def stop(*args, **kwargs):
        raise RuntimeError("stopped while writing tags")

    ld.builder._write_tags = stop
    with pytest.raises(RuntimeError):
        ld.run(job, lambda p: None)
    assert not db.get_vocabulary(conn, "aops")  # a later load reads the Hub again


def test_an_entry_with_an_unusable_row_is_skipped_before_anything_is_added(tmp_path, monkeypatch):
    # d1 has a solution, so it needs an abstracted form; finding that only after the add
    # would fail the load instead of building locally.
    rows = [{**ROWS[0], "principle": None}, ROWS[1]]
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch, rows=rows)
    result = ld.run(job, lambda p: None)
    assert result["hub"]["skipped"].startswith("the entry is unusable: item ba-d1")
    assert result["tagged"] == 2  # built locally instead


def test_a_load_free_because_of_an_entry_never_builds_locally(tmp_path, monkeypatch):
    # It was queued without confirm because the entry made it free: a local build would
    # spend without anyone having agreed to it.
    conn, ix, ld, job, _ = hub_load(tmp_path, monkeypatch, newest=None, held=True)
    with pytest.raises(ValueError, match="load again to see what it would spend"):
        ld.run(job, lambda p: None)
    assert ix.store.ids("aops") == []  # nothing was added
    assert conn.execute("select count(*) from ai_ledger")[0][0] == 0


# What build_for gives each recipe, with and without a summary. A Hub entry"s build must
# equal it to be applied, so a change here leaves every published entry unused.
BUILD_SETTINGS = {
    ("worked_examples", False): {
        "units": "whole",
        "added_text": "none",
        "vocabulary": "induced",
        "describes": "solution",
        "tagging": "ai",
        "abstracted": True,
        "classifier": True,
        "profile": True,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("worked_examples", True): {
        "units": "whole",
        "added_text": "none",
        "vocabulary": "induced",
        "describes": "solution",
        "tagging": "ai",
        "abstracted": True,
        "classifier": True,
        "profile": True,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("reference", False): {
        "units": "chunks",
        "added_text": "title_prefix",
        "vocabulary": "induced",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": True,
        "profile": True,
        "chunk_words": 100,
        "chunk_overlap": 10,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("reference", True): {
        "units": "chunks",
        "added_text": "title_prefix",
        "vocabulary": "induced",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": True,
        "profile": True,
        "chunk_words": 100,
        "chunk_overlap": 10,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("paper_2024", False): {
        "units": "chunks",
        "added_text": "none",
        "vocabulary": "lists",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": False,
        "profile": True,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("paper_2024", True): {
        "units": "chunks",
        "added_text": "summary",
        "vocabulary": "lists",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": False,
        "profile": True,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("documents", False): {
        "units": "whole",
        "added_text": "none",
        "vocabulary": "induced",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": False,
        "profile": False,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("documents", True): {
        "units": "whole",
        "added_text": "none",
        "vocabulary": "induced",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": False,
        "profile": False,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("documents_tagged", False): {
        "units": "whole",
        "added_text": "none",
        "vocabulary": "induced",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": False,
        "profile": False,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
    ("documents_tagged", True): {
        "units": "whole",
        "added_text": "none",
        "vocabulary": "induced",
        "describes": "text",
        "tagging": "similarity",
        "abstracted": False,
        "classifier": False,
        "profile": False,
        "chunk_words": 100,
        "chunk_overlap": 25,
        "tag_k": 3,
        "tag_cutoff": 0.5,
    },
}


def test_hub_build_settings_are_pinned():
    assert {
        (r, s): asdict(build_for(r, s)) for r in RECIPES for s in (False, True)
    } == BUILD_SETTINGS
