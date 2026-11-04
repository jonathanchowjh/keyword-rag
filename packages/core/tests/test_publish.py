import json
import shutil

import httpx
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from ragqa_core import db, loader
from ragqa_core.builder import Builder, abstracted_parts, abstracted_text
from ragqa_core.db import connect, create_job, get_job, update_job
from ragqa_core.fakes import fake_ai, fake_hub, fake_models
from ragqa_core.hub import MANIFEST, Hub, sha256
from ragqa_core.indexer import Indexer
from ragqa_core.loader import Loader
from ragqa_core.publish import Refused, dataset_card, merge_manifest, publish
from ragqa_core.store import Store
from ragqa_core.types import Item
from ragqa_core.vocabulary import read_vocabulary

TERMS = ["relative speed", "unit conversion"]
SECRET = "sk-ant-not-a-real-key-0123456789"
MARK = "marker-that-must-stay-home"
ITEMS = [
    Item("ba-d1", "two trains approach", "4", "add the speeds"),
    Item("ba-d2", "kilometres to metres", "2000", None, meta={"note": MARK}),
]


def built(tmp_path):
    """Collection aops, built from authored outputs, after a finished bright_aops load."""
    conn = connect(tmp_path / "r.db")
    ai = fake_ai(conn, lambda p: "x", api_key=SECRET)
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("aops", "science", False, "worked_examples")
    ix.add(spec, ITEMS, "bright_aops")
    builder = Builder(conn, ix.store, ix.models, ai, ix)
    outputs = {
        "ba-d1": {"tags": ["relative speed"], "principle": "P", "method": "M", "answer": "4"},
        "ba-d2": {"tags": ["unit conversion"]},
    }
    builder.apply_outputs(ix.spec("aops"), TERMS, outputs)
    job = create_job(conn, "load", "aops", {"dataset": "bright_aops", "limit": 2})
    update_job(conn, job, status="done")
    hub = Hub("owner/builds", tmp_path / "cache")
    return conn, builder, hub


def sending(hub: Hub, owner: str = "owner") -> list[tuple]:
    """The Hub's token belongs to `owner`; every send is recorded instead of made."""
    sent: list[tuple] = []
    hub.owner = lambda: owner  # type: ignore[method-assign]
    hub.send = lambda files, message, parent: sent.append((files, message, parent))  # type: ignore[method-assign]
    return sent


def run(tmp_path, hub, builder, **kw):
    return publish(
        builder,
        hub,
        "aops",
        authored_by="Claude Opus 5.5, in-session",
        out=tmp_path / "out",
        home=tmp_path,
        **kw,
    )


def no_repo(hub: Hub) -> None:
    def missing() -> str:
        request = httpx.Request("HEAD", "https://huggingface.co/datasets/owner/builds")
        response = httpx.Response(404, request=request)
        raise RepositoryNotFoundError("404 owner/builds", response=response)

    hub._commit = missing  # type: ignore[method-assign]


def test_dry_run_sends_nothing(tmp_path):
    conn, builder, hub = built(tmp_path)
    no_repo(hub)
    sent = sending(hub)
    done = run(tmp_path, hub, builder)
    entry = done["entry"]
    assert entry["id"] == "bright_aops-2-worked_examples" and entry["items"] == 2
    assert entry["limit"] == 2 and entry["source"]["repo"] == "xlangai/BRIGHT"
    assert entry["authored_by"] == "Claude Opus 5.5, in-session"
    assert entry["build"]["tagging"] == "ai" and entry["build"]["abstracted"] is True
    assert entry["vocabulary_version"] == db.get_collection(conn, "aops").vocabulary_version
    assert done["sent"] is False and done["parent"] is None and sent == []
    for path, digest in entry["files"].items():
        assert sha256(done["files"][path]) == digest
    assert set(done["files"]) == {*entry["files"], MANIFEST, "README.md"}
    manifest = json.loads(done["files"][MANIFEST].read_text())
    assert manifest == {"entries": [entry]}


def test_a_published_build_is_what_a_load_applies(tmp_path, monkeypatch):
    # Round trip: publish, serve the files as the Hub would, load the same dataset into a
    # new collection with no AI call, and get the same tags and abstracted forms back.
    conn, builder, hub = built(tmp_path)
    no_repo(hub)
    done = run(tmp_path, hub, builder)
    for path, local in done["files"].items():
        (tmp_path / "hub" / "c1" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, tmp_path / "hub" / "c1" / path)
    other = connect(tmp_path / "other.db")
    ai = fake_ai(other, lambda p: "x")
    ix = Indexer(other, Store(None, None), fake_models(), ai)
    ix.create("aops", "science", False, "worked_examples")
    builder_there = Builder(other, ix.store, ix.models, ai, ix)
    ld = Loader(other, ix, builder_there, ai, tmp_path, hub_repo="owner/builds")
    fake_hub(ld.hub, tmp_path / "hub", "c1")
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (ITEMS, []))
    job = get_job(other, create_job(other, "load", "aops", {"dataset": "bright_aops", "limit": 2}))
    assert ld.run(job, lambda p: None)["hub"] == {"entry": done["entry"]["id"], "items": 2}
    assert other.execute("select count(*) from ai_ledger")[0][0] == 0
    ids = ["ba-d1", "ba-d2"]
    here = {h.id: (h.tags, h.abstracted) for h in builder.store.get("aops", ids)}
    there = {h.id: (h.tags, h.abstracted) for h in ix.store.get("aops", ids)}
    assert there == here and db.get_vocabulary(other, "aops") == TERMS


def test_publish_merges_the_hubs_manifest_in_one_commit(tmp_path):
    conn, builder, hub = built(tmp_path)
    other = {"id": "bright_theoremqa-all-worked_examples", "files": {}}
    (tmp_path / "hub" / "c7").mkdir(parents=True)
    (tmp_path / "hub" / "c7" / MANIFEST).write_text(json.dumps({"entries": [other]}))
    fake_hub(hub, tmp_path / "hub", "c7")
    sent = sending(hub)
    done = run(tmp_path, hub, builder, send=True)
    assert done["sent"] is True
    [(files, message, parent)] = sent
    assert parent == "c7"  # refused by the Hub if main moved meanwhile
    assert files == done["files"] and message == "Publish bright_aops-2-worked_examples"
    ids = [e["id"] for e in json.loads(done["files"][MANIFEST].read_text())["entries"]]
    assert ids == ["bright_theoremqa-all-worked_examples", "bright_aops-2-worked_examples"]


def test_refuses_when_the_token_owner_is_not_the_repo_owner(tmp_path):
    _, builder, hub = built(tmp_path)
    no_repo(hub)
    sent = sending(hub, owner="someone-else")
    with pytest.raises(Refused, match="someone-else.*owner/builds"):
        run(tmp_path, hub, builder, send=True)
    assert sent == []


def test_publish_without_a_usable_token_is_refused_before_anything_is_sent(tmp_path):
    _, builder, hub = built(tmp_path)
    sent = sending(hub)

    def owner():
        raise OSError("Invalid user token")

    hub.owner = owner  # type: ignore[method-assign]
    with pytest.raises(Refused, match="set HF_TOKEN"):
        run(tmp_path, hub, builder, send=True)
    assert sent == []


def test_refuses_when_the_hubs_manifest_cannot_be_read(tmp_path):
    # Only a repo or manifest that does not exist starts a new manifest: a Hub that is
    # down must not have its other entries replaced.
    _, builder, hub = built(tmp_path)
    fake_hub(hub, tmp_path / "hub", None)
    sent = sending(hub)
    with pytest.raises(Refused, match="could not be read"):
        run(tmp_path, hub, builder, send=True)
    assert sent == []


def test_refuses_while_a_job_works_on_the_collection(tmp_path):
    conn, builder, hub = built(tmp_path)
    create_job(conn, "retag", "aops", {})
    with pytest.raises(Refused, match="a retag job is queued or running"):
        run(tmp_path, hub, builder)


def published_text(files: dict) -> str:
    """Every published file as text, parquet decoded (its pages are compressed)."""
    return "\n".join(
        json.dumps(pq.read_table(path).to_pylist())
        if path.suffix == ".parquet"
        else path.read_text()
        for path in files.values()
    )


def test_published_files_hold_only_the_build(tmp_path):
    conn, builder, hub = built(tmp_path)
    builder.ai.sql("insert into ai_cache values (?, ?, 0, 0)", (MARK, MARK))
    builder.ai.sql(
        "insert into ai_ledger (at, model, input_tokens, output_tokens, usd, batch, purpose)"
        " values ('t', ?, 0, 0, 0, 0, ?)",
        (MARK, MARK),
    )
    no_repo(hub)
    done = run(tmp_path, hub, builder)
    text = published_text(done["files"])
    assert SECRET not in text and MARK not in text  # no key, item metadata, cache or ledger
    entry = done["entry"]
    [vocabulary] = [p for p in entry["files"] if p.endswith("vocabulary.parquet")]
    [outputs] = [p for p in entry["files"] if p.endswith("outputs.parquet")]
    assert pq.read_schema(done["files"][vocabulary]).names == ["position", "term", "definition"]
    table = pq.read_table(done["files"][outputs])
    assert table.schema.names == ["doc_id", "tags", "principle", "method", "answer"]
    rows = {r["doc_id"]: r for r in table.to_pylist()}
    assert set(rows) == {"d1", "d2"}  # the dataset's own ids, without the collection's prefix
    assert rows["d1"] == {
        "doc_id": "d1",
        "tags": ["relative speed"],
        "principle": "P",
        "method": "M",
        "answer": "4",
    }
    assert rows["d2"]["principle"] is None


def test_a_collection_with_untagged_items_is_refused(tmp_path):
    conn, builder, hub = built(tmp_path)
    builder.indexer.add(builder.indexer.spec("aops"), [Item("ba-d3", "new", "1", None)], "x")
    no_repo(hub)
    with pytest.raises(Refused, match="a load could not apply this build: item ba-d3 without tags"):
        run(tmp_path, hub, builder)


def test_a_worked_example_without_its_abstracted_form_is_refused(tmp_path):
    # A load would skip such an entry, so it is not published.
    conn, builder, hub = built(tmp_path)
    builder.store.set_payloads("aops", {"ba-d1": {"abstracted": None}})
    no_repo(hub)
    with pytest.raises(Refused, match="a load could not apply this build: item ba-d1"):
        run(tmp_path, hub, builder)


def test_abstracted_parts_read_back_the_stored_form():
    stored = abstracted_text({"principle": "P", "method": "1. a\n2. b", "answer": "x"})
    assert stored is not None
    assert abstracted_text(abstracted_parts(stored)) == stored
    assert abstracted_parts(None) == {"principle": None, "method": None, "answer": None}


def test_definitions_are_matched_by_the_term_as_stored(tmp_path):
    # The collection keeps cleaned, lower-case terms; an authored file need not.
    listed = [{"term": "Pythagorean  Theorem", "definition": "a² + b² = c²"}, {"term": "x"}]
    (tmp_path / "v.json").write_text(json.dumps({"terms": listed}))
    terms, definitions = read_vocabulary(tmp_path / "v.json")
    assert terms == ["Pythagorean  Theorem", "x"]
    assert definitions == {"pythagorean theorem": "a² + b² = c²"}


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
    card = dataset_card({"entries": [entry]}, "owner/builds")
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert "data_files: builds/bright_aops-2000-worked_examples/outputs.parquet" in card
    assert "Claude Opus 5.5, in-session" in card
    for column in ("position", "term", "definition", "doc_id", "tags", "principle", "method"):
        assert f"`{column}`" in card
    assert (
        "pd.read_parquet" in card and "hf://datasets/owner/builds/builds/bright_aops-2000" in card
    )
    assert "Su et al." in card and "xlangai/BRIGHT" in card and "CC BY 4.0" in card
    assert "AQuA-RAT" in card and "TheoremQA" in card
