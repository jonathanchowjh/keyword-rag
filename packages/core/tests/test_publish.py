import json

import httpx
import pyarrow.parquet as pq
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from ragqa_core import db
from ragqa_core.builder import Builder, abstracted_text
from ragqa_core.db import connect, create_job, update_job
from ragqa_core.fakes import fake_ai, fake_hub, fake_models
from ragqa_core.hub import MANIFEST, Hub, sha256
from ragqa_core.indexer import Indexer
from ragqa_core.publish import Refused, abstracted_parts, publish, read_definitions
from ragqa_core.store import Store
from ragqa_core.types import Item

TERMS = ["relative speed", "unit conversion"]
SECRET = "sk-ant-not-a-real-key-0123456789"


class Api:
    """Records what would be sent to the Hub; whoami names the token's owner."""

    def __init__(self, owner: str = "owner"):
        self.owner = owner
        self.calls: list[tuple[str, dict]] = []

    def whoami(self) -> dict:
        return {"name": self.owner}

    def create_repo(self, repo_id: str, **kw) -> None:
        self.calls.append(("create_repo", {"repo_id": repo_id, **kw}))

    def create_commit(self, repo_id: str, **kw) -> None:
        self.calls.append(("create_commit", {"repo_id": repo_id, **kw}))


def built(tmp_path):
    """Collection aops, built from authored outputs, after a finished bright_aops load."""
    conn = connect(tmp_path / "r.db")
    ai = fake_ai(conn, lambda p: "x", api_key=SECRET)
    ix = Indexer(conn, Store(None, None), fake_models(), ai)
    spec = ix.create("aops", "science", False, "worked_examples")
    ix.add(
        spec,
        [
            Item("ba-d1", "two trains approach", "4", "add the speeds"),
            Item("ba-d2", "kilometres to metres", "2000", None),
        ],
        "bright_aops",
    )
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


def run(tmp_path, hub, builder, api=None, **kw):
    return publish(
        builder,
        hub,
        builder.conn,
        "aops",
        authored_by="Claude Opus 5.5, in-session",
        out=tmp_path / "out",
        home=tmp_path,
        api=api,
        **kw,
    )


def no_repo(hub: Hub) -> None:
    def missing() -> str:
        request = httpx.Request("HEAD", "https://huggingface.co/datasets/owner/builds")
        response = httpx.Response(404, request=request)
        raise RepositoryNotFoundError("404 owner/builds", response=response)

    hub._commit = missing  # type: ignore[method-assign]


def test_dry_run_uploads_nothing(tmp_path):
    conn, builder, hub = built(tmp_path)
    no_repo(hub)
    done = run(tmp_path, hub, builder)
    entry = done["entry"]
    assert entry["id"] == "bright_aops-2-worked_examples" and entry["items"] == 2
    assert entry["limit"] == 2 and entry["source"]["repo"] == "xlangai/BRIGHT"
    assert entry["authored_by"] == "Claude Opus 5.5, in-session"
    assert entry["vocabulary_version"] == db.get_collection(conn, "aops").vocabulary_version
    assert done["uploaded"] is False and done["parent"] is None
    for path, digest in entry["files"].items():
        assert sha256(done["files"][path]) == digest
    assert set(done["files"]) == {*entry["files"], MANIFEST, "README.md"}
    manifest = json.loads(done["files"][MANIFEST].read_text())
    assert manifest == {"entries": [entry]}


def test_publish_merges_the_hubs_manifest_in_one_commit(tmp_path):
    conn, builder, hub = built(tmp_path)
    other = {"id": "bright_theoremqa-all-worked_examples", "files": {}}
    (tmp_path / "hub" / "c7").mkdir(parents=True)
    (tmp_path / "hub" / "c7" / MANIFEST).write_text(json.dumps({"entries": [other]}))
    fake_hub(hub, tmp_path / "hub", "c7")
    api = Api()
    done = run(tmp_path, hub, builder, api)
    assert done["uploaded"] is True
    [create, commit] = api.calls
    made = {"repo_id": "owner/builds", "repo_type": "dataset", "exist_ok": True}
    assert create == ("create_repo", made)
    assert commit[1]["parent_commit"] == "c7"  # refused by the Hub if main moved meanwhile
    sent = {op.path_in_repo for op in commit[1]["operations"]}
    assert sent == set(done["files"])
    ids = [e["id"] for e in json.loads(done["files"][MANIFEST].read_text())["entries"]]
    assert ids == ["bright_theoremqa-all-worked_examples", "bright_aops-2-worked_examples"]


def test_refuses_when_the_token_owner_is_not_the_repo_owner(tmp_path):
    _, builder, hub = built(tmp_path)
    no_repo(hub)
    api = Api(owner="someone-else")
    with pytest.raises(Refused, match="someone-else.*owner/builds"):
        run(tmp_path, hub, builder, api)
    assert api.calls == []


def test_refuses_when_the_hubs_manifest_cannot_be_read(tmp_path):
    # Only a repo or manifest that does not exist starts a new manifest: a Hub that is
    # down must not have its other entries replaced.
    _, builder, hub = built(tmp_path)
    fake_hub(hub, tmp_path / "hub", None)
    api = Api()
    with pytest.raises(Refused, match="could not be read"):
        run(tmp_path, hub, builder, api)
    assert api.calls == []


def test_published_files_hold_no_keys_ledger_or_embeddings(tmp_path):
    conn, builder, hub = built(tmp_path)
    no_repo(hub)
    done = run(tmp_path, hub, builder)
    for path in done["files"].values():
        assert SECRET.encode() not in path.read_bytes()
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
    with pytest.raises(Refused, match="1 items have no tags"):
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
    assert read_definitions(tmp_path / "v.json") == {"pythagorean theorem": "a² + b² = c²"}
