"""The keyword-list routes: tags, proposed lists, versions and the vocabulary."""

from dataclasses import asdict

from api_helpers import (
    EVAL,
    LISTS,
    NO_KEYWORDS_MODEL,
    OTHER_LISTS,
    WORKED,
    add_collection,
    add_documents,
    add_tags,
    documents_with_passages,
    free_disabled_model,
    ledger_purposes,
    make_client,
    passages,
    reembed,
    small_model,
    stale,
    stored_worked,
    terms_reply,
    unloadable,
    wait,
)

from ragqa_api.guards import PRICE_CHANGED
from ragqa_core import db
from ragqa_core.fakes import ITEMS, fake_replies
from ragqa_core.types import Item


def test_propose_lists(client):
    add_collection(client, lists=False)
    proposal = '{"topic": ["enzymes"], "method": ["assay"], "common": ["rate"]}'
    fake_replies(client.app.state.ai, lambda p: proposal)
    url = "/collections/c/keyword-lists/propose"
    spent = client.get("/budget").json()["spent_usd"]
    r = client.post(url)
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1
    assert r.json()["estimate"]["est_usd"] > 0
    assert client.get("/budget").json()["spent_usd"] == spent  # the estimate spent nothing
    lists = client.post(url, json={"confirm": True}).json()
    assert lists == {"topic": ["enzymes"], "method": ["assay"], "common": ["rate"]}


def test_vocabulary_export_import_queues_retag(client):
    client.app.state.worker.stop()  # the retag job stays queued
    r = client.post("/collections", json=WORKED)
    assert r.status_code == 201, r.text
    empty = client.get("/collections/w/vocabulary").json()
    assert empty == {"version": None, "terms": None, "versions": []}
    terms = {"terms": [" Ohm's Law ", "ideal gas law"]}
    r = client.put("/collections/w/vocabulary", json=terms)
    assert r.status_code == 200 and r.json() == {
        "estimate": {"est_usd": 0.0, "items": 0, "calls": 0}
    }
    assert client.get("/collections/w/vocabulary").json() == empty  # nothing saved
    assert client.get("/jobs").json() == []  # nothing queued
    r = client.put("/collections/w/vocabulary", json={**terms, "confirm": True})
    assert r.status_code == 202, r.text
    setup = r.json()
    job = client.get(f"/jobs/{setup['job_id']}").json()
    assert (job["kind"], job["status"], job["setup"]) == (
        "retag",
        "queued",
        {"version": setup["version"]},
    )
    exported = client.get("/collections/w/vocabulary").json()
    assert exported["version"] is None  # in force only once the retag has run
    assert [(v["version"], v["count"], v["active"]) for v in exported["versions"]] == [
        (setup["version"], 2, False)
    ]
    db.set_vocabulary_version(client.app.state.conn, "w", setup["version"])
    exported = client.get("/collections/w/vocabulary").json()
    assert exported["version"] == setup["version"]
    assert exported["terms"] == ["ohm's law", "ideal gas law"]
    assert exported["versions"][0]["active"] is True
    # The export is the import: the GET body PUTs back as the same version.
    client.post(f"/jobs/{setup['job_id']}/cancel")
    again = client.put("/collections/w/vocabulary", json={**exported, "confirm": True})
    assert again.status_code == 202 and again.json()["version"] == setup["version"]


def test_vocabulary_put_refuses_a_misspelt_field(client):
    client.app.state.worker.stop()
    assert client.post("/collections", json=WORKED).status_code == 201
    exported = client.get("/collections/w/vocabulary").json()
    body = {**exported, "terms": ["ohm's law"], "confirmed": True}  # not confirm
    r = client.put("/collections/w/vocabulary", json=body)
    assert r.status_code == 422 and "confirmed" in r.text
    assert client.get("/jobs").json() == []


def test_vocabulary_estimate_prices_ai_tagging_without_spending(client):
    client.app.state.worker.stop()
    stored_worked(client)
    spent = client.get("/budget").json()["spent_usd"]
    r = client.put("/collections/w/vocabulary", json={"terms": ["enzymes", "markets"]})
    assert r.status_code == 200, r.text
    estimate = r.json()["estimate"]
    assert estimate["est_usd"] > 0 and estimate["items"] == 3 and estimate["calls"] == 3
    assert client.get("/budget").json()["spent_usd"] == spent
    assert client.get("/collections/w/vocabulary").json()["versions"] == []
    assert client.put("/collections/w/vocabulary", json={"terms": []}).status_code == 422


def test_vocabulary_refuses_a_changed_price(client):
    client.app.state.worker.stop()
    stored_worked(client)  # store w: AI tagging
    terms = ["enzymes", "markets"]
    seen = client.put("/collections/w/vocabulary", json={"terms": terms}).json()["estimate"]
    r = client.put("/collections/w/vocabulary", json={"terms": terms, **stale(seen)})
    assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
    assert db.list_jobs(client.app.state.conn, kind="retag") == []
    r = client.put("/collections/w/vocabulary", json={"terms": terms, "confirm": True})
    assert r.status_code == 202


def test_vocabulary_confirm_with_the_shown_price_is_accepted(client):
    client.app.state.worker.stop()
    stored_worked(client)
    terms = ["enzymes", "markets"]
    seen = client.put("/collections/w/vocabulary", json={"terms": terms}).json()["estimate"]
    shown = {"terms": terms, "confirm": True, "calls": seen["calls"], "est_usd": seen["est_usd"]}
    r = client.put("/collections/w/vocabulary", json=shown)
    assert r.status_code == 202, r.text
    job = client.get(f"/jobs/{r.json()['job_id']}").json()
    assert (job["kind"], job["status"]) == ("retag", "queued")
    versions = client.get("/collections/w/vocabulary").json()["versions"]
    assert [(v["version"], v["count"]) for v in versions] == [(r.json()["version"], 2)]


def test_vocabulary_put_with_ai_tagging_and_no_model_is_503(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        client.app.state.worker.stop()
        stored_worked(client)
        for body in ({"terms": ["enzymes"]}, {"terms": ["enzymes"], "confirm": True}):
            r = client.put("/collections/w/vocabulary", json=body)
            assert r.status_code == 503 and "'keywords'" in r.json()["detail"], body
        assert client.get("/collections/w/vocabulary").json()["versions"] == []
        assert client.get("/jobs").json() == []
        # Similarity tagging spends nothing, so it needs no model.
        ref = {"name": "r", "qa_type": "multihop", "recipe": "reference"}
        client.post("/collections", json=ref)
        r = client.put("/collections/r/vocabulary", json={"terms": ["x"], "confirm": True})
        assert r.status_code == 202, r.text


def test_cached_ai_retag_needs_no_model(client):
    stored_worked(client)  # store w: worked examples, AI tagging, ITEMS stored
    ai = client.app.state.ai
    free_disabled_model(client, "keywords")
    # The first retag runs on gw/llama: its prompts cache under it.
    ai.update_model("gw", "llama", enabled=True)
    terms = {"terms": ["enzymes", "markets"], "confirm": True}
    r = client.put("/collections/w/vocabulary", json=terms)
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"
    ai.update_model("gw", "llama", enabled=False)
    seen = client.put("/collections/w/vocabulary", json={"terms": terms["terms"]})
    assert seen.json()["estimate"]["calls"] == 0, seen.text
    r = client.put("/collections/w/vocabulary", json=terms)
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"


def test_empty_ai_retag_needs_no_model(client):
    client.post("/collections", json=WORKED)  # store w: AI tagging, nothing stored
    ai = client.app.state.ai
    ai.set_role("default", None)
    ai.set_role("keywords", None)
    r = client.put("/collections/w/vocabulary", json={"terms": ["enzymes"], "confirm": True})
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"  # nothing to tag: no model
    assert client.get("/collections/w/vocabulary").json()["version"] == r.json()["version"]


UNPRICED = "gw/nop has no price; enter one first"


def unpriced_keywords_model(client):
    ai = client.app.state.ai
    ai.add_connection("gw", "openai", "Gateway", base_url="https://gw.example/v1")
    ai.add_model("gw", "nop", price_in=None, price_out=None)
    ai.set_role("keywords", "gw/nop")


def test_vocabulary_with_an_unpriced_model_is_422(client):
    client.app.state.worker.stop()
    stored_worked(client)
    unpriced_keywords_model(client)
    for body in ({"terms": ["enzymes"]}, {"terms": ["enzymes"], "confirm": True}):
        r = client.put("/collections/w/vocabulary", json=body)
        assert r.status_code == 422 and r.json()["detail"] == UNPRICED, r.text


def test_vocabulary_retag_runs_and_switches(client):
    stored_worked(client)
    r = client.put("/collections/w/vocabulary", json={"terms": ["enzymes"], "confirm": True})
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done" and job["result"] == {"version": r.json()["version"]}
    assert client.get("/collections/w/vocabulary").json()["version"] == r.json()["version"]
    hits = client.app.state.store.get("w", [i["id"] for i in ITEMS])
    assert all(h.tags == ["enzymes"] for h in hits)


def test_vocabulary_put_on_lists_recipe_is_409(client):
    client.post("/collections", json={"name": "p", "qa_type": "mcq"})
    for body in ({"terms": ["ohm's law"]}, {"terms": ["ohm's law"], "confirm": True}):
        r = client.put("/collections/p/vocabulary", json=body)
        assert r.status_code == 409 and "lists" in r.json()["detail"]
    assert client.get("/collections/p/vocabulary").json()["versions"] == []


def test_vocabulary_put_refuses_while_busy_and_an_empty_list(client):
    client.app.state.worker.stop()
    client.post("/collections", json=WORKED)
    yes = {"terms": ["ohm's law"], "confirm": True}
    assert client.put("/collections/w/vocabulary", json={"terms": ["  "]}).status_code == 422
    assert (
        client.put("/collections/w/vocabulary", json={"terms": ["  "], "confirm": True}).status_code
        == 422
    )
    assert client.get("/collections/nope/vocabulary").status_code == 404
    assert client.put("/collections/nope/vocabulary", json={"terms": ["x"]}).status_code == 404
    for kind in ("load", "import", "retag", "eval"):
        job_id = db.create_job(client.app.state.conn, kind, "w", {})
        for body in ({"terms": ["ohm's law"]}, yes):
            r = client.put("/collections/w/vocabulary", json=body)
            assert r.status_code == 409 and kind in r.json()["detail"], kind
        db.request_cancel(client.app.state.conn, job_id)
    assert client.put("/collections/w/vocabulary", json=yes).status_code == 202


def test_propose_lists_needs_a_key(client):
    add_collection(client, lists=False)
    client.app.state.ai.update_connection("anthropic", api_key=None)
    assert client.post("/collections/c/keyword-lists/propose").status_code == 503


def test_lists_save_is_busy_only_while_an_eval_runs(client):
    add_collection(client)
    conn, before = client.app.state.conn, db.get_collection(client.app.state.conn, "c")
    client.app.state.worker.stop()
    j = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    r = client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS))
    assert r.status_code == 202, r.text  # a queued eval does not count
    version, rekey = r.json()["version"], r.json()["job_id"]
    job = db.get_job(conn, rekey)
    assert job is not None and job["kind"] == "rekey" and job["status"] == "queued"
    assert job["setup"] == {"version": version}
    after = db.get_collection(conn, "c")
    assert before and after and after.keyword_version == before.keyword_version != version
    db.update_job(conn, j, status="running")
    r = client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS))
    assert r.status_code == 409 and "an eval is running on 'c'" in r.json()["detail"]
    db.update_job(conn, j, status="cancelled")
    client.app.state.worker.start()
    assert wait(client, rekey)["status"] == "done"
    after = db.get_collection(conn, "c")
    assert after and after.keyword_version == version


def test_saving_lists_is_refused_when_the_stores_model_cannot_load(client):
    add_collection(client, lists=False)
    unloadable(client)
    r = client.put("/collections/c/keyword-lists", json=asdict(LISTS))
    assert r.status_code == 409 and "this service can't load" in r.json()["detail"]
    assert client.get("/collections/c/keyword-lists").json() == []


def test_keyword_list_versions_newest_first_with_counts_and_active(client):
    add_collection(client)
    first = db.get_collection(client.app.state.conn, "c")
    assert first
    put = client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS)).json()
    wait(client, put["job_id"])
    rows = client.get("/collections/c/keyword-lists").json()
    assert [r["version"] for r in rows] == [put["version"], first.keyword_version]
    assert [r["active"] for r in rows] == [True, False]
    assert rows[0]["counts"] == {"topic": 2, "method": 1, "common": 1}
    assert rows[1]["counts"] == {k: len(v) for k, v in asdict(LISTS).items()}
    assert rows[0]["created_at"]


def test_keyword_list_versions_of_a_collection_without_lists_is_empty(client):
    add_collection(client, lists=False)
    assert client.get("/collections/c/keyword-lists").json() == []


def test_one_keyword_list_version(client):
    add_collection(client)
    version = db.get_collection(client.app.state.conn, "c").keyword_version
    r = client.get(f"/collections/c/keyword-lists/{version}")
    assert r.status_code == 200
    assert r.json() == {
        "version": version,
        "created_at": r.json()["created_at"],
        "active": True,
        "lists": asdict(LISTS),
    }


def test_keyword_list_version_404s(client):
    add_collection(client)
    assert client.get("/collections/c/keyword-lists/nope").status_code == 404
    assert client.get("/collections/ghost/keyword-lists").status_code == 404
    assert client.get("/collections/ghost/keyword-lists/x").status_code == 404


def test_worked_documents_are_built_in_the_request(tmp_path):
    def reply(prompt: str) -> str:
        if "skills or principles" in prompt:
            return '{"terms": ["rates"]}'
        if '"principle"' in prompt:
            return '{"principle": "P.", "method": "M.", "answer": "A"}'
        return '{"terms": ["rates"]}'

    with make_client(tmp_path, reply=reply) as client:
        client.post("/collections", json=WORKED)
        r = client.post("/collections/w/documents", json={"items": ITEMS})
        estimate = r.json()["estimate"]
        assert r.status_code == 200 and estimate["est_usd"] > 0 and ledger_purposes(client) == []
        r = client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True})
        assert r.status_code == 200 and r.json()["added"] == 3
        hits = client.app.state.store.get("w", [i["id"] for i in ITEMS])
        assert all(h.tags == ["rates"] and h.abstracted for h in hits)
        assert ledger_purposes(client) == ["induce"] + ["tag"] * 3 + ["abstract"] * 3
        # Tagging again with the same terms is answered from the cache: priced at nothing.
        same = client.put("/collections/w/vocabulary", json={"terms": ["rates"]}).json()
        other = client.put("/collections/w/vocabulary", json={"terms": ["rates", "speed"]}).json()
        assert same["estimate"]["est_usd"] == 0 and other["estimate"]["est_usd"] > 0


def test_writes_are_refused_while_a_rebuild_is_queued(tmp_path, monkeypatch):
    client = make_client(tmp_path)  # not entered: the rebuild stays queued
    small_model(client, monkeypatch)
    add_collection(client, lists=False)
    app = client.app.state
    app.indexer.add(
        db.get_collection(app.indexer.conn, "c"), [Item("x-1", "optics light lens", "D")], "x"
    )
    r = reembed(client)
    assert r.status_code == 202, r.text
    job = db.get_job(app.conn, r.json()["job_id"])
    assert job["kind"] == "reembed" and job["status"] == "queued"
    assert job["setup"] == {"model": "fake/small", "from": "fake/dense"}
    assert client.get("/collections/c").json()["state"] == "queued"
    r = client.post("/collections/c/documents", json={"items": ITEMS[:1], "confirm": True})
    assert r.status_code == 409 and "reembed" in r.json()["detail"]
    assert client.delete("/collections/c/sources/x").status_code == 409
    body = {"name": "cc", "recipe": "paper_2024", "confirm": True}
    assert client.post("/collections/c/copy", json=body).status_code == 409
    assert reembed(client).status_code == 409  # one rebuild at a time
    assert client.post("/collections", json=WORKED).status_code == 201
    assert reembed(client, "w").status_code == 202
    r = client.put("/collections/w/vocabulary", json={"terms": ["ohm's law"], "confirm": True})
    assert r.status_code == 409 and "reembed" in r.json()["detail"]


NOT_UNTAGGED = "only a Documents store without keyword tags can add them"


def keyword_tags(client, name="c"):
    return client.get(f"/collections/{name}").json()["keyword_tags"]


def test_add_keyword_tags_estimates_then_builds(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        conn = client.app.state.conn
        client.put("/collections/c/search", json={"weights": {"keywords": 2}})
        saved = db.get_collection(conn, "c").search
        r = add_tags(client, False)
        assert r.status_code == 200, r.text
        estimate = r.json()["estimate"]
        assert set(estimate) == {"est_usd", "calls"}
        assert estimate["est_usd"] > 0 and estimate["calls"] >= 1
        assert ledger_purposes(client) == [] and db.get_collection(conn, "c").recipe == "documents"
        r = add_tags(client, True)
        assert r.status_code == 202, r.text
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        spec = db.get_collection(conn, "c")
        assert spec.recipe == "documents_tagged" and spec.vocabulary_version is not None
        assert spec.search == saved  # the switch keeps a saved search override
        assert len(passages(client)) == 3 and all(h.tags for h in passages(client))
        body = client.get("/collections/c").json()
        assert "keyword_rag" in body["methods_available"]
        terms = db.get_vocabulary(conn, "c")
        assert terms and body["keyword_tags"] == {"state": "built", "terms": len(terms)}
        query = {"collection": "c", "question": "alpha", "methods": ["keyword_rag"]}
        result = client.post("/query", json=query).json()["results"]["keyword_rag"]
        assert result["error"] is None and result["hits"]


def test_add_keyword_tags_without_ai_is_refused_and_changes_nothing(client):
    documents_with_passages(client)
    client.app.state.ai.set_role("default", None)  # no model for keywords at all
    for confirm in (False, True):
        r = add_tags(client, confirm)
        assert r.status_code == 503 and r.json()["detail"] == NO_KEYWORDS_MODEL, r.text
    spec = db.get_collection(client.app.state.conn, "c")
    assert spec.recipe == "documents" and spec.vocabulary_version is None
    assert db.list_jobs(client.app.state.conn, kind="tag_build") == []
    assert ledger_purposes(client) == [] and keyword_tags(client) == {"state": "none"}


def test_add_keyword_tags_is_refused_on_other_stores(client):
    add_collection(client, name="m", lists=False, questions=False)
    tagged = {"name": "t", "qa_type": "documents", "recipe": "documents_tagged"}
    assert client.post("/collections", json=tagged).status_code == 201
    for name in ("m", "t"):
        for confirm in (False, True):
            r = client.post(f"/collections/{name}/keyword-tags", json={"confirm": confirm})
            assert r.status_code == 409 and r.json()["detail"] == NOT_UNTAGGED, (name, r.text)
    add_documents(client, "e")
    r = client.post("/collections/e/keyword-tags", json={"confirm": True})
    assert r.status_code == 409 and r.json()["detail"] == "add passages first"
    assert client.post("/collections/nope/keyword-tags", json={}).status_code == 404
    r = client.post("/collections/e/keyword-tags", json={"confirm": True, "x": 1})
    assert r.status_code == 422  # extra fields are refused
    assert db.list_jobs(client.app.state.conn, kind="tag_build") == []


def test_a_tag_build_stopped_after_its_vocabulary_resumes_with_no_keywords_model(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        conn, builder = client.app.state.conn, client.app.state.builder

        def stop(spec, ids, tags, progress=lambda p: None):
            raise RuntimeError("stopped after the vocabulary")

        builder._write_tags = stop
        job_id = add_tags(client, True).json()["job_id"]
        assert wait(client, job_id)["status"] == "failed"
        del builder._write_tags
        assert db.get_vocabulary(conn, "c") and not any(h.tags for h in passages(client))
        client.app.state.ai.set_role("default", None)  # what is left is free: no model needed
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": False})
        assert r.status_code == 200 and r.json()["estimate"] == {"est_usd": 0, "calls": 0}, r.text
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": True})
        assert r.status_code == 202, r.text
        assert wait(client, job_id)["status"] == "done"
        assert all(h.tags for h in passages(client)) and keyword_tags(client)["state"] == "built"


def test_zero_priced_tag_build_that_calls_needs_its_model(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        client.app.state.worker.stop()
        free_disabled_model(client, "keywords")  # $0, off: today estimate == 0 skips need_ai
        r = add_tags(client, True)
        assert r.status_code == 503 and "'keywords'" in r.json()["detail"]
        assert db.list_jobs(client.app.state.conn, kind="tag_build") == []


def test_a_tag_build_with_no_usable_terms_goes_back_to_none(tmp_path):
    with make_client(tmp_path, reply=lambda p: '{"terms": []}') as client:
        documents_with_passages(client)
        job = wait(client, add_tags(client, True).json()["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["note"] == "no usable terms: nothing was tagged"
        spec = db.get_collection(client.app.state.conn, "c")
        assert spec.recipe == "documents" and spec.vocabulary_version is None
        assert keyword_tags(client) == {"state": "none"}
        assert add_tags(client, False).status_code == 200  # Add keyword tags can be tried again
        assert add_tags(client, True).status_code == 202


def test_tag_build_and_proposal_refuse_a_changed_price(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)  # store c, Documents
        seen = add_tags(client, False).json()["estimate"]
        r = client.post("/collections/c/keyword-tags", json=stale(seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        assert db.list_jobs(client.app.state.conn, kind="tag_build") == []
        add_collection(client, lists=False, name="p")  # no lists: a proposal prices
        seen = client.post("/collections/p/keyword-lists/propose", json={}).json()["estimate"]
        r = client.post("/collections/p/keyword-lists/propose", json=stale(seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        assert ledger_purposes(client) == []


def test_confirmed_proposal_with_an_unpriced_model_is_422(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client, lists=False, name="p")
        unpriced_keywords_model(client)
        shown = {"confirm": True, "calls": 1, "est_usd": 0.0}
        r = client.post("/collections/p/keyword-lists/propose", json=shown)
        assert r.status_code == 422 and r.json()["detail"] == UNPRICED


def test_every_keyword_route_answers_422_for_an_unpriced_model(tmp_path):
    with make_client(tmp_path) as client:
        documents_with_passages(client)  # store c, Documents
        add_collection(client, lists=False, name="p")  # no lists: a proposal prices
        unpriced_keywords_model(client)
        for body in ({}, {"confirm": True}):
            r = client.post("/collections/c/keyword-tags", json=body)
            assert r.status_code == 422 and r.json()["detail"] == UNPRICED, r.text
        for body in (None, {"confirm": True}):
            r = client.post("/collections/p/keyword-lists/propose", json=body)
            assert r.status_code == 422 and r.json()["detail"] == UNPRICED, r.text


def test_a_disabled_unpriced_model_is_still_503(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client, lists=False, name="p")
        unpriced_keywords_model(client)
        client.app.state.ai.update_model("gw", "nop", enabled=False)
        r = client.post("/collections/p/keyword-lists/propose")
        assert r.status_code == 503 and "'keywords'" in r.json()["detail"], r.text


def test_a_disabled_or_keyless_unpriced_model_is_503_on_every_keyword_route(tmp_path):
    for cut in ("disabled", "keyless"):
        with make_client(tmp_path / cut) as client:
            documents_with_passages(client)  # store c, Documents
            add_collection(client, lists=False, name="p")
            client.app.state.worker.stop()
            stored_worked(client)  # store w, which takes a vocabulary
            ai = client.app.state.ai
            if cut == "disabled":
                unpriced_keywords_model(client)
                ai.update_model("gw", "nop", enabled=False)
            else:
                ai.add_connection("ant", "anthropic", "Anthropic")  # needs a key; none set
                ai.add_model("ant", "nop", price_in=None, price_out=None)
                ai.set_role("keywords", "ant/nop")
            shown = {"confirm": True, "calls": 1, "est_usd": 0.0}
            calls = [
                ("post", "/collections/p/keyword-lists/propose", shown),
                ("put", "/collections/w/vocabulary", {"terms": ["a"]}),
                ("put", "/collections/w/vocabulary", {"terms": ["a"], "confirm": True}),
                ("post", "/collections/c/keyword-tags", {}),
                ("post", "/collections/c/keyword-tags", {"confirm": True}),
            ]
            for method, url, body in calls:
                r = getattr(client, method)(url, json=body)
                detail = r.json()["detail"]
                assert r.status_code == 503 and "keywords" in detail, (cut, url, r.text)


def test_put_vocabulary_is_refused_on_an_untagged_documents_store(client):
    documents_with_passages(client)
    message = "'c' has no keyword tags yet; use Add keyword tags on the store page"
    for confirm in (False, True):
        r = client.put("/collections/c/vocabulary", json={"terms": ["alpha"], "confirm": confirm})
        assert r.status_code == 409 and r.json()["detail"] == message, r.text
    assert db.list_vocabularies(client.app.state.conn, "c") == []
    assert db.list_jobs(client.app.state.conn, kind="retag") == []


def test_keyword_tags_state_is_reported(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        add_collection(client, name="m", lists=False, questions=False)
        assert keyword_tags(client, "m") is None  # not a Documents store
        documents_with_passages(client)
        conn, worker = client.app.state.conn, client.app.state.worker
        assert keyword_tags(client) == {"state": "none"}
        worker.stop()
        job_id = add_tags(client, True).json()["job_id"]
        assert keyword_tags(client) == {"state": "building", "job_id": job_id, "progress": 0}
        r = add_tags(client, False)
        assert r.status_code == 409 and "a tag_build job is queued" in r.json()["detail"]
        db.update_job(conn, job_id, status="running", progress=0.25)
        assert keyword_tags(client) == {"state": "building", "job_id": job_id, "progress": 0.25}
        assert client.get("/collections/c").json()["state"] == "tagging"
        db.update_job(conn, job_id, status="cancelled")
        assert keyword_tags(client) == {"state": "stopped", "job_id": job_id}
        worker.start()
        assert client.post(f"/jobs/{job_id}/resume", json={"confirm": True}).status_code == 202
        assert wait(client, job_id)["status"] == "done"
        terms = db.get_vocabulary(conn, "c")
        assert terms and keyword_tags(client) == {"state": "built", "terms": len(terms)}
        worker.stop()
        later = db.create_job(conn, "tag_build", "c", {})  # building outranks built
        assert keyword_tags(client)["state"] == "building"
        db.update_job(conn, later, status="interrupted")  # and so does stopped
        assert keyword_tags(client) == {"state": "stopped", "job_id": later}
        db.update_job(conn, later, status="done")
        assert keyword_tags(client)["state"] == "built"
