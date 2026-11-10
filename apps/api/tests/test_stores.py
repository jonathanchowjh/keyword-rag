"""The store routes: a store's spec, state and sources, its loads, copies and rebuilds."""

from dataclasses import asdict

import pytest
from api_helpers import (
    EVAL,
    HAND,
    LISTS,
    MCQ,
    OTHER,
    OTHER_LISTS,
    PAIR,
    PROPOSING,
    TIDES,
    WORKED,
    add_collection,
    add_documents,
    building,
    do_import,
    free_disabled_model,
    import_items,
    jsonl,
    ledger_purposes,
    make_client,
    new_chat,
    put_questions,
    reembed,
    run_pair,
    scibench_source,
    set_running,
    small_model,
    three_items,
    upload,
    wait,
)

from ragqa_api.guards import PRICE_CHANGED
from ragqa_api.routes.stores import collection_state
from ragqa_core import db
from ragqa_core.fakes import ITEMS
from ragqa_core.types import Item, KeywordLists, Question


def test_create_without_recipe_is_paper_2024(client):
    add_collection(client)
    body = client.get("/collections/c").json()
    assert (body["recipe"], body["corpus_type"]) == ("paper_2024", "worked")
    assert body["build"]["added_text"] == "none" and body["build"]["chunk_overlap"] == 25
    assert body["search"]["weights"] == {"meaning": 1.0, "words": 1.0, "keywords": 1.0}
    assert body["vocabulary_version"] is None and "profile" not in body
    assert "recipes" in client.get("/catalog").json()


def test_create_with_recipe(client):
    r = client.post("/collections", json={"name": "w", "qa_type": "science", "recipe": "reference"})
    assert r.status_code == 201
    assert r.json()["recipe"] == "reference" and r.json()["search"]["query_tags"] == "classifier"


def test_unknown_recipe_is_422(client):
    r = client.post("/collections", json={"name": "z", "qa_type": "mcq", "recipe": "nope"})
    assert r.status_code == 422


def test_put_search_merges_and_validates(client):
    add_collection(client)
    r = client.put("/collections/c/search", json={"weights": {"keywords": 2}})
    assert r.status_code == 200
    assert r.json()["weights"] == {"meaning": 1.0, "words": 1.0, "keywords": 2.0}
    assert client.get("/collections/c").json()["search"]["weights"]["keywords"] == 2.0
    r = client.put("/collections/c/search", json={"program": True})
    assert r.json()["weights"]["keywords"] == 2.0 and r.json()["program"] is True
    assert client.put("/collections/c/search", json={"example": "x"}).status_code == 422
    assert client.put("/collections/nope/search", json={}).status_code == 404


def test_collection_detail(client):
    add_collection(client, lists=False)
    detail = client.get("/collections/c").json()
    assert detail["count"] == 3 and detail["keyword_lists"] is None
    assert "hybrid" in detail["methods_available"] and "kw_all" not in detail["methods_available"]
    wait(client, client.put("/collections/c/keyword-lists", json=asdict(LISTS)).json()["job_id"])
    detail = client.get("/collections/c").json()
    assert "kw_all" in detail["methods_available"] and detail["keyword_lists"]["topic"]
    assert "keyword_rag" not in detail["methods_available"]  # no vocabulary
    assert [c["name"] for c in client.get("/collections").json()] == ["c"]


def test_duplicate_collection_409(client):
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    assert client.post("/collections", json={"name": "c", "qa_type": "mcq"}).status_code == 409


def test_errors_say_store(client):
    assert client.get("/collections/nope").json()["detail"] == "no store named 'nope'"
    add_collection(client)
    r = client.post("/collections/c/load", json={"dataset": "tatqa"})
    assert "store 'c'" in r.json()["detail"] and "collection" not in r.json()["detail"]


def test_load_rejects_wrong_qa_type(client):
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    assert client.post("/collections/c/load", json={"dataset": "musique"}).status_code == 409


def test_load_rejects_the_other_qa_type_for_science_datasets(client):
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    r = client.post("/collections/c/load", json={"dataset": "scibench"})
    assert r.status_code == 409 and "dataset scibench is science" in r.json()["detail"]
    client.post("/collections", json={"name": "s", "qa_type": "science"})
    r = client.post("/collections/s/load", json={"dataset": "scienceqa"})
    assert r.status_code == 409 and "dataset scienceqa is mcq" in r.json()["detail"]


def test_load_summarize_without_key_503(tmp_path):
    client = make_client(tmp_path, api_key=None)
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/load", json={"dataset": "scienceqa"})
    assert r.status_code == 503 and "'summary' job" in r.json()["detail"]


def test_load_without_ai_runs_and_saves_no_lists(tmp_path, monkeypatch):
    from ragqa_core import loader
    from ragqa_core.types import Item, Question

    def fake_load(name, cache):
        items = [ragqa_item(i) for i in ITEMS]
        return items, [Question("q1", "enzyme rate", None, ["x"], ["enzyme-item"], {})]

    def ragqa_item(d):
        return Item(d["id"], d["text"] + " shared words", d["answer"], d["explanation"], d["meta"])

    monkeypatch.setattr(loader, "load_dataset", fake_load)
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        r = client.post("/collections/m/load", json={"dataset": "musique"})
        assert r.status_code == 202 and set(r.json()) == {"job_id"}
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["added"] == 3 and job["result"]["lists_source"] is None
        got = client.get("/collections/m").json()
        assert got["keyword_lists"] is None and got["state"] == "no_lists"


def test_load_saves_no_lists_when_the_ai_proposes_nothing(tmp_path, monkeypatch):
    from ragqa_core import loader
    from ragqa_core.types import Item

    items = [Item(d["id"], d["text"] + " shared words", d["answer"]) for d in ITEMS]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    with make_client(tmp_path, reply=lambda p: "I cannot help with that.") as client:
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        r = client.post("/collections/m/load", json={"dataset": "musique", "confirm": True})
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["lists_source"] == "none (ai proposal unusable)"
        assert client.get("/collections/m").json()["keyword_lists"] is None


def test_load_shows_its_propose_cost_first(tmp_path, monkeypatch):
    three_items(monkeypatch)
    reply = lambda p: '{"topic": ["enzymes"], "method": ["assay"], "common": ["rate"]}'  # noqa: E731
    with make_client(tmp_path, reply=reply) as client:
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        r = client.post("/collections/m/load", json={"dataset": "musique"})
        assert r.status_code == 200 and set(r.json()) == {"estimate"}
        estimate = r.json()["estimate"]
        assert estimate["est_usd"] > 0 and estimate["items"] == 3 and estimate["calls"] == 1
        assert ledger_purposes(client) == [] and client.get("/jobs").json() == []
        r = client.post("/collections/m/load", json={"dataset": "musique", "confirm": True})
        assert r.status_code == 202 and set(r.json()) == {"job_id"}
        job = wait(client, r.json()["job_id"])
        assert job["result"]["lists_source"] == "ai"
        assert ledger_purposes(client) == ["propose"]


def test_induced_recipe_load_prices_its_induction_and_never_proposes(tmp_path, monkeypatch):
    three_items(monkeypatch)
    reply = lambda p: '{"terms": ["enzymes"], "topic": ["x"], "method": ["y"], "common": ["z"]}'  # noqa: E731
    with make_client(tmp_path, reply=reply) as client:
        body = {"name": "r", "qa_type": "multihop", "recipe": "reference"}
        assert client.post("/collections", json=body).status_code == 201
        r = client.post("/collections/r/load", json={"dataset": "musique"})
        assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1  # induction only
        r = client.post("/collections/r/load", json={"dataset": "musique", "confirm": True})
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done" and job["result"]["lists_source"] is None
        assert ledger_purposes(client) == ["induce"]  # similarity tags: no tag calls
        assert job["result"]["tagged"] == 3
        # With a vocabulary in force, a reference load spends nothing: queued at once.
        r = client.post("/collections/r/load", json={"dataset": "musique"})
        assert r.status_code == 202 and "estimate" not in r.json()


def test_a_load_free_only_because_of_a_hub_entry_holds_to_it(tmp_path, monkeypatch):
    # Queued without confirm because the entry made it free, it must not build locally at a
    # price nobody agreed to; a confirmed load has agreed to spend, so it may.
    three_items(monkeypatch)
    with make_client(tmp_path) as client:
        client.app.state.worker.stop()  # the jobs stay queued, to read their setup
        assert client.post("/collections", json=WORKED).status_code == 201
        hub = {"entry": "scibench-all-worked_examples", "items": 3}
        free = {"est_usd": 0.0, "items": 3, "calls": 0, "summary_calls": 0, "keyword_calls": 0}
        quote = {**free, "hub": hub}
        client.app.state.loader.estimate_load = lambda spec, dataset, limit: quote
        for confirm in (False, True):
            r = client.post("/collections/w/load", json={"dataset": "scibench", "confirm": confirm})
            assert r.status_code == 202, r.text
            setup = client.get(f"/jobs/{r.json()['job_id']}").json()["setup"]
            assert setup.get("hub_entry") == (None if confirm else hub["entry"])


def test_a_stale_price_is_refused_before_a_missing_model(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        body = {"dataset": "scienceqa", "confirm": True, "calls": 99}
        r = client.post("/collections/s/load", json=body)
        assert r.status_code == 409
        assert r.json()["detail"] == "the price changed since you saw it: check it again"
        state = client.app.state
        now = state.loader.estimate_load(state.retriever.spec("s"), "scienceqa", None)
        shown = {"calls": 3, "est_usd": now["est_usd"]}  # what it costs now
        r = client.post("/collections/s/load", json={**body, **shown})
        assert r.status_code == 503 and "'summary'" in r.json()["detail"]
        assert client.get("/jobs").json() == []


def test_a_load_free_because_of_a_hub_build_needs_no_keywords_model(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, api_key=None) as client:
        client.app.state.worker.stop()
        assert client.post("/collections", json=WORKED).status_code == 201
        r = client.post("/collections/w/load", json={"dataset": "scibench"})
        assert r.status_code == 503 and "'keywords'" in r.json()["detail"]  # it would build
        loader = client.app.state.loader
        published = {"entry": {"id": "scibench-all-worked_examples"}}
        hub = {"entry": "scibench-all-worked_examples", "items": 3}
        loader.from_hub = lambda spec, items, dataset, limit: (published, hub)
        for confirm in (False, True):
            r = client.post("/collections/w/load", json={"dataset": "scibench", "confirm": confirm})
            assert r.status_code == 202, r.text


def test_load_that_spends_nothing_queues_without_confirm(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, api_key=None) as client:  # no AI: no lists, nothing spent
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        assert client.post("/collections/m/load", json={"dataset": "musique"}).status_code == 202
    with make_client(tmp_path / "k") as client:  # lists exist: no propose
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        lists = {"topic": ["enzyme"], "method": ["assay"], "common": ["shared"]}
        wait(client, client.put("/collections/m/keyword-lists", json=lists).json()["job_id"])
        r = client.post("/collections/m/load", json={"dataset": "musique"})
        assert r.status_code == 202 and "estimate" not in r.json()


def test_summarising_load_estimate_includes_the_propose(tmp_path, monkeypatch):
    items = three_items(monkeypatch)
    with make_client(tmp_path) as client:
        client.post("/collections", json={"name": "s", "qa_type": "multihop", "summarize": True})
        spec = client.app.state.retriever.spec("s")
        indexer = client.app.state.indexer
        summaries = (
            indexer.plan_summaries(indexer.changed(spec, items, "hand")[0])[0]
            if spec.summarize
            else 0.0
        )
        r = client.post("/collections/s/load", json={"dataset": "musique"})
        estimate = r.json()["estimate"]
        assert r.status_code == 200 and estimate["est_usd"] > summaries
        assert estimate["calls"] == 3 + 1 and ledger_purposes(client) == []
        # With lists already saved, only the summaries are priced.
        lists = KeywordLists(["enzyme"], ["assay"], ["shared"])
        db.save_keyword_lists(client.app.state.conn, "s", lists, "v1")
        db.set_keyword_version(client.app.state.conn, "s", "v1")
        estimate = client.post("/collections/s/load", json={"dataset": "musique"}).json()
        assert estimate["estimate"]["est_usd"] == pytest.approx(summaries)
        assert estimate["estimate"]["calls"] == 3


def test_unknown_collection_404(client):
    assert client.post("/query", json={"collection": "nope", "question": "x"}).status_code == 404
    assert client.get("/collections/nope").status_code == 404


def test_bad_name_422(client):
    assert client.post("/collections", json={"name": "a b", "qa_type": "mcq"}).status_code == 422


def test_delete_collection_cancels_queued_and_refuses_running(client):  # Review focus 3
    add_collection(client)
    client.app.state.worker.stop()
    j = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    assert client.delete("/collections/c").status_code == 204
    assert client.get(f"/jobs/{j}").status_code == 404
    assert client.get("/collections/c").status_code == 404
    assert not client.app.state.store.client.collection_exists("c")
    assert len(db.get_questions(client.app.state.conn, "c")) == 1  # the eval set stays
    add_collection(client, name="d", lists=False)
    running = db.create_job(client.app.state.conn, "load", "d", {})
    set_running(client.app.state.conn, running)
    r = client.delete("/collections/d")
    assert r.status_code == 409 and "running" in r.json()["detail"]
    assert client.get("/collections/d").json()["count"] == 3
    assert client.delete("/collections/nope").status_code == 404


def test_answer_carries_proxy_trusted(client):
    add_collection(client)
    run = client.post("/eval", json={**EVAL, "methods": PAIR, "confirm": True}).json()["job_id"]
    wait(client, run)
    answer = client.get("/collections/c").json()["answer"]
    assert "proxy_trusted" in answer
    assert answer["proxy_trusted"] == client.app.state.evaluator.collection_proxy_trusted("c")


def test_collection_detail_has_state_and_answer(client):
    conn = client.app.state.conn
    client.post("/collections", json={"name": "e", "qa_type": "mcq"})
    e = client.get("/collections/e").json()
    assert e["state"] == "empty" and e["job"] is None and e["progress"] is None
    assert e["questions"] == 0 and e["dev_questions"] == 0 and e["runs"] == 0
    assert e["answer"] is None
    add_collection(client, lists=False)
    assert client.get("/collections/c").json()["state"] == "no_lists"
    wait(client, client.put("/collections/c/keyword-lists", json=asdict(LISTS)).json()["job_id"])
    c = client.get("/collections/c").json()
    assert c["state"] == "ready" and c["job"] is None
    assert c["questions"] == 1 and c["dev_questions"] == 1 and c["runs"] == 0
    r = client.post("/eval", json={**EVAL, "methods": PAIR, "answer": True, "confirm": True})
    run = r.json()["job_id"]
    assert wait(client, run)["status"] == "done"
    c = client.get("/collections/c").json()
    assert c["runs"] == 1 and c["answer"]["run_id"] == run and c["answer"]["answered"] is True
    assert c["answer"] == client.app.state.evaluator.collection_answer("c")
    client.app.state.worker.stop()
    queued = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    c = client.get("/collections/c").json()
    assert c["state"] == "queued" and c["job"]["id"] == queued and c["runs"] == 2
    rekey = db.create_job(conn, "rekey", "c", {"version": "v"})
    set_running(conn, rekey, 0.4)
    c = client.get("/collections/c").json()
    assert c["state"] == "rekeying" and c["progress"] == 0.4 and c["job"]["id"] == rekey
    db.update_job(conn, rekey, status="done")
    set_running(conn, queued, 0.5)
    assert client.get("/collections/c").json()["state"] == "evaluating"
    db.update_job(conn, queued, status="done")
    load = db.create_job(conn, "load", "c", {})
    set_running(conn, load)
    assert client.get("/collections/c").json()["state"] == "loading"


def test_since_is_empty_when_nothing_changed(client):
    add_collection(client)
    job_id = run_pair(client)
    expected = {"untracked": False, "changes": []}
    assert client.get(f"/eval/{job_id}").json()["since"] == expected
    assert client.get("/collections/c").json()["answer"]["since"] == expected


def test_since_names_changed_keyword_lists(client):
    add_collection(client)
    job_id = run_pair(client)
    r = client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS))
    assert wait(client, r.json()["job_id"])["status"] == "done"
    expected = {"untracked": False, "changes": ["keyword lists"]}
    assert client.get(f"/eval/{job_id}").json()["since"] == expected
    assert client.get("/collections/c").json()["answer"]["since"] == expected


def test_since_marks_a_run_without_recorded_versions_untracked(client):
    add_collection(client)
    job_id = run_pair(client)
    conn = client.app.state.conn
    setup = db.get_job(conn, job_id)["setup"]
    for key in ("keyword_version", "dense_model"):
        del setup[key]
    db.set_job_setup(conn, job_id, setup)
    client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS))
    expected = {"untracked": True, "changes": []}
    assert client.get(f"/eval/{job_id}").json()["since"] == expected
    assert client.get("/collections/c").json()["answer"]["since"] == expected


def test_every_import_and_load_path_records_that_it_priced_the_proposal(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, reply=PROPOSING) as client:
        add_collection(client, lists=False)
        client.app.state.worker.stop()  # the jobs stay queued, to read their setup
        assert client.app.state.ai.ready("keywords")
        many = [{"id": f"m{i}", "text": f"note {i} about reefs"} for i in range(21)]
        r = client.post("/collections/c/documents", json={"items": many, "confirm": True})
        assert r.status_code == 202, r.text
        jobs = [r.json()["job_id"]]
        r = do_import(client, upload(client).json(), confirm=True)
        assert r.status_code == 202, r.text
        jobs.append(r.json()["job_id"])
        r = client.post("/collections/c/load", json={"dataset": "scienceqa", "confirm": True})
        assert r.status_code == 202, r.text
        jobs.append(r.json()["job_id"])
        assert [client.get(f"/jobs/{j}").json()["setup"]["propose"] for j in jobs] == [True] * 3
        client.app.state.worker.start()
        done = [wait(client, j) for j in jobs]
        assert [j["status"] for j in done] == ["done"] * 3, done
        assert done[0]["result"]["lists_source"] == "ai"  # the first makes them; then none
        assert client.get("/collections/c").json()["keyword_lists"] is not None
        assert ledger_purposes(client) == ["propose"]


def test_store_replies_have_no_profile(client):
    add_collection(client)
    assert "profile" not in client.get("/collections/c").json()
    assert all("profile" not in c for c in client.get("/collections").json())


def test_sources_list_switch_and_query_skips_disabled(client):
    add_collection(client)
    import_items(client)
    sources = client.get("/collections/c/sources").json()
    assert [s["source"] for s in sources] == ["hand", "upload:notes"]
    r = client.patch("/collections/c/sources/upload:notes", json={"enabled": False})
    assert r.status_code == 200 and r.json() == {
        "source": "upload:notes",
        "items": 3,
        "enabled": False,
    }
    for method in ("hybrid", "random_example"):
        body = {"collection": "c", "question": "tides moon", "methods": [method], "k": 10}
        hits = client.post("/query", json=body).json()["results"][method]["hits"]
        assert hits and not {h["id"] for h in hits} & {"n1", "n2", "n3"}
    assert client.get("/collections/c/sources").json()[1]["enabled"] is False
    client.patch("/collections/c/sources/upload:notes", json={"enabled": True})
    assert client.get("/collections/c/sources").json()[1]["enabled"] is True


def test_unknown_source_is_404(client):
    add_collection(client)
    assert client.patch("/collections/c/sources/nope", json={"enabled": False}).status_code == 404
    assert client.delete("/collections/c/sources/nope").status_code == 404


def test_switch_refused_while_eval_running(client):
    add_collection(client)
    import_items(client)
    conn = client.app.state.conn
    with conn.lock:  # the worker can't claim the job between the two calls
        job_id = db.create_job(conn, "eval", "c", {})
        db.update_job(conn, job_id, status="running")
    r = client.patch("/collections/c/sources/upload:notes", json={"enabled": False})
    assert r.status_code == 409 and "eval is running" in r.json()["detail"]


def test_switch_refused_while_backfill_pending(client):
    add_collection(client)
    import_items(client)
    client.app.state.worker.stop()  # the backfill job stays queued
    db.create_job(client.app.state.conn, "backfill", "c", {})
    r = client.patch("/collections/c/sources/upload:notes", json={"enabled": False})
    assert r.status_code == 409 and "still being labelled" in r.json()["detail"]


def test_last_source_cannot_be_switched_off(client):
    add_collection(client)
    only = client.get("/collections/c/sources").json()
    assert [s["source"] for s in only] == ["hand"]
    r = client.patch("/collections/c/sources/hand", json={"enabled": False})
    assert r.status_code == 409 and "at least one source" in r.json()["detail"]


def test_store_sources_count_items_and_deleting_one_keeps_questions(client):
    add_collection(client)
    import_items(client)
    uq = Question("uq", "do tides follow the moon?", ["yes", "no"], ["A"], [], {})
    db.save_questions(client.app.state.conn, "c", [uq])
    store = client.app.state.store
    assert store.count("c") == 6
    assert client.delete("/collections/c/sources/upload:notes").status_code == 204
    assert store.count("c") == 3 and "upload:notes" not in store.source_counts("c")
    ids = [q.id for q in db.get_questions(client.app.state.conn, "c")]
    assert ids == ["q1", "uq"]  # the questions are the eval set's, not the store's
    assert client.get("/collections/c/sources").json() == [
        {"source": "hand", "items": 3, "enabled": True}
    ]


def test_delete_source_refused_while_a_job_is_queued(client):
    add_collection(client)
    import_items(client)
    client.app.state.worker.stop()  # else the worker may claim and fail the job first
    db.create_job(client.app.state.conn, "rekey", "c", {})
    assert client.delete("/collections/c/sources/upload:notes").status_code == 409


def test_disabled_source_without_items_still_lists(client):
    add_collection(client)
    db.set_source_enabled(client.app.state.conn, "c", "gone", False)
    sources = client.get("/collections/c/sources").json()
    assert {"source": "gone", "items": 0, "enabled": False} in sources


def test_collection_reports_fields_and_measurable(client):
    client.post("/collections", json={"name": "sci", "qa_type": "science"})
    client.post("/collections/sci/documents", json={"items": ITEMS, "confirm": True})
    plain = Question("p1", "enzyme rate", None, ["3 J"], [], {})
    db.save_questions(client.app.state.conn, "sci", [plain])
    detail = client.get("/collections/sci").json()
    assert detail["fields"] == {
        "questions": {"total": 1, "solution": 0, "skill": 0, "gold_ids": 0},
        "items": {"total": 3, "explanation": 3, "skill": 3},
    }
    m = detail["measurable"]
    assert m["solution_sim@1"] == {"measurable": False, "note": "no question has a worked solution"}
    assert m["accuracy"]["measurable"] is True
    assert detail["sources"] == [{"source": "hand", "items": 3, "enabled": True}]
    assert detail["unlabelled"] == 0 and detail["leaked"] == 0


def import_questions_only(client):
    """Imports one question into store c's eval set (named c), on the eval set's page."""
    rows = jsonl({"id": "uq", "text": "do tides follow?", "choices": ["yes", "no"], "answer": "A"})
    up = client.post("/eval-sets/c/uploads", params={"filename": "qs.jsonl"}, content=rows).json()
    body = {"upload_id": up["upload_id"], "mapping": up["preview"]["mapping"], "source_name": "qs"}
    assert client.post("/eval-sets/c/imports", json=body).status_code == 200


def test_a_question_import_adds_no_source_row(client):
    add_collection(client)
    import_questions_only(client)
    assert [s["source"] for s in client.get("/collections/c/sources").json()] == ["hand"]
    assert [q.id for q in db.get_questions(client.app.state.conn, "c")] == ["q1", "uq"]


def test_deleting_a_source_cannot_leave_every_item_source_off(client):
    add_collection(client)
    import_items(client)
    assert client.patch("/collections/c/sources/hand", json={"enabled": False}).status_code == 200
    r = client.delete("/collections/c/sources/upload:notes")
    assert r.status_code == 409 and "at least one source must stay on" in r.json()["detail"]
    assert client.get("/collections/c").json()["count"] == 6
    client.patch("/collections/c/sources/hand", json={"enabled": True})
    assert client.delete("/collections/c/sources/upload:notes").status_code == 204


def test_deleting_a_collection_removes_its_uploads(client):
    add_collection(client)
    add_collection(client, name="keep", questions=False, lists=False)
    gone = upload(client).json()["upload_id"]
    kept = upload(client, name="keep").json()["upload_id"]
    assert client.delete("/collections/c").status_code == 204
    uploads = client.app.state.settings.home / "uploads"
    assert not (uploads / gone).exists() and (uploads / kept).exists()


def test_zero_priced_disabled_summary_model_is_refused(tmp_path):
    with make_client(tmp_path) as client:
        scibench_source(client)
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        client.app.state.worker.stop()
        free_disabled_model(client, "summary", "keywords")  # keywords: s proposes nothing
        conn = client.app.state.conn

        def refused(r) -> None:
            assert r.status_code == 503, r.text
            assert "'summary'" in r.json()["detail"]

        up = upload(client, name="s").json()
        assert up["estimate"]["est_usd"] == 0 and up["estimate"]["calls"] > 0
        refused(do_import(client, up, name="s", confirm=True))
        two = [{"id": f"t{i}", "text": f"tide note {i}"} for i in range(2)]
        refused(client.post("/collections/s/documents", json={"items": two}))
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        refused(client.post("/collections/s/documents", json={"items": many}))
        refused(client.post("/collections/s/documents", json={"items": many, "confirm": True}))
        body = {"name": "x", "recipe": "paper_2024", "build": {"added_text": "summary"}}
        for confirm in (False, True):
            refused(client.post("/collections/p/copy", json={**body, "confirm": confirm}))
        assert client.get("/collections/s").json()["count"] == 0
        assert client.get("/collections/x").status_code == 404
        assert db.list_jobs(conn, limit=db.ALL) == [] and ledger_purposes(client) == []


def test_zero_priced_disabled_keywords_model_is_refused(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, reply=building) as client:
        scibench_source(client)
        client.post("/collections", json=WORKED)
        client.app.state.worker.stop()
        free_disabled_model(client, "keywords")

        def refused(r) -> None:
            assert r.status_code == 503, r.text
            assert "'keywords'" in r.json()["detail"]

        refused(client.post("/collections/w/load", json={"dataset": "scibench"}))
        refused(client.post("/collections/w/documents", json={"items": ITEMS}))
        for confirm in (False, True):
            body = {"name": "x", "recipe": "worked_examples", "confirm": confirm}
            refused(client.post("/collections/p/copy", json=body))
        assert db.list_jobs(client.app.state.conn, limit=db.ALL) == []


def test_unconfirmed_load_with_a_stale_price_gets_the_estimate(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, reply=PROPOSING) as client:
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        client.app.state.worker.stop()
        seen = client.post("/collections/m/load", json={"dataset": "musique"}).json()["estimate"]
        assert seen["calls"] == 1 and seen["est_usd"] > 0  # the proposal
        half = seen["est_usd"] / 2
        for stale in ({"calls": 0}, {"calls": 0, "est_usd": seen["est_usd"]}, {"est_usd": half}):
            r = client.post("/collections/m/load", json={"dataset": "musique", **stale})
            assert r.status_code == 200 and r.json() == {"estimate": seen}, (stale, r.text)
        conn = client.app.state.conn
        assert db.list_jobs(conn, limit=db.ALL) == [] and ledger_purposes(client) == []


def test_unconfirmed_stale_free_load_queues(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, api_key=None) as client:  # no AI: the load is free
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        client.app.state.worker.stop()
        r = client.post(
            "/collections/m/load", json={"dataset": "musique", "calls": 5, "est_usd": 1}
        )
        assert r.status_code == 202 and "estimate" not in r.json(), r.text
        assert len(db.list_jobs(client.app.state.conn, limit=db.ALL)) == 1
        assert ledger_purposes(client) == []


def test_copy_estimate_then_build(tmp_path):
    with make_client(tmp_path, reply=building) as client:
        items = scibench_source(client)
        body = {"name": "w", "recipe": "worked_examples"}
        r = client.post("/collections/p/copy", json=body)
        assert r.status_code == 200, r.text
        estimate = r.json()["estimate"]
        assert estimate["est_usd"] > 0 and estimate["items"] == len(items) + len(ITEMS)
        assert client.get("/collections/w").status_code == 404
        assert client.get("/jobs").json() == [] and ledger_purposes(client) == []
        r = client.post("/collections/p/copy", json={**body, "confirm": True})
        assert r.status_code == 202, r.text
        assert r.json()["name"] == "w"
        job = wait(client, r.json()["job_id"])
        assert (job["kind"], job["status"]) == ("copy", "done"), job
        assert job["result"]["sources"] == ["hand", "scibench"]
        w, p = client.get("/collections/w").json(), client.get("/collections/p").json()
        assert (w["recipe"], w["qa_type"], w["corpus_type"]) == (
            "worked_examples",
            "science",
            "worked",
        )
        assert w["search"]["example"] == "abstracted"
        assert w["count"] == p["count"] and w["questions"] == p["questions"]
        assert w["vocabulary_version"]
        tagged = [h for batch in client.app.state.store.scroll("w") for h in batch]
        assert len(tagged) == w["count"] and all(h.tags == ["kinematics"] for h in tagged)
        conn, store = client.app.state.conn, client.app.state.store
        assert sorted(store.ids("w")) == sorted(store.ids("p"))
        assert db.default_eval_set(conn, "w") == "p" and db.get_questions(conn, "w") == []
        assert set(ledger_purposes(client)) == {"induce", "tag", "abstract"}


def test_copy_keeps_sources_and_disabled_state(tmp_path):
    with make_client(tmp_path, reply=building) as client:
        scibench_source(client)
        r = client.post(
            "/collections/p/copy", json={"name": "r", "recipe": "reference", "confirm": True}
        )
        assert r.status_code == 202, r.text
        assert wait(client, r.json()["job_id"])["status"] == "done"
        sources = client.get("/collections/r/sources").json()
        assert sources == client.get("/collections/p/sources").json()
        assert [(s["source"], s["enabled"]) for s in sources] == [
            ("hand", False),
            ("scibench", True),
        ]
        assert client.get("/collections/r").json()["eval_set"] == "p"


def test_copy_to_existing_name_is_409(client):
    scibench_source(client)
    client.app.state.worker.stop()
    client.post("/collections", json={"name": "w", "qa_type": "mcq"})
    for confirm in (False, True):
        body = {"name": "w", "recipe": "reference", "confirm": confirm}
        r = client.post("/collections/p/copy", json=body)
        assert r.status_code == 409 and "already exists" in r.json()["detail"]
        r = client.post("/collections/p/copy", json={**body, "name": "p"})  # onto itself
        assert r.status_code == 409
    r = client.post("/collections/nope/copy", json={"name": "x", "recipe": "reference"})
    assert r.status_code == 404
    # A job on the source could change what is copied while the copy reads it.
    job_id = db.create_job(client.app.state.conn, "load", "p", {})
    r = client.post("/collections/p/copy", json={"name": "x", "recipe": "reference"})
    assert r.status_code == 409 and "load" in r.json()["detail"]
    db.request_cancel(client.app.state.conn, job_id)
    assert client.get("/collections/x").status_code == 404


def test_copy_to_a_name_whose_eval_set_is_another_type_is_409(client):
    scibench_source(client)
    client.app.state.worker.stop()
    client.post("/collections", json={"name": "m", "qa_type": "mcq"})
    client.post("/eval-sets", json={"name": "w", "qa_type": "science"})
    for confirm in (False, True):
        body = {"name": "w", "recipe": "reference", "confirm": confirm}
        r = client.post("/collections/m/copy", json=body)
        assert r.status_code == 409
        assert r.json()["detail"] == "eval set 'w' is science; store 'w' is mcq"
    assert client.get("/collections/w").status_code == 404
    assert db.list_jobs(client.app.state.conn, kind="copy") == []


def test_copy_build_override_validated(client):
    scibench_source(client)
    client.app.state.worker.stop()
    bad = (
        ("reference", {"tagging": "magic"}),
        ("reference", {"colour": "red"}),
        ("reference", {"chunk_overlap": 100}),
        ("reference", {"added_text": "summary"}),
        ("worked_examples", {"vocabulary": "lists"}),
        ("nope", None),
    )
    for recipe, build in bad:
        for confirm in (False, True):
            body = {"name": "x", "recipe": recipe, "build": build, "confirm": confirm}
            r = client.post("/collections/p/copy", json=body)
            assert r.status_code == 422, (recipe, build, r.text)
    r = client.post("/collections/p/copy", json={"name": "bad name!", "recipe": "reference"})
    assert r.status_code == 422
    assert client.get("/collections/x").status_code == 404 and client.get("/jobs").json() == []
    body = {"name": "x", "recipe": "reference", "build": {"chunk_words": 50, "chunk_overlap": 5}}
    r = client.post("/collections/p/copy", json={**body, "confirm": True})
    assert r.status_code == 202, r.text
    build = client.get("/collections/x").json()["build"]
    assert (build["chunk_words"], build["chunk_overlap"], build["tagging"]) == (50, 5, "similarity")


def test_copy_without_a_model_is_503_when_it_would_spend(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        scibench_source(client)
        client.app.state.worker.stop()
        for recipe, build, role in (
            ("worked_examples", None, "keywords"),
            ("paper_2024", {"added_text": "summary"}, "summary"),
        ):
            for confirm in (False, True):
                body = {"name": "x", "recipe": recipe, "build": build, "confirm": confirm}
                r = client.post("/collections/p/copy", json=body)
                assert r.status_code == 503, r.text
                assert f"'{role}'" in r.json()["detail"] and "Settings" in r.json()["detail"]
        assert client.get("/collections/x").status_code == 404 and client.get("/jobs").json() == []
        # Embedding is free: a plain paper 2024 copy needs no model.
        r = client.post("/collections/p/copy", json={"name": "x", "recipe": "paper_2024"})
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] == 0
        r = client.post(
            "/collections/p/copy", json={"name": "x", "recipe": "paper_2024", "confirm": True}
        )
        assert r.status_code == 202, r.text


def test_copy_state_and_problem_wording(client):
    client.post("/collections", json=WORKED)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    copy = db.create_job(conn, "copy", "w", {"source": "p"})
    set_running(conn, copy, 0.4)
    assert client.get("/collections/w").json()["state"] == "copying"
    db.update_job(conn, copy, status="failed", error="boom", progress=0.4)
    problems = client.get("/status").json()["problems"]
    assert "The copy into w failed; delete w and copy it again" in problems
    # A collection being copied into takes no hand-added items or vocabulary meanwhile.
    db.update_job(conn, db.create_job(conn, "copy", "w", {"source": "p"}), status="queued")
    r = client.post("/collections/w/documents", json={"items": ITEMS})
    assert r.status_code == 409 and "copy" in r.json()["detail"]
    r = client.put("/collections/w/vocabulary", json={"terms": ["x"]})
    assert r.status_code == 409 and "copy" in r.json()["detail"]


def test_copy_confirm_with_the_shown_price_is_accepted(client):
    add_collection(client, summarize=True, name="s")
    body = {"name": "x", "recipe": "paper_2024"}
    seen = client.post("/collections/s/copy", json=body).json()["estimate"]
    shown = {**body, "confirm": True, "calls": seen["calls"], "est_usd": seen["est_usd"]}
    r = client.post("/collections/s/copy", json=shown)
    assert r.status_code == 202, r.text
    job = db.get_job(client.app.state.conn, r.json()["job_id"])
    assert (job["kind"], job["collection"]) == ("copy", "x")
    assert job["setup"]["source"] == "s"
    assert client.get("/collections/x").status_code == 200


def test_copy_refuses_a_changed_price(client):
    add_collection(client, summarize=True, name="s")
    body = {"name": "x", "recipe": "paper_2024"}
    seen = client.post("/collections/s/copy", json=body).json()["estimate"]
    stale_body = {**body, "confirm": True, "calls": seen["calls"] + 1, "est_usd": seen["est_usd"]}
    r = client.post("/collections/s/copy", json=stale_body)
    assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
    assert client.get("/collections/x").status_code == 404
    assert client.post("/collections/s/copy", json={**body, "confirm": True}).status_code == 202


def test_the_store_page_scans_for_leaked_questions_once_per_change(client):
    add_collection(client)
    tides = jsonl({"id": "n1", "text": "so the tides follow the moon each day we saw"})
    import_items(client, body=tides)
    meta = {"source": "upload:qs"}  # as the eval set's import stamps it
    u1 = Question("u1", "the tides follow the moon each day", ["x", "y"], ["A"], [], meta)
    db.save_questions(client.app.state.conn, "c", [u1])
    store, scans = client.app.state.store, []
    scroll = store.scroll

    def counted(name, batch=256, skip=()):
        scans.append(1)
        return scroll(name, batch, skip)

    store.scroll = counted
    assert client.get("/collections/c").json()["leaked"] == 1
    assert client.get("/collections/c").json()["leaked"] == 1
    assert len(scans) == 1
    import_items(
        client, body=jsonl({"id": "n2", "text": "glaciers carve valleys"}), source_name="more"
    )
    scans.clear()
    assert client.get("/collections/c").json()["leaked"] == 1 and len(scans) == 1


def test_patch_sets_and_clears_the_store_default_eval_set(client):
    add_collection(client)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    assert (
        put_questions(client, "e", [{**MCQ, "meta": {"solution": "Rates rise."}}]).status_code
        == 200
    )
    body = {**EVAL, "methods": PAIR, "answer": True, "eval_set": "e", "confirm": True}
    run = client.post("/eval", json=body).json()["job_id"]
    assert wait(client, run)["status"] == "done"
    assert client.get("/collections/c").json()["answer"] is None  # not the default's run
    r = client.patch("/collections/c", json={"eval_set": "e"})
    assert r.status_code == 200, r.text
    assert r.json() == client.get("/collections/c").json()
    assert r.json()["eval_set"] == "e" and r.json()["answer"]["run_id"] == run
    cleared = client.patch("/collections/c", json={"eval_set": None}).json()
    assert cleared["eval_set"] == "c" and cleared["answer"] is None  # the name fallback
    client.post("/eval-sets", json={"name": "s", "qa_type": "science"})
    clash = client.patch("/collections/c", json={"eval_set": "s"})
    assert clash.status_code == 409
    assert clash.json()["detail"] == "eval set 's' is science; store 'c' is mcq"
    assert client.patch("/collections/c", json={"eval_set": "nope"}).status_code == 404
    assert client.patch("/collections/nope", json={"eval_set": "e"}).status_code == 404
    assert client.get("/collections/c").json()["eval_set"] == "c"


def test_patch_is_refused_while_a_load_is_queued(client):
    add_collection(client)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    client.app.state.worker.stop()
    conn = client.app.state.conn
    db.create_job(conn, "load", "c", {"dataset": "scienceqa"})
    r = client.patch("/collections/c", json={"eval_set": "e"})
    assert r.status_code == 409 and r.json()["detail"] == "a load is under way on 'c'; wait"
    assert db.default_eval_set(conn, "c") == "c"


def science_default(client):
    """Store c (mcq) whose default eval set is a science one, as an old database may have."""
    add_collection(client, questions=False)
    client.app.state.worker.stop()
    client.post("/eval-sets", json={"name": "s", "qa_type": "science"})
    db.set_default_eval_set(client.app.state.conn, "c", "s")


def test_load_is_refused_before_anything_when_the_default_set_is_another_type(client):
    science_default(client)
    for confirm in (False, True):
        r = client.post("/collections/c/load", json={"dataset": "scienceqa", "confirm": confirm})
        assert r.status_code == 409
        assert r.json()["detail"] == "eval set 's' is science; store 'c' is mcq"
    conn = client.app.state.conn
    assert db.list_jobs(conn, kind="load") == []
    assert db.default_eval_set(conn, "c") == "s" and db.get_eval_set(conn, "c") is None


def test_a_deleted_store_leaves_no_leak_scan_behind(client):
    def store_with(body):
        add_collection(client, questions=False, lists=False)
        import_items(client, body=body)
        assert client.patch("/collections/c", json={"eval_set": "e"}).status_code == 200
        return client.get("/collections/c").json()

    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    assert put_questions(client, "e", [HAND, OTHER]).status_code == 200
    assert store_with(TIDES)["leaked"] == 1
    assert client.delete("/collections/c").status_code == 204
    # the same count of items and the same set version, but nothing leaks into these
    assert store_with(jsonl({"id": "n1", "text": "glaciers carve valleys"}))["leaked"] == 0


def test_a_deleted_source_leaves_no_leak_scan_behind(client):
    add_collection(client, questions=False, lists=False)
    import_items(client, body=TIDES)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    assert put_questions(client, "e", [HAND, OTHER]).status_code == 200
    assert client.patch("/collections/c", json={"eval_set": "e"}).json()["leaked"] == 1
    assert client.delete("/collections/c/sources/upload:notes").status_code == 204
    # the same count of items and the same set version, but nothing leaks into these
    import_items(client, body=jsonl({"id": "n1", "text": "glaciers carve valleys"}))
    assert client.get("/collections/c").json()["leaked"] == 0


def test_store_chat_settings_show_their_source_and_reset_to_settings(client):
    add_collection(client)  # keyword lists, no vocabulary
    recommended = {
        "method": {"value": "hybrid", "from": "recommended"},
        "candidates": {"value": 20, "from": "recommended"},
    }
    assert client.get("/collections/c").json()["chat_settings"] == recommended
    r = client.patch("/collections/c", json={"chat_method": "vector", "chat_k": 30})
    assert r.status_code == 200, r.text
    assert r.json()["chat_settings"] == {
        "method": {"value": "vector", "from": "store"},
        "candidates": {"value": 30, "from": "store"},
    }
    assert (r.json()["chat_method"], r.json()["chat_k"]) == ("vector", 30)
    r = client.patch("/collections/c", json={"chat_method": None, "chat_k": None})
    assert r.json()["chat_settings"] == recommended
    assert client.put("/search-defaults", json={"method": "keyword"}).status_code == 200
    method = client.get("/collections/c").json()["chat_settings"]["method"]
    assert method == {"value": "keyword", "from": "settings"}

    assert client.patch("/collections/c", json={"chat_method": "hybrid_rerank"}).status_code == 422
    for k in (0, 61):
        assert client.patch("/collections/c", json={"chat_k": k}).status_code == 422
    r = client.patch("/collections/c", json={"chat_method": "keyword_rag", "chat_k": 9})
    assert r.status_code == 409 and "needs a vocabulary" in r.json()["detail"]
    assert client.get("/collections/c").json()["chat_k"] is None  # nothing was saved

    # A method saved before it left the chat list reads as null, here and from Set as default.
    db.set_chat_method(client.app.state.conn, "c", "hybrid_rerank")
    detail = client.get("/collections/c").json()
    assert detail["chat_method"] is None
    assert detail["chat_settings"]["method"] == {"value": "keyword", "from": "settings"}
    chat = new_chat(client)
    client.post(f"/chats/{chat['id']}/messages", json={"content": "enzyme rate"})
    url = f"/chats/{chat['id']}/messages/1/attempts"
    assert client.post(url, json={"stores": {"c": {"candidates": 7}}}).status_code == 200
    got = client.post(f"/chats/{chat['id']}/stores/c/default").json()
    assert (got["chat_method"], got["chat_k"]) == (None, 7)


def test_store_lists_why_each_method_cannot_run(client):
    add_collection(client, name="bare", lists=False)
    why = client.get("/collections/bare").json()["methods_unavailable"]
    assert why["keyword_rag"] == "needs a vocabulary"
    assert why["kw_individual"] == "needs keyword lists"
    assert "hybrid" not in why


def test_removed_routes_are_gone(client):
    add_collection(client)
    chat = new_chat(client)
    assert client.get("/collections/c/bar").status_code == 404
    assert client.post("/collections/c/route-profile/draft").status_code == 404
    assert client.post("/routing/eval", json={"stores": ["c", "c"]}).status_code == 404
    assert client.patch("/collections/c", json={"route_profile": "x"}).status_code == 422
    assert client.patch(f"/chats/{chat['id']}", json={"routing": "all"}).status_code == 422
    detail = client.get("/collections/c").json()
    assert not {"chat_bar", "route_profile", "route_profile_drafted"} & set(detail)


def test_a_store_is_created_with_the_settings_model_or_the_one_named(client):
    service = client.app.state.models.dense_model
    r = client.post("/collections", json={"name": "a", "qa_type": "mcq"})
    assert r.status_code == 201 and r.json()["dense_model"] == service
    r = client.put("/search-defaults", json={"dense_model": "BAAI/bge-base-en-v1.5"})
    assert r.status_code == 200, r.text
    r = client.post("/collections", json={"name": "b", "qa_type": "mcq"})
    assert r.status_code == 201 and r.json()["dense_model"] == "BAAI/bge-base-en-v1.5"
    body = {"name": "c", "qa_type": "mcq", "dense_model": "BAAI/bge-small-en-v1.5"}
    r = client.post("/collections", json=body)
    assert r.status_code == 201 and r.json()["dense_model"] == "BAAI/bge-small-en-v1.5"
    body = {"name": "d", "qa_type": "mcq", "dense_model": service}  # the service's own
    assert client.post("/collections", json=body).json()["dense_model"] == service
    r = client.post("/collections", json={"name": "e", "qa_type": "mcq", "dense_model": "nope/y"})
    assert r.status_code == 422 and "unknown embedding model nope/y" in r.text
    assert client.get("/collections/e").status_code == 404


def test_rebuild_with_another_model_re_embeds_and_switches(client, monkeypatch):
    small_model(client, monkeypatch)
    add_collection(client)
    run_pair(client)
    r = reembed(client)
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"
    body = client.get("/collections/c").json()
    assert (body["dense_model"], body["dense_dim"]) == ("fake/small", 64)
    assert body["count"] == len(ITEMS) and body["rebuild"] is None
    ask = {"collection": "c", "question": "enzyme rate", "methods": ["hybrid"]}
    hits = client.post("/query", json=ask).json()["results"]["hybrid"]["hits"]
    assert hits[0]["id"] == "enzyme-item"
    assert "dense model" in body["answer"]["since"]["changes"]


def test_a_store_whose_search_data_is_gone_still_reads(client):
    add_collection(client)
    client.app.state.store.drop("c")
    r = client.get("/collections/c")
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 0


def test_rebuild_is_refused_for_an_unknown_or_the_same_model_or_a_busy_store(tmp_path, monkeypatch):
    client = make_client(tmp_path)  # not entered: no worker, so a queued job stays queued
    small_model(client, monkeypatch)
    add_collection(client, lists=False)
    r = reembed(client, model="nope/y")
    assert r.status_code == 422 and r.json()["detail"] == "unknown embedding model nope/y"
    r = reembed(client, model="fake/dense")
    assert r.status_code == 409 and r.json()["detail"] == "'c' is already built with fake/dense"
    assert reembed(client, "nope").status_code == 404
    db.create_job(client.app.state.conn, "load", "c", {})
    r = reembed(client)
    assert r.status_code == 409 and "a load job is queued" in r.json()["detail"]
    r = client.post("/collections/c/reembed", json={"model": "fake/small", "confirm": True})
    assert r.status_code == 422  # nothing to confirm: a rebuild is free


def test_a_stopped_switch_refuses_deleting_a_source_and_copying(tmp_path, monkeypatch):
    client = make_client(tmp_path)  # not entered: the stopped switch stays stopped
    small_model(client, monkeypatch)
    add_collection(client, lists=False)
    app = client.app.state
    app.indexer.add(app.indexer.spec("c"), [Item("x-1", "optics light lens", "D")], "x")
    setup = {"model": "fake/small", "from": "fake/dense", "switching": True}
    db.update_job(app.conn, db.create_job(app.conn, "reembed", "c", setup), status="interrupted")
    stopped = "the switch to fake/small stopped part-way; resume it first"
    r = client.delete("/collections/c/sources/x")
    assert r.status_code == 409 and r.json()["detail"] == stopped
    r = client.post("/collections/c/copy", json={"name": "cc", "recipe": "paper_2024"})
    assert r.status_code == 409 and r.json()["detail"] == stopped
    assert app.store.count("c") == len(ITEMS) + 1 and db.get_collection(app.conn, "cc") is None


def test_a_running_rebuild_reports_its_state(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    small_model(client, monkeypatch)
    add_collection(client, lists=False)
    conn = client.app.state.conn
    job_id = reembed(client).json()["job_id"]
    db.update_job(conn, job_id, status="running", progress=0.5)
    body = client.get("/collections/c").json()
    assert (body["state"], body["progress"]) == ("rebuilding", 0.5)
    assert body["rebuild"]["progress"] == 0.5 and body["rebuild"]["switching"] is False


def test_new_store_names_may_not_end_in_a_search_data_suffix(client):
    client.app.state.worker.stop()
    assert client.post("/collections", json={"name": "x", "qa_type": "mcq"}).status_code == 201
    message = "store names may not end in __rebuild or __chunks"
    r = client.post("/collections", json={"name": "x__rebuild", "qa_type": "mcq"})
    assert r.status_code == 422 and message in r.json()["detail"]
    r = client.post("/collections/x/copy", json={"name": "x__chunks", "recipe": "paper_2024"})
    assert r.status_code == 422 and message in r.json()["detail"]
    assert db.get_collection(client.app.state.conn, "x__chunks") is None


def test_a_stopped_switch_reads_as_rebuilding():
    def job(status, switching):
        setup = {"model": "m", "from": "o", "switching": switching}
        return {"id": "j", "kind": "reembed", "status": status, "progress": 0, "setup": setup}

    for count, lists in ((0, False), (3, True), (3, False)):
        state, behind = collection_state([job("interrupted", True)], count, lists)
        assert state == "rebuilding" and behind["id"] == "j"
    assert collection_state([job("interrupted", False)], 3, True)[0] == "ready"
    assert collection_state([job("done", True)], 3, True)[0] == "ready"


def test_a_documents_store_defaults_to_the_documents_recipe_and_refuses_others(client):
    client.app.state.worker.stop()
    assert add_documents(client, "d")["recipe"] == "documents"
    tagged = {"name": "t", "qa_type": "documents", "recipe": "documents_tagged"}
    assert client.post("/collections", json=tagged).json()["recipe"] == "documents_tagged"
    for name, qa_type, recipe in (("e", "documents", "reference"), ("m", "mcq", "documents")):
        body = {"name": name, "qa_type": qa_type, "recipe": recipe}
        r = client.post("/collections", json=body)
        assert r.status_code == 422, r.text
        assert db.get_collection(client.app.state.conn, name) is None
    r = client.post("/collections", json={"name": "m", "qa_type": "mcq"})
    assert r.json()["recipe"] == "paper_2024"  # the other types keep their default
    r = client.post("/collections/d/copy", json={"name": "d2", "recipe": "paper_2024"})
    assert r.status_code == 422, r.text
    r = client.post("/collections/m/copy", json={"name": "m2", "recipe": "documents_tagged"})
    assert r.status_code == 422, r.text
    assert db.get_collection(client.app.state.conn, "d2") is None


def test_the_store_list_names_each_stores_default_eval_set(client):
    client.app.state.worker.stop()
    for name in ("h", "k", "n"):
        add_documents(client, name)
    body = {"name": "hq", "qa_type": "documents", "store": "h"}
    assert client.post("/eval-sets", json=body).status_code == 201
    assert client.post("/eval-sets", json={"name": "n", "qa_type": "documents"}).status_code == 201
    got = {c["name"]: c["eval_set"] for c in client.get("/collections").json()}
    assert got == {"h": "hq", "k": None, "n": "n"}  # n's is the set named after it


def test_collection_reports_the_newest_load_error(client):
    add_collection(client)
    client.app.state.worker.stop()
    conn = client.app.state.conn

    def load(status, error=""):
        db.update_job(conn, db.create_job(conn, "load", "c", {}), status=status, error=error)
        return client.get("/collections/c").json()["load_error"]

    assert load("failed", "HF said no") == "HF said no"
    assert load("done") is None  # a later try that finished clears it
    assert load("cancelled") is None
    assert load("failed") == "the load failed"
    assert load("interrupted") == "the load was interrupted by a restart"
    assert load("stopped_budget") == "the load stopped at the monthly cap"
    assert load("stopped_odd") == "the load stopped_odd"  # a status with no phrasing yet
