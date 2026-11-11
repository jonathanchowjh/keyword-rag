"""The eval-set routes: sets, questions and versions, the passage pool, reference answers."""

import json
from dataclasses import asdict, replace

from api_helpers import (
    EVAL,
    HAND,
    MCQ,
    OTHER,
    QUESTION,
    TIDES,
    add_collection,
    add_documents,
    add_tags,
    answer_store,
    ask_answer,
    count_ai,
    free_disabled_model,
    handbook,
    ids_of,
    import_items,
    jsonl,
    ledger_purposes,
    long_text,
    make_client,
    new_set,
    passages,
    put_questions,
    stale,
    store_import,
    stored_worked,
    terms_reply,
    wait,
)

from ragqa_api.guards import PRICE_CHANGED
from ragqa_core import db, imports
from ragqa_core.chats import relevance
from ragqa_core.indexer import ModelMismatch
from ragqa_core.jobs import no_ai
from ragqa_core.types import Question, Stale


def test_eval_set_crud_and_versions(client):
    assert client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"}).json() == {"name": "e"}
    assert client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"}).status_code == 409
    assert (
        client.post("/eval-sets", json={"name": "bad name!", "qa_type": "mcq"}).status_code == 422
    )
    first = {**MCQ, "meta": {"skill": "s", "odd": [1]}, "gold_ids": []}
    r = put_questions(client, "e", [first, {**MCQ, "id": "b", "gold_answers": ["B"]}])
    assert r.status_code == 200, r.text
    version = r.json()["version"]
    got = client.get("/eval-sets/e").json()
    assert got["version"] == version and got["qa_type"] == "mcq"
    assert [q["id"] for q in got["questions"]] == ["a", "b"]
    assert got["questions"][0]["meta"] == {"skill": "s", "odd": [1], "source": "hand"}
    assert [v["version"] for v in got["versions"]] == [version]
    assert client.get("/eval-sets").json()[0]["questions"] == 2
    bad = put_questions(client, "e", [{**MCQ, "id": "z", "gold_answers": ["Z"]}], version)
    assert bad.status_code == 422 and "'z'" in bad.json()["detail"]
    repeated = put_questions(client, "e", [MCQ, MCQ], version)
    assert repeated.status_code == 422 and "'a'" in repeated.json()["detail"]
    client.post("/eval-sets", json={"name": "s", "qa_type": "science"})
    plain = {"id": "p", "text": "how warm?", "choices": None, "gold_answers": []}
    assert put_questions(client, "s", [plain]).status_code == 200
    r = client.patch("/eval-sets/e", json={"include_leaked": True})
    assert r.status_code == 200 and r.json()["include_leaked"] is True
    assert client.get("/eval-sets/e").json()["include_leaked"] is True
    assert client.get("/eval-sets/nope").status_code == 404


def test_a_stale_or_busy_save_is_refused_and_writes_nothing(client):
    add_collection(client, questions=False)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    version = put_questions(client, "e", [MCQ]).json()["version"]
    stale = put_questions(client, "e", [MCQ, {**MCQ, "id": "b"}], "old")
    assert stale.status_code == 409
    client.app.state.worker.stop()
    db.set_default_eval_set(client.app.state.conn, "c", "e")
    db.create_job(client.app.state.conn, "load", "c", {"dataset": "scibench"})
    busy = put_questions(client, "e", [MCQ, {**MCQ, "id": "b"}], version)
    assert busy.status_code == 409
    assert client.get("/eval-sets/e").json()["version"] == version
    assert [q.id for q in db.get_questions(client.app.state.conn, "e")] == ["a"]


def test_eval_set_import_runs_in_the_request(client, monkeypatch):
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    client.post("/collections", json={"name": "e", "qa_type": "mcq"})
    rows = jsonl(
        *(
            {"id": f"u{i}", "text": f"question {i}?", "choices": ["x", "y"], "answer": "A"}
            for i in range(3)
        )
    )
    r = client.post("/eval-sets/e/uploads", params={"filename": "quiz.jsonl"}, content=rows)
    assert r.status_code == 200, r.text
    up = r.json()
    assert up["preview"]["ready"] and up["preview"]["usable"] == 3 and up["source_name"] == "quiz"
    from ragqa_core import imports

    seen = []
    real = imports.preview
    monkeypatch.setattr(imports, "preview", lambda *a, **k: seen.append(a[-1]) or real(*a, **k))
    named = client.post(
        "/eval-sets/e/uploads", params={"filename": "Q3 (final).jsonl"}, content=rows
    )
    assert seen == ["upload:Q3 -final"] and named.json()["source_name"] == "Q3 -final"
    body = {
        "upload_id": up["upload_id"],
        "mapping": up["preview"]["mapping"],
        "source_name": "quiz",
    }
    done = client.post("/eval-sets/e/imports", json=body)
    assert done.status_code == 200, done.text
    out = done.json()
    assert {k: out[k] for k in ("new", "updated", "unchanged", "skipped")} == {
        "new": 3,
        "updated": 0,
        "unchanged": 0,
        "skipped": 0,
    }
    assert out["version"] == client.get("/eval-sets/e").json()["version"]
    assert client.get("/jobs").json() == []
    got = client.get("/eval-sets/e").json()["questions"]
    assert {q["meta"]["source"] for q in got} == {"upload:quiz"} and len(got) == 3
    again = client.post("/eval-sets/e/imports", json=body).json()
    assert again["unchanged"] == 3 and again["new"] == 0
    # a store of the same name does not see the upload
    assert client.post("/collections/e/imports", json=body).status_code == 404


def test_deleting_an_eval_set_deletes_its_runs_and_a_store_keeps_its_sets(client):
    add_collection(client)
    conn = client.app.state.conn
    with conn.lock:  # the worker can't claim the run between the two calls
        run = db.create_job(conn, "eval", "c", {"eval_set": "c"})
        db.update_job(conn, run, status="done")
    r = client.delete("/eval-sets/c")
    assert r.status_code == 200 and r.json() == {"runs": 1}
    assert db.get_job(conn, run) is not None and db.get_eval_set(conn, "c") is not None
    assert client.delete("/eval-sets/c", params={"confirm": True}).status_code == 204
    assert db.get_job(conn, run) is None and db.get_eval_set(conn, "c") is None
    assert client.get("/collections/c").json()["eval_set"] is None
    assert client.delete("/eval-sets/c", params={"confirm": True}).status_code == 404
    add_collection(client, name="d")
    assert client.delete("/collections/d").status_code == 204
    assert db.get_eval_set(conn, "d") is not None


def test_deleting_an_eval_set_is_refused_while_its_run_is_queued(client):
    add_collection(client)
    client.app.state.worker.stop()
    conn = client.app.state.conn
    run = db.create_job(conn, "eval", "c", {"eval_set": "c"})
    assert client.delete("/eval-sets/c", params={"confirm": True}).status_code == 409
    assert db.get_job(conn, run) is not None and db.get_eval_set(conn, "c") is not None


def test_an_eval_set_is_not_deleted_or_imported_into_while_a_load_writes_to_it(client):
    add_collection(client)
    client.app.state.worker.stop()
    conn = client.app.state.conn
    db.create_job(conn, "load", "c", {"dataset": "musique", "limit": None})
    assert client.delete("/eval-sets/c", params={"confirm": True}).status_code == 409
    assert db.get_eval_set(conn, "c") is not None
    body = {"upload_id": "0" * 32, "source_name": "qs"}
    assert client.post("/eval-sets/c/imports", json=body).status_code == 409


def test_a_hand_written_question_is_checked_for_leaks_per_store(client):
    add_collection(client, questions=False)
    import_items(client, body=TIDES)
    add_collection(client, name="d", questions=False)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    kept = {**MCQ, "id": "k1", "text": "already here", "meta": {}}
    assert put_questions(client, "e", [kept]).status_code == 200
    version = client.get("/eval-sets/e").json()["version"]
    r = put_questions(client, "e", [kept, HAND, OTHER], version)
    assert r.status_code == 200, r.text
    meta = {q["id"]: q["meta"] for q in client.get("/eval-sets/e").json()["questions"]}
    assert meta["h1"]["source"] == "hand" and meta["h2"]["source"] == "hand"
    assert meta["k1"] == {}  # not new to the set: stored as sent
    for store, excluded, asked in (("c", 1, 2), ("d", 0, 3)):
        body = {**EVAL, "collection": store, "eval_set": "e", "confirm": True}
        r = client.post("/eval", json=body)
        assert r.status_code == 202, r.text
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        assert job["setup"]["leaked_excluded"] == excluded
        assert len(job["setup"]["question_ids"]) == asked


def test_a_dataset_question_put_back_is_not_marked_hand_written(client):
    add_collection(client, lists=False)  # its eval set c holds QUESTION, as a dataset left it
    assert put_questions(client, "c", []).status_code == 200
    restored = {**asdict(QUESTION), "meta": {"skill": "enzymes"}}
    assert put_questions(client, "c", [restored]).status_code == 200
    assert client.get("/eval-sets/c").json()["questions"][0]["meta"] == {"skill": "enzymes"}


def test_a_store_and_an_eval_set_of_one_name_have_one_type(client):
    client.post("/eval-sets", json={"name": "x", "qa_type": "science"})
    r = client.post("/collections", json={"name": "x", "qa_type": "mcq"})
    assert r.status_code == 409
    assert r.json()["detail"] == "eval set 'x' is science; store 'x' is mcq"
    assert client.post("/collections", json={"name": "x", "qa_type": "science"}).status_code == 201
    client.post("/collections", json={"name": "y", "qa_type": "mcq"})
    r = client.post("/eval-sets", json={"name": "y", "qa_type": "table"})
    assert r.status_code == 409
    assert r.json()["detail"] == "eval set 'y' is table; store 'y' is mcq"
    assert client.post("/eval-sets", json={"name": "y", "qa_type": "mcq"}).status_code == 201


def test_an_empty_question_id_is_refused(client):
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    for blank in ("", "  "):
        r = put_questions(client, "e", [{**MCQ, "id": blank}])
        assert r.status_code == 422 and r.json()["detail"].endswith(": no id")
    assert client.get("/eval-sets/e").json()["questions"] == []


def test_stale_confirm_with_the_model_off_is_409_too(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Ten days [1].") as client:
        add_collection(client, summarize=True, name="s")
        # Summaries made by another model: a copy that spends, so it needs the model.
        copy = {"name": "x", "recipe": "paper_2024", "build": {"added_text": "summary"}}
        copy_seen = client.post("/collections/s/copy", json=copy).json()["estimate"]
        stored_worked(client)
        terms, vocabulary = ["enzymes", "markets"], "/collections/w/vocabulary"
        vocab_seen = client.put(vocabulary, json={"terms": terms}).json()["estimate"]
        ids = answer_store(client)
        answer_seen = ask_answer(client, ids[:2]).json()["estimate"]
        client.app.state.worker.stop()  # a confirm let through would queue, not run
        # Off: a fresh confirm would be 503.
        free_disabled_model(client, "summary", "keywords", "answer")
        r = client.post("/collections/s/copy", json={**copy, **stale(copy_seen)})
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        r = client.put(vocabulary, json={"terms": terms, **stale(vocab_seen)})
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        r = ask_answer(client, ids[:2], **stale(answer_seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED


def test_put_questions_keeps_judged_ids(client):
    client.post("/eval-sets", json={"name": "d", "qa_type": "documents"})
    labelled = {"id": "a", "text": "one?", "judged_ids": ["a"], "gold_ids": ["b"]}
    r = put_questions(client, "d", [labelled, {"id": "b", "text": "two?"}])
    assert r.status_code == 200, r.text
    got = {q["id"]: q for q in client.get("/eval-sets/d").json()["questions"]}
    assert got["a"]["judged_ids"] == ["a", "b"]
    assert got["b"]["judged_ids"] is None


def test_a_documents_set_round_trips_through_its_json(client, tmp_path):
    for name in ("d1", "d2"):
        client.post("/eval-sets", json={"name": name, "qa_type": "documents"})
    qs = [
        {"id": "a", "text": "one?", "gold_ids": ["p1"], "judged_ids": ["p1", "p2"]},
        {
            "id": "b",
            "text": "two?",
            "gold_ids": ["p3"],
            "judged_ids": ["p3"],
            "gold_answers": ["x"],
        },
        {"id": "c", "text": "three?"},
    ]
    assert put_questions(client, "d1", qs).status_code == 200
    first = client.get("/eval-sets/d1").json()["questions"]
    (tmp_path / "qs.json").write_text(json.dumps(first))
    up = client.post(
        "/eval-sets/d2/uploads",
        params={"filename": "qs.json"},
        content=(tmp_path / "qs.json").read_bytes(),
    )
    assert up.status_code == 200, up.text
    body = {
        "upload_id": up.json()["upload_id"],
        "mapping": up.json()["preview"]["mapping"],
        "source_name": "qs",
    }
    assert client.post("/eval-sets/d2/imports", json=body).status_code == 200
    second = client.get("/eval-sets/d2").json()["questions"]
    keys = ("id", "text", "gold_ids", "judged_ids", "gold_answers")
    assert [{k: q[k] for k in keys} for q in second] == [{k: q[k] for k in keys} for q in first]
    assert second[2]["judged_ids"] is None


def test_a_label_saved_while_an_import_merges_is_never_overwritten(client, monkeypatch):
    import threading

    new_set(client, "d", "documents")
    q = {"id": "q1", "text": "one?", "gold_ids": ["p1"], "judged_ids": ["p1"]}
    assert put_questions(client, "d", [q]).status_code == 200
    read = Question(**client.get("/eval-sets/d").json()["questions"][0])
    conn = client.app.state.conn
    counts, saved, threads = db.question_counts, [], []

    def label() -> None:  # a labeller's PUT of q1, sent while the import is merging
        marked = replace(read, gold_ids=["p9"], judged_ids=["p1", "p9"])
        try:
            db.upsert_question(conn, "d", marked, read)
            saved.append("saved")
        except Stale:
            saved.append("refused")

    def merge_slowly(conn, name, questions):
        out = counts(conn, name, questions)
        threads.append(threading.Thread(target=label))
        threads[0].start()
        threads[0].join(0.3)  # time for the PUT to land between the merge and its write
        return out

    monkeypatch.setattr(db, "question_counts", merge_slowly)
    row = json.dumps({"id": "q1", "text": "one?", "gold_ids": ["p2"]})
    up = client.post("/eval-sets/d/uploads", params={"filename": "qs.jsonl"}, content=row)
    assert up.status_code == 200, up.text
    body = {
        "upload_id": up.json()["upload_id"],
        "mapping": up.json()["preview"]["mapping"],
        "source_name": "qs",
    }
    assert client.post("/eval-sets/d/imports", json=body).status_code == 200
    threads[0].join(5)
    assert saved == ["refused"]  # it waited for the import, then found q1 had changed
    assert client.get("/eval-sets/d").json()["questions"][0]["gold_ids"] == ["p1", "p2"]


def test_creating_a_set_for_a_store_links_both_ways(client):
    client.app.state.worker.stop()
    add_documents(client, "h")
    body = {"name": "hq", "qa_type": "documents", "store": "h"}
    assert client.post("/eval-sets", json=body).status_code == 201
    assert client.get("/eval-sets/hq").json()["store"] == "h"
    assert client.get("/collections/h").json()["eval_set"] == "hq"
    assert client.post("/eval-sets", json={**body, "name": "hq2"}).status_code == 201
    assert client.get("/eval-sets/hq2").json()["store"] == "h"
    assert client.get("/collections/h").json()["eval_set"] == "hq"  # the first stays


def test_creating_a_linked_set_checks_the_store(client):
    client.app.state.worker.stop()
    add_documents(client, "h")
    add_collection(client, name="m", questions=False, lists=False)
    body = {"name": "x", "qa_type": "documents", "store": "nope"}
    assert client.post("/eval-sets", json=body).status_code == 404
    r = client.post("/eval-sets", json={**body, "store": "m"})
    assert r.status_code == 409
    assert r.json()["detail"] == "eval set 'x' is documents; store 'm' is mcq"
    db.create_job(client.app.state.conn, "load", "h", {"dataset": "scienceqa"})
    r = client.post("/eval-sets", json={**body, "store": "h"})
    assert r.status_code == 409
    assert r.json()["detail"] == "a load is under way on 'h'; wait"
    assert db.get_eval_set(client.app.state.conn, "x") is None


def test_patch_eval_set_links_and_unlinks_a_store(client):
    client.app.state.worker.stop()
    add_documents(client, "h")
    add_collection(client, name="m", questions=False, lists=False)
    client.post("/eval-sets", json={"name": "e", "qa_type": "documents"})
    assert client.get("/eval-sets/e").json()["store"] is None
    r = client.patch("/eval-sets/e", json={"store": "h"})
    assert r.status_code == 200 and r.json()["store"] == "h"
    r = client.patch("/eval-sets/e", json={"include_leaked": True})
    assert r.json()["store"] == "h" and r.json()["include_leaked"] is True
    r = client.patch("/eval-sets/e", json={"store": None})
    assert r.status_code == 200 and r.json()["store"] is None
    assert r.json()["include_leaked"] is True
    for bad in ({"include_leaked": None}, {"nonsense": 1}):
        assert client.patch("/eval-sets/e", json=bad).status_code == 422
    r = client.patch("/eval-sets/e", json={"include_leaked": None})
    assert r.json()["detail"] == "include_leaked cannot be null: leave it out to keep it"
    assert client.patch("/eval-sets/e", json={"store": "nope"}).status_code == 404
    r = client.patch("/eval-sets/e", json={"store": "m"})
    assert r.status_code == 409
    assert r.json()["detail"] == "eval set 'e' is documents; store 'm' is mcq"
    assert client.get("/eval-sets/e").json()["store"] is None


def test_deleting_a_store_unlinks_its_eval_sets_and_keeps_their_labels(client):
    client.app.state.worker.stop()
    add_documents(client, "h")
    body = {"name": "hq", "qa_type": "documents", "store": "h"}
    assert client.post("/eval-sets", json=body).status_code == 201
    q = {"id": "q1", "text": "where?", "gold_ids": ["a"], "judged_ids": ["a", "b"]}
    assert put_questions(client, "hq", [q]).status_code == 200
    assert client.delete("/collections/h").status_code == 204
    got = client.get("/eval-sets/hq").json()
    assert got["store"] is None
    assert (got["questions"][0]["gold_ids"], got["questions"][0]["judged_ids"]) == (
        ["a"],
        ["a", "b"],
    )


def test_get_eval_set_names_labelled_passages_no_longer_in_its_store(client):
    add_collection(client, questions=False, lists=False)
    old = handbook("")
    store_import(client, "x.md", old)
    ids = ids_of("x.md", old)
    body = {"name": "lq", "qa_type": "mcq", "store": "c"}
    assert client.post("/eval-sets", json=body).status_code == 201
    assert client.get("/eval-sets/lq").json()["missing"] == {}
    mcq = {"choices": ["x", "y"], "gold_answers": ["A"]}
    qs = [
        {"id": "q1", "text": "where?", "gold_ids": [ids[0]], **mcq},
        {"id": "q2", "text": "none?", **mcq},
    ]
    r = put_questions(client, "lq", qs)
    assert r.status_code == 200, r.text
    got = client.get("/eval-sets/lq").json()
    assert got["missing"] == {}
    store_import(client, "x.md", handbook("A new first paragraph."))
    got = client.get("/eval-sets/lq").json()
    assert got["missing"] == {"q1": [ids[0]]}
    assert got["questions"][0]["gold_ids"] == [ids[0]]
    client.patch("/eval-sets/lq", json={"store": None})
    assert client.get("/eval-sets/lq").json()["missing"] == {}


def test_an_eval_set_whose_store_lost_its_search_data_lists_every_passage_missing(client):
    add_collection(client, questions=False, lists=False)
    body = handbook("")
    store_import(client, "x.md", body)
    ids = ids_of("x.md", body)
    new = {"name": "lq", "qa_type": "mcq", "store": "c"}
    assert client.post("/eval-sets", json=new).status_code == 201
    mcq = {"choices": ["x", "y"], "gold_answers": ["A"]}
    qs = [
        {"id": "q1", "text": "where?", "gold_ids": [ids[0]], **mcq},
        {"id": "q2", "text": "and here?", "gold_ids": [ids[1], ids[2]], **mcq},
        {"id": "q3", "text": "none?", **mcq},
    ]
    assert put_questions(client, "lq", qs).status_code == 200
    client.app.state.store.drop("c")
    r = client.get("/eval-sets/lq")
    assert r.status_code == 200, r.text
    assert r.json()["missing"] == {"q1": [ids[0]], "q2": [ids[1], ids[2]]}


def qpath(name, qid=None):
    return f"/eval-sets/{name}/questions" + (f"/{qid}" if qid else "")


def test_post_question_names_it_by_its_text(client):
    new_set(client)
    body = {
        "question": {"text": "How long is leave?", "choices": ["x", "y"], "gold_answers": ["A"]}
    }
    r = client.post(qpath("e"), json=body)
    assert r.status_code == 201, r.text
    want = imports.make_id("q-", "How long is leave?")
    assert r.json()["id"] == want
    again = client.post(qpath("e"), json=body)
    assert again.status_code == 201 and again.json()["id"] == want + "-2"
    got = client.get("/eval-sets/e").json()
    assert got["version"] == again.json()["version"]
    assert {q["id"]: q["meta"]["source"] for q in got["questions"]} == {
        want: "hand",
        want + "-2": "hand",
    }
    sent = {"question": {**body["question"], "meta": {"source": "upload:x"}, "text": "other?"}}
    r = client.post(qpath("e"), json=sent)
    q = next(q for q in client.get("/eval-sets/e").json()["questions"] if q["id"] == r.json()["id"])
    assert q["meta"]["source"] == "upload:x"
    assert (
        client.post(qpath("e"), json={"question": {**body["question"], "text": ""}}).status_code
        == 422
    )
    assert client.post(qpath("nope"), json=body).status_code == 404


def test_two_labellers_on_different_questions_both_keep_their_marks(client):
    new_set(client)
    q1, q2 = MCQ, {**MCQ, "id": "b"}
    assert put_questions(client, "e", [q1, q2]).status_code == 200
    read = {q["id"]: q for q in client.get("/eval-sets/e").json()["questions"]}
    n1 = {**read["a"], "gold_ids": ["p1"], "judged_ids": ["p1", "p2"]}
    n2 = {**read["b"], "gold_ids": ["p3"], "judged_ids": ["p3"]}
    r1 = client.put(qpath("e", "a"), json={"question": n1, "base": read["a"]})
    r2 = client.put(qpath("e", "b"), json={"question": n2, "base": read["b"]})
    assert r1.status_code == 200 and r2.status_code == 200, (r1.text, r2.text)
    got = {q["id"]: q for q in client.get("/eval-sets/e").json()["questions"]}
    assert got["a"]["judged_ids"] == ["p1", "p2"] and got["b"]["gold_ids"] == ["p3"]
    assert client.get("/eval-sets/e").json()["version"] == r2.json()["version"]


def test_a_stale_question_save_is_409_and_changes_nothing(client):
    new_set(client)
    assert put_questions(client, "e", [MCQ]).status_code == 200
    got = client.get("/eval-sets/e").json()
    read = got["questions"][0]
    first = {**read, "gold_ids": ["p1"]}
    assert client.put(qpath("e", "a"), json={"question": first, "base": read}).status_code == 200
    second = {**read, "gold_ids": ["p2"]}
    r = client.put(qpath("e", "a"), json={"question": second, "base": read})
    assert r.status_code == 409 and "changed by someone else" in r.json()["detail"]
    assert client.get("/eval-sets/e").json()["questions"][0]["gold_ids"] == ["p1"]
    old = put_questions(client, "e", [MCQ], got["version"])
    assert old.status_code == 409


def test_put_or_delete_of_a_gone_question_is_404(client):
    new_set(client)
    assert put_questions(client, "e", [MCQ]).status_code == 200
    read = client.get("/eval-sets/e").json()["questions"][0]
    assert client.delete(qpath("e", "a")).status_code == 200
    gone = "This question was deleted since you opened it"
    r = client.put(qpath("e", "a"), json={"question": read, "base": read})
    assert r.status_code == 404 and r.json()["detail"] == gone
    r = client.put(qpath("e", "other"), json={"question": read, "base": read})
    assert r.status_code == 422 and "must match the address" in r.json()["detail"]
    r = client.delete(qpath("e", "a"))
    assert r.status_code == 404 and r.json()["detail"] == gone
    assert client.delete(qpath("nope", "a")).status_code == 404


def test_a_question_id_with_a_slash_saves_and_deletes_one_at_a_time(client):
    from urllib.parse import quote

    new_set(client, "d", "documents")
    one = {"id": "hr/leave-1", "text": "how long is leave?"}
    two = {"id": "hr/leave-2", "text": "who approves leave?"}
    assert put_questions(client, "d", [one, two]).status_code == 200
    read = {q["id"]: q for q in client.get("/eval-sets/d").json()["questions"]}
    marked = {**read["hr/leave-1"], "gold_ids": ["x.md#aaaaaaaaaaaa"]}
    r = client.put(
        qpath("d", quote("hr/leave-1", safe="")),
        json={"question": marked, "base": read["hr/leave-1"]},
    )
    assert r.status_code == 200, r.text
    got = {q["id"]: q for q in client.get("/eval-sets/d").json()["questions"]}
    assert got["hr/leave-1"]["gold_ids"] == ["x.md#aaaaaaaaaaaa"]
    r = client.delete(qpath("d", quote("hr/leave-2", safe="")))
    assert r.status_code == 200, r.text
    assert [q["id"] for q in client.get("/eval-sets/d").json()["questions"]] == ["hr/leave-1"]
    gone = client.delete(qpath("d", quote("hr/leave-2", safe="")))
    assert gone.status_code == 404
    assert gone.json()["detail"] == "This question was deleted since you opened it"


def test_question_writes_wait_for_a_load(client):
    add_collection(client)
    client.app.state.worker.stop()
    conn = client.app.state.conn
    db.create_job(conn, "load", "c", {"dataset": "musique", "limit": None})
    stored = client.get("/eval-sets/c").json()["questions"][0]
    new = {"question": {"text": "x?", "choices": ["x", "y"], "gold_answers": ["A"]}}
    assert client.post(qpath("c"), json=new).status_code == 409
    put = {"question": stored, "base": stored}
    assert client.put(qpath("c", stored["id"]), json=put).status_code == 409
    assert client.delete(qpath("c", stored["id"])).status_code == 409


def test_get_eval_set_reads_an_old_version(client):
    new_set(client)
    v1 = put_questions(client, "e", [MCQ]).json()["version"]
    v2 = put_questions(client, "e", [MCQ, {**MCQ, "id": "b"}], v1).json()["version"]
    got = client.get("/eval-sets/e", params={"version": v1}).json()
    assert [q["id"] for q in got["questions"]] == ["a"]
    assert got["version"] == v2 and got["questions_version"] == v1
    now = client.get("/eval-sets/e").json()
    assert now["questions_version"] == v2 and len(now["questions"]) == 2
    r = client.get("/eval-sets/e", params={"version": "nope"})
    assert r.status_code == 404 and r.json()["detail"] == "That version of 'e' doesn't exist"


def test_restoring_an_old_version_puts_its_questions_back(client):
    new_set(client)
    conn = client.app.state.conn
    q = Question("a", "which one?", ["x", "y"], ["A"], ["p1"], judged_ids=["p1", "p2"])
    db.save_questions(conn, "e", [q])
    v1 = client.get("/eval-sets/e").json()["version"]
    changed = {**MCQ, "text": "changed?"}
    cur = put_questions(client, "e", [changed], v1).json()["version"]
    assert cur != v1
    old = client.get("/eval-sets/e", params={"version": v1}).json()["questions"]
    r = put_questions(client, "e", old, cur)
    assert r.status_code == 200 and r.json()["version"] == v1
    got = client.get("/eval-sets/e").json()["questions"]
    assert got[0]["judged_ids"] == ["p1", "p2"] and "source" not in got[0]["meta"]


def test_restoring_a_version_from_before_judged_ids_leaves_them_unknown(client):
    new_set(client)
    row = [
        {
            "id": "a",
            "text": "t",
            "choices": ["x", "y"],
            "gold_answers": ["A"],
            "gold_ids": [],
            "meta": {},
        }
    ]
    client.app.state.conn.execute(
        "insert into eval_set_versions values (?, ?, ?, ?)", ("e", "old", json.dumps(row), "now")
    )
    old = client.get("/eval-sets/e", params={"version": "old"}).json()["questions"]
    assert put_questions(client, "e", old).status_code == 200
    assert [q["judged_ids"] for q in client.get("/eval-sets/e").json()["questions"]] == [None]


POOL_WORDS = [
    "alpha",
    "brine",
    "cocoa",
    "delta",
    "ember",
    "fjord",
    "gusto",
    "haven",
    "irony",
    "jumbo",
    "kiosk",
    "lemon",
    "mango",
    "nylon",
    "otter",
]


POOL_KEYS = {"id", "text", "file", "part", "parts", "relevance"}


POOL_LOADING_NOTE = (
    "The ranking model is still loading, so passages are in search order, not best first"
)


POOL_UNAVAILABLE_NOTE = (
    "The ranking model is unavailable, so passages are in search order, not best first"
)


KEYWORD_TAGS_NOTE = (
    "Keyword-tag search was left out: its tags would use AI, and Find passages is free"
)


FREE = ["vector", "keyword", "hybrid"]


def pool_store(client) -> None:
    """Store c: a Documents store of 15 one-word passages from x.md, and the Documents eval
    set s labelled against it."""
    add_documents(client)
    store_import(client, "x.md", long_text(POOL_WORDS))
    assert client.app.state.store.count("c") == 15
    r = client.post("/eval-sets", json={"name": "s", "qa_type": "documents", "store": "c"})
    assert r.status_code == 201, r.text


def count_reranks(client) -> list[str]:
    models, calls = client.app.state.models, []
    real = models.rerank

    def rerank(query, texts, model=None):
        calls.append(query)
        return real(query, texts, model)

    models.rerank = rerank
    return calls


def find_passages(client, question: str, labelled: list[str] | None = None, name="s") -> dict:
    body = {"question": question, "labelled": labelled or []}
    r = client.post(f"/eval-sets/{name}/pool", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def top_ids(client, question: str, method: str) -> list[str]:
    return [h.id for h in client.app.state.retriever.retrieve("c", question, method, 10)]


def test_pool_takes_the_top_10_of_each_free_method_once_in_rerank_order(client):
    pool_store(client)
    calls = count_ai(client)
    retriever = client.app.state.retriever
    real = retriever.retrieve
    searched = set()

    def retrieve(collection, question, method, k=5, *args, **kw):
        searched.add((method, k))
        return real(collection, question, method, k, *args, **kw)

    retriever.retrieve = retrieve
    question = "alpha brine cocoa"
    body = find_passages(client, question)
    assert searched == {("vector", 10), ("keyword", 10), ("hybrid", 10)}
    assert body["store"] == "c" and body["methods"] == FREE
    assert body["merged_by"] == "rerank" and body["notes"] == []
    found = body["passages"]
    ids = [p["id"] for p in found]
    assert len(ids) == len(set(ids)) >= 10
    scores = client.app.state.models.rerank(question, [p["text"] for p in found])
    assert scores == sorted(scores, reverse=True) and scores[0] > 0
    assert [p["relevance"] for p in found] == [relevance(s) for s in scores]
    assert all(set(p) == POOL_KEYS and p["file"] == "x.md" for p in found)
    assert all(p["parts"] == 15 and 1 <= p["part"] <= 15 for p in found)
    assert body["labelled"] == [] and body["missing"] == []
    assert calls == []


def test_pool_adds_keyword_rag_only_once_tags_are_built(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        pool_store(client)
        r = add_tags(client, True)
        assert r.status_code == 202, r.text
        assert wait(client, r.json()["job_id"])["status"] == "done"
        calls = count_ai(client)
        body = find_passages(client, "alpha brine")
        assert body["methods"] == [*FREE, "keyword_rag"] and body["notes"] == []
        conn = client.app.state.conn
        newest = db.list_jobs(conn, kind="tag_build", collection="c")[0]
        db.update_job(conn, newest["id"], status="interrupted")
        body = find_passages(client, "alpha brine")
        assert body["methods"] == FREE and body["notes"] == []
        assert calls == []


def test_pool_leaves_out_keyword_rag_when_its_tags_would_call_ai(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        pool_store(client)
        r = add_tags(client, True)
        assert wait(client, r.json()["job_id"])["status"] == "done"
        r = client.put("/collections/c/search", json={"query_tags": "model"})
        assert r.status_code == 200, r.text
        calls = count_ai(client)
        body = find_passages(client, "alpha brine")
        assert body["methods"] == FREE and body["notes"] == [KEYWORD_TAGS_NOTE]
        assert body["passages"]
        client.app.state.ai.set_role("query", None)
        body = find_passages(client, "alpha brine")
        assert body["methods"] == FREE and body["notes"] == [KEYWORD_TAGS_NOTE]
        assert calls == []


def test_pool_searches_keyword_rag_with_the_settings_it_checked(tmp_path):
    from ragqa_core.types import SearchSettings

    with make_client(tmp_path, reply=terms_reply) as client:
        pool_store(client)
        r = add_tags(client, True)
        assert wait(client, r.json()["job_id"])["status"] == "done"
        chats, conn = client.app.state.chats, client.app.state.conn
        why_not = chats.why_not

        def flip(spec, method):  # PUT /collections/c/search {"query_tags": "model"} mid-pool
            if method == "keyword_rag":
                search = spec.search or SearchSettings()
                db.set_search(conn, "c", replace(search, query_tags="model"))
            return why_not(spec, method)

        chats.why_not = flip
        calls = count_ai(client)
        body = find_passages(client, "alpha brine")
        assert body["methods"] == [*FREE, "keyword_rag"] and body["notes"] == []
        assert calls == []
        assert db.get_collection(conn, "c").search.query_tags == "model"


def test_pool_falls_back_to_search_order_while_the_reranker_loads(client):
    pool_store(client)
    models = client.app.state.models
    reranked = count_reranks(client)
    question = "alpha brine"
    lists = {m: top_ids(client, question, m) for m in FREE}
    expected: list[str] = []
    for i in range(10):
        for m in FREE:
            if i < len(lists[m]) and lists[m][i] not in expected:
                expected.append(lists[m][i])
    for status, note in (("loading", POOL_LOADING_NOTE), ("failed", POOL_UNAVAILABLE_NOTE)):
        models.rerank_status = lambda model=None, status=status: status
        body = find_passages(client, question)
        assert body["merged_by"] == "rank" and body["notes"] == [note], status
        assert [p["id"] for p in body["passages"]] == expected
        assert all(p["relevance"] is None for p in body["passages"])
    assert reranked == []


def test_pool_on_an_empty_store_finds_nothing_and_never_reranks(client):
    add_documents(client)
    r = client.post("/eval-sets", json={"name": "s", "qa_type": "documents", "store": "c"})
    assert r.status_code == 201, r.text
    reranked = count_reranks(client)
    body = find_passages(client, "anything")
    assert body["passages"] == [] and body["notes"] == ["No passages found in c"]
    assert reranked == []


def test_pool_leaves_out_a_method_that_fails_with_a_note(client):
    pool_store(client)
    question = "alpha brine"
    wanted = set(top_ids(client, question, "keyword")) | set(top_ids(client, question, "hybrid"))
    retriever = client.app.state.retriever
    real = retriever.retrieve

    def retrieve(collection, question, method, *args, **kw):
        if method == "vector":
            raise ModelMismatch("x")
        return real(collection, question, method, *args, **kw)

    retriever.retrieve = retrieve
    body = find_passages(client, question)
    assert body["methods"] == ["keyword", "hybrid"]
    assert body["notes"] == ["Vector search was left out: x"]
    assert {p["id"] for p in body["passages"]} == wanted


def test_pool_returns_labelled_passages_it_did_not_find_and_names_missing_ones(client):
    pool_store(client)
    every = sorted(h.id for h in passages(client))
    pooled = find_passages(client, "alpha")["passages"][0]["id"]
    gone = "x.md#000000000000"
    labelled = [pooled, gone, *[i for i in every if i != pooled]]
    body = find_passages(client, "alpha", labelled)
    found = [p["id"] for p in body["passages"]]
    extra = [p["id"] for p in body["labelled"]]
    assert extra and sorted(found + extra) == every  # each held id once
    assert extra == [i for i in labelled if i in set(extra)]  # in the order given
    assert all(set(p) == POOL_KEYS and p["relevance"] is None for p in body["labelled"])
    assert all(p["file"] == "x.md" for p in body["labelled"])
    assert body["missing"] == [gone]


def test_pool_refuses_without_a_documents_set_and_store(client):
    pool_store(client)
    assert client.post("/eval-sets", json={"name": "m", "qa_type": "mcq"}).status_code == 201
    assert client.post("/eval-sets", json={"name": "d", "qa_type": "documents"}).status_code == 201
    for name, status, detail in (
        ("m", 409, "Find passages is for Documents eval sets"),
        ("d", 409, "pick a store for this eval set first"),
        ("nope", 404, "no eval set 'nope'"),
    ):
        r = client.post(f"/eval-sets/{name}/pool", json={"question": "alpha"})
        assert (r.status_code, r.json()["detail"]) == (status, detail), name
    for bad in ({"question": ""}, {"question": "alpha", "x": 1}, {}):
        assert client.post("/eval-sets/s/pool", json=bad).status_code == 422, bad


def record_purposes(client) -> list[str]:
    """Every complete_many purpose from here on; the calls still go through."""
    ai, seen = client.app.state.ai, []
    real = ai.complete_many

    def many(prompts, **kw):
        seen.append(kw["purpose"])
        return real(prompts, **kw)

    ai.complete_many = many
    return seen


def test_reference_answer_estimate_spends_nothing_and_confirm_makes_one_call(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Ten days [1].") as client:
        ids = answer_store(client)
        before = client.get("/eval-sets/cq").json()
        seen = record_purposes(client)
        r = ask_answer(client, ids[:2])
        assert r.status_code == 200, r.text
        estimate = r.json()["estimate"]
        assert estimate["calls"] == 1 and estimate["est_usd"] > 0
        assert seen == []
        r = ask_answer(client, ids[:2], True)
        assert r.status_code == 200, r.text
        assert r.json()["answer"] == "Ten days." and r.json()["usd"] >= 0
        assert seen == ["reference_answer"]
        r = ask_answer(client, ids[:2])
        assert r.json()["estimate"] == {
            "est_usd": 0,
            "calls": 0,
            "roles": ["answer"],
            "missing": [],
        }
        assert seen == ["reference_answer"]
        after = client.get("/eval-sets/cq").json()
        assert (after["version"], after["questions"]) == (before["version"], before["questions"])


def test_reference_answer_over_the_cap_is_402_and_saves_nothing(tmp_path):
    with make_client(tmp_path) as client:
        ids = answer_store(client)
        before = client.get("/eval-sets/cq").json()
        client.app.state.ai.set_cap(0)
        assert ask_answer(client, ids[:1], True).status_code == 402
        assert client.get("/eval-sets/cq").json() == before


def test_reference_answer_with_no_answer_model(tmp_path):
    with make_client(tmp_path) as client:
        ids = answer_store(client)
        client.app.state.ai.set_role("default", None)
        assert ask_answer(client, ids[:1]).json()["estimate"]["est_usd"] is None
        r = ask_answer(client, ids[:1], True)
        assert r.status_code == 503 and r.json()["detail"] == no_ai("answer")
        assert client.get("/eval-sets/cq").json()["answer_available"] is False


def test_reference_answer_refuses_a_changed_price(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Ten days [1].") as client:
        ids = answer_store(client)  # store c and Documents set cq
        seen = ask_answer(client, ids[:2]).json()["estimate"]
        r = ask_answer(client, ids[:2], **stale(seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        assert ledger_purposes(client) == []
        assert ask_answer(client, ids[:2], True).status_code == 200


def test_reference_answer_confirm_with_the_shown_price_is_accepted(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Ten days [1].") as client:
        ids = answer_store(client)
        seen = ask_answer(client, ids[:2]).json()["estimate"]
        shown = {"calls": seen["calls"], "est_usd": seen["est_usd"]}
        r = ask_answer(client, ids[:2], True, **shown)
        assert r.status_code == 200 and r.json()["answer"] == "Ten days.", r.text


def test_reference_answer_refuses_what_it_cannot_answer(client):
    ids = answer_store(client)
    assert client.post("/eval-sets", json={"name": "m", "qa_type": "mcq"}).status_code == 201
    assert client.post("/eval-sets", json={"name": "d", "qa_type": "documents"}).status_code == 201
    seen = record_purposes(client)
    for name, status, detail in (
        ("m", 409, "The AI's answer is for Documents eval sets"),
        ("d", 409, "pick a store for this eval set first"),
        ("nope", 404, "no eval set 'nope'"),
    ):
        r = ask_answer(client, ids[:1], True, name=name)
        assert (r.status_code, r.json()["detail"]) == (status, detail), name
    assert ask_answer(client, []).status_code == 422
    assert ask_answer(client, [f"p{i}" for i in range(21)]).status_code == 422
    assert ask_answer(client, ids[:1], extra=1).status_code == 422
    r = ask_answer(client, ids[:1], question="  ")
    assert (r.status_code, r.json()["detail"]) == (422, "a question needs some text")
    r = ask_answer(client, ["gone"], True)
    assert (r.status_code, r.json()["detail"]) == (
        422,
        "mark at least one passage Right that is in c first",
    )
    assert seen == []


def test_pool_and_answer_on_another_type_are_409(client):
    assert client.post("/eval-sets", json={"name": "m", "qa_type": "mcq"}).status_code == 201
    r = client.post("/eval-sets/m/pool", json={"question": "alpha"})
    assert (r.status_code, r.json()["detail"]) == (409, "Find passages is for Documents eval sets")
    for confirm in (False, True):
        r = ask_answer(client, ["a"], confirm, name="m")
        detail = "The AI's answer is for Documents eval sets"
        assert (r.status_code, r.json()["detail"]) == (409, detail), confirm
    assert ledger_purposes(client) == []


def test_get_eval_set_says_whether_an_answer_model_is_connected(client):
    answer_store(client)
    assert client.get("/eval-sets/cq").json()["answer_available"] is True
