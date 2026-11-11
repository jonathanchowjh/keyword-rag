"""The run and job routes: starting, listing, resuming and cancelling runs, and /status."""

from dataclasses import asdict

import pytest
from api_helpers import (
    EVAL,
    MCQ,
    NO_KEYWORDS_MODEL,
    NOTES,
    OTHER_LISTS,
    PAIR,
    QUESTION,
    TURBO,
    WORKED,
    add_collection,
    add_documents,
    add_tags,
    bare_client,
    building,
    count_ai,
    do_import,
    documents_with_passages,
    free_disabled_model,
    import_items,
    ledger_purposes,
    make_client,
    passages,
    put_questions,
    reembed,
    reference_with_vocabulary,
    row,
    run_pair,
    scibench_source,
    set_running,
    small_model,
    stale,
    stopped_eval_on,
    store_import,
    terms_reply,
    three_items,
    unloadable,
    upload,
    wait,
)
from fastapi.testclient import TestClient

from ragqa_api.guards import PRICE_CHANGED
from ragqa_core import db, search_defaults
from ragqa_core.fakes import ITEMS, until
from ragqa_core.loader import QUESTION_IMPORT_GONE
from ragqa_core.types import Question


def test_eval_needs_confirm_when_it_costs(client):
    add_collection(client)
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"], "answer": True})
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1
    assert [j["kind"] for j in client.get("/jobs").json()] == ["rekey"]  # no eval queued


def test_eval_of_an_empty_split_is_422(client):
    add_collection(client, questions=False)
    client.post("/eval-sets", json={"name": "c", "qa_type": "mcq"})
    r = client.post("/eval", json={**EVAL, "confirm": True})
    assert r.status_code == 422 and r.json()["detail"] == "eval set 'c' has no questions"


def test_eval_flow(client):
    add_collection(client)
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"], "answer": True})
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1
    r = client.post(
        "/eval", json={"collection": "c", "methods": ["hybrid"], "answer": True, "confirm": True}
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    job = wait(client, job_id)
    assert job["status"] == "done"
    table = client.get(f"/eval/{job_id}").json()
    assert table["methods"]["hybrid"]["accuracy"]["mean"] == 1.0
    assert table["log"] == job["log"] and table["log"][0]["message"] == "started"
    assert table["created_at"] == job["created_at"] and table["question_count"] == 1
    items = client.get(f"/eval/{job_id}/items", params={"method": "hybrid"}).json()
    assert items[0]["question"] == "enzyme rate"
    assert (
        client.get(f"/eval/{job_id}/items", params={"method": "hybrid", "wrong": True}).json() == []
    )


def test_an_incidental_key_error_is_a_500(tmp_path):
    with make_client(tmp_path) as c:
        add_collection(c)
        job = c.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
        wait(c, job)
        c.app.state.evaluator.items = lambda *a: {}["x"]  # a bug, not a not-found
        raw = TestClient(c.app, raise_server_exceptions=False)
        assert raw.get(f"/eval/{job}/items", params={"method": "hybrid"}).status_code == 500


def test_a_run_whose_eval_set_version_is_gone_is_a_404(client):
    add_collection(client)
    setup = {"collection": "c", "eval_set": "c", "eval_set_version": "gone", "methods": ["hybrid"]}
    job = db.create_job(client.app.state.conn, "eval", "c", setup)
    r = client.get(f"/eval/{job}/questions")
    assert r.status_code == 404 and r.json()["detail"] == "That version of 'c' doesn't exist"


SPENDING = {
    "answer": lambda c: c.post("/eval", json={**EVAL, "answer": True, "confirm": True}),
    "summary": lambda c: c.post("/collections/s/load", json={"dataset": "scienceqa"}),
    "keywords": lambda c: c.post("/collections/c/keyword-lists/propose"),
}


@pytest.mark.parametrize("role", SPENDING)
def test_no_key_spending_request_is_503(tmp_path, role):
    with make_client(tmp_path, api_key=None) as client:
        add_collection(client, lists=role != "keywords")
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        client.app.state.worker.stop()
        r = SPENDING[role](client)
        assert r.status_code == 503, r.text
        assert f"'{role}'" in r.json()["detail"] and "Settings" in r.json()["detail"]
        assert client.get("/eval").json() == []


def test_query_and_eval_validate_search(client):
    reference_with_vocabulary(client)
    for search in ({"weights": {"vibes": 1}}, {"colour": "red"}, {"weights": {"words": -1}}):
        body = {"collection": "r", "question": "enzyme", "methods": ["hybrid"], "search": search}
        assert client.post("/query", json=body).status_code == 422, search
        run = {"collection": "r", "methods": ["keyword_rag"], "search": search}
        assert client.post("/eval", json=run).status_code == 422, search
        assert client.post("/eval", json={**run, "confirm": True}).status_code == 422, search


def test_eval_keyword_rag_without_vocabulary_is_refused_before_queueing(client):
    add_collection(client)
    run = {"collection": "c", "methods": ["hybrid", "keyword_rag"], "answer": True}
    assert "estimate" in client.post("/eval", json=run).json()
    r = client.post("/eval", json={**run, "confirm": True})
    assert r.status_code == 409 and "no vocabulary for store 'c'" in r.json()["detail"]
    assert client.get("/eval?collection=c").json() == []


def test_eval_records_the_search_override(client):
    reference_with_vocabulary(client)
    db.save_questions(
        client.app.state.conn, "r", [Question("q1", "enzyme rate", None, ["A"], ["enzyme-item"])]
    )
    run = {
        "collection": "r",
        "methods": ["keyword_rag"],
        "search": {"weights": {"keywords": 2}},
        "confirm": True,
    }
    r = client.post("/eval", json=run)
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"
    table = client.get(f"/eval/{r.json()['job_id']}").json()
    assert table["setup"]["search"]["weights"]["keywords"] == 2.0
    assert table["setup"]["vocabulary_version"]
    items = client.get(f"/eval/{r.json()['job_id']}/items?method=keyword_rag").json()
    assert items[0]["query_tags"][0] == "enzyme"


def test_retag_state_and_problem_wording(client):
    client.post("/collections", json=WORKED)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    retag = db.create_job(conn, "retag", "w", {"version": "v"})
    set_running(conn, retag, 0.4)
    assert client.get("/collections/w").json()["state"] == "retagging"
    db.update_job(conn, retag, status="failed", error="boom", progress=0.4)
    problems = client.get("/status").json()["problems"]
    # still choosing tags at 0.4: nothing was written, so nothing is partly re-tagged
    assert not any("partly re-tagged" in p for p in problems)


def test_a_retag_stopped_while_writing_is_partly_re_tagged(client):
    client.post("/collections", json=WORKED)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    retag = db.create_job(conn, "retag", "w", {"version": "v"})
    db.update_job(conn, retag, status="cancelled", progress=0.5)
    problems = client.get("/status").json()["problems"]
    assert "The vocabulary on w is partly re-tagged; save it again" in problems


def test_unknown_job_404(client):
    assert client.get("/jobs/missing").status_code == 404
    assert client.get("/eval/missing").status_code == 404


def test_model_mismatch_on_query_and_eval(client):
    add_collection(client)
    unloadable(client)
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["hybrid"]})
    assert "built with gone/model" in r.json()["results"]["hybrid"]["error"]
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"]})
    assert r.status_code == 409 and "this service can't load" in r.json()["detail"]


def test_confirmed_paid_eval_without_a_key_is_503_and_queues_nothing(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        add_collection(client)
        client.app.state.worker.stop()
        body = {**EVAL, "answer": True, "confirm": True}
        r = client.post("/eval", json=body)
        assert r.status_code == 503 and "'answer' job" in r.json()["detail"]
        assert client.get("/eval").json() == []
        free = client.post("/eval", json={**EVAL, "confirm": True})  # nothing to pay for
        assert free.status_code == 202


def test_eval_naming_an_unpriced_model_is_422_and_queues_nothing(client):
    add_collection(client)
    client.app.state.worker.stop()
    ai = client.app.state.ai
    ai.add_connection("gw", "openai", "Gateway", base_url="https://gw.example/v1")
    ai.add_model("gw", "nop", price_in=None, price_out=None)
    body = {**EVAL, "answer": True, "model": "gw/nop", "confirm": True}
    r = client.post("/eval", json=body)
    assert r.status_code == 422 and r.json()["detail"] == "gw/nop has no price; enter one first"
    assert client.get("/eval").json() == []


def test_eval_refuses_a_changed_price(client):
    add_collection(client)
    seen = client.post("/eval", json={**EVAL, "answer": True}).json()["estimate"]
    base = {**EVAL, "answer": True, "confirm": True}
    r = client.post("/eval", json={**base, "calls": seen["calls"] + 1, "est_usd": seen["est_usd"]})
    assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
    assert client.get("/eval").json() == []
    r = client.post("/eval", json={**base, "calls": seen["calls"], "est_usd": seen["est_usd"]})
    assert r.status_code == 202


def test_eval_confirm_without_a_shown_price_is_unchanged(client):
    add_collection(client)
    assert client.post("/eval", json={**EVAL, "confirm": True}).status_code == 202


def test_rerun_that_saw_no_price_is_refused_once_priced(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        ai = client.app.state.ai
        model = ai.roles()["default"]
        ai.set_role("default", None)  # nothing can price the answers
        seen = client.post("/eval", json={**EVAL, "answer": True}).json()["estimate"]
        assert seen["est_usd"] is None and seen["calls"] >= 1
        shown = {**EVAL, "answer": True, "confirm": True, "calls": seen["calls"], "est_usd": None}
        r = client.post("/eval", json=shown)
        assert r.status_code == 503 and "'answer' job" in r.json()["detail"]
        ai.set_role("default", model)  # now priced, above the nothing that was shown
        r = client.post("/eval", json=shown)
        assert r.status_code == 409 and client.get("/eval").json() == []


def test_stale_eval_confirm_with_the_model_off_is_409(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        client.app.state.ai.set_role("default", None)  # the answer model is off: a 503 waits
        seen = client.post("/eval", json={**EVAL, "answer": True}).json()["estimate"]
        stale = {**EVAL, "answer": True, "confirm": True, "calls": seen["calls"] + 1}
        r = client.post("/eval", json=stale)
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED  # before any 503
        assert client.get("/eval").json() == []


def test_free_eval_queues_with_no_ai_set_up(tmp_path):
    with bare_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        r = client.post("/eval", json={**EVAL, "confirm": True})
        assert r.status_code == 202, r.text
        assert db.get_job(client.app.state.conn, r.json()["job_id"])["setup"]["model"] is None


def test_paid_eval_with_no_ai_estimates_unpriced_and_refuses_the_confirm(tmp_path):
    with bare_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        r = client.post("/eval", json={**EVAL, "answer": True})
        assert r.status_code == 200, r.text
        assert r.json()["estimate"]["est_usd"] is None and r.json()["estimate"]["calls"] == 1
        r = client.post("/eval", json={**EVAL, "answer": True, "confirm": True})
        assert r.status_code == 503 and "'answer' job" in r.json()["detail"]
        assert client.get("/eval").json() == []


def test_eval_and_resume_check_the_query_job_too(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client, summarize=True)
        ai = client.app.state.ai
        ai.set_role("query", "anthropic/claude-sonnet-5-5")
        ai.update_model("anthropic", "claude-sonnet-5-5", enabled=False)
        client.app.state.worker.stop()
        body = {**EVAL, "methods": ["kw_individual"], "answer": True, "confirm": True}
        r = client.post("/eval", json=body)
        assert r.status_code == 503 and "'query' job" in r.json()["detail"]
        assert client.get("/eval").json() == []
        conn = client.app.state.conn
        job = client.app.state.evaluator.queue("c", "dev", ["kw_individual"], True, None)
        db.update_job(conn, job, status="stopped_no_key")
        r = client.post(f"/jobs/{job}/resume", json={"confirm": True})
        assert r.status_code == 503 and "'query' job" in r.json()["detail"]
        assert db.get_job(conn, job)["status"] == "stopped_no_key"


def test_confirmed_paid_resume_without_a_key_is_503(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        add_collection(client)
        client.app.state.worker.stop()
        conn = client.app.state.conn
        job = client.app.state.evaluator.queue("c", "dev", ["hybrid"], True, None)
        db.update_job(conn, job, status="stopped_no_key")
        r = client.post(f"/jobs/{job}/resume", json={"confirm": True})
        assert r.status_code == 503 and db.get_job(conn, job)["status"] == "stopped_no_key"


def test_cancel_a_finished_job_is_409(client):  # Review focus 1
    add_collection(client)
    job = client.get("/jobs", params={"kind": "rekey"}).json()[0]
    r = client.post(f"/jobs/{job['id']}/cancel")
    assert r.status_code == 409 and r.json()["detail"] == "job is done"


def test_cancel_a_queued_or_running_job(client):
    add_collection(client)
    client.app.state.worker.stop()
    conn = client.app.state.conn
    queued = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    assert client.post(f"/jobs/{queued}/cancel").json() == {"status": "cancelled"}
    running = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    set_running(conn, running)
    r = client.post(f"/jobs/{running}/cancel")
    assert r.status_code == 200 and r.json() == {"status": "running"}
    assert db.is_cancel_requested(conn, running)
    assert client.post("/jobs/missing/cancel").status_code == 404


def test_jobs_list_filters(client):
    add_collection(client)
    add_collection(client, name="d", lists=False)
    client.app.state.worker.stop()
    e = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    assert [j["id"] for j in client.get("/jobs", params={"kind": "eval"}).json()] == [e]
    assert [j["kind"] for j in client.get("/jobs", params={"collection": "c"}).json()] == [
        "eval",
        "rekey",
    ]
    assert [j["id"] for j in client.get("/jobs", params={"status": "queued"}).json()] == [e]
    assert client.get("/jobs", params={"collection": "d"}).json() == []


def test_runs_list_and_delete(client):
    add_collection(client)
    r = client.post("/eval", json={**EVAL, "methods": PAIR, "confirm": True})
    done = r.json()["job_id"]
    assert wait(client, done)["status"] == "done"
    runs = client.get("/eval", params={"collection": "c"}).json()
    assert [run["id"] for run in runs] == [done]
    run = runs[0]
    assert set(run) == {
        "id",
        "collection",
        "status",
        "progress",
        "created_at",
        "methods",
        "split",
        "answer",
        "model",
        "question_count",
        "eval_set",
        "eval_set_version",
        "evidence_named",
        "evidence_held",
        "usd_total",
        "answer_summary",
    }
    assert run["status"] == "done" and run["methods"] == PAIR and run["split"] == "dev"
    assert run["answer"] is False and run["question_count"] == 1 and run["usd_total"] == 0.0
    assert run["model"] == client.app.state.ai.model_for("answer")
    detail = client.get(f"/eval/{done}").json()
    # The verdict carries what its notes need: the proxy check and the headline's scale.
    assert run["answer_summary"] == {
        **detail["answer"],
        "proxy_trusted": detail["proxy_trusted"],
        "scale": detail["scale"],
    }
    assert client.get("/eval", params={"collection": "other"}).json() == []
    client.app.state.worker.stop()
    queued = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    assert [run["id"] for run in client.get("/eval").json()] == [queued, done]
    r = client.delete(f"/eval/{queued}")
    assert r.status_code == 409 and "cancel it first" in r.json()["detail"]
    client.post(f"/jobs/{queued}/cancel")
    assert client.delete(f"/eval/{queued}").status_code == 204
    assert client.delete(f"/eval/{done}").status_code == 204
    assert client.get(f"/eval/{done}").status_code == 404
    assert db.run_items(client.app.state.conn, done) == []
    assert client.get("/eval", params={"collection": "c"}).json() == []
    assert client.delete("/eval/missing").status_code == 404


def test_resume_flow_estimates_then_requeues(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    body = {**EVAL, "methods": ["kw_individual"], "answer": True, "confirm": True}
    j = client.post("/eval", json=body).json()["job_id"]
    db.update_job(conn, j, status="interrupted")
    r = client.post(f"/jobs/{j}/resume", json={})
    assert r.status_code == 200 and r.json()["estimate"]["questions"] == 1
    assert r.json()["estimate"]["calls"] == 1 and r.json()["estimate"]["est_usd"] > 0
    assert db.get_job(conn, j)["status"] == "interrupted"  # an estimate changes nothing
    r = client.post(f"/jobs/{j}/resume", json={"confirm": True})
    assert r.status_code == 202 and r.json() == {"job_id": j}
    assert db.get_job(conn, j)["status"] == "queued"
    client.app.state.worker.start()
    assert wait(client, j)["status"] == "done"
    r = client.post(f"/jobs/{j}/resume", json={"confirm": True})
    assert r.status_code == 409 and r.json()["detail"] == "a done run cannot be resumed"
    # A keyword run whose lists changed since it ran is refused.
    db.update_job(conn, j, status="failed")
    put = client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS)).json()
    assert wait(client, put["job_id"])["status"] == "done"
    r = client.post(f"/jobs/{j}/resume")
    assert r.status_code == 409 and "keyword lists changed since this run" in r.json()["detail"]
    rekey = client.get("/jobs", params={"kind": "rekey"}).json()[0]["id"]
    r = client.post(f"/jobs/{rekey}/resume")
    assert r.status_code == 409 and r.json()["detail"] == "only evals can be resumed"
    assert client.post("/jobs/missing/resume").status_code == 404


def stopped_eval(client) -> str:
    return stopped_eval_on(client, "c")


def test_resume_refuses_a_changed_price(client):
    job = stopped_eval(client)
    seen = client.post(f"/jobs/{job}/resume", json={"confirm": False}).json()["estimate"]
    r = client.post(f"/jobs/{job}/resume", json=stale(seen))
    assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
    assert db.get_job(client.app.state.conn, job)["status"] == "interrupted"
    shown = {"confirm": True, "calls": seen["calls"], "est_usd": seen["est_usd"]}
    assert client.post(f"/jobs/{job}/resume", json=shown).status_code == 202


def test_resume_confirm_without_a_shown_price_is_unchanged(client):
    job = stopped_eval(client)
    assert client.post(f"/jobs/{job}/resume", json={"confirm": True}).status_code == 202


def test_status_reports_problems(client):
    add_collection(client)
    body = client.get("/status").json()
    assert body["worker"] == {"alive": True, "running": None, "error": None}
    assert body["queue"] == [] and body["batches"] == [] and body["problems"] == []
    add_collection(client, name="d", lists=False)
    worker, conn = client.app.state.worker, client.app.state.conn
    worker.stop()
    stopped = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    db.update_job(conn, stopped, status="stopped_budget", error="monthly cap reached")
    failed = db.create_job(conn, "load", "c", {})
    db.update_job(conn, failed, status="failed", error="boom")
    db.update_job(conn, db.create_job(conn, "rekey", "c", {}), status="interrupted")
    db.update_job(conn, db.create_job(conn, "rekey", "c", {}), status="done")  # newer: fine
    queued = client.post("/eval", json={**EVAL, "collection": "d", "confirm": True}).json()
    running = db.create_job(conn, "load", "d", {})
    set_running(conn, running, 0.25)
    conn.execute(
        "insert into ai_batches (id, model, purpose, created_at, usd, job_id, connection_id)"
        " values ('b1', 'm', 'answer', 't', 0.5, ?, 'anthropic')",
        (queued["job_id"],),
    )
    worker.error = "RuntimeError('x')"
    body = client.get("/status").json()
    assert body["worker"]["alive"] is False and body["worker"]["error"] == "RuntimeError('x')"
    assert body["worker"]["running"]["id"] == running
    assert body["worker"]["running"]["progress"] == 0.25
    assert [j["id"] for j in body["queue"]] == [queued["job_id"]]
    assert body["batches"] == [
        {
            "id": "b1",
            "model": "m",
            "purpose": "answer",
            "created_at": "t",
            "usd": 0.5,
            "job_id": queued["job_id"],
            "connection_id": "anthropic",
        }
    ]
    assert body["problems"] == [
        "The worker thread has stopped; restart the service",
        "Worker error: RuntimeError('x')",
        f"Load {failed[:8]} on c failed",
        f"Eval {stopped[:8]} on c stopped at the monthly cap",
    ]


def test_resume_requeues_only_a_resumable_job(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    j = client.post("/eval", json={**EVAL, "answer": True, "confirm": True}).json()["job_id"]
    for status in ("queued", "running"):
        db.update_job(conn, j, status=status)
        r = client.post(f"/jobs/{j}/resume", json={"confirm": True})
        assert r.status_code == 409 and db.get_job(conn, j)["status"] == status
    db.update_job(conn, j, status="interrupted")
    assert client.post(f"/jobs/{j}/resume", json={"confirm": True}).status_code == 202
    # A second confirm (a double click) finds it queued and moves nothing.
    r = client.post(f"/jobs/{j}/resume", json={"confirm": True})
    assert r.status_code == 409 and db.get_job(conn, j)["status"] == "queued"


def test_status_reports_partly_rekeyed_lists(client):
    add_collection(client)
    conn = client.app.state.conn
    line = "Keyword lists on c are partly re-keyed; save them again"
    for status in ("failed", "cancelled", "interrupted"):
        rekey = db.create_job(conn, "rekey", "c", {})
        db.update_job(conn, rekey, status=status, progress=0.5)
        assert line in client.get("/status").json()["problems"], status
    db.update_job(conn, db.create_job(conn, "rekey", "c", {}), status="done")
    assert not any("re-keyed" in p for p in client.get("/status").json()["problems"])


def test_status_reports_a_running_job_with_no_progress_for_10_minutes(client):
    from datetime import UTC, datetime, timedelta

    conn = client.app.state.conn
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    job = db.create_job(conn, "load", "c", {})
    set_running(conn, job)
    assert client.get("/status").json()["problems"] == []  # just updated
    for minutes, stalled in ((9, False), (11, True)):
        at = datetime.now(UTC) - timedelta(minutes=minutes)
        conn.execute(
            "update jobs set updated_at = ? where id = ?", (at.isoformat(timespec="seconds"), job)
        )
        line = f"Load {job[:8]} on c has made no progress for {minutes} minutes"
        assert (line in client.get("/status").json()["problems"]) is stalled
    # Waiting on an AI batch can take hours: not a stall.
    conn.execute(
        "insert into ai_batches (id, model, purpose, created_at, usd, job_id, connection_id)"
        " values ('b1', 'm', 'answer', 't', 0.5, ?, 'anthropic')",
        (job,),
    )
    assert client.get("/status").json()["problems"] == []


def test_status_says_partly_rekeyed_only_when_the_rekey_started(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    partly = "Keyword lists on c are partly re-keyed; save them again"

    def problems() -> list[str]:
        found = client.get("/status").json()["problems"]
        return [p for p in found if "Rekey" in p or "re-keyed" in p]

    never = db.create_job(conn, "rekey", "c", {})
    assert db.request_cancel(conn, never) == "cancelled"
    assert problems() == []  # cancelled while queued: no line at all
    failed = db.create_job(conn, "rekey", "c", {})
    db.update_job(conn, failed, status="failed", error="boom")
    assert problems() == [f"Rekey {failed[:8]} on c failed"]
    started = db.create_job(conn, "rekey", "c", {})
    db.update_job(conn, started, status="failed", progress=0.4)
    assert problems() == [partly]


def test_since_names_a_changed_dense_model(client):
    add_collection(client)
    job_id = run_pair(client)
    conn = client.app.state.conn
    conn.execute("update collections set dense_model = 'other/model' where name = 'c'")
    expected = {"untracked": False, "changes": ["dense model"]}
    assert client.get(f"/eval/{job_id}").json()["since"] == expected


def seed_rows(client, job_id, rows):
    """Replace a run's rows: {method: {question_id: row}}."""
    conn = client.app.state.conn
    conn.execute("delete from run_items where job_id = ?", (job_id,))
    for method, per_q in rows.items():
        for qid, values in per_q.items():
            db.upsert_run_item(conn, job_id, method, qid, values, complete=True)


def test_questions_lists_every_methods_row_per_question(client):
    add_collection(client)
    db.save_questions(
        client.app.state.conn,
        "c",
        [QUESTION, Question("q2", "market price", ["x", "y"], ["B"], [], {})],
    )
    job_id = run_pair(client)
    seed_rows(
        client,
        job_id,
        {
            "kw_individual": {"q1": row("enzyme-item", 1), "q2": row("market-item", 1)},
            "hybrid": {"q1": row("enzyme-item", 1), "q2": row("volcano-item", 0)},
            "random_example": {"q1": row("volcano-item", 0), "q2": row("volcano-item", 0)},
        },
    )
    out = client.get(f"/eval/{job_id}/questions").json()
    assert [q["question_id"] for q in out] == ["q1", "q2"]
    first = out[0]
    assert first["question"] == "enzyme rate" and first["gold_answers"] == ["A"]
    assert set(first["methods"]) == {"kw_individual", "hybrid", "random_example"}
    assert first["methods"]["kw_individual"]["hit_ids"] == ["enzyme-item"]
    assert first["methods"]["kw_individual"]["hit_matched"] == [["rate"]]
    assert first["methods"]["hybrid"]["outcome"] == "right"
    assert first["methods"]["random_example"]["outcome"] == "wrong"
    assert client.get("/eval/nope/questions").status_code == 404


def test_questions_disagree_filter_ignores_controls_and_absent_outcomes(client):
    add_collection(client)
    skill = {"skill": "enzymes"}  # measurable without evidence ids, so a free run is allowed
    qs = [Question(f"q{i}", f"enzyme {i}", ["x", "y"], ["A"], [], skill) for i in range(1, 5)]
    db.save_questions(client.app.state.conn, "c", qs)
    job_id = run_pair(client)
    seed_rows(
        client,
        job_id,
        {
            "kw_individual": {
                "q1": row("a", 1),  # q1: both right, control wrong -> agree
                "q2": row("a", 1),  # q2: differ
                "q3": row("a", 0),  # q3: both wrong
                "q4": row("a", 1),  # q4: hybrid has no outcome -> nothing to differ from
            },
            "hybrid": {
                "q1": row("a", 1),
                "q2": row("a", 0),
                "q3": row("a", 0),
                "q4": {"hit_ids": [], "hit_skills": [], "retrieval": {}},
            },
            "random_example": {f"q{i}": row("a", 0) for i in range(1, 5)},
        },
    )
    got = client.get(f"/eval/{job_id}/questions", params={"disagree": True}).json()
    assert [q["question_id"] for q in got] == ["q2"]
    assert "outcome" not in client.get(f"/eval/{job_id}/questions").json()[3]["methods"]["hybrid"]


def test_questions_outcome_falls_back_to_the_free_metric(client):
    add_collection(client)
    job_id = run_pair(client)
    seed_rows(
        client,
        job_id,
        {
            "kw_individual": {"q1": row("a", retrieval={"same_skill@1": 1.0})},
            "hybrid": {"q1": row("a", retrieval={"same_skill@1": 0.0})},
            "random_example": {"q1": row("a", retrieval={})},
        },
    )
    got = client.get(f"/eval/{job_id}/questions", params={"disagree": True}).json()
    assert len(got) == 1
    assert "outcome" not in got[0]["methods"]["random_example"]


def test_old_question_import_job_fails_and_wont_resume(client):
    add_collection(client)
    conn = client.app.state.conn
    setup = {"upload_id": "x", "kind": "questions", "mapping": {}, "source": "upload:q"}
    job = db.create_job(conn, "import", "c", setup)
    row = wait(client, job)
    assert row["status"] == "failed" and QUESTION_IMPORT_GONE in row["error"]
    r = client.post(f"/jobs/{job}/resume", json={"confirm": True})
    assert r.status_code == 409 and QUESTION_IMPORT_GONE in r.json()["detail"]


def test_role_with_disabled_model_is_503_naming_job(client):
    add_collection(client)
    client.put("/ai/models/anthropic/claude-haiku-5-5", json={"enabled": False})
    r = client.post("/eval", json={**EVAL, "answer": True, "confirm": True})
    assert r.status_code == 503, r.text
    assert "'answer'" in r.json()["detail"] and "Settings" in r.json()["detail"]


def eval_with_model(client, model, confirm=True):
    body = {**EVAL, "answer": True, "model": model, "confirm": confirm}
    return client.post("/eval", json=body)


def test_confirm_checks_the_named_model_not_the_answer_job(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        ai = client.app.state.ai
        ai.update_model("anthropic", "claude-sonnet-5-5", enabled=False)
        r = eval_with_model(client, "anthropic/claude-sonnet-5-5")
        assert r.status_code == 503 and "'answer'" in r.json()["detail"]
        assert client.get("/eval").json() == []
        # The other way round: the job's model is off, the named one works.
        ai.update_model("anthropic", "claude-sonnet-5-5", enabled=True)
        ai.update_model("anthropic", "claude-haiku-5-5", enabled=False)
        r = eval_with_model(client, "anthropic/claude-sonnet-5-5")
        assert r.status_code == 202, r.text


def test_confirm_refuses_a_disabled_free_model(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        ai = client.app.state.ai
        ai.add_connection("gw", "openai", "Gateway", base_url="https://gw.example/v1")
        ai.add_model("gw", "llama", price_in=0, price_out=0)  # a price of 0, entered by hand
        ai.update_model("gw", "llama", enabled=False)
        assert eval_with_model(client, "gw/llama", confirm=False).json()["estimate"]["est_usd"] == 0
        r = eval_with_model(client, "gw/llama")
        assert r.status_code == 503 and "'answer'" in r.json()["detail"]
        assert client.get("/eval").json() == []


def test_resume_checks_the_runs_own_model(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client)
        client.app.state.worker.stop()
        ai, conn = client.app.state.ai, client.app.state.conn
        job = client.app.state.evaluator.queue(
            "c", "dev", ["hybrid"], True, "anthropic/claude-sonnet-5-5"
        )
        db.update_job(conn, job, status="stopped_no_key")
        ai.update_model("anthropic", "claude-sonnet-5-5", enabled=False)
        r = client.post(f"/jobs/{job}/resume", json={"confirm": True})
        assert r.status_code == 503 and "'answer'" in r.json()["detail"]
        assert db.get_job(conn, job)["status"] == "stopped_no_key"
        ai.update_model("anthropic", "claude-sonnet-5-5", enabled=True)
        ai.update_model("anthropic", "claude-haiku-5-5", enabled=False)  # the job's own model
        r = client.post(f"/jobs/{job}/resume", json={"confirm": True})
        assert r.status_code == 202, r.text


def test_cached_summaries_need_no_summary_model(tmp_path):
    with make_client(tmp_path) as client:
        add_collection(client, summarize=True, lists=False)
        scibench_source(client)
        body = {"name": "x", "recipe": "paper_2024", "build": {"added_text": "summary"}}
        job_id = client.post("/collections/p/copy", json={**body, "confirm": True}).json()["job_id"]
        assert wait(client, job_id)["status"] == "done"
        import_items(client)  # the notes' summaries are made, and cached
        client.app.state.worker.stop()
        app = client.app.state
        # A second copy of p's items into a summarising store, stopped before it ran: its
        # summaries are the ones the first copy made.
        again = app.loader.copy_spec(app.retriever.spec("p"), "y", "paper_2024", body["build"])
        assert again.summarize
        stopped = app.loader.start_copy("p", again)
        db.update_job(app.conn, stopped, status="interrupted")
        r = client.post(f"/jobs/{stopped}/resume")
        assert r.status_code == 200 and r.json()["estimate"]["calls"] == 0
        client.app.state.ai.update_connection("anthropic", api_key=None)  # no model can run
        same = NOTES.replace(b'"n', b'"again-n')  # the same texts under new ids
        up = upload(client, body=same, filename="again.jsonl").json()
        assert up["estimate"]["calls"] == 0 and up["estimate"]["new"] == 3
        r = do_import(client, up, "again", confirm=True)
        assert r.status_code == 202, r.text
        r = client.post(f"/jobs/{stopped}/resume", json={"confirm": True})
        assert r.status_code == 202, r.text


def test_build_without_keywords_model_is_refused_before_spending(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, api_key=None) as client:
        client.app.state.worker.stop()
        client.post("/collections", json=WORKED)

        def refused(r) -> None:
            assert r.status_code == 503, r.text
            assert "'keywords'" in r.json()["detail"] and "Settings" in r.json()["detail"]

        for body in ({"dataset": "scibench"}, {"dataset": "scibench", "confirm": True}):
            refused(client.post("/collections/w/load", json=body))
        refused(client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True}))
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        refused(client.post("/collections/w/documents", json={"items": many, "confirm": True}))
        up = upload(client, name="w").json()
        assert up["estimate"]["est_usd"] > 0
        refused(do_import(client, up, name="w", confirm=True))
        # With no model for keywords at all, nothing can price the build: the estimate is None.
        client.app.state.ai.set_role("default", None)
        r = client.post("/collections/w/documents", json={"items": ITEMS})
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] is None
        refused(client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True}))
        refused(client.post("/collections/w/load", json={"dataset": "scibench"}))
        assert client.get("/collections/w").json()["count"] == 0
        assert client.get("/jobs").json() == [] and ledger_purposes(client) == []


def test_a_stopped_copy_resumes_in_place_and_prices_only_what_is_left(tmp_path):
    broken = {"abstract": True}

    def reply(prompt: str) -> str:
        if '"principle"' in prompt and broken["abstract"]:
            raise RuntimeError("provider down")
        return building(prompt)

    with make_client(tmp_path, reply=reply) as client:
        scibench_source(client)
        body = {"name": "w", "recipe": "worked_examples", "confirm": True}
        job_id = client.post("/collections/p/copy", json=body).json()["job_id"]
        assert wait(client, job_id)["status"] == "failed"
        whole = client.post("/collections/p/copy", json={**body, "name": "w2", "confirm": False})
        r = client.post(f"/jobs/{job_id}/resume")
        assert r.status_code == 200, r.text
        left = r.json()["estimate"]
        assert left["items"] == 0  # every item was added; only abstraction is left
        assert 0 < left["calls"] < whole.json()["estimate"]["calls"]
        broken["abstract"] = False
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": True})
        assert r.status_code == 202 and r.json() == {"job_id": job_id}
        assert wait(client, job_id)["status"] == "done"
        clean = client.post("/collections/p/copy", json={**body, "name": "w2"}).json()["job_id"]
        assert wait(client, clean)["status"] == "done"
        store = client.app.state.store

        def built(name: str) -> list:
            return sorted((h.id, h.tags, h.abstracted) for h in store.get(name, store.ids("p")))

        assert built("w") == built("w2")  # as a copy that never stopped
        assert any(abstracted for _, _, abstracted in built("w"))  # the tags were not paid twice
        r = client.post(f"/jobs/{job_id}/resume")
        assert r.status_code == 409 and r.json()["detail"] == "a done copy cannot be resumed"


def test_resuming_a_stopped_copy_refuses_a_changed_price(tmp_path):
    with make_client(tmp_path, reply=building) as client:
        scibench_source(client)
        conn = client.app.state.conn
        client.app.state.worker.stop()
        body = {"name": "w", "recipe": "worked_examples", "confirm": True}
        job_id = client.post("/collections/p/copy", json=body).json()["job_id"]
        db.update_job(conn, job_id, status="interrupted")
        seen = client.post(f"/jobs/{job_id}/resume", json={"confirm": False}).json()["estimate"]
        r = client.post(f"/jobs/{job_id}/resume", json=stale(seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        assert db.get_job(conn, job_id)["status"] == "interrupted"
        shown = {"confirm": True, "calls": seen["calls"], "est_usd": seen["est_usd"]}
        assert client.post(f"/jobs/{job_id}/resume", json=shown).status_code == 202


def test_eval_refuses_an_eval_set_of_another_qa_type(client):
    add_collection(client)
    client.app.state.worker.stop()
    client.post("/eval-sets", json={"name": "s", "qa_type": "science"})
    sci = {"id": "s1", "text": "how warm is it?", "gold_answers": ["3"]}
    assert put_questions(client, "s", [sci]).status_code == 200
    for confirm in (False, True):
        r = client.post("/eval", json={**EVAL, "eval_set": "s", "confirm": confirm})
        assert r.status_code == 409
        assert r.json()["detail"] == "eval set 's' is science; store 'c' is mcq"
    assert db.list_jobs(client.app.state.conn, kind="eval") == []


def test_estimate_counts_the_chosen_eval_set(client):
    add_collection(client)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    questions = [
        {**MCQ, "id": "a", "gold_ids": ["enzyme-item"]},
        {**MCQ, "id": "b", "gold_ids": ["nowhere"]},
        {**MCQ, "id": "c"},
    ]
    assert put_questions(client, "e", questions).status_code == 200
    est = client.post("/eval", json={**EVAL, "eval_set": "e"}).json()["estimate"]
    assert (est["questions"], est["dev_questions"]) == (3, 3)
    assert (est["evidence_named"], est["evidence_held"]) == (2, 1)
    own = client.post("/eval", json=EVAL).json()["estimate"]
    assert (own["questions"], own["dev_questions"], own["evidence_named"]) == (1, 1, 0)


def test_eval_without_eval_set_uses_the_default_and_says_when_there_is_none(client):
    add_collection(client)
    r = client.post("/eval", json={**EVAL, "confirm": True})
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done" and job["setup"]["eval_set"] == "c"
    version = db.get_eval_set(client.app.state.conn, "c")["version"]
    (summary,) = client.get("/eval", params={"collection": "c"}).json()
    detail = client.get(f"/eval/{job['id']}").json()
    for run in (summary, detail):
        assert run["eval_set"] == "c" and run["eval_set_version"] == version
        assert (run["evidence_named"], run["evidence_held"]) == (0, 0)
    add_collection(client, name="d", questions=False)
    r = client.post("/eval", json={**EVAL, "collection": "d", "confirm": True})
    assert r.status_code == 422 and r.json()["detail"] == "store 'd' has no eval set; choose one"


def spy_retrieve(client) -> list[dict]:
    """Every retrieve call's pool and rerank_model, with the real search still running."""
    retriever, seen = client.app.state.retriever, []
    real = retriever.retrieve

    def retrieve(*args, **kw):
        seen.append({"pool": kw.get("pool"), "rerank_model": kw.get("rerank_model")})
        return real(*args, **kw)

    retriever.retrieve = retrieve
    return seen


def test_a_run_records_and_uses_its_rerank_pool(client):
    add_collection(client)
    seen = spy_retrieve(client)
    body = {**EVAL, "methods": ["hybrid_rerank"], "rerank_pool": 35}  # not the default, 20
    r = client.post("/eval", json={**body, "confirm": True})
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done" and job["setup"]["rerank_pool"] == 35
    assert seen and {s["pool"] for s in seen} == {35}
    for bad in (0, 51):
        assert client.post("/eval", json={**body, "rerank_pool": bad}).status_code == 422


def test_a_run_recorded_before_the_reranker_and_pool_keeps_bge_at_50(client):
    add_collection(client)
    conn = client.app.state.conn
    body = {**EVAL, "methods": ["hybrid_rerank"], "confirm": True}
    done = wait(client, client.post("/eval", json=body).json()["job_id"])
    old = {k: v for k, v in done["setup"].items() if k not in ("rerank_model", "rerank_pool")}
    seen = spy_retrieve(client)
    job = wait(client, db.create_job(conn, "eval", "c", old))
    assert job["status"] == "done", job
    assert seen and {(s["rerank_model"], s["pool"]) for s in seen} == {
        ("BAAI/bge-reranker-v2-m3-int8", 50)
    }


def test_a_run_records_and_uses_its_rerank_model(client):
    add_collection(client)
    conn = client.app.state.conn
    seen = spy_retrieve(client)
    body = {**EVAL, "methods": ["hybrid_rerank"], "confirm": True}
    search_defaults.save(conn, {"rerank_model": TURBO})
    job = wait(client, client.post("/eval", json=body).json()["job_id"])
    assert job["setup"]["rerank_model"] == TURBO
    assert {s["rerank_model"] for s in seen} == {TURBO}
    search_defaults.save(conn, {"rerank_model": None})
    assert db.get_job(conn, job["id"])["setup"]["rerank_model"] == TURBO  # what it recorded
    tiny = "jinaai/jina-reranker-v1-tiny-en"
    job = wait(client, client.post("/eval", json={**body, "rerank_model": tiny}).json()["job_id"])
    assert job["setup"]["rerank_model"] == tiny
    assert client.post("/eval", json={**body, "rerank_model": "nope/y"}).status_code == 422
    job = wait(client, client.post("/eval", json={**body, "methods": ["hybrid"]}).json()["job_id"])
    assert "rerank_model" not in job["setup"]


def test_a_store_reads_cleanly_through_a_rebuild_switch(client, monkeypatch):
    small_model(client, monkeypatch)
    add_collection(client)
    r = reembed(client, name="c", model="fake/small")
    assert r.status_code == 202, r.text
    job_id, statuses = r.json()["job_id"], []

    def finished():
        statuses.append(client.get("/collections/c").status_code)
        return client.get(f"/jobs/{job_id}").json()["status"] not in ("queued", "running")

    until(finished, 10, 0.01, "the rebuild finishing")
    job = client.get(f"/jobs/{job_id}").json()
    assert job["status"] == "done", job["error"]
    assert set(statuses) == {200}, [s for s in statuses if s != 200]


def test_a_stopped_rebuild_resumes_through_the_jobs_route(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    small_model(client, monkeypatch)
    add_collection(client, lists=False)
    conn = client.app.state.conn
    job_id = db.create_job(conn, "reembed", "c", {"model": "fake/small", "from": "fake/dense"})
    db.update_job(conn, job_id, status="cancelled")
    rebuild = client.get("/collections/c").json()["rebuild"]
    assert rebuild == {
        "job_id": job_id,
        "status": "cancelled",
        "model": "fake/small",
        "from": "fake/dense",
        "progress": 0.0,
        "switching": False,
    }
    r = client.post(f"/jobs/{job_id}/resume")
    assert r.status_code == 202 and r.json() == {"job_id": job_id}
    with client:
        assert wait(client, job_id)["status"] == "done"
        body = client.get("/collections/c").json()
        assert (body["dense_model"], body["rebuild"]) == ("fake/small", None)
        r = client.post(f"/jobs/{job_id}/resume")
        assert r.status_code == 409 and r.json()["detail"] == "a done rebuild cannot be resumed"


def test_a_rebuild_resumes_only_when_its_store_is_idle_and_it_is_the_newest(tmp_path, monkeypatch):
    client = make_client(tmp_path)  # not entered: queued jobs stay queued
    small_model(client, monkeypatch)
    add_collection(client, lists=False)
    conn = client.app.state.conn
    older = db.create_job(conn, "reembed", "c", {"model": "fake/small", "from": "fake/dense"})
    db.update_job(conn, older, status="cancelled")
    load = db.create_job(conn, "load", "c", {})
    r = client.post(f"/jobs/{older}/resume")
    assert r.status_code == 409 and "a load job is queued" in r.json()["detail"]
    db.update_job(conn, load, status="done")
    newer = db.create_job(conn, "reembed", "c", {"model": "fake/small", "from": "fake/dense"})
    db.update_job(conn, newer, status="failed")
    r = client.post(f"/jobs/{older}/resume")
    assert r.status_code == 409 and "a newer rebuild" in r.json()["detail"]
    assert client.post(f"/jobs/{newer}/resume").status_code == 202


def test_a_documents_run_may_ask_for_answers(client):
    add_documents(client)
    store_import(client, "x.md", b"some text about passwords")
    r = client.post("/eval-sets", json={"name": "s", "qa_type": "documents", "store": "c"})
    assert r.status_code == 201, r.text
    held = passages(client)[0].id
    r = put_questions(client, "s", [{"id": "a", "text": "passwords?", "gold_ids": [held]}])
    assert r.status_code == 200, r.text
    calls = count_ai(client)
    body = {"collection": "c", "methods": ["hybrid"], "answer": True, "eval_set": "s"}
    r = client.post("/eval", json=body)
    assert r.status_code == 200, r.text
    estimate = r.json()["estimate"]
    assert estimate["calls"] == 1 and estimate["evidence_size"] == 8
    assert calls == []
    evaluator, conn = client.app.state.evaluator, client.app.state.conn
    estimate_now = evaluator.estimate

    def then_settings_change(*args, **kw):  # after the estimate priced 8, before queue
        out = estimate_now(*args, **kw)
        search_defaults.save(conn, {"evidence_size": 5})
        return out

    evaluator.estimate = then_settings_change
    r = client.post("/eval", json={**body, "confirm": True})
    assert r.status_code == 202, r.text
    assert r.json()["estimate"]["evidence_size"] == 8
    job_id = r.json()["job_id"]
    job = wait(client, job_id)
    assert job["status"] == "done", job
    assert db.get_job(conn, job_id)["setup"]["evidence_size"] == 8  # what the estimate priced
    (row,) = db.run_items(conn, job_id)
    assert row["pred"] and row["answer"] == {}
    assert calls == ["complete_many"]


def test_judge_answers_over_the_api(client):
    add_documents(client)
    store_import(client, "x.md", b"some text about passwords")
    r = client.post("/eval-sets", json={"name": "s", "qa_type": "documents", "store": "c"})
    assert r.status_code == 201, r.text
    held = passages(client)[0].id
    question = {"id": "a", "text": "passwords?", "gold_ids": [held], "gold_answers": ["Ask IT"]}
    r = put_questions(client, "s", [question])
    assert r.status_code == 200, r.text
    calls = count_ai(client)
    body = {
        "collection": "c",
        "methods": ["hybrid"],
        "answer": True,
        "judge": True,
        "eval_set": "s",
    }
    r = client.post("/eval", json=body)
    assert r.status_code == 200, r.text
    estimate = r.json()["estimate"]
    assert estimate["calls"] == 2 and estimate["references"] == 1
    assert calls == []
    r = client.post("/eval", json={**body, "confirm": True})
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert calls == ["complete_many", "complete_many"]
    table = client.get(f"/eval/{r.json()['job_id']}").json()
    assert table["answers"]["judged"] is True
    r = client.post("/eval", json={**body, "answer": False})
    assert r.status_code == 422 and r.json()["detail"] == (
        "Judge answers needs the AI's answers: tick Have the AI answer the questions"
    )


NEEDS_TAGS = "needs keyword tags: Add keyword tags on the store page"


def test_a_stopped_tag_build_resumes_and_finishes(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        conn, builder = client.app.state.conn, client.app.state.builder
        real = builder._write_tags

        def first_only(spec, ids, tags, progress=lambda p: None):
            real(spec, ids[:1], tags[:1])
            raise RuntimeError("stopped part-way")

        builder._write_tags = first_only  # writes 1 of 3, then the job stops
        job_id = add_tags(client, True).json()["job_id"]
        assert wait(client, job_id)["status"] == "failed"
        del builder._write_tags
        db.update_job(conn, job_id, status="interrupted")  # as a restart leaves it
        spec = db.get_collection(conn, "c")
        assert spec.recipe == "documents_tagged" and spec.vocabulary_version is not None
        assert sum(1 for h in passages(client) if h.tags) == 1
        body = client.get("/collections/c").json()
        assert body["keyword_tags"] == {"state": "stopped", "job_id": job_id}
        assert body["methods_unavailable"]["keyword_rag"] == NEEDS_TAGS
        query = {"collection": "c", "question": "alpha", "methods": ["keyword_rag"]}
        result = client.post("/query", json=query).json()["results"]["keyword_rag"]
        assert result["error"] == NEEDS_TAGS and result["hits"] == []
        assert client.app.state.chats.why_not(spec, "keyword_rag") == NEEDS_TAGS
        r = client.patch("/collections/c", json={"chat_method": "keyword_rag"})
        assert r.status_code == 409 and NEEDS_TAGS in r.json()["detail"]
        assert add_tags(client, False).status_code == 409  # switched already: resume it
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": False})
        assert r.status_code == 200, r.text
        assert set(r.json()["estimate"]) == {"est_usd", "calls"}
        assert db.get_job(conn, job_id)["status"] == "interrupted"  # only estimated
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": True})
        assert r.status_code == 202 and r.json() == {"job_id": job_id}
        assert wait(client, job_id)["status"] == "done"
        assert all(h.tags for h in passages(client))
        body = client.get("/collections/c").json()
        assert "keyword_rag" in body["methods_available"]
        assert body["keyword_tags"]["state"] == "built"
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": True})
        assert r.status_code == 409  # done: nothing to resume


def test_cached_tag_build_resume_needs_no_model(tmp_path):
    # Passes on today's code too ($0 skips need_ai): it pins that the calls rule does not
    # over-refuse.
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        ai, builder = client.app.state.ai, client.app.state.builder
        free_disabled_model(client, "keywords")
        ai.update_model("gw", "llama", enabled=True)  # at $0 on gw/llama: replies cache under it

        def stop(spec, terms):  # after induction's reply is cached, before the vocabulary
            raise RuntimeError("stopped before the vocabulary")

        builder.save_vocabulary = stop  # the build's call after induction
        job_id = add_tags(client, True).json()["job_id"]
        assert wait(client, job_id)["status"] == "failed"
        del builder.save_vocabulary
        ai.update_model("gw", "llama", enabled=False)
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": False})
        assert r.status_code == 200 and r.json()["estimate"]["calls"] == 0, r.text
        r = client.post(f"/jobs/{job_id}/resume", json={"confirm": True})
        assert r.status_code == 202, r.text
        assert wait(client, job_id)["status"] == "done"


def test_only_the_newest_tag_build_resumes(client):
    documents_with_passages(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    older = db.create_job(conn, "tag_build", "c", {})
    db.update_job(conn, older, status="interrupted")
    newer = db.create_job(conn, "tag_build", "c", {})
    db.update_job(conn, newer, status="cancelled")
    for confirm in (False, True):
        r = client.post(f"/jobs/{older}/resume", json={"confirm": confirm})
        assert r.status_code == 409, r.text
        assert r.json()["detail"] == "a newer tag build of 'c' exists; resume that one"
    assert db.get_job(conn, older)["status"] == "interrupted"
    assert client.post(f"/jobs/{newer}/resume", json={"confirm": False}).status_code == 200


def test_resuming_a_stopped_tag_build_refuses_a_changed_price(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        conn = client.app.state.conn
        client.app.state.worker.stop()
        job_id = db.create_job(conn, "tag_build", "c", {})
        db.update_job(conn, job_id, status="interrupted")
        seen = client.post(f"/jobs/{job_id}/resume", json={"confirm": False}).json()["estimate"]
        r = client.post(f"/jobs/{job_id}/resume", json=stale(seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        assert db.get_job(conn, job_id)["status"] == "interrupted"
        shown = {"confirm": True, "calls": seen["calls"], "est_usd": seen["est_usd"]}
        assert client.post(f"/jobs/{job_id}/resume", json=shown).status_code == 202


def test_stale_confirm_with_the_model_off_is_409(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        tags_seen = add_tags(client, False).json()["estimate"]
        job = stopped_eval_on(client, "r")  # a second store, so c stays a Documents store
        resume_seen = client.post(f"/jobs/{job}/resume", json={"confirm": False}).json()["estimate"]
        add_collection(client, lists=False, name="p")  # no lists: a proposal prices
        propose = "/collections/p/keyword-lists/propose"
        propose_seen = client.post(propose, json={}).json()["estimate"]
        free_disabled_model(client, "keywords", "answer")  # off: a fresh confirm would be 503
        r = client.post("/collections/c/keyword-tags", json=stale(tags_seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        r = client.post(f"/jobs/{job}/resume", json=stale(resume_seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        r = client.post(propose, json=stale(propose_seen))
        assert r.status_code == 409 and r.json()["detail"] == PRICE_CHANGED
        assert ledger_purposes(client) == []


def test_no_keywords_model_is_503(client):
    documents_with_passages(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    client.app.state.ai.set_role("default", None)  # no model for keywords at all
    shown = {"confirm": True, "calls": 3, "est_usd": 0.01}  # what a page drew before
    for body in ({"confirm": False}, {"confirm": True}, shown):
        r = client.post("/collections/c/keyword-tags", json=body)
        assert (r.status_code, r.json()["detail"]) == (503, NO_KEYWORDS_MODEL), body
    job_id = db.create_job(conn, "tag_build", "c", {})
    db.update_job(conn, job_id, status="interrupted")
    for body in ({"confirm": False}, {"confirm": True}, shown):
        r = client.post(f"/jobs/{job_id}/resume", json=body)
        assert (r.status_code, r.json()["detail"]) == (503, NO_KEYWORDS_MODEL), body
    assert db.get_job(conn, job_id)["status"] == "interrupted"
    assert ledger_purposes(client) == []
