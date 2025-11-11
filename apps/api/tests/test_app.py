import json
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragqa_api.app import create_app
from ragqa_core import db
from ragqa_core.fakes import fake_embeddings, fake_replies
from ragqa_core.retriever import METHOD_NAMES
from ragqa_core.settings import load_settings
from ragqa_core.types import KeywordLists, Question

ITEMS = [
    {
        "id": "enzyme-item",
        "text": "enzyme kinetics rate experiment",
        "answer": "A",
        "explanation": "Rates rise.",
        "meta": {"skill": "enzymes"},
    },
    {
        "id": "market-item",
        "text": "stock market price demand",
        "answer": "B",
        "explanation": "Prices fall.",
        "meta": {"skill": "markets"},
    },
    {
        "id": "volcano-item",
        "text": "volcano magma lava pressure",
        "answer": "C",
        "explanation": "Magma rises.",
        "meta": {"skill": "volcanoes"},
    },
]
LISTS = KeywordLists(
    topic=["enzyme", "market", "volcano", "optics"],
    method=["experiment", "survey", "sampling"],
    common=["rate", "price", "lava", "light"],
)
OTHER_LISTS = KeywordLists(topic=["enzyme", "market"], method=["experiment"], common=["rate"])
QUESTION = Question(
    "q1", "enzyme rate", ["x", "y"], ["A"], [], {"skill": "enzymes", "solution": "Rates rise."}
)
EVAL = {"collection": "c", "methods": ["hybrid"], "answer": False}
PAIR = ["kw_individual", "hybrid"]


def settings(tmp_path: Path, **overrides):
    base = load_settings({"ANTHROPIC_API_KEY": "test"})
    fake = {"dense_model": "fake/dense", "judge_model": "fake/judge", "rerank_model": "fake/rerank"}
    return replace(base, home=tmp_path, **{**fake, **overrides})


def make_client(tmp_path, reply=lambda p: "Answer: A", **overrides) -> TestClient:
    """Not started: enter it (`with`) for the lifespan, which runs the worker."""
    app = create_app(settings(tmp_path, **overrides))
    fake_embeddings(app.state.models)
    fake_replies(app.state.claude, reply)
    app.state.worker.idle_seconds = 0.01
    return TestClient(app)


def wait(client, job_id, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def add_collection(client, summarize=False, lists=True, name="c", questions=True):
    r = client.post("/collections", json={"name": name, "qa_type": "mcq", "summarize": summarize})
    assert r.status_code == 201, r.text
    r = client.post(f"/collections/{name}/documents", json={"items": ITEMS, "confirm": True})
    assert r.status_code == 200, r.text
    if questions:
        db.save_questions(client.app.state.conn, name, [QUESTION])
    if lists:
        r = client.put(f"/collections/{name}/keyword-lists", json=asdict(LISTS))
        assert r.status_code == 202, r.text
        assert wait(client, r.json()["job_id"])["status"] == "done"


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"ok": True, "api_key": True}


def test_health_reports_missing_key(tmp_path):
    assert make_client(tmp_path, api_key=None).get("/health").json()["api_key"] is False


def test_create_add_query_flow(client):
    add_collection(client)
    body = {"collection": "c", "question": "enzyme rate", "methods": ["hybrid", "kw_individual"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["hybrid"]["hits"][0]["id"] == "enzyme-item"
    assert r["results"]["kw_individual"]["hits"][0]["matched"]
    assert r["results"]["hybrid"]["answer"] is None and r["results"]["hybrid"]["error"] is None


def test_query_with_answer(client):
    add_collection(client)
    body = {
        "collection": "c",
        "question": "enzyme rate?",
        "choices": ["x", "y"],
        "methods": ["hybrid"],
        "answer": True,
    }
    result = client.post("/query", json=body).json()["results"]["hybrid"]
    assert result["answer"] == "A" and result["usd"] > 0


def test_collection_detail(client):
    add_collection(client, lists=False)
    detail = client.get("/collections/c").json()
    assert detail["count"] == 3 and detail["keyword_lists"] is None
    assert "hybrid" in detail["methods_available"] and "kw_all" not in detail["methods_available"]
    wait(client, client.put("/collections/c/keyword-lists", json=asdict(LISTS)).json()["job_id"])
    detail = client.get("/collections/c").json()
    assert "kw_all" in detail["methods_available"] and detail["keyword_lists"]["topic"]
    assert [c["name"] for c in client.get("/collections").json()] == ["c"]


def test_duplicate_collection_409(client):
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    assert client.post("/collections", json={"name": "c", "qa_type": "mcq"}).status_code == 409


def test_documents_need_confirm_when_they_cost(client):
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/documents", json={"items": ITEMS})
    assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0
    assert client.get("/collections/s").json()["count"] == 0


def test_query_reports_unavailable_method_per_method(client):
    add_collection(client, lists=False)
    body = {"collection": "c", "question": "x", "methods": ["hybrid", "kw_all"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["hybrid"]["error"] is None
    assert "keyword lists not set" in r["results"]["kw_all"]["error"]


def test_query_without_key_keeps_free_methods(client):
    add_collection(client, summarize=True)
    client.app.state.claude.api_key = None
    body = {"collection": "c", "question": "enzyme", "methods": ["keyword", "kw_individual"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["keyword"]["hits"]
    assert r["results"]["kw_individual"]["error"] == "needs ANTHROPIC_API_KEY"


def test_query_unknown_method_422(client):
    add_collection(client)
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["nope"]})
    assert r.status_code == 422


def test_eval_needs_confirm_when_it_costs(client):
    add_collection(client)
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"], "answer": True})
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1
    assert [j["kind"] for j in client.get("/jobs").json()] == ["rekey"]  # no eval queued


def test_eval_of_an_empty_split_is_422(client):
    add_collection(client, questions=False)
    r = client.post("/eval", json={**EVAL, "confirm": True})
    assert r.status_code == 422 and r.json()["detail"] == "collection 'c' has no questions"


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


def test_load_rejects_wrong_qa_type(client):
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    assert client.post("/collections/c/load", json={"dataset": "musique"}).status_code == 409


def test_load_summarize_without_key_503(tmp_path):
    client = make_client(tmp_path, api_key=None)
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/load", json={"dataset": "scienceqa"})
    assert r.status_code == 503 and r.json()["detail"] == "needs ANTHROPIC_API_KEY"


def test_load_runs_job_and_falls_back_to_tfidf_lists(tmp_path, monkeypatch):
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
        assert job["result"]["added"] == 3 and job["result"]["lists_source"] == "tfidf"
        assert client.get("/collections/m").json()["keyword_lists"]["common"]


def test_load_falls_back_to_tfidf_when_claude_proposes_nothing(tmp_path, monkeypatch):
    from ragqa_core import loader
    from ragqa_core.types import Item

    items = [Item(d["id"], d["text"] + " shared words", d["answer"]) for d in ITEMS]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    with make_client(tmp_path, reply=lambda p: "I cannot help with that.") as client:
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        r = client.post("/collections/m/load", json={"dataset": "musique"})
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["lists_source"] == "tfidf (claude proposal unusable)"
        assert client.get("/collections/m").json()["keyword_lists"]["topic"]


def test_propose_lists(client):
    add_collection(client, lists=False)
    fake_replies(client.app.state.claude, lambda p: '{"topic": ["enzymes"], "method": ["assay"]}')
    lists = client.post("/collections/c/keyword-lists/propose").json()
    assert lists["topic"] == ["enzymes"] and lists["method"] == ["assay"]


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


def test_unknown_collection_404(client):
    assert client.post("/query", json={"collection": "nope", "question": "x"}).status_code == 404
    assert client.get("/collections/nope").status_code == 404


def test_unknown_job_404(client):
    assert client.get("/jobs/missing").status_code == 404
    assert client.get("/eval/missing").status_code == 404


def test_bad_name_422(client):
    assert client.post("/collections", json={"name": "a b", "qa_type": "mcq"}).status_code == 422


def test_budget_note(client):
    body = client.get("/budget").json()
    assert "Console" in body["note"] and body["cap_usd"] == 20.0 and body["spent_usd"] == 0.0


def test_budget_exceeded_is_402(tmp_path):
    client = make_client(tmp_path, cap_usd=0.0)
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/documents", json={"items": ITEMS, "confirm": True})
    assert r.status_code == 402 and "monthly cap reached" in r.json()["detail"]


def test_basic_auth(tmp_path):
    c = TestClient(create_app(settings(tmp_path, username="u", password="p")))
    assert c.get("/health").status_code == 401
    assert c.get("/health").headers["www-authenticate"] == "Basic"
    assert c.get("/health", auth=("u", "wrong")).status_code == 401
    assert c.get("/health", auth=("u", "p")).status_code == 200


def test_basic_auth_handles_non_ascii(tmp_path):
    import base64

    c = TestClient(create_app(settings(tmp_path, username="ü", password="p")))

    def header(user, password):
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    assert c.get("/health", headers=header("é", "x")).status_code == 401
    assert c.get("/health", headers=header("ü", "p")).status_code == 200


def test_half_set_auth_is_503(tmp_path):
    r = TestClient(create_app(settings(tmp_path, username="u"))).get("/health")
    assert r.status_code == 503 and "RAGQA_PASSWORD" in r.json()["detail"]


STATIC = Path(__file__).resolve().parents[1] / "src/ragqa_api/static"
PUBLIC = {"/collections", "/query", "/eval", "/jobs", "/budget", "/health", "/catalog", "/status"}
PAGES = ("/", "/c/x", "/c/x/runs/abc", "/worker", "/lab")


def test_pages_are_served(client):
    for path in PAGES:
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    board = client.get("/board", follow_redirects=False)
    assert board.status_code == 307 and board.headers["location"] == "/"
    assert not (STATIC / "board.html").exists()


def test_static_assets_are_served(client):
    assert client.get("/static/ui.css").status_code == 200
    assert "export" in client.get("/static/ui.js").text


def test_static_is_behind_basic_auth(tmp_path):
    c = TestClient(create_app(settings(tmp_path, username="u", password="p")))
    for path in ("/static/ui.js", "/static/ui.css", "/", "/lab"):
        assert c.get(path).status_code == 401
        assert c.get(path, auth=("u", "p")).status_code == 200


def test_pages_only_call_public_endpoints():
    import re

    for path in STATIC.glob("*.[hj][ts]*"):
        text = path.read_text()
        calls = re.findall(r"""(?:fetch|api)\(\s*[`'"](/[a-z/]+)""", text)
        assert {"/" + p.split("/")[1] for p in calls} <= PUBLIC, (path.name, calls)
        assert "http://" not in text and "https://" not in text  # no external requests
    assert {p.name for p in STATIC.glob("*.html")} >= {"collections.html", "lab.html"}


NODE = shutil.which("node")
CHECK = """
import * as u from './ui.mjs';
const out = {
  a: u.pts(6.2, 1.8, 10.4), b: u.pts(-7, -12, -2), c: u.pts(1, -2, 4), d: u.pts(0.4, -3, 3),
  e: u.pct(0.776), f: u.esc('<a href="x">&</a>'),
  g: u.clock('2026-10-23T14:02:00', new Date('2026-10-23T09:00:00')),
  h: u.clock('2026-10-21T14:02:00', new Date('2026-10-23T09:00:00')),
  i: u.verdictText(null, {}),
};
const labels = {methods: {kw_individual: {label: 'Keywords per list'}, hybrid: {label: 'Hybrid'}},
                metrics: {accuracy: {label: 'Answered correctly'}}};
const ans = (verdict, diff, lo, hi) => u.verdictText(
  {candidate: 'kw_individual', reference: 'hybrid', metric: 'accuracy', verdict, diff, lo, hi},
  labels);
out.j = ans('yes', 6.2, 1.8, 10.4);
out.k = ans('no', -7, -12, -2);
out.l = ans('cant_tell', 1, -2, 4);
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_ui_formatting(tmp_path):
    (tmp_path / "ui.mjs").write_text((STATIC / "ui.js").read_text())
    (tmp_path / "check.mjs").write_text(CHECK)
    r = subprocess.run([NODE, "check.mjs"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["a"] == "+6 pts (2–10)"
    assert got["b"] == "−7 pts (−12 to −2)"
    assert got["c"] == "+1 pt (−2 to 4)"
    assert got["d"] == "+0 pts (−3 to 3)"
    assert got["e"] == "78%"
    assert got["f"] == "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;"
    assert got["g"] == "14:02" and got["h"] == "21 Oct 14:02"
    assert got["i"] == "No answer yet"
    assert got["j"] == "Yes: Keywords per list +6 pts (2–10) answered correctly vs Hybrid"
    assert got["k"].startswith("No: Keywords per list −7 pts")
    assert got["l"].startswith("Can't tell: Keywords per list +1 pt")


def test_model_mismatch_on_query_and_eval(client):
    add_collection(client)
    client.app.state.models.dense_model = "other/model"
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["hybrid"]})
    assert "built with fake/dense" in r.json()["results"]["hybrid"]["error"]
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"]})
    assert r.status_code == 409 and "service is running other/model" in r.json()["detail"]


def set_running(conn, job_id, progress=0.0):
    conn.execute("update jobs set status='running', progress=? where id=?", (progress, job_id))


def test_restart_keeps_queued_jobs_and_interrupts_running(tmp_path):  # Review focus 5
    app = create_app(settings(tmp_path))
    conn = app.state.conn
    q = db.create_job(conn, "rekey", "c", {"version": "v"})
    r = db.create_job(conn, "load", "c", {})
    conn.execute("update jobs set status='running' where id=?", (r,))
    db.interrupt_running(conn)
    assert (
        db.get_job(conn, r)["status"] == "interrupted" and db.get_job(conn, q)["status"] == "queued"
    )
    # The lifespan does the same before it starts the worker (kept off here, so q stays queued).
    r2 = db.create_job(conn, "load", "c", {})
    set_running(conn, r2)
    app.state.worker.start = lambda: None
    with TestClient(app):
        assert db.get_job(conn, r2)["status"] == "interrupted"
        assert db.get_job(conn, q)["status"] == "queued"


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


def test_delete_collection_cancels_queued_and_refuses_running(client):  # Review focus 3
    add_collection(client)
    client.app.state.worker.stop()
    j = client.post("/eval", json={**EVAL, "confirm": True}).json()["job_id"]
    assert client.delete("/collections/c").status_code == 204
    assert client.get(f"/jobs/{j}").status_code == 404
    assert client.get("/collections/c").status_code == 404
    assert not client.app.state.store.client.collection_exists("c")
    assert db.get_questions(client.app.state.conn, "c") == []
    add_collection(client, name="d", lists=False)
    running = db.create_job(client.app.state.conn, "load", "d", {})
    set_running(client.app.state.conn, running)
    r = client.delete("/collections/d")
    assert r.status_code == 409 and "running" in r.json()["detail"]
    assert client.get("/collections/d").json()["count"] == 3
    assert client.delete("/collections/nope").status_code == 404


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
        "usd_total",
        "answer_summary",
    }
    assert run["status"] == "done" and run["methods"] == PAIR and run["split"] == "dev"
    assert run["answer"] is False and run["question_count"] == 1 and run["usd_total"] == 0.0
    assert run["model"] == client.app.state.claude.default_model
    assert run["answer_summary"] == client.get(f"/eval/{done}").json()["answer"]
    assert run["answer_summary"] is not None
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
    j = client.post("/eval", json={**EVAL, "answer": True, "confirm": True}).json()["job_id"]
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
    # A run whose keyword lists changed since it ran is refused.
    db.update_job(conn, j, status="failed")
    put = client.put("/collections/c/keyword-lists", json=asdict(OTHER_LISTS)).json()
    assert wait(client, put["job_id"])["status"] == "done"
    r = client.post(f"/jobs/{j}/resume")
    assert r.status_code == 409 and "keyword lists changed since this run" in r.json()["detail"]
    rekey = client.get("/jobs", params={"kind": "rekey"}).json()[0]["id"]
    r = client.post(f"/jobs/{rekey}/resume")
    assert r.status_code == 409 and r.json()["detail"] == "only evals can be resumed"
    assert client.post("/jobs/missing/resume").status_code == 404


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
        "insert into pending_batches (id, model, purpose, created_at, usd, job_id)"
        " values ('b1', 'm', 'answer', 't', 0.5, ?)",
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
        }
    ]
    assert body["problems"] == [
        "The worker thread has stopped; restart the service",
        "Worker error: RuntimeError('x')",
        f"Load {failed[:4]} on c failed",
        f"Eval {stopped[:4]} on c stopped at the monthly cap",
    ]


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
        line = f"Load {job[:8]} on c has made no progress since {at:%H:%M}"
        assert (line in client.get("/status").json()["problems"]) is stalled
    # Waiting on a Claude batch can take hours: not a stall.
    conn.execute(
        "insert into pending_batches (id, model, purpose, created_at, usd, job_id)"
        " values ('b1', 'm', 'answer', 't', 0.5, ?)",
        (job,),
    )
    assert client.get("/status").json()["problems"] == []


def test_catalog_lists_every_method(client):
    body = client.get("/catalog").json()
    assert set(body["methods"]) == set(METHOD_NAMES)
    assert body["methods"]["kw_individual"]["label"] == "Keywords per list"
    assert body["answer_pair"] == PAIR and body["qa_types"]["mcq"]["label"]


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
