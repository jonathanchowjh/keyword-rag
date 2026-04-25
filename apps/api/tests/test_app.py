import json
import re
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
from ragqa_core.types import Item, KeywordLists, Question

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


def test_query_on_a_science_collection_without_choices_returns_free_text(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Answer: 3.1 J") as client:
        client.post("/collections", json={"name": "sci", "qa_type": "science", "summarize": False})
        client.post("/collections/sci/documents", json={"items": ITEMS, "confirm": True})
        body = {"collection": "sci", "question": "enzyme rate?", "methods": ["vector"]}
        result = client.post("/query", json={**body, "answer": True}).json()["results"]["vector"]
    assert result["answer"] == "3.1 J"


def test_query_usd_counts_every_claude_call_of_the_method(client):
    add_collection(client, summarize=True)
    body = {"collection": "c", "question": "enzyme rate", "methods": ["kw_individual", "hybrid"]}
    results = client.post("/query", json=body).json()["results"]
    assert results["kw_individual"]["usd"] > 0  # the question's summary
    assert results["hybrid"]["usd"] == 0


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
        r = client.post("/collections/m/load", json={"dataset": "musique", "confirm": True})
        job = wait(client, r.json()["job_id"])
        assert job["status"] == "done", job
        assert job["result"]["lists_source"] == "tfidf (claude proposal unusable)"
        assert client.get("/collections/m").json()["keyword_lists"]["topic"]


def ledger_purposes(client):
    rows = client.app.state.conn.execute("select purpose from ledger")
    return [r["purpose"] for r in rows]


def three_items(monkeypatch):
    from ragqa_core import loader
    from ragqa_core.types import Item

    items = [Item(d["id"], d["text"] + " shared words", d["answer"]) for d in ITEMS]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    monkeypatch.setattr("ragqa_api.app.load_dataset", lambda name, cache: (items, []))
    return items


def test_load_shows_its_propose_cost_first(tmp_path, monkeypatch):
    three_items(monkeypatch)
    reply = lambda p: '{"topic": ["enzymes"], "method": ["assay"]}'  # noqa: E731
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
        assert job["result"]["lists_source"] == "claude"
        assert ledger_purposes(client) == ["propose"]


def test_load_that_spends_nothing_queues_without_confirm(tmp_path, monkeypatch):
    three_items(monkeypatch)
    with make_client(tmp_path, api_key=None) as client:  # tfidf lists, no Claude
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
        summaries = client.app.state.indexer.estimate_add_usd(spec, items, "hand")
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


def test_propose_lists(client):
    add_collection(client, lists=False)
    fake_replies(client.app.state.claude, lambda p: '{"topic": ["enzymes"], "method": ["assay"]}')
    url = "/collections/c/keyword-lists/propose"
    spent = client.get("/budget").json()["spent_usd"]
    r = client.post(url)
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1
    assert r.json()["estimate"]["est_usd"] > 0
    assert client.get("/budget").json()["spent_usd"] == spent  # the estimate spent nothing
    lists = client.post(url, json={"confirm": True}).json()
    assert lists["topic"] == ["enzymes"] and lists["method"] == ["assay"]


def test_propose_lists_needs_a_key(client):
    add_collection(client, lists=False)
    client.app.state.claude.api_key = ""
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
    for path in PAGES:  # /favicon.ico is not served, so every page opts out of asking for it
        assert 'rel="icon"' in client.get(path).text, path
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

    called: set[str] = set()
    for path in STATIC.glob("*.[hj][ts]*"):
        text = path.read_text()
        assert not re.search(r"(?:fetch|api)\(\s*`\$\{", text), path.name  # unreadable target
        for _, target in re.findall(r"""(?:fetch|api)\(\s*([`'"])(/[^`'"]*)""", text):
            called.add("/" + re.match(r"/?([A-Za-z0-9_-]*)", target).group(1))
        assert "http://" not in text and "https://" not in text  # no external requests
    assert called <= PUBLIC, called - PUBLIC
    assert {"/collections", "/query", "/catalog", "/status", "/budget"} <= called
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
  n: [u.pct(null), u.pct(undefined), u.pct(NaN), u.pct(Infinity), u.pct(0)],
  o: [u.pts(undefined, 1, 2), u.pts(null, 1, 2), u.pts(NaN, 1, 2), u.pts(-2.5, -4, -1)],
  p: [u.clock('garbage'), u.clock(null), u.clock(undefined)],
  q: [u.pts(2.5, 1.5, 3.5), u.pts(5, null, undefined)],
};
const labels = {methods: {kw_individual: {label: 'Keywords per list'}, hybrid: {label: 'Hybrid'}},
                metrics: {accuracy: {label: 'Answered correctly'}, em: {label: 'Exact match'}},
                metric_labels_by_type: {table: {em: 'Exact answer'}}};
const ans = (verdict, diff, lo, hi, metric = 'accuracy', qaType = 'mcq') => u.verdictText(
  {candidate: 'kw_individual', reference: 'hybrid', metric, verdict, diff, lo, hi},
  qaType, labels);
out.j = ans('yes', 6.2, 1.8, 10.4);
out.k = ans('no', -7, -12, -2);
out.l = ans('cant_tell', 1, -2, 4);
out.t = ans('yes', 6, 2, 10, 'em', 'table');
out.u = ans('yes', 6, 2, 10, 'em', 'mcq');
out.v = ['rekeying', 'stopped_budget', 'stopped_no_key', 'cancelled'].map(u.pill);
out.w = [u.count(11656), u.count(0)];
out.x = [0.004, 0, 0.4141, 12].map(u.usd);
const sim = {metric: 'solution_sim@1', proxy_trusted: true};
out.y = [{unit: 'raw'}, {unit: 'match'}].map((scale) => u.verdictNotes({...sim, scale}));
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
    assert got["n"] == ["–", "–", "–", "–", "0%"]
    assert got["o"] == ["–", "–", "–", "−3 pts (−4 to −1)"]
    assert got["p"] == ["–", "–", "–"]
    assert got["q"] == ["+3 pts (2–4)", "+5 pts"]
    assert got["j"] == "Yes: Keywords per list +6 pts (2–10) answered correctly vs Hybrid"
    assert got["k"].startswith("No: Keywords per list −7 pts")
    assert got["l"].startswith("Can't tell: Keywords per list +1 pt")
    assert got["t"] == "Yes: Keywords per list +6 pts (2–10) exact answer vs Hybrid"
    assert got["u"].endswith("exact match vs Hybrid")  # no label for this type: the plain one
    assert [re.sub("<[^>]*>", "", v) for v in got["v"]] == [
        "Re-keying",
        "Stopped budget",
        "Stopped no key",
        "Cancelled",
    ]
    assert got["w"] == ["11,656", "0"]
    assert got["x"] == ["< $0.01", "$0.00", "$0.41", "$12.00"]
    raw, match = got["y"]  # a verdict's own scale marks a raw reasoning match
    assert len(raw) == 1 and "no chance level" in raw[0] and match == []


def test_model_mismatch_on_query_and_eval(client):
    add_collection(client)
    client.app.state.models.dense_model = "other/model"
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["hybrid"]})
    assert "built with fake/dense" in r.json()["results"]["hybrid"]["error"]
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"]})
    assert r.status_code == 409 and "service is running other/model" in r.json()["detail"]


def test_saving_lists_is_refused_when_the_dense_model_changed(client):
    add_collection(client, lists=False)
    client.app.state.models.dense_model = "other/model"
    r = client.put("/collections/c/keyword-lists", json=asdict(LISTS))
    assert r.status_code == 409 and "service is running other/model" in r.json()["detail"]
    assert client.get("/collections/c/keyword-lists").json() == []


def test_confirmed_paid_eval_without_a_key_is_503_and_queues_nothing(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        add_collection(client)
        client.app.state.worker.stop()
        body = {**EVAL, "answer": True, "confirm": True}
        r = client.post("/eval", json=body)
        assert r.status_code == 503 and r.json()["detail"] == "needs ANTHROPIC_API_KEY"
        assert client.get("/eval").json() == []
        free = client.post("/eval", json={**EVAL, "confirm": True})  # nothing to pay for
        assert free.status_code == 202


def test_confirmed_paid_resume_without_a_key_is_503(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        add_collection(client)
        client.app.state.worker.stop()
        conn = client.app.state.conn
        job = client.app.state.evaluator.queue("c", "dev", ["hybrid"], True, None)
        db.update_job(conn, job, status="stopped_no_key")
        r = client.post(f"/jobs/{job}/resume", json={"confirm": True})
        assert r.status_code == 503 and db.get_job(conn, job)["status"] == "stopped_no_key"


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


def test_answer_carries_proxy_trusted(client):
    add_collection(client)
    run = client.post("/eval", json={**EVAL, "methods": PAIR, "confirm": True}).json()["job_id"]
    wait(client, run)
    answer = client.get("/collections/c").json()["answer"]
    assert "proxy_trusted" in answer
    assert answer["proxy_trusted"] == client.app.state.evaluator.collection_proxy_trusted("c")


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


def test_documents_are_refused_while_a_load_rekey_or_eval_is_queued_or_running(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    body = {"items": ITEMS[:1], "confirm": True}
    for kind in ("load", "rekey", "eval"):
        for status in ("queued", "running"):
            job = db.create_job(conn, kind, "c", {})
            db.update_job(conn, job, status=status)
            r = client.post("/collections/c/documents", json=body)
            assert r.status_code == 409 and f"a {kind} job" in r.json()["detail"], (kind, status)
            db.update_job(conn, job, status="done")
    assert client.post("/collections/c/documents", json=body).status_code == 200


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


def test_a_recreated_collection_does_not_reuse_the_controls_id_cache(client):
    add_collection(client, lists=False, questions=False)
    query = {"collection": "c", "question": "enzyme", "methods": ["random_example"], "k": 3}

    def hit_ids() -> set[str]:
        hits = client.post("/query", json=query).json()["results"]["random_example"]["hits"]
        return {h["id"] for h in hits}

    assert hit_ids() == {i["id"] for i in ITEMS}
    assert client.delete("/collections/c").status_code == 204
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    fresh = [{**i, "id": "new-" + i["id"]} for i in ITEMS]
    client.post("/collections/c/documents", json={"items": fresh, "confirm": True})
    assert hit_ids() == {i["id"] for i in fresh}


def run_pair(client) -> str:
    body = {"collection": "c", "methods": PAIR + ["random_example"], "answer": False}
    job_id = client.post("/eval", json={**body, "confirm": True}).json()["job_id"]
    assert wait(client, job_id)["status"] == "done"
    return job_id


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


def test_since_names_a_changed_dense_model(client):
    add_collection(client)
    job_id = run_pair(client)
    conn = client.app.state.conn
    conn.execute("update collections set dense_model = 'other/model' where name = 'c'")
    expected = {"untracked": False, "changes": ["dense model"]}
    assert client.get(f"/eval/{job_id}").json()["since"] == expected


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


def seed_rows(client, job_id, rows):
    """Replace a run's rows: {method: {question_id: row}}."""
    conn = client.app.state.conn
    conn.execute("delete from run_items where job_id = ?", (job_id,))
    for method, per_q in rows.items():
        for qid, row in per_q.items():
            db.upsert_run_item(conn, job_id, method, qid, row, complete=True)


def row(hit, right=None, retrieval=None):
    out = {"hit_ids": [hit], "hit_skills": ["s"], "hit_matched": [["rate"]], "retrieval": {}}
    if retrieval is not None:
        out["retrieval"] = retrieval
    if right is not None:
        out["pred"] = "A"
        out["answer"] = {"accuracy": float(right)}
    return out


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
    qs = [Question(f"q{i}", f"enzyme {i}", ["x", "y"], ["A"], [], {}) for i in range(1, 5)]
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


def write_legacy(app, name="c"):
    models, store = app.state.models, app.state.store
    app.state.indexer.create(name, "mcq", False)
    items = [Item("sqa-1", "enzyme rate", "A", None, {}), Item("note-1", "market price", "B")]
    texts = [i.text for i in items]
    store.upsert_items(name, items, models.embed(texts), models.sparse(texts), [None, None])
    store.upsert_chunks(name, [i.id for i in items], texts, models.embed(texts), [None, None])


def backfills(conn):
    return [j for j in db.list_jobs(conn) if j["kind"] == "backfill"]


def test_startup_queues_one_backfill_for_unlabelled_collections(tmp_path):
    client = make_client(tmp_path)
    write_legacy(client.app)
    with client:
        jobs = backfills(client.app.state.conn)
        assert len(jobs) == 1
        assert wait(client, jobs[0]["id"])["status"] == "done"
        assert client.app.state.store.unsourced("c") == 0
    with client:
        assert len(backfills(client.app.state.conn)) == 1


def test_labelled_collections_queue_no_backfill(tmp_path):
    client = make_client(tmp_path)
    with client:
        add_collection(client, lists=False)
    with client:
        assert backfills(client.app.state.conn) == []
