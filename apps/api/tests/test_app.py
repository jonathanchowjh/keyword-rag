import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ragqa_api.app import create_app
from ragqa_api.config import load_settings
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.store import Store
from ragqa_core.types import KeywordLists

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


def settings(tmp_path: Path, **overrides):
    base = load_settings({"ANTHROPIC_API_KEY": "test"})
    return replace(base, home=tmp_path, **overrides)


def make_client(tmp_path, reply=lambda p: "Answer: A", **overrides) -> TestClient:
    app = create_app(settings(tmp_path, **overrides))
    app.state.store = Store(None, None)
    app.state.models = fake_models()
    s = app.state.settings
    app.state.claude = fake_claude(app.state.conn, reply, api_key=s.api_key, cap_usd=s.cap_usd)
    app.state.rewire()
    return TestClient(app)


def add_collection(client, summarize=False, lists=True, name="c"):
    r = client.post("/collections", json={"name": name, "qa_type": "mcq", "summarize": summarize})
    assert r.status_code == 201, r.text
    r = client.post(f"/collections/{name}/documents", json={"items": ITEMS, "confirm": True})
    assert r.status_code == 200, r.text
    if lists:
        assert (
            client.put(f"/collections/{name}/keyword-lists", json=asdict(LISTS)).status_code == 200
        )


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


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
    client.put("/collections/c/keyword-lists", json=asdict(LISTS))
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
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 0


def wait(client, job_id, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_eval_flow(client):
    add_collection(client)
    from ragqa_core.db import save_questions
    from ragqa_core.types import Question

    save_questions(
        client.app.state.conn,
        "c",
        [
            Question(
                "q1",
                "enzyme rate",
                ["x", "y"],
                ["A"],
                [],
                {"skill": "enzymes", "solution": "Rates rise."},
            )
        ],
    )
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"], "answer": True})
    assert r.status_code == 200 and r.json()["estimate"]["calls"] == 1
    r = client.post(
        "/eval", json={"collection": "c", "methods": ["hybrid"], "answer": True, "confirm": True}
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert wait(client, job_id)["status"] == "done"
    table = client.get(f"/eval/{job_id}").json()
    assert table["methods"]["hybrid"]["accuracy"]["mean"] == 1.0
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
    from ragqa_api import app as app_module
    from ragqa_core.types import Item, Question

    def fake_load(name, cache):
        items = [ragqa_item(i) for i in ITEMS]
        return items, [Question("q1", "enzyme rate", None, ["x"], ["enzyme-item"], {})]

    def ragqa_item(d):
        return Item(d["id"], d["text"] + " shared words", d["answer"], d["explanation"], d["meta"])

    monkeypatch.setattr(app_module, "load_dataset", fake_load)
    client = make_client(tmp_path, api_key=None)
    client.post("/collections", json={"name": "m", "qa_type": "multihop"})
    r = client.post("/collections/m/load", json={"dataset": "musique"})
    assert r.status_code == 202
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    assert job["result"]["added"] == 3 and job["result"]["lists_source"] == "tfidf"
    assert client.get("/collections/m").json()["keyword_lists"]["common"]


def test_load_falls_back_to_tfidf_when_claude_proposes_nothing(tmp_path, monkeypatch):
    from ragqa_api import app as app_module
    from ragqa_core.types import Item

    items = [Item(d["id"], d["text"] + " shared words", d["answer"]) for d in ITEMS]
    monkeypatch.setattr(app_module, "load_dataset", lambda name, cache: (items, []))
    client = make_client(tmp_path, reply=lambda p: "I cannot help with that.")
    client.post("/collections", json={"name": "m", "qa_type": "multihop"})
    job = wait(
        client, client.post("/collections/m/load", json={"dataset": "musique"}).json()["job_id"]
    )
    assert job["status"] == "done", job
    assert job["result"]["lists_source"] == "tfidf (claude proposal unusable)"
    assert client.get("/collections/m").json()["keyword_lists"]["topic"]


def test_propose_lists(client):
    add_collection(client, lists=False)
    client.app.state.claude = fake_claude(
        client.app.state.conn, lambda p: '{"topic": ["enzymes"], "method": ["assay"]}'
    )
    client.app.state.rewire()
    lists = client.post("/collections/c/keyword-lists/propose").json()
    assert lists["topic"] == ["enzymes"] and lists["method"] == ["assay"]


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


def test_half_set_auth_is_503(tmp_path):
    r = TestClient(create_app(settings(tmp_path, username="u"))).get("/health")
    assert r.status_code == 503 and "RAGQA_PASSWORD" in r.json()["detail"]


STATIC = Path(__file__).resolve().parents[1] / "src/ragqa_api/static"
PUBLIC = {"/collections", "/query", "/eval", "/jobs", "/budget", "/health"}


def test_pages_are_served(client):
    for path in ("/lab", "/board"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")


def test_pages_only_call_public_endpoints():
    import re

    for page in ("lab.html", "board.html"):
        text = (STATIC / page).read_text()
        paths = {"/" + p.split("/")[1] for p in re.findall(r"""fetch\(\s*[`'"](/[a-z/]+)""", text)}
        assert paths and paths <= PUBLIC, (page, paths)
        assert "http://" not in text and "https://" not in text  # no external requests


def test_model_mismatch_on_query_and_eval(client):
    add_collection(client)
    client.app.state.models.dense_model = "other/model"
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["hybrid"]})
    assert "built with fake/dense" in r.json()["results"]["hybrid"]["error"]
    r = client.post("/eval", json={"collection": "c", "methods": ["hybrid"]})
    assert r.status_code == 409 and "service is running other/model" in r.json()["detail"]


def test_load_limit_keeps_gold_passages_and_only_answerable_questions():
    from ragqa_api.app import limited
    from ragqa_core.types import Item, Question

    items = [Item(f"i{n}", "t") for n in range(6)]
    questions = [
        Question("q1", "x", None, ["a"], ["i4", "i5"]),
        Question("q2", "x", None, ["a"], ["i0", "i3"]),  # would pass the limit
        Question("q3", "x", None, ["a"], ["i1"]),
        Question("mcq", "x", ["a"], ["A"], []),  # retrieves examples, not gold passages
    ]
    kept, answerable = limited(items, questions, 3)
    assert [i.id for i in kept] == ["i1", "i4", "i5"]
    assert [q.id for q in answerable] == ["q1", "q3", "mcq"]
    assert limited(items, questions, None) == (items, questions)
