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
from ragqa_core.fakes import fake_ai, fake_embeddings, fake_replies
from ragqa_core.retriever import METHOD_NAMES
from ragqa_core.settings import load_settings
from ragqa_core.types import CollectionSpec, Item, KeywordLists, Question

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
    base = load_settings({})
    fake = {
        "dense_model": "fake/dense",
        "judge_model": "fake/judge",
        "rerank_model": "fake/rerank",
        "build_hub": "",  # tests never read the real Hub
    }
    return replace(base, home=tmp_path, **{**fake, **overrides})


def make_client(
    tmp_path,
    reply=lambda p: "Answer: A",
    api_key: str | None = "test",
    cap_usd: float = 20.0,
    **overrides,
) -> TestClient:
    """Not started: enter it (`with`) for the lifespan, which runs the worker. The key is
    stored, not from the environment, so a test can clear it; with none, models are still
    set up and priced, so estimates work and spending is refused."""
    app = create_app(settings(tmp_path, **overrides))
    fake_embeddings(app.state.models)
    fake_ai(app.state.conn, reply, api_key, cap_usd)  # stored in the app's file
    fake_replies(app.state.ai, reply)
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
    assert client.get("/health").json() == {"ok": True, "ai": True}


def test_health_reports_missing_key(tmp_path):
    assert make_client(tmp_path, api_key=None).get("/health").json()["ai"] is False


def test_create_add_query_flow(client):
    add_collection(client)
    body = {"collection": "c", "question": "enzyme rate", "methods": ["hybrid", "kw_individual"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["hybrid"]["hits"][0]["id"] == "enzyme-item"
    assert r["results"]["kw_individual"]["hits"][0]["matched"]
    assert r["results"]["hybrid"]["answer"] is None and r["results"]["hybrid"]["error"] is None


def test_create_without_recipe_is_paper_2024(client):
    add_collection(client)
    body = client.get("/collections/c").json()
    assert (body["recipe"], body["corpus_type"]) == ("paper_2024", "worked")
    assert body["build"]["added_text"] == "none" and body["build"]["chunk_overlap"] == 25
    assert body["search"]["weights"] == {"meaning": 1.0, "words": 1.0, "keywords": 1.0}
    assert body["vocabulary_version"] is None and body["profile"] == {}
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


def test_query_answers_from_the_program_and_the_example_setting(tmp_path):
    sent = []

    def reply(prompt):
        sent.append(prompt)
        return "plan\n```python\nprint(6*7)\n```\nAnswer: 41"

    with make_client(tmp_path, reply=reply) as client:
        client.post("/collections", json={"name": "sci", "qa_type": "science", "summarize": False})
        client.post("/collections/sci/documents", json={"items": ITEMS, "confirm": True})
        body = {"collection": "sci", "question": "enzyme rate?", "methods": ["vector"]}
        ask = {**body, "answer": True, "search": {"program": True, "example": "off"}}
        result = client.post("/query", json=ask).json()["results"]["vector"]
        assert result["answer"] == "42" and result["program"]["ran"] is True
        assert "```python" in sent[-1] and "Example" not in sent[-1]
        chosen = {**ask, "choices": ["x", "y"], "search": {"program": True}}
        result = client.post("/query", json=chosen).json()["results"]["vector"]
        assert result["answer"] == "" and result["program"] is None
        assert "```python" not in sent[-1] and "Example problem" in sent[-1]


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
    assert "keyword_rag" not in detail["methods_available"]  # no vocabulary
    assert [c["name"] for c in client.get("/collections").json()] == ["c"]


def test_duplicate_collection_409(client):
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    assert client.post("/collections", json={"name": "c", "qa_type": "mcq"}).status_code == 409


def test_documents_need_confirm_when_they_cost(client):
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/documents", json={"items": ITEMS})
    assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0
    assert client.get("/collections/s").json()["count"] == 0


def test_a_query_answered_by_hand_says_its_prompt_waits(client):
    add_collection(client)
    ai = client.app.state.ai
    client.post("/ai/connections", json={"preset": "manual", "id": "hand"})
    client.post(
        "/ai/connections/hand/models", json={"model_id": "m", "price_in": 0, "price_out": 0}
    )
    ai.set_role("answer", "hand/m")
    body = {"collection": "c", "question": "enzyme?", "choices": ["x", "y"], "answer": True}
    result = client.post("/query", json={**body, "methods": ["hybrid"]}).json()["results"]
    assert "1 prompt wait for replies by hand" in result["hybrid"]["error"]
    [waiting] = ai.waiting()
    ai.answer({waiting["key"]: "Answer: B"})
    result = client.post("/query", json={**body, "methods": ["hybrid"]}).json()["results"]
    assert result["hybrid"]["answer"] == "B" and result["hybrid"]["usd"] == 0


def test_query_reports_unavailable_method_per_method(client):
    add_collection(client, lists=False)
    body = {"collection": "c", "question": "x", "methods": ["hybrid", "kw_all"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["hybrid"]["error"] is None
    assert "keyword lists not set" in r["results"]["kw_all"]["error"]


def test_query_without_key_keeps_free_methods(client):
    add_collection(client, summarize=True)
    client.app.state.ai.update_connection("anthropic", api_key=None)
    body = {"collection": "c", "question": "enzyme", "methods": ["keyword", "kw_individual"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["keyword"]["hits"]
    assert "'query' job" in r["results"]["kw_individual"]["error"]


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


def test_env_key_becomes_anthropic_connection(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456"))
    assert app.state.ai.connections()[0]["id"] == "anthropic"
    assert app.state.ai.connections()[0]["from_env"] == ["api_key", "base_url"]
    assert app.state.ai.ready("answer")


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


def ledger_purposes(client):
    rows = client.app.state.conn.execute("select purpose from ai_ledger")
    return [r["purpose"] for r in rows]


def three_items(monkeypatch):
    from ragqa_core import loader
    from ragqa_core.types import Item

    items = [Item(d["id"], d["text"] + " shared words", d["answer"]) for d in ITEMS]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    monkeypatch.setattr("ragqa_core.loader.load_dataset", lambda name, cache: (items, []))
    return items


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
        quote = {"est_usd": 0.0, "items": 3, "calls": 0, "hub": hub}
        client.app.state.loader.estimate_load = lambda spec, dataset, limit: quote
        for confirm in (False, True):
            r = client.post("/collections/w/load", json={"dataset": "scibench", "confirm": confirm})
            assert r.status_code == 202, r.text
            setup = client.get(f"/jobs/{r.json()['job_id']}").json()["setup"]
            assert setup.get("hub_entry") == (None if confirm else hub["entry"])


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


WORKED = {"name": "w", "qa_type": "science", "recipe": "worked_examples"}


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


def stored_worked(client) -> None:
    """Collection w (worked examples, AI tagging) with ITEMS stored."""
    client.post("/collections", json=WORKED)
    app = client.app.state
    spec = app.indexer.spec("w")
    app.indexer.add(spec, [Item(**i) for i in ITEMS], "hand")


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


def test_vocabulary_retag_runs_and_switches(client):
    stored_worked(client)
    r = client.put("/collections/w/vocabulary", json={"terms": ["enzymes"], "confirm": True})
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done" and job["result"] == {"version": r.json()["version"]}
    assert client.get("/collections/w/vocabulary").json()["version"] == r.json()["version"]
    assert client.get("/collections/w").json()["profile"] == {"enzymes": 3}


def reference_with_vocabulary(client) -> None:
    """Collection r (reference recipe) with ITEMS stored and tagged from a vocabulary."""
    client.post("/collections", json={"name": "r", "qa_type": "multihop", "recipe": "reference"})
    app = client.app.state
    app.indexer.add(app.indexer.spec("r"), [Item(**i) for i in ITEMS], "hand")
    terms = {"terms": ["enzyme", "market", "volcano"], "confirm": True}
    r = client.put("/collections/r/vocabulary", json=terms)
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"


def test_query_returns_channels_and_tags(client):
    reference_with_vocabulary(client)
    assert "keyword_rag" in client.get("/collections/r").json()["methods_available"]
    body = {"collection": "r", "question": "enzyme rate", "methods": ["keyword_rag", "hybrid"]}
    results = client.post("/query", json=body).json()["results"]
    found = results["keyword_rag"]
    assert found["error"] is None and found["query_tags"][0] == "enzyme"
    top = found["hits"][0]
    assert top["id"] == "enzyme-item" and top["tags"][0] == "enzyme"
    assert top["channels"] == ["meaning", "words", "keywords"] and "enzyme" in top["matched"]
    assert "query_tags" not in results["hybrid"]
    weighted = {**body, "search": {"weights": {"meaning": 0, "words": 0}}}
    found = client.post("/query", json=weighted).json()["results"]["keyword_rag"]
    assert {tuple(h["channels"]) for h in found["hits"]} == {("keywords",)}


def test_query_model_tags_are_the_ais_pick_and_counted_in_its_usd(client):
    reference_with_vocabulary(client)
    fake_replies(client.app.state.ai, lambda p: '{"terms": ["volcano"]}')
    body = {"collection": "r", "question": "enzyme rate", "methods": ["keyword_rag"]}
    found = client.post("/query", json={**body, "search": {"query_tags": "model"}}).json()
    found = found["results"]["keyword_rag"]
    assert found["error"] is None and found["query_tags"] == ["volcano"] and found["usd"] > 0


def test_query_oracle_is_422(client):
    reference_with_vocabulary(client)
    body = {"collection": "r", "question": "enzyme", "methods": ["keyword_rag"]}
    r = client.post("/query", json={**body, "search": {"query_tags": "oracle"}})
    assert r.status_code == 422 and "oracle tags need a question's gold" in r.json()["detail"]


def test_query_and_eval_validate_search(client):
    reference_with_vocabulary(client)
    for search in ({"weights": {"vibes": 1}}, {"colour": "red"}, {"weights": {"words": -1}}):
        body = {"collection": "r", "question": "enzyme", "methods": ["hybrid"], "search": search}
        assert client.post("/query", json=body).status_code == 422, search
        run = {"collection": "r", "methods": ["keyword_rag"], "search": search}
        assert client.post("/eval", json=run).status_code == 422, search
        assert client.post("/eval", json={**run, "confirm": True}).status_code == 422, search


def test_query_keyword_rag_without_vocabulary_is_a_method_error(client):
    add_collection(client)
    body = {"collection": "c", "question": "enzyme", "methods": ["keyword_rag", "hybrid"]}
    results = client.post("/query", json=body).json()["results"]
    assert "no vocabulary for collection 'c'" in results["keyword_rag"]["error"]
    assert results["hybrid"]["hits"]


def test_eval_keyword_rag_without_vocabulary_is_refused_before_queueing(client):
    add_collection(client)
    run = {"collection": "c", "methods": ["hybrid", "keyword_rag"], "answer": True}
    assert "estimate" in client.post("/eval", json=run).json()
    r = client.post("/eval", json={**run, "confirm": True})
    assert r.status_code == 409 and "no vocabulary for collection 'c'" in r.json()["detail"]
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


def test_retag_state_and_problem_wording(client):
    client.post("/collections", json=WORKED)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    retag = db.create_job(conn, "retag", "w", {"version": "v"})
    set_running(conn, retag, 0.4)
    assert client.get("/collections/w").json()["state"] == "retagging"
    db.update_job(conn, retag, status="failed", error="boom", progress=0.4)
    problems = client.get("/status").json()["problems"]
    assert "The vocabulary on w is partly re-tagged; save it again" in problems


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
    assert "own spend" in body["note"] and body["cap_usd"] == 20.0 and body["spent_usd"] == 0.0


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
PUBLIC = {
    "/collections",
    "/eval-sets",
    "/query",
    "/eval",
    "/jobs",
    "/budget",
    "/health",
    "/catalog",
    "/status",
    "/ai",
    "/chat",
}
PAGES = ("/", "/c/x", "/c/x/runs/abc", "/e/x", "/worker", "/lab", "/chat")


def test_pages_are_served(client):
    for path in PAGES:
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    board = client.get("/board", follow_redirects=False)
    assert board.status_code == 307 and board.headers["location"] == "/"
    for path in PAGES:  # /favicon.ico is not served, so every page opts out of asking for it
        assert 'rel="icon"' in client.get(path).text, path
    assert not (STATIC / "board.html").exists()


def test_eval_sets_path_is_a_page_for_a_browser_and_the_list_for_the_api(client):
    page = client.get("/eval-sets", headers={"accept": "text/html,application/xhtml+xml"})
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["vary"] == "Accept"
    api = client.get("/eval-sets")
    assert api.json() == [] and api.headers["vary"] == "Accept"


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
out.v = ['rekeying', 'retagging', 'copying', 'stopped_budget', 'stopped_no_key', 'cancelled']
  .map(u.pill);
out.w = [u.count(11656), u.count(0)];
out.x = [0.004, 0, 0.4141, 12].map(u.usd);
const box = () => ({innerHTML: '', querySelector: () => ({addEventListener() {}})});
out.z = [{est_usd: null, calls: 3}, {est_usd: 0.4141, calls: 1}].map((estimate) => {
  const el = box();
  u.confirmSpend(el, estimate, () => {});
  return el.innerHTML;
});
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
        "Re-tagging",
        "Copying",
        "Stopped budget",
        "Stopped no key",
        "Cancelled",
    ]
    assert "pill run" in got["v"][2]
    assert got["w"] == ["11,656", "0"]
    assert got["x"] == ["< $0.01", "$0.00", "$0.41", "$12.00"]
    unpriced, priced = got["z"]  # no model to price with: unknown, not $0.00
    assert "cost unknown" in unpriced and "$0.00" not in unpriced and "≈ $0.41" in priced
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
        assert r.status_code == 503 and "'answer' job" in r.json()["detail"]
        assert client.get("/eval").json() == []
        free = client.post("/eval", json={**EVAL, "confirm": True})  # nothing to pay for
        assert free.status_code == 202


def bare_client(tmp_path) -> TestClient:
    """A fresh install: no connection, no model and no job set up at all."""
    app = create_app(settings(tmp_path))
    fake_embeddings(app.state.models)
    app.state.worker.idle_seconds = 0.01
    return TestClient(app)


def uploads_left(tmp_path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "uploads").glob("*"))


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


def test_summarising_previews_work_with_no_ai_and_leave_no_upload(tmp_path):
    with bare_client(tmp_path) as client:
        client.post("/collections", json={"name": "c", "qa_type": "mcq", "summarize": True})
        up = upload(client)
        assert up.status_code == 200, up.text
        assert up.json()["estimate"]["est_usd"] is None and up.json()["estimate"]["calls"] == 3
        r = do_import(client, up.json(), confirm=True)
        assert r.status_code == 503 and "'summary' job" in r.json()["detail"]
        assert uploads_left(tmp_path) == [up.json()["upload_id"]]  # kept for a retry
        few = {"items": ITEMS}
        r = client.post("/collections/c/documents", json=few)
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] is None
        r = client.post("/collections/c/documents", json={**few, "confirm": True})
        assert r.status_code == 503 and "'summary' job" in r.json()["detail"]
        many = {"items": [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]}
        r = client.post("/collections/c/documents", json=many)
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] is None
        r = client.post("/collections/c/documents", json={**many, "confirm": True})
        assert r.status_code == 503 and "'summary' job" in r.json()["detail"]
        assert uploads_left(tmp_path) == [up.json()["upload_id"]]
        assert client.get("/collections/c").json()["count"] == 0
        assert db.list_jobs(client.app.state.conn, collection="c", limit=db.ALL) == []


def test_a_failed_upload_estimate_leaves_no_upload(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/collections", json={"name": "c", "qa_type": "mcq", "summarize": True})

        def broken(*args, **kwargs):
            raise RuntimeError("estimate failed")

        client.app.state.loader.estimate_import = broken
        with pytest.raises(RuntimeError):
            upload(client)
        many = {"items": [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]}
        with pytest.raises(RuntimeError):
            client.post("/collections/c/documents", json=many)
        assert uploads_left(tmp_path) == []


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
    assert len(db.get_questions(client.app.state.conn, "c")) == 1  # the eval set stays
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
    # Waiting on an AI batch can take hours: not a stall.
    conn.execute(
        "insert into ai_batches (id, model, purpose, created_at, usd, job_id, connection_id)"
        " values ('b1', 'm', 'answer', 't', 0.5, ?, 'anthropic')",
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


def test_startup_skips_collections_qdrant_lacks(tmp_path):
    client = make_client(tmp_path)
    conn = client.app.state.conn
    db.save_collection(conn, CollectionSpec("ghost", "mcq", False, "fake/dense", 256, None))
    with client:
        assert backfills(conn) == []


# -- uploads, imports, sources, questions ---------------------------------------


def jsonl(*rows) -> bytes:
    return "\n".join(json.dumps(r) for r in rows).encode()


NOTES = jsonl(
    {"id": "n1", "text": "tides follow the moon", "answer": "x"},
    {"id": "n2", "text": "glaciers carve valleys", "answer": "y"},
    {"id": "n3", "text": "reefs need warm shallow water", "answer": "z"},
)


def upload(client, body=NOTES, filename="notes.jsonl", kind="items", name="c"):
    return client.post(
        f"/collections/{name}/uploads", params={"filename": filename, "kind": kind}, content=body
    )


def do_import(client, up, source_name="notes", kind="items", name="c", **extra):
    body = {"upload_id": up["upload_id"], "kind": kind, "source_name": source_name, **extra}
    body["mapping"] = body.get("mapping", up["preview"]["mapping"])
    return client.post(f"/collections/{name}/imports", json=body)


def import_items(client, source_name="notes", body=NOTES, name="c"):
    up = upload(client, body, name=name).json()
    r = do_import(client, up, source_name, name=name, confirm=True)
    assert r.status_code == 202, r.text
    return wait(client, r.json()["job_id"])


def test_upload_jsonl_previews_with_a_guessed_mapping(client):
    add_collection(client)
    r = upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "notes.jsonl" and len(body["upload_id"]) == 32
    assert body["preview"]["ready"] and body["preview"]["mapping"]["text"] == "text"
    assert body["estimate"]["new"] == 3 and body["estimate"]["est_usd"] == 0


def test_upload_not_ready_has_no_estimate(client):
    add_collection(client)
    body = upload(client, jsonl({"foo": "bar"})).json()
    assert body["preview"]["ready"] is False and body["estimate"] is None


def test_upload_over_limit_is_413(client, monkeypatch):
    from ragqa_core import imports

    add_collection(client)
    monkeypatch.setattr(imports, "UPLOAD_LIMIT", 10)
    assert upload(client).status_code == 413
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))


def test_upload_unsupported_type_is_422(client):
    add_collection(client)
    r = upload(client, b"x", filename="notes.exe")
    assert r.status_code == 422
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))


def test_store_upload_with_accept_text_refuses_other_files(client):
    add_collection(client)
    r = client.post(
        "/collections/c/uploads",
        params={"filename": "notes.pdf", "kind": "items", "accept": "text"},
        content=b"x",
    )
    assert r.status_code == 422 and "only .txt and .md" in r.json()["detail"]
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))
    ok = client.post(
        "/collections/c/uploads",
        params={"filename": "Q3 (final).md", "kind": "items", "accept": "text"},
        content=b"some text",
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["source_name"] == "Q3 -final"
    assert upload(client, filename="notes.jsonl").status_code == 200  # no accept: as before


def test_upload_to_an_unknown_collection_is_404(client):
    assert upload(client, name="nope").status_code == 404


def test_import_unknown_upload_is_404(client):
    add_collection(client)
    body = {"upload_id": "../x", "kind": "items", "mapping": None, "source_name": "notes"}
    assert client.post("/collections/c/imports", json=body).status_code == 404
    body["upload_id"] = "0" * 32
    assert client.post("/collections/c/imports", json=body).status_code == 404


def test_import_rejects_a_bad_source_name_and_mapping(client):
    add_collection(client)
    up = upload(client).json()
    assert do_import(client, up, "../evil").status_code == 422
    assert do_import(client, up, mapping={"bogus": "text"}).status_code == 422
    assert do_import(client, up, mapping={"text": 3}).status_code == 422


def test_import_estimate_then_confirm_then_reimport_unchanged(client):
    add_collection(client)
    up = upload(client).json()
    r = do_import(client, up)
    assert r.status_code == 200 and r.json()["estimate"]["new"] == 3
    assert r.json()["preview"]["ready"]
    assert client.get("/collections/c").json()["count"] == 3  # nothing yet
    job = import_items(client)
    assert job["status"] == "done" and job["setup"]["source"] == "upload:notes"
    assert job["result"]["new"] == 3
    detail = client.get("/collections/c").json()
    assert detail["count"] == 6
    again = import_items(client)
    assert again["result"]["new"] == 0 and again["result"]["updated"] == 0
    assert again["result"]["unchanged"] == 3
    sources = {s["source"]: s for s in client.get("/collections/c/sources").json()}
    assert sources["upload:notes"] == {"source": "upload:notes", "items": 3, "enabled": True}


def test_import_not_ready_is_422(client):
    add_collection(client)
    up = upload(client, jsonl({"foo": "bar"})).json()
    r = do_import(client, up, confirm=True)
    assert r.status_code == 422 and "text" in str(r.json()["detail"])


def test_import_that_costs_needs_a_key(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "c", "qa_type": "mcq", "summarize": True})
        up = upload(client).json()
        assert up["estimate"]["est_usd"] > 0
        assert do_import(client, up, confirm=True).status_code == 503


def test_import_questions_with_a_mapping(client):
    client.post("/collections", json={"name": "sci", "qa_type": "science"})
    rows = jsonl(
        {"qid": "a", "ask": "How warm is the lagoon?", "ans": "28 C"},
        {"qid": "b", "ask": "How deep is the trench?", "ans": "10 km"},
    )
    up = upload(client, rows, kind="questions", name="sci").json()
    assert up["preview"]["ready"] is False and up["preview"]["missing"] == ["text", "answer"]
    mapping = {"id": "qid", "text": "ask", "answer": "ans"}
    r = do_import(client, up, "qs", kind="questions", name="sci", mapping=mapping)
    assert r.status_code == 200 and r.json()["preview"]["ready"]
    r = do_import(client, up, "qs", kind="questions", name="sci", mapping=mapping, confirm=True)
    assert wait(client, r.json()["job_id"])["status"] == "done"
    est = client.post("/eval", json={"collection": "sci", "methods": ["vector"]}).json()
    assert est["estimate"]["questions"] == 2
    page = client.get("/collections/sci/questions").json()
    assert page["total"] == 2 and page["questions"][0]["source"] == "upload:qs"


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
    job_id = db.create_job(client.app.state.conn, "eval", "c", {})
    db.update_job(client.app.state.conn, job_id, status="running")
    r = client.patch("/collections/c/sources/upload:notes", json={"enabled": False})
    assert r.status_code == 409 and "eval is running" in r.json()["detail"]


def test_switch_refused_while_backfill_pending(client):
    add_collection(client)
    import_items(client)
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
    qs = jsonl(
        {"id": "uq", "text": "do tides follow the moon?", "choices": ["yes", "no"], "answer": "A"}
    )
    up = upload(client, qs, kind="questions").json()
    r = do_import(client, up, "notes", kind="questions", confirm=True)  # the same source
    wait(client, r.json()["job_id"])
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


def test_questions_list_and_include_leaked(client):
    add_collection(client)
    rows = jsonl(
        {
            "id": "u1",
            "text": "the tides follow the moon each day",
            "choices": ["x", "y"],
            "answer": "A",
            "gold_ids": ["n1"],
        },
        {
            "id": "u2",
            "text": "an unrelated question entirely",
            "choices": ["x", "y"],
            "answer": "B",
        },
    )
    tides = jsonl({"id": "n1", "text": "so the tides follow the moon each day we saw"})
    import_items(client, body=tides)
    up = upload(client, rows, kind="questions").json()
    r = do_import(client, up, "qs", kind="questions", confirm=True)
    assert "leaked" not in wait(client, r.json()["job_id"])["result"]
    conn = client.app.state.conn
    assert all("leaked" not in q.meta for q in db.get_questions(conn, "c"))
    page = client.get("/collections/c/questions").json()
    assert page["total"] == 3 and page["leaked"] == 1 and page["include_leaked"] is False
    row = next(q for q in page["questions"] if q["id"] == "u1")
    assert row["leaked"] is True and row["source"] == "upload:qs" and row["has_gold"] is True
    assert row["has_solution"] is False
    other = next(q for q in page["questions"] if q["id"] == "u2")
    assert other["has_gold"] is False and other["leaked"] is False
    only = client.get("/collections/c/questions", params={"leaked": True}).json()
    assert [q["id"] for q in only["questions"]] == ["u1"]
    paged = client.get("/collections/c/questions", params={"limit": 1, "offset": 1}).json()
    assert len(paged["questions"]) == 1 and paged["total"] == 3
    assert client.get("/collections/c").json()["questions"] == 2
    r = client.patch("/collections/c/questions/leaked", json={"include": True})
    assert r.json() == {"include_leaked": True}
    detail = client.get("/collections/c").json()
    assert detail["questions"] == 3 and detail["leaked"] == 1 and detail["include_leaked"]


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


def test_documents_over_20_items_queue_an_import(client):
    add_collection(client, questions=False)
    many = [
        {"id": f"m{i}", "text": f"note number {i} about reefs", "answer": "a"} for i in range(21)
    ]
    r = client.post("/collections/c/documents", json={"items": many})
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["kind"] == "import" and job["status"] == "done"
    assert job["setup"]["source"] == "hand" and job["result"]["new"] == 21
    assert client.get("/collections/c").json()["count"] == 24


def test_documents_over_20_items_estimate_first(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        r = client.post("/collections/s/documents", json={"items": many})
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0
        assert client.get("/collections/s").json()["count"] == 0


def import_questions_only(client):
    rows = jsonl({"id": "uq", "text": "do tides follow?", "choices": ["yes", "no"], "answer": "A"})
    up = upload(client, rows, kind="questions").json()
    r = do_import(client, up, "qs", kind="questions", confirm=True)
    assert wait(client, r.json()["job_id"])["status"] == "done"


def test_a_question_import_adds_no_source_row(client):
    add_collection(client)
    import_questions_only(client)
    assert [s["source"] for s in client.get("/collections/c/sources").json()] == ["hand"]
    assert [q.id for q in db.get_questions(client.app.state.conn, "c")] == ["q1", "uq"]


def test_queued_documents_need_a_key_when_they_cost(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        r = client.post("/collections/s/documents", json={"items": many, "confirm": True})
        assert r.status_code == 503
        assert db.list_jobs(client.app.state.conn, collection="s", limit=db.ALL) == []


def test_startup_ensures_the_source_indexes_of_every_existing_collection(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    store = client.app.state.store
    store.create("old", 256)
    db.save_collection(client.app.state.conn, CollectionSpec("old", "mcq", False, "f", 256, None))
    db.save_collection(client.app.state.conn, CollectionSpec("ghost", "mcq", False, "f", 256, None))
    seen = []
    monkeypatch.setattr(store, "ensure_indexes", seen.append)
    with client:
        assert seen == ["old"]


def test_preview_warns_when_ids_would_move_from_another_source(client):
    add_collection(client)
    import_items(client)
    up = upload(client, filename="again.jsonl").json()
    warning = "3 items already belong to upload:notes and would move to this source"
    assert warning in up["preview"]["problems"]
    same = do_import(client, up, "notes").json()["preview"]["problems"]
    assert not [p for p in same if "already belong" in p]
    other = do_import(client, up, "other").json()["preview"]["problems"]
    assert warning in other


def test_preview_warns_when_question_ids_would_move_from_another_source(client):
    add_collection(client)
    import_questions_only(client)
    rows = jsonl({"id": "uq", "text": "do tides follow?", "choices": ["yes", "no"], "answer": "A"})
    up = upload(client, rows, kind="questions").json()
    shown = do_import(client, up, "elsewhere", kind="questions").json()["preview"]
    assert (
        "1 question already belongs to upload:qs and would move to this source" in shown["problems"]
    )


def test_documents_are_refused_while_an_import_or_backfill_is_queued_or_running(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    body = {"items": ITEMS[:1], "confirm": True}
    for kind in ("import", "backfill"):
        for status in ("queued", "running"):
            job = db.create_job(conn, kind, "c", {})
            db.update_job(conn, job, status=status)
            r = client.post("/collections/c/documents", json=body)
            assert r.status_code == 409 and f"a {kind} job" in r.json()["detail"], (kind, status)
            db.update_job(conn, job, status="done")


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


# -- /ai settings ---------------------------------------------------------------


def test_ai_settings_round_trip(client):
    ai = client.app.state.ai
    ai._list_models = lambda connection: ["llama3.1", "qwen2"]  # type: ignore[method-assign]
    r = client.post("/ai/connections", json={"preset": "openrouter", "id": "router"})
    assert r.status_code == 201, r.text
    assert client.post("/ai/connections/router/models/fetch").json() == {
        "added": ["llama3.1", "qwen2"]
    }
    assert client.put("/ai/roles/answer", json={"model": "router/llama3.1"}).status_code == 200
    body = client.get("/ai").json()
    router = next(c for c in body["connections"] if c["id"] == "router")
    assert router["kind"] == "openai" and router["label"] == "OpenRouter" and "local" not in router
    # Fetched models start at $0 until a price is entered.
    assert {m["ref"]: m["price_in"] for m in body["models"] if m["connection_id"] == "router"} == {
        "router/llama3.1": 0,
        "router/qwen2": 0,
    }
    assert {"label": "Haiku 5.5", "price_in": 0.10, "price_out": 0.50} in body["prices"]
    assert body["roles"]["answer"] == "router/llama3.1"
    assert body["known_roles"] == ["default", "answer", "summary", "keywords", "query"]
    assert body["budget"] == {"cap_usd": 20.0, "spent_usd": 0.0, "from_env": False}
    assert "ollama" not in {p["id"] for p in body["presets"]}
    assert {"id": "openrouter", "label": "OpenRouter", "kind": "openai"}.items() <= next(
        p for p in body["presets"] if p["id"] == "openrouter"
    ).items()
    assert client.put("/ai/budget", json={"cap_usd": 7.5}).status_code == 200
    assert client.get("/ai").json()["budget"]["cap_usd"] == 7.5
    r = client.post("/ai/connections/router/models", json={"model_id": "x", "price_in": 1})
    assert r.status_code == 200, r.text
    r = client.patch("/ai/connections/router", json={"label": "Mine", "api_key": "abc"})
    assert r.status_code == 200 and client.get("/ai").json()["connections"][1]["label"] == "Mine"
    assert client.put("/ai/roles/answer", json={"model": None}).status_code == 200
    assert client.delete("/ai/connections/router").status_code == 204


def test_ai_validation_errors_are_422(client):
    r = client.post("/ai/connections", json={"id": "x"})  # no preset, no kind
    assert r.status_code == 422
    assert client.post("/ai/connections", json={"preset": "nope", "id": "x"}).status_code == 422
    client.post("/ai/connections", json={"preset": "openai", "id": "oa"})
    assert client.patch("/ai/connections/oa", json={"label": ""}).status_code == 422
    r = client.post("/ai/connections/oa/models", json={"model_id": "m", "price_in": -1})
    assert r.status_code == 422
    assert client.put("/ai/models/oa/ghost", json={"enabled": False}).status_code == 422
    client.post("/ai/connections/oa/models", json={"model_id": "m"})
    assert client.put("/ai/models/oa/m", json={"enabled": None}).status_code == 422
    assert client.post("/ai/connections", json={"preset": "openai", "id": "oa"}).status_code == 409


def test_key_never_leaves_the_server(client):
    client.post(
        "/ai/connections", json={"preset": "openai", "id": "oa", "api_key": "sk-proj-SECRET123456"}
    )
    for path in ("/ai", "/status", "/health", "/budget"):
        assert "SECRET" not in client.get(path).text
    assert "SECRET" not in client.post("/ai/connections/oa/test").text


def test_price_route_accepts_slash_in_model_id(client):
    client.post("/ai/connections", json={"preset": "openrouter", "id": "or"})
    client.post("/ai/connections/or/models", json={"model_id": "meta-llama/llama-3.1-8b"})
    r = client.put(
        "/ai/models/or/meta-llama/llama-3.1-8b", json={"price_in": 0.02, "price_out": 0.05}
    )
    assert r.status_code == 200, r.text
    model = next(m for m in client.get("/ai").json()["models"] if m["ref"].startswith("or/"))
    assert model["model_id"] == "meta-llama/llama-3.1-8b"
    assert (model["price_in"], model["price_out"]) == (0.02, 0.05)


def test_delete_connection_in_use_is_409_naming_role(client):
    r = client.delete("/ai/connections/anthropic")
    assert r.status_code == 409 and "default" in r.json()["detail"]


def test_env_locked_key_is_409(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456", cap_usd=5.0))
    client = TestClient(app)
    r = client.patch("/ai/connections/anthropic", json={"api_key": "other"})
    assert r.status_code == 409 and "environment" in r.json()["detail"]
    assert client.put("/ai/budget", json={"cap_usd": 9}).status_code == 409
    assert client.get("/ai").json()["budget"]["from_env"] is True


def test_unknown_role_is_422(client):
    r = client.put("/ai/roles/nonsense", json={"model": None})
    assert r.status_code == 422 and "known" in r.json()["detail"]
    r = client.put("/ai/roles/answer", json={"model": "anthropic/ghost"})
    assert r.status_code == 422


def test_an_unreachable_provider_is_one_methods_error_in_a_query(client):
    # Each method reports its own error, so what the others spent is still shown.
    add_collection(client)

    def down(*args, **kwargs):
        raise ConnectionError("connection refused")

    client.app.state.ai._send = down
    r = client.post(
        "/query",
        json={"collection": "c", "question": "enzyme rate", "methods": ["hybrid"], "answer": True},
    )
    assert r.status_code == 200, r.text
    assert "connection refused" in r.json()["results"]["hybrid"]["error"]


def test_role_with_disabled_model_is_503_naming_job(client):
    add_collection(client)
    client.put("/ai/models/anthropic/claude-haiku-5-5", json={"enabled": False})
    r = client.post("/eval", json={**EVAL, "answer": True, "confirm": True})
    assert r.status_code == 503, r.text
    assert "'answer'" in r.json()["detail"] and "Settings" in r.json()["detail"]


def test_env_model_locks_the_default_job(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456", model="claude-sonnet-5-5"))
    client = TestClient(app)
    assert client.get("/ai").json()["default_from_env"] is True
    r = client.put("/ai/roles/default", json={"model": "anthropic/claude-haiku-5-5"})
    assert r.status_code == 409 and "environment" in r.json()["detail"]
    assert client.put("/ai/roles/answer", json={"model": "claude-haiku-5-5"}).status_code == 200


def test_env_connection_keeps_its_address_and_cannot_be_removed(tmp_path):
    app = create_app(settings(tmp_path, api_key="sk-env-key-123456"))
    client = TestClient(app)
    assert client.get("/ai").json()["default_from_env"] is False
    r = client.patch("/ai/connections/anthropic", json={"base_url": "http://elsewhere"})
    assert r.status_code == 409 and "environment" in r.json()["detail"]
    assert client.put("/ai/roles/default", json={"model": None}).status_code == 200
    r = client.delete("/ai/connections/anthropic")
    assert r.status_code == 409 and "environment" in r.json()["detail"]


def test_new_address_without_a_new_key_drops_the_stored_key(client):
    client.post(
        "/ai/connections", json={"preset": "openai", "id": "oa", "api_key": "sk-proj-SECRET123456"}
    )
    r = client.patch("/ai/connections/oa", json={"base_url": "http://other/v1"})
    assert r.status_code == 200, r.text
    oa = next(c for c in client.get("/ai").json()["connections"] if c["id"] == "oa")
    assert oa["key"] is None and oa["base_url"] == "http://other/v1"


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
        assert client.get("/collections/w").json()["profile"] == {"rates": 3}
        hits = client.app.state.store.get("w", [i["id"] for i in ITEMS])
        assert all(h.tags == ["rates"] and h.abstracted for h in hits)
        assert ledger_purposes(client) == ["induce"] + ["tag"] * 3 + ["abstract"] * 3
        # Tagging again with the same terms is answered from the cache: priced at nothing.
        same = client.put("/collections/w/vocabulary", json={"terms": ["rates"]}).json()
        other = client.put("/collections/w/vocabulary", json={"terms": ["rates", "speed"]}).json()
        assert same["estimate"]["est_usd"] == 0 and other["estimate"]["est_usd"] > 0


def test_worked_documents_whose_build_failed_are_built_by_sending_them_again(tmp_path):
    def reply(prompt: str) -> str:
        if '"principle"' in prompt:
            return '{"principle": "P.", "method": "M.", "answer": "A"}'
        return '{"terms": ["rates"]}'

    with make_client(tmp_path, reply=reply) as client:
        client.post("/collections", json=WORKED)
        builder = client.app.state.builder

        def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        builder.abstract = boom
        with pytest.raises(RuntimeError):
            client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True})
        del builder.abstract
        assert client.get("/collections/w").json()["count"] == 3  # added and tagged only
        r = client.post("/collections/w/documents", json={"items": ITEMS})
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0  # the abstracts
        r = client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True})
        assert r.status_code == 200
        hits = client.app.state.store.get("w", [i["id"] for i in ITEMS])
        assert all(h.tags == ["rates"] and h.abstracted for h in hits)
        r = client.post("/collections/w/documents", json={"items": ITEMS})
        assert r.status_code == 200 and r.json() == {"added": 3}  # built: free, nothing to confirm


SCIBENCH = Path(__file__).parents[3] / "packages" / "core" / "tests" / "data" / "scibench_mini"


def building(prompt: str) -> str:
    if '"principle"' in prompt:
        return '{"principle": "P.", "method": "M.", "answer": "A"}'
    return '{"terms": ["kinematics"]}'


def scibench_source(client) -> list[Item]:
    """Collection p (paper 2024, science): the scibench mini items, and ITEMS under a second
    source "hand" that is switched off; a question asked of each source."""
    from ragqa_core.datasets import load_scibench

    app = client.app.state
    r = client.post("/collections", json={"name": "p", "qa_type": "science"})
    assert r.status_code == 201, r.text
    items, _ = load_scibench(app.settings.home, {p.name: p for p in SCIBENCH.glob("*.json")})
    spec = app.indexer.spec("p")
    app.indexer.add(spec, items, "scibench")
    app.indexer.add(spec, [Item(**i) for i in ITEMS], "hand")
    db.save_questions(
        app.conn,
        "p",
        [
            Question("q1", "a ball falls", None, ["3"], [items[0].id], {"source": "scibench"}),
            Question("q2", "enzyme rate", None, ["A"], ["enzyme-item"], {"source": "hand"}),
        ],
    )
    db.set_source_enabled(app.conn, "p", "hand", False)
    return items


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
        assert w["vocabulary_version"] and w["profile"] == {"kinematics": w["count"]}
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


def test_a_sync_add_holds_off_jobs_on_its_collection_until_it_ends(client):
    """A sync add runs outside the worker, so the jobs table can't show it: a retag claimed
    during it would leave the new items tagged under the old vocabulary."""
    add_collection(client)
    client.post("/collections", json=WORKED)
    state = client.app.state
    state.worker.stop()
    seen = []
    build = state.builder.build
    state.builder.build = lambda spec, todo: (seen.append(set(state.adding)), build(spec, todo))
    r = client.post("/collections/c/documents", json={"items": ITEMS[:1], "confirm": True})
    assert r.status_code == 200 and seen == [{"c"}] and state.adding == set()

    state.adding.update({"c", "w"})  # as while each one's add runs

    def versions():
        lists = state.conn.execute("select count(*) as n from keyword_lists")[0]["n"]
        return lists, db.list_vocabularies(state.conn, "w")

    before = versions()
    refused = {
        "rekey": client.put("/collections/c/keyword-lists", json=asdict(LISTS)),
        "retag": client.put("/collections/w/vocabulary", json={"terms": ["x"], "confirm": True}),
        "load": client.post("/collections/c/load", json={"dataset": "scienceqa", "confirm": True}),
        "copy": client.post(
            "/collections/c/copy", json={"name": "c2", "recipe": "paper_2024", "confirm": True}
        ),
        "eval": client.post(
            "/eval", json={"collection": "c", "methods": ["vector"], "confirm": True}
        ),
        "add": client.post("/collections/c/documents", json={"items": ITEMS[:1], "confirm": True}),
    }
    for kind, r in refused.items():
        assert r.status_code == 409 and "being added" in r.json()["detail"], (kind, r.text)
    assert db.list_jobs(state.conn, status="queued", limit=db.ALL) == []
    assert versions() == before  # a refused rekey or retag saves no version
    state.adding.clear()
    assert (
        client.put("/collections/w/vocabulary", json={"terms": ["x"], "confirm": True}).status_code
        == 202
    )


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


# -- eval sets -------------------------------------------------------------------

MCQ = {"id": "a", "text": "which one?", "choices": ["x", "y"], "gold_answers": ["A"]}


def put_questions(client, name, questions, base=None):
    body = {"questions": questions, "base_version": base}
    return client.put(f"/eval-sets/{name}/questions", json=body)


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
    assert up["preview"]["ready"] and up["preview"]["usable"] == 3
    from ragqa_core import imports

    seen = []
    real = imports.preview
    monkeypatch.setattr(imports, "preview", lambda *a, **k: seen.append(a[-1]) or real(*a, **k))
    client.post("/eval-sets/e/uploads", params={"filename": "Q3 (final).jsonl"}, content=rows)
    assert seen == ["upload:Q3 -final"]
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
    r = client.post(
        "/collections/e/imports",
        json={**body, "kind": "questions"},
    )
    assert r.status_code == 404


def test_deleting_an_eval_set_deletes_its_runs_and_a_store_keeps_its_sets(client):
    add_collection(client)
    conn = client.app.state.conn
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


def test_store_question_routes_use_the_default_eval_set(client):
    add_collection(client, questions=False)
    conn = client.app.state.conn
    assert client.get("/collections/c").json()["eval_set"] is None
    assert client.get("/collections/c").json()["questions"] == 0
    rows = jsonl({"id": "u1", "text": "do tides follow?", "choices": ["y", "n"], "answer": "A"})
    up = upload(client, rows, kind="questions").json()
    r = do_import(client, up, "qs", kind="questions", confirm=True)
    assert wait(client, r.json()["job_id"])["status"] == "done"
    assert db.default_eval_set(conn, "c") == "c"
    assert [q.id for q in db.get_questions(conn, "c")] == ["u1"]
    assert client.get("/eval-sets/c").json()["qa_type"] == "mcq"
    page = client.get("/collections/c/questions").json()
    assert [q["id"] for q in page["questions"]] == ["u1"]
    client.post("/eval-sets", json={"name": "other", "qa_type": "mcq"})
    db.set_default_eval_set(conn, "c", "other")
    put_questions(client, "other", [MCQ])
    assert [q["id"] for q in client.get("/collections/c/questions").json()["questions"]] == ["a"]
    detail = client.get("/collections/c").json()
    assert detail["eval_set"] == "other" and detail["questions"] == 1
    r = client.patch("/collections/c/questions/leaked", json={"include": True})
    assert r.json() == {"include_leaked": True}
    assert (
        db.get_include_leaked(conn, "other") is True and db.get_include_leaked(conn, "c") is False
    )


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


def test_an_eval_set_is_not_deleted_or_imported_into_while_a_load_writes_to_it(client):
    add_collection(client)
    client.app.state.worker.stop()
    conn = client.app.state.conn
    db.create_job(conn, "load", "c", {"dataset": "musique", "limit": None})
    assert client.delete("/eval-sets/c", params={"confirm": True}).status_code == 409
    assert db.get_eval_set(conn, "c") is not None
    body = {"upload_id": "0" * 32, "source_name": "qs"}
    assert client.post("/eval-sets/c/imports", json=body).status_code == 409


def test_the_store_page_scans_for_leaked_questions_once_per_change(client):
    add_collection(client)
    tides = jsonl({"id": "n1", "text": "so the tides follow the moon each day we saw"})
    import_items(client, body=tides)
    rows = jsonl(
        {
            "id": "u1",
            "text": "the tides follow the moon each day",
            "choices": ["x", "y"],
            "answer": "A",
        }
    )
    up = upload(client, rows, kind="questions").json()
    wait(client, do_import(client, up, "qs", kind="questions", confirm=True).json()["job_id"])
    store, scans = client.app.state.store, []
    scroll = store.scroll
    store.scroll = lambda *a, **kw: scans.append(1) or scroll(*a, **kw)
    assert client.get("/collections/c").json()["leaked"] == 1
    assert client.get("/collections/c").json()["leaked"] == 1
    assert client.get("/collections/c/questions").json()["leaked"] == 1
    assert len(scans) == 1
    import_items(
        client, body=jsonl({"id": "n2", "text": "glaciers carve valleys"}), source_name="more"
    )
    scans.clear()
    assert client.get("/collections/c").json()["leaked"] == 1 and len(scans) == 1


TIDES = jsonl({"id": "n1", "text": "so the tides follow the moon each day we saw"})
HAND = {
    **MCQ,
    "id": "h1",
    "text": "the tides follow the moon each day",
    "meta": {"solution": "Rates rise."},
}
OTHER = {**MCQ, "id": "h2", "text": "an unrelated question", "meta": {"solution": "Rates rise."}}


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


def test_patch_is_refused_while_a_load_or_question_import_is_queued(client):
    add_collection(client)
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    client.app.state.worker.stop()
    conn = client.app.state.conn
    job = db.create_job(conn, "import", "c", {"kind": "questions"})
    r = client.patch("/collections/c", json={"eval_set": "e"})
    assert r.status_code == 409 and "under way on 'c'" in r.json()["detail"]
    db.request_cancel(conn, job)
    db.create_job(conn, "load", "c", {"dataset": "scienceqa"})
    assert client.patch("/collections/c", json={"eval_set": "e"}).status_code == 409
    assert db.default_eval_set(conn, "c") == "c"


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


def test_a_question_import_is_refused_when_the_default_set_is_another_type(client):
    science_default(client)
    rows = jsonl({"id": "u1", "text": "do tides follow?", "choices": ["y", "n"], "answer": "A"})
    up = upload(client, rows, kind="questions").json()
    for confirm in (False, True):
        r = do_import(client, up, "qs", kind="questions", confirm=confirm)
        assert r.status_code == 409
        assert r.json()["detail"] == "eval set 's' is science; store 'c' is mcq"
    conn = client.app.state.conn
    assert db.list_jobs(conn, kind="import") == []
    assert db.default_eval_set(conn, "c") == "s" and db.get_eval_set(conn, "c") is None
    items = do_import(client, upload(client).json(), "notes", confirm=False)
    assert items.status_code == 200  # items do not go to the eval set


def test_an_empty_question_id_is_refused(client):
    client.post("/eval-sets", json={"name": "e", "qa_type": "mcq"})
    for blank in ("", "  "):
        r = put_questions(client, "e", [{**MCQ, "id": blank}])
        assert r.status_code == 422 and r.json()["detail"].endswith(": no id")
    assert client.get("/eval-sets/e").json()["questions"] == []


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


def chat_body(question="enzyme rate", **extra):
    return {"collection": "c", "messages": [{"role": "user", "content": question}], **extra}


def test_chat_answers_with_citations(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Rates rise [1]. Answer: x") as client:
        add_collection(client)
        r = client.post("/chat", json=chat_body())
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["answer"].startswith("Rates rise") and got["error"] is None
        assert got["citations"][0]["id"] == got["hit_ids"][0] == "enzyme-item"
        assert got["citations"][0]["n"] == 1 and "enzyme" in got["citations"][0]["text"]
        assert got["usd"] > 0 and got["method"] == "hybrid" and got["query_tags"] is None


def test_chat_cites_what_the_model_saw(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Long [1]") as client:
        add_collection(client, name="long", questions=False, lists=False)
        long_text = "enzyme " + "x" * 4993
        r = client.post(
            "/collections/long/documents",
            json={"items": [{"id": "big", "text": long_text}], "confirm": True},
        )
        assert r.status_code == 200, r.text
        body = {**chat_body("enzyme x"), "collection": "long", "method": "vector"}
        got = client.post("/chat", json=body).json()
        big = [c for c in got["citations"] if c["id"] == "big"]
        assert big and len(big[0]["text"]) == 1500


def test_chat_method_defaults_and_refusals(client):
    add_collection(client)
    assert client.post("/chat", json=chat_body()).json()["method"] == "hybrid"
    allowed = client.post("/chat", json=chat_body(method="kw_individual"))
    assert allowed.status_code == 200
    add_collection(client, name="bare", lists=False)
    r = client.post("/chat", json={**chat_body(method="kw_individual"), "collection": "bare"})
    assert r.status_code == 409 and "needs keyword lists" in r.json()["detail"]
    assert client.post("/chat", json=chat_body(method="nope")).status_code == 422
    reference_with_vocabulary(client)
    body = {**chat_body(), "collection": "r"}
    got = client.post("/chat", json=body).json()
    assert got["method"] == "keyword_rag" and got["query_tags"][0] == "enzyme"
    put = client.put("/collections/r/search", json={"query_tags": "oracle"})
    assert put.status_code == 200
    r = client.post("/chat", json=body)
    assert r.status_code == 200 and r.json()["error"] is None and r.json()["query_tags"]


def test_chat_over_budget_is_that_replys_error(tmp_path):
    with make_client(tmp_path, cap_usd=0) as client:
        add_collection(client)
        r = client.post("/chat", json=chat_body())
        assert r.status_code == 200
        got = r.json()
        assert got["answer"] is None and "cap" in got["error"]


def test_chat_sends_only_the_last_six_turns(tmp_path):
    seen = []

    def reply(prompt):
        seen.append(prompt)
        return "ok"

    with make_client(tmp_path, reply=reply) as client:
        add_collection(client)
        turns = [
            {"role": "user" if i % 2 else "assistant", "content": f"turn {i}"} for i in range(1, 9)
        ]
        last = {"role": "user", "content": "enzyme rate"}
        r = client.post("/chat", json={"collection": "c", "messages": [*turns, last]})
        assert r.status_code == 200, r.text
    prompt = seen[-1]
    assert "turn 2" not in prompt and "turn 3" in prompt and "turn 8" in prompt


def test_chat_needs_the_user_to_speak_last(client):
    add_collection(client)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert client.post("/chat", json={"collection": "c", "messages": messages}).status_code == 422
    assert client.post("/chat", json={"collection": "c", "messages": []}).status_code == 422
