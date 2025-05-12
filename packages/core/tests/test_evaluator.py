import time

import pytest

from ragqa_core.db import connect, create_job, get_job, run_items, save_questions
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.retriever import Retriever
from ragqa_core.store import Store
from ragqa_core.types import Item, KeywordLists, Question

TOPICS = ["enzyme", "market", "volcano", "optics", "genetics"]
ITEMS = [
    Item(f"{t}-item", f"{t} question about {t} things", "A", f"{t} explanation", {"skill": t})
    for t in TOPICS
]
QUESTIONS = [
    Question(
        f"q-{t}",
        f"a {t} question",
        ["yes", "no"],
        ["A"],
        [],
        {"skill": t, "solution": f"{t} explanation"},
    )
    for t in TOPICS
]
LISTS = KeywordLists(topic=TOPICS, method=["survey", "assay"], common=["things", "question"])


def setup_for(methods, answer=False, model=None, split="dev"):
    return {
        "collection": "c",
        "split": split,
        "methods": methods,
        "answer": answer,
        "model": model,
        "k": 5,
        "rrf_k": 60,
    }


@pytest.fixture
def env(tmp_path):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    claude = fake_claude(conn, lambda p: "Answer: A")
    ix = Indexer(conn, store, models, claude)
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS)
    ix.set_lists(spec, LISTS)
    save_questions(conn, "c", QUESTIONS)
    retriever = Retriever(conn, store, models, claude, ix)
    return conn, Evaluator(conn, retriever, models, claude)


def wait_until_done(conn, job, seconds=10):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if get_job(conn, job)["status"] not in ("queued", "running"):
            return
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_estimate_counts_calls(env):
    _, ev = env
    est = ev.estimate("c", "dev", ["hybrid", "kw_individual"], answer=True, model=None)
    assert est["questions"] == 5 and est["calls"] == 5 * 2 and est["est_usd"] > 0


def test_estimate_free_retrieval_is_zero(env):
    _, ev = env
    est = ev.estimate("c", "dev", ["hybrid"], answer=False, model=None)
    assert est == {"questions": 5, "calls": 0, "est_usd": 0.0}


def test_estimate_counts_iterative_hops(env):
    _, ev = env
    assert ev.estimate("c", "dev", ["iterative"], answer=False, model=None)["calls"] == 15


def test_estimate_sonnet_costs_more(env):
    _, ev = env
    haiku = ev.estimate("c", "dev", ["hybrid"], answer=True, model=None)["est_usd"]
    sonnet = ev.estimate("c", "dev", ["hybrid"], answer=True, model="claude-sonnet-5-5")["est_usd"]
    assert sonnet > haiku


def test_run_writes_rows_and_table_has_cis(env):
    conn, ev = env
    job = create_job(
        conn, "eval", "c", setup_for(["hybrid", "vector", "kw_individual"], answer=True)
    )
    ev.run(job)
    t = ev.table(job)
    assert t["status"] == "done" and t["progress"] == 1.0
    hybrid = t["methods"]["hybrid"]
    assert set(hybrid) >= {"solution_sim@1", "same_skill@1", "accuracy", "usd", "n"}
    assert hybrid["n"] == 5 and hybrid["same_skill@1"]["mean"] == 1.0
    acc = hybrid["accuracy"]
    assert acc["lo"] <= acc["mean"] <= acc["hi"] and acc["mean"] == 1.0
    assert any(p["metric"] == "solution_sim@1" for p in t["pairs"])
    assert any(p["metric"] == "accuracy" for p in t["pairs"])
    assert t["proxy"] is not None and set(t["proxy"]) == {"spearman", "trusted"}


def test_run_without_answers_has_no_answer_metrics(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["hybrid"]))
    ev.run(job)
    t = ev.table(job)
    assert "accuracy" not in t["methods"]["hybrid"] and t["proxy"] is None
    assert t["methods"]["hybrid"]["usd"] == 0.0


def test_run_stops_on_budget_and_keeps_rows(env):
    conn, ev = env
    ev.claude = fake_claude(conn, lambda p: "Answer: A", cap_usd=1e-9)
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_budget"
    assert len(run_items(conn, job)) == 5
    assert ev.table(job)["methods"]["hybrid"]["solution_sim@1"]["mean"] > 0


def test_run_stops_without_key(env):
    conn, ev = env
    ev.claude = fake_claude(conn, lambda p: "Answer: A", api_key=None)
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_no_key"


def test_run_records_failure(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["nope"]))
    ev.run(job)
    row = get_job(conn, job)
    assert row["status"] == "failed" and "nope" in row["error"]


def test_setup_records_full_configuration(env):
    conn, ev = env
    job = ev.start("c", "dev", ["hybrid"], False, None)
    wait_until_done(conn, job)
    setup = get_job(conn, job)["setup"]
    assert {"dense_model", "judge_model", "keyword_version", "rrf_k", "k", "seed"} <= set(setup)
    assert {"dataset_filter", "collection", "split", "methods"} <= set(setup)
    assert setup["keyword_version"] and setup["seed"] == 472


def test_items_wrong_only(env):
    conn, ev = env
    ev.claude = fake_claude(conn, lambda p: "Answer: B")
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    ev.run(job)
    wrong = ev.items(job, "hybrid", wrong_only=True)
    assert len(wrong) == 5 and wrong[0]["question"] and wrong[0]["gold_answers"] == ["A"]
    assert wrong[0]["pred"] == "B"
    # Same prompts would be served "B" from the cache; another model is another key.
    ev.claude = fake_claude(conn, lambda p: "Answer: A")
    right = create_job(
        conn, "eval", "c", setup_for(["hybrid"], answer=True, model="claude-sonnet-5-5")
    )
    ev.run(right)
    assert ev.items(right, "hybrid", wrong_only=True) == []
    assert len(ev.items(right, "hybrid", wrong_only=False)) == 5
