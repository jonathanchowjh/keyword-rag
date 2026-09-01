import time

import pytest

from aiconnect.ai import BudgetExceeded, NoConnection
from ragqa_core import db
from ragqa_core.db import connect, get_job, save_questions
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_ai, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.loader import Loader
from ragqa_core.retriever import Retriever
from ragqa_core.store import Store
from ragqa_core.types import Item, KeywordLists, Question
from ragqa_worker.worker import Worker

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
OTHER_LISTS = KeywordLists(topic=TOPICS[:3], method=["survey"], common=["things"])


@pytest.fixture
def env(tmp_path):
    conn = connect(tmp_path / "r.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "Answer: A")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    ix.set_lists(spec, LISTS)
    save_questions(conn, "c", QUESTIONS)
    ev = Evaluator(conn, Retriever(conn, store, models, ai, ix), models, ai)
    w = Worker(conn, ai, ix, Loader(conn, ix, ai, tmp_path), ev, idle_seconds=0.01)
    w.other_version = ix.save_lists(spec, OTHER_LISTS)
    return w, conn


def queue_eval(w):
    return w.evaluator.queue("c", "dev", ["hybrid"], False, None)


def log_text(conn, job_id):
    return "\n".join(map(str, get_job(conn, job_id)["log"]))


def test_jobs_run_in_queue_order(env):
    w, conn = env
    v = w.other_version
    a = db.create_job(conn, "rekey", "c", {"version": v})
    b = queue_eval(w)
    assert w.run_one() == a and db.get_collection(conn, "c").keyword_version == v
    assert w.run_one() == b and get_job(conn, b)["setup"]["keyword_version"] == v
    assert w.run_one() is None


def test_cancel_requested_mid_job_ends_cancelled(env):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    w.indexer.rekey = lambda name, version, progress: (db.request_cancel(conn, j), progress(0.5))
    w.run_one()
    assert get_job(conn, j)["status"] == "cancelled"


def test_failure_is_recorded_and_loop_continues(env):
    w, conn = env
    setup = {
        "collection": "c",
        "split": "dev",
        "methods": ["nope"],
        "answer": False,
        "model": None,
        "k": 5,
        "rrf_k": 60,
    }
    bad = db.create_job(conn, "eval", "c", setup)
    good = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    w.run_one()
    w.run_one()
    assert get_job(conn, bad)["status"] == "failed" and get_job(conn, good)["status"] == "done"


def test_job_log_records_start_and_finish(env):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    w.run_one()
    text = log_text(conn, j)
    assert "started" in text and "finished: done" in text
    assert text.index("started") < text.index("finished: done")


def test_rekey_result_and_progress(env):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    w.run_one()
    job = get_job(conn, j)
    assert job["progress"] == 1.0 and job["result"] == {"version": w.other_version}


def test_progress_logs_each_ten_percent_step_once(env):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})

    def fake(name, version, progress):
        for p in (0.41, 0.42, 0.45, 0.5):
            progress(p)

    w.indexer.rekey = fake
    w.run_one()
    text = log_text(conn, j)
    assert text.count("progress 40%") == 1 and text.count("progress 50%") == 1


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (BudgetExceeded("cap"), "stopped_budget"),
        (NoConnection("answer"), "stopped_no_key"),
        (RuntimeError("boom"), "failed"),
    ],
)
def test_exceptions_map_to_statuses(env, exc, status):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})

    def boom(name, version, progress):
        raise exc

    w.indexer.rekey = boom
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == status
    expected = {"failed": repr(exc), "stopped_budget": str(exc)}.get(status)
    assert job["error"] == expected


def test_unknown_kind_fails(env):
    w, conn = env
    j = db.create_job(conn, "mystery", "c", {})
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "failed" and job["error"] == "unknown job kind 'mystery'"


def test_load_job_runs_loader_and_records_result(env):
    w, conn = env
    j = db.create_job(conn, "load", "c", {"dataset": "x"})
    w.loader.run = lambda job, progress: (progress(0.0), {"added": 3})[1]
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done" and job["progress"] == 1.0 and job["result"] == {"added": 3}


def test_eval_final_status_is_not_overwritten(env):
    w, conn = env
    j = queue_eval(w)
    w.run_one()
    assert get_job(conn, j)["status"] == "done"
    assert "finished: done" in log_text(conn, j)


def test_running_job_id_is_set_only_while_running(env):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    seen = []
    w.indexer.rekey = lambda name, version, progress: seen.append(w.running_job_id)
    assert w.running_job_id is None
    w.run_one()
    assert seen == [j] and w.running_job_id is None


def wait_for(cond, seconds=3.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_loop_collects_pending_when_idle_and_survives_errors(env):
    w, _ = env
    calls = []

    def collect():
        calls.append(1)
        raise RuntimeError("collect failed")

    w.ai.collect_pending = collect
    w.start()
    assert wait_for(lambda: len(calls) >= 2)
    assert w.alive() and w.error == repr(RuntimeError("collect failed"))
    w.stop()


def test_loop_survives_unexpected_error_outside_a_job(env):
    w, conn = env
    real, state = w.run_one, {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("db gone")
        out = real()
        if out is not None:
            state["ran"] = out
        return out

    w.run_one = flaky
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    w.start()
    wait_for(lambda: state.get("ran") == j)
    assert state.get("ran") == j
    w.stop()
    assert get_job(conn, j)["status"] == "done"
    assert w.error is None  # the clean cycle that ran the job cleared it


def test_an_error_clears_after_the_next_clean_cycle(env):
    w, _ = env
    state = {"n": 0}

    def collect():
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("network blip")

    w.ai.collect_pending = collect
    w.cycle()
    assert w.error == repr(RuntimeError("network blip"))
    w.cycle()
    assert w.error is None


def test_start_and_stop(env):
    w, _ = env
    w.start()
    assert w.alive()
    w.stop()
    assert not w.alive()


def test_logging_failure_still_ends_in_a_final_status(env, monkeypatch):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    real, state = db.log_job, {"n": 0}

    def flaky(conn, job_id, message):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("log broke")
        real(conn, job_id, message)

    monkeypatch.setattr(db, "log_job", flaky)
    assert w.run_one() == j
    job = get_job(conn, j)
    assert job["status"] == "failed" and "log broke" in job["error"]


def test_eval_returning_without_final_status_is_failed(env):
    w, conn = env
    j = queue_eval(w)
    w.evaluator.run = lambda job_id, check: None
    assert w.run_one() == j
    job = get_job(conn, j)
    assert job["status"] == "failed" and job["error"] == "job ended without a final status"
    assert "finished: failed" in log_text(conn, j)


def test_a_job_whose_read_after_the_claim_fails_ends_failed_not_running(env, monkeypatch):
    w, conn = env
    j = db.create_job(conn, "rekey", "c", {"version": w.other_version})
    real, calls = db.get_job, []

    def flaky(conn, job_id):
        calls.append(job_id)
        if len(calls) == 1:  # the read right after the claim
            raise RuntimeError("disk hiccup")
        return real(conn, job_id)

    monkeypatch.setattr(db, "get_job", flaky)
    assert w.run_one() == j
    job = get_job(conn, j)
    assert job["status"] == "failed" and "disk hiccup" in job["error"]
    assert w.running_job_id is None


def test_load_job_logs_progress_steps(env, monkeypatch):
    w, conn = env
    from ragqa_core import indexer, loader

    # Over two batches: the collection has lists, so adding is the whole load and
    # each batch is a step (64 of 150 is 42%).
    items = [Item(f"i{n}", f"text {n}", "A") for n in range(indexer.BATCH * 2 + 22)]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    j = db.create_job(conn, "load", "c", {"dataset": "x", "limit": None})
    w.ai.update_connection("anthropic", api_key=None)
    w.run_one()
    assert get_job(conn, j)["status"] == "done"
    text = log_text(conn, j)
    assert "progress 40%" in text and "progress 80%" in text and "progress 100%" in text


def test_job_logs_when_a_model_loads(env, monkeypatch):
    w, conn = env
    from ragqa_core import models as models_module

    class Stub:
        def __init__(self, model_name, cache_dir):
            pass

    monkeypatch.setattr(models_module, "TextEmbedding", Stub)
    models = w.indexer.models
    models.embed = lambda texts: models._load("text", "BAAI/test") and None  # type: ignore[assignment]
    j = db.create_job(conn, "load", "c", {"dataset": "x"})
    w.loader.run = lambda job, progress: (models.embed(["a"]), models.embed(["b"]), {})[2]
    w.run_one()
    text = log_text(conn, j)
    assert text.index("loading model BAAI/test") < text.index("model BAAI/test ready")
    assert text.count("loading model BAAI/test") == 1  # cached after the first load


def test_backfill_job_ends_done(env):
    w, conn = env
    j = db.create_job(conn, "backfill", "c", {})
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done" and job["result"] == {"labelled": 0}


def test_import_job_runs_and_is_done(env):
    import json

    from ragqa_core import imports

    w, conn = env
    row = {"id": "n1", "text": "a fresh note about tides"}
    upload_id = imports.save_upload(w.loader.home, "c", "n.jsonl", [json.dumps(row).encode()])
    setup = {
        "upload_id": upload_id,
        "kind": "items",
        "mapping": {"id": "id", "text": "text"},
        "source": "upload:notes",
    }
    j = db.create_job(conn, "import", "c", setup)
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done" and job["result"]["new"] == 1
