import threading
import time

import pytest

from aiconnect.ai import AwaitingReplies, BudgetExceeded, NoConnection
from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.db import connect, get_job, save_questions
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_ai, fake_embeddings, fake_models, fake_replies
from ragqa_core.indexer import Indexer, ModelMismatch
from ragqa_core.loader import Loader
from ragqa_core.recipes import build_for
from ragqa_core.retriever import Retriever
from ragqa_core.store import Store, rebuild_of
from ragqa_core.types import Item, KeywordLists, Question, SearchSettings
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
    builder = Builder(conn, store, models, ai, ix)
    ev = Evaluator(conn, Retriever(conn, store, models, ai, ix, builder), models, ai)
    loader = Loader(conn, ix, builder, ai, tmp_path)
    w = Worker(conn, ai, ix, loader, ev, builder, idle_seconds=0.01)
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
        (AwaitingReplies(3), "stopped_replies"),
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
    expected = {
        "failed": repr(exc),
        "stopped_budget": str(exc),
        "stopped_replies": f"{exc}: export them with `docker compose exec api python -m"
        " ragqa_core.replies export /home/app/.ragqa/prompts.jsonl`, import the replies,"
        " then resume the run or start the job again; replies already imported stay cached",
        "stopped_no_key": "no AI model is set up for the 'answer' job; choose one on the"
        " Settings page",
    }.get(status)
    assert job["error"] == expected


def test_unknown_kind_fails(env):
    w, conn = env
    j = db.create_job(conn, "mystery", "c", {})
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "failed" and job["error"] == "unknown job kind 'mystery'"


def test_job_on_a_deleted_store_names_it(env):
    w, conn = env
    w.indexer.create("x", "mcq", False)
    setups = {
        "reembed": {"model": "fake/small", "from": "fake/dense"},
        "load": {"dataset": "x"},
        "import": {"kind": "items"},
        "tag_build": {},
    }
    jobs = [db.create_job(conn, kind, "x", setup) for kind, setup in setups.items()]
    conn.execute("delete from collections where name = 'x'")  # its queued jobs are kept
    while w.run_one() is not None:
        pass
    named = "UnknownCollection(\"no store named 'x'\")"
    for j in jobs:
        job = get_job(conn, j)
        assert job["status"] == "failed" and job["error"] == named


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
    models.embed = lambda texts, model=None: models._load("text", "BAAI/test") and None  # type: ignore[assignment]
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


def test_worker_runs_retag(env):
    w, conn = env
    spec = w.indexer.create("w", "science", False, recipe="worked_examples")
    w.indexer.add(spec, ITEMS, "x")
    version = w.builder.save_vocabulary(spec, TOPICS)
    j = db.create_job(conn, "retag", "w", {"version": version})
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done", job["error"]
    assert job["progress"] == 1.0 and job["result"] == {"version": version}
    after = db.get_collection(conn, "w")
    assert after is not None and after.vocabulary_version == version
    tagged = [h for batch in w.indexer.store.scroll("w") for h in batch]
    assert sum(len(h.tags) for h in tagged) == 3 * len(ITEMS)  # the AI's reply is unusable: top 3


def test_worker_runs_copy(env):
    w, conn = env
    source = db.get_collection(conn, "c")
    assert source is not None
    j = w.loader.start_copy("c", w.loader.copy_spec(source, "r", "reference", None))
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done", job["error"]
    assert job["progress"] == 1.0 and job["result"]["added"] == len(ITEMS)
    assert job["result"]["sources"] == ["x"]
    assert w.indexer.store.count("r") == len(ITEMS)


def queue_reembed(w, conn):
    fake_embeddings(w.indexer.models, dims={"fake/small": 64})
    return db.create_job(conn, "reembed", "c", {"model": "fake/small", "from": "fake/dense"})


def test_worker_runs_a_reembed_job_and_records_its_switch(env):
    w, conn = env
    j = queue_reembed(w, conn)
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done", job["error"]
    assert job["setup"]["switching"] is True and job["progress"] == 1.0
    assert job["result"]["model"] == "fake/small" and job["result"]["items"] == len(ITEMS)
    after = db.get_collection(conn, "c")
    assert after is not None and (after.dense_model, after.dense_dim) == ("fake/small", 64)


def test_a_cancelled_reembed_job_is_cancelled_and_changes_nothing(env):
    w, conn = env
    j = queue_reembed(w, conn)
    models, embed = w.indexer.models, w.indexer.models.embed
    started, release = threading.Event(), threading.Event()

    def held(texts, model=None):
        started.set()
        release.wait(5)
        return embed(texts, model)

    models.embed = held
    runner = threading.Thread(target=w.run_one)
    runner.start()
    assert started.wait(5)
    db.request_cancel(conn, j)
    release.set()
    runner.join(5)
    assert get_job(conn, j)["status"] == "cancelled"
    after = db.get_collection(conn, "c")
    assert after is not None and (after.dense_model, after.dense_dim) == ("fake/dense", 256)
    assert not w.indexer.store.exists(rebuild_of("c")) and w.indexer.store.count("c") == len(ITEMS)


def test_a_reembed_stopped_during_its_switch_is_refused_until_resumed(env):
    w, conn = env
    j = queue_reembed(w, conn)
    store, copy = w.indexer.store, w.indexer.store.copy

    def dies(source, target):
        raise RuntimeError("the process stopped")

    store.copy = dies  # the store is emptied and recreated, then the copy back stops
    w.run_one()
    assert get_job(conn, j)["status"] == "failed"
    assert get_job(conn, j)["setup"]["switching"] is True
    spec = db.get_collection(conn, "c")
    assert spec is not None and store.count("c") == 0
    with pytest.raises(ModelMismatch, match="store 'c' is switching to fake/small"):
        w.indexer.check_model(spec)
    store.copy = copy
    assert db.requeue(conn, j)
    w.run_one()
    assert get_job(conn, j)["status"] == "done", get_job(conn, j)["error"]
    spec = db.get_collection(conn, "c")
    assert spec is not None and (spec.dense_model, spec.dense_dim) == ("fake/small", 64)
    assert store.count("c") == len(ITEMS) and not store.exists(rebuild_of("c"))
    w.indexer.check_model(spec)


def documents_store(w) -> None:
    """Store d: a Documents store of three passages, without keyword tags."""
    spec = w.indexer.create("d", "documents", False, recipe="documents")
    w.indexer.add(spec, [Item(f"p-{t}", f"a passage about {t} things") for t in TOPICS[:3]], "x")


def test_worker_runs_a_tag_build_switching_the_recipe_in_place(env):
    w, conn = env
    documents_store(w)
    saved = SearchSettings(weights={"meaning": 1.0, "words": 1.0, "keywords": 2.0})
    db.set_search(conn, "d", saved)  # an override the switch must keep
    fake_replies(w.ai, lambda p: '{"terms": ["enzyme", "market", "volcano"]}')
    j = db.create_job(conn, "tag_build", "d", {})
    assert w.run_one() == j
    job = get_job(conn, j)
    assert job["status"] == "done", job["error"]
    assert job["progress"] == 1.0 and job["result"]["tagged"] == 3
    after = db.get_collection(conn, "d")
    assert after is not None and after.recipe == "documents_tagged"
    assert after.build == build_for("documents_tagged") and after.search == saved
    assert after.vocabulary_version == job["result"]["vocabulary_version"] is not None
    assert all(h.tags for batch in w.builder.store.scroll("d") for h in batch)
    assert not db.tag_build_unfinished(conn, "d")


def test_a_tag_build_with_no_usable_terms_switches_back_and_ends_done(env):
    w, conn = env
    documents_store(w)
    fake_replies(w.ai, lambda p: '{"terms": []}')
    j = db.create_job(conn, "tag_build", "d", {})
    w.run_one()
    job = get_job(conn, j)
    assert job["status"] == "done", job["error"]
    assert job["result"]["note"] == "no usable terms: nothing was tagged"
    after = db.get_collection(conn, "d")
    assert after is not None and after.recipe == "documents" and after.vocabulary_version is None
    assert after.build == build_for("documents")
    assert not any(h.tags for batch in w.builder.store.scroll("d") for h in batch)
