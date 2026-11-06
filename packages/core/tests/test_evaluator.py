import json
from dataclasses import replace

import pytest

from aiconnect.ai import Cancelled
from ragqa_core import db
from ragqa_core.builder import Builder
from ragqa_core.datasets import DEV_SLICE, dev_slice
from ragqa_core.db import (
    connect,
    create_job,
    get_job,
    list_jobs,
    requeue,
    run_items,
    save_questions,
)
from ragqa_core.evaluator import EvalSetMismatch, Evaluator, changes_since
from ragqa_core.fakes import fake_ai, fake_models, fake_replies
from ragqa_core.indexer import Indexer
from ragqa_core.retriever import Retriever
from ragqa_core.store import Store
from ragqa_core.types import CollectionSpec, Item, KeywordLists, Question

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
    ai = fake_ai(conn, lambda p: "Answer: A")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS, "x")
    ix.set_lists(spec, LISTS)
    save_questions(conn, "c", QUESTIONS)
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    return conn, Evaluator(conn, retriever, models, ai)


def test_estimate_counts_calls(env):
    _, ev = env
    est = ev.estimate("c", "dev", ["hybrid", "kw_individual"], answer=True, model=None)
    assert est["questions"] == 5 and est["calls"] == 5 * 2 and est["est_usd"] > 0


def test_estimate_free_retrieval_is_zero(env):
    _, ev = env
    est = ev.estimate("c", "dev", ["hybrid"], answer=False, model=None)
    assert est == {
        "questions": 5,
        "dev_questions": 5,
        "evidence_named": 0,
        "evidence_held": 0,
        "calls": 0,
        "est_usd": 0.0,
        "roles": [],
    }


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
        conn, "eval", "c", setup_for(["hybrid", "vector", "kw_individual", "keyword"], answer=True)
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
    assert t["proxy"] is not None and t["proxy"]["methods"] == 4
    assert all(r["complete"] for r in run_items(conn, job))


def test_run_without_answers_has_no_answer_metrics(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["hybrid"]))
    ev.run(job)
    t = ev.table(job)
    assert "accuracy" not in t["methods"]["hybrid"] and t["proxy"] is None
    assert t["methods"]["hybrid"]["usd"] == 0.0


def test_run_stops_on_budget_and_keeps_rows(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "Answer: A", cap_usd=1e-9)
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_budget"
    assert len(run_items(conn, job)) == 5
    assert ev.table(job)["methods"]["hybrid"]["solution_sim@1"]["mean"] > 0


def test_run_stops_without_key(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "Answer: A", api_key=None)
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_no_key"


def test_a_run_answered_by_hand_stops_for_replies_and_resumes_on_them(env):
    conn, ev = env
    ev.ai.add_connection("hand", "manual", "Answered by hand")
    ev.ai.add_model("hand", "m", price_in=0.0, price_out=0.0)
    ev.ai.set_role("answer", "hand/m")
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    with ev.ai.for_job(job, lambda: None, lambda m: None):  # as the worker runs it
        ev.run(job)
    row = get_job(conn, job)
    assert row["status"] == "stopped_replies" and "ragqa_core.replies export" in row["error"]
    assert ev.resume_refusal(job) is None
    waiting = ev.ai.waiting(job)
    assert len(waiting) == 5 and {w["purpose"] for w in waiting} == {"answer"}
    ev.ai.answer({w["key"]: "Answer: A" for w in waiting})
    ev.run(job)
    t = ev.table(job)
    assert t["status"] == "done" and t["methods"]["hybrid"]["accuracy"]["mean"] == 1.0


@pytest.mark.parametrize("qa_type", ["mcq", "science", "multihop", "table"])
def test_a_question_with_no_gold_answer_is_unmeasured_not_wrong(qa_type):
    # BRIGHT's questions have none: an answering run on them must not read "0% right".
    from ragqa_core.evaluator import score

    spec = CollectionSpec("c", qa_type, False, "fake/dense", 4)
    q = Question(id="q", text="?", choices=None, gold_answers=[], gold_ids=[])
    assert score(spec, q, "42") == {}


def test_run_records_failure(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["nope"]))
    ev.run(job)
    row = get_job(conn, job)
    assert row["status"] == "failed" and "nope" in row["error"]


def test_setup_records_full_configuration(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert {"dense_model", "judge_model", "keyword_version", "rrf_k", "k", "seed"} <= set(setup)
    assert {"dataset_filter", "collection", "split", "methods"} <= set(setup)
    assert setup["keyword_version"] and setup["seed"] == 472
    assert setup["dataset_filter"] == "items you added or imported"


def test_items_wrong_only(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "Answer: B")
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    ev.run(job)
    wrong = ev.items(job, "hybrid", wrong_only=True)
    assert len(wrong) == 5 and wrong[0]["question"] and wrong[0]["gold_answers"] == ["A"]
    assert wrong[0]["pred"] == "B"
    # Same prompts would be served "B" from the cache; another model is another key.
    ev.ai = fake_ai(conn, lambda p: "Answer: A")
    right = create_job(
        conn, "eval", "c", setup_for(["hybrid"], answer=True, model="claude-sonnet-5-5")
    )
    ev.run(right)
    assert ev.items(right, "hybrid", wrong_only=True) == []
    assert len(ev.items(right, "hybrid", wrong_only=False)) == 5


def test_items_wrong_only_without_answers_uses_skill_miss(env):
    conn, ev = env
    save_questions(
        conn,
        "c",
        [
            Question(
                "q-odd",
                "enzyme question",
                None,
                ["A"],
                [],
                {"skill": "odd", "solution": "enzyme explanation"},
            )
        ],
    )
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], split="full"))
    ev.run(job)
    assert [row["question_id"] for row in ev.items(job, "hybrid", wrong_only=True)] == ["q-odd"]


def test_estimate_matches_how_calls_are_sent(env):
    conn, ev = env
    many = [
        Question(f"m{i}", f"enzyme question {i}", ["yes", "no"], ["A"], [], {"skill": "enzyme"})
        for i in range(60)
    ]
    save_questions(conn, "c", many)
    methods = ["hybrid", "vector", "kw_all", "kw_individual", "kw_rrf", "keyword"]
    small = ev.estimate("c", "full", methods[:6], answer=True, model=None)
    # 65 questions per method: each method's answers exceed 50, so each is a batch.
    job = create_job(conn, "eval", "c", setup_for(methods, answer=True, split="full"))
    ev.run(job)
    modes = {r["batch"] for r in conn.execute("select batch from ai_ledger where purpose='answer'")}
    assert modes == {1} and small["calls"] == 65 * 6


def test_estimate_does_not_halve_for_runs_whose_methods_stay_under_50(env):
    conn, ev = env
    extra = [
        Question(f"x{i}", f"market question {i}", ["yes", "no"], ["A"], [], {"skill": "market"})
        for i in range(5)
    ]
    save_questions(conn, "c", extra)  # 10 questions
    methods = ["hybrid", "vector", "kw_all", "kw_individual", "kw_rrf", "keyword"]
    est = ev.estimate("c", "full", methods, answer=True, model=None)  # 60 calls in all
    single = ev.estimate("c", "full", ["hybrid"], answer=True, model=None)  # 10 calls
    assert est["est_usd"] == pytest.approx(6 * single["est_usd"], rel=0.2)


def test_rows_keep_the_raw_answer_line(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "thinking\nAnswer: The answer is A")
    job = create_job(
        conn, "eval", "c", setup_for(["hybrid"], answer=True, model="claude-sonnet-5-5")
    )
    ev.run(job)
    row = run_items(conn, job)[0]
    assert row["pred"] == "A" and row["raw"] == "The answer is A"


def test_estimate_prices_hops_at_the_model_that_sends_them(env):
    # Hop and query-summary calls run on the service's default model; only the
    # answers use the requested one.
    _, ev = env
    default = ev.estimate("c", "dev", ["iterative"], answer=False, model=None)["est_usd"]
    sonnet = ev.estimate("c", "dev", ["iterative"], answer=False, model="claude-sonnet-5-5")
    assert sonnet["est_usd"] == pytest.approx(default)


def test_setup_is_recorded_on_first_run_only(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    assert "question_ids" not in get_job(conn, job)["setup"]
    ev.run(job)
    ids = get_job(conn, job)["setup"]["question_ids"]
    assert sorted(ids) == sorted(q.id for q in QUESTIONS)
    new = Question("q-new", "new", ["a"], ["A"], [], {"skill": "x", "solution": "y"})
    save_questions(conn, "c", [new])
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["setup"]["question_ids"] == ids
    assert "q-new" not in {r["question_id"] for r in run_items(conn, job)}


def test_queue_stores_only_the_request(env):
    conn, ev = env
    search = {"weights": {"words": 0.5}}
    job = ev.queue("c", "dev", ["hybrid"], True, None, k=3, rrf_k=10, search=search)
    assert get_job(conn, job)["setup"] == {
        "collection": "c",
        "split": "dev",
        "methods": ["hybrid"],
        "answer": True,
        "model": ev.ai.model_for("answer"),
        "k": 3,
        "rrf_k": 10,
        "search": search,  # the override as asked; the effective settings are recorded on run
        "eval_set": "c",
    }
    assert get_job(conn, job)["status"] == "queued"
    ev.run(job)
    version = db.get_eval_set(conn, "c")["version"]
    assert get_job(conn, job)["setup"]["eval_set_version"] == version


def test_run_skips_recorded_ids_whose_question_is_gone(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    setup = get_job(conn, job)["setup"]
    ids = ["q-gone", "q-market", "q-enzyme"]
    versions = ev._record_setup(job, ev.retriever.spec("c"), setup)
    db.set_job_setup(conn, job, {**versions, "question_ids": ids})
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [r["question_id"] for r in run_items(conn, job)] == ["q-market", "q-enzyme"]


def test_retrieval_only_rows_are_complete_from_the_start(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid", "vector"], False, None)
    seen = []

    def check():
        seen.append([r["complete"] for r in run_items(conn, job, "hybrid")])

    ev.run(job, check=check)
    assert seen == [[], [True] * 5]


def test_resume_redoes_only_incomplete_rows(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid", "vector"], True, None)

    def stop_after_first_method():
        hybrid = run_items(conn, job, "hybrid")
        if hybrid and all(r["complete"] for r in hybrid):
            raise Cancelled()

    ev.run(job, check=stop_after_first_method)
    assert get_job(conn, job)["status"] == "cancelled"
    assert ev.estimate_resume(job)["calls"] == 5  # vector's 5 answers only
    sent, judged = [], []
    complete_many, judge = ev.ai.complete_many, ev.models.judge

    def counting_complete_many(prompts, **kw):
        sent.append(len(prompts))
        return complete_many(prompts, **kw)

    def counting_judge(texts):
        judged.append(len(texts))
        return judge(texts)

    ev.ai.complete_many = counting_complete_many
    ev.models.judge = counting_judge
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert sent == [5] and judged == [5, 5]  # vector only: answers, then mine and gold
    assert len(run_items(conn, job)) == 10 and all(r["complete"] for r in run_items(conn, job))


def test_resume_of_a_complete_run_estimates_zero(env):  # Review focus 2
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], True, None)
    ev.run(job)
    assert ev.estimate_resume(job) == {"questions": 0, "calls": 0, "est_usd": 0.0, "roles": []}
    rows = run_items(conn, job)
    ledger = conn.execute("select count(*) from ai_ledger")[0][0]
    db.update_job(conn, job, status="interrupted")
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert run_items(conn, job) == rows
    assert conn.execute("select count(*) from ai_ledger")[0][0] == ledger


def test_resume_estimate_of_a_run_that_never_started_is_the_full_estimate(env):
    _, ev = env
    job = ev.queue("c", "dev", ["hybrid", "kw_individual"], True, None)
    full = ev.estimate("c", "dev", ["hybrid", "kw_individual"], answer=True, model=None)
    assert ev.estimate_resume(job) == {
        k: full[k] for k in ("questions", "calls", "est_usd", "roles")
    }


def test_resume_estimate_counts_a_method_with_no_rows(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid", "vector"], True, None)
    ev.run(job)
    conn.execute("delete from run_items where job_id = ? and method = 'vector'", (job,))
    conn.execute(
        "update run_items set complete = 0 where job_id = ? and question_id = 'q-enzyme'", (job,)
    )
    est = ev.estimate_resume(job)
    assert est["calls"] == 6 and est["questions"] == 5 and est["est_usd"] > 0


def test_resume_refused_when_lists_changed(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["kw_individual"], False, None)
    other = ev.queue("c", "dev", ["hybrid"], False, None)
    for j in (job, other):
        ev.run(j)
        db.update_job(conn, j, status="interrupted")
    assert ev.resume_refusal(job) is None
    old = ev.retriever.spec("c").keyword_version
    new = ev.retriever.indexer.set_lists(ev.retriever.spec("c"), OTHER_LISTS)
    assert ev.resume_refusal(job) == (
        f"keyword lists changed since this run ({old} → {new}); start a new run"
    )
    assert ev.resume_refusal(other) is None  # hybrid never searches the lists


def test_resume_refused_when_dense_model_changed(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    db.update_job(conn, job, status="failed")
    conn.execute("update collections set dense_model = 'other/dense' where name = 'c'")
    assert ev.resume_refusal(job) == (
        "the dense model changed since this run (fake/dense → other/dense); start a new run"
    )


def test_resume_refused_by_kind_and_status(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    assert ev.resume_refusal(job) == "a done run cannot be resumed"
    other = create_job(conn, "load", "c", {})
    db.update_job(conn, other, status="interrupted")
    assert ev.resume_refusal(other) == "only evals can be resumed"


def test_a_run_that_never_started_is_resumable_after_list_changes(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    db.update_job(conn, job, status="cancelled")
    ev.retriever.indexer.set_lists(ev.retriever.spec("c"), OTHER_LISTS)
    assert ev.resume_refusal(job) is None


def test_question_only_has_no_retrieval_metrics(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["question_only", "hybrid"], False, None)
    ev.run(job)
    rows = run_items(conn, job, "question_only")
    assert len(rows) == 5 and all(r["retrieval"] == {} for r in rows)
    t = ev.table(job)
    assert t["methods"]["question_only"] == {"usd": 0.0, "n": 5}
    assert all("question_only" not in (p["a"], p["b"]) for p in t["pairs"])
    assert ev.items(job, "question_only", wrong_only=True) == []


def test_legacy_question_only_zeros_are_cleared_on_connect(env, tmp_path):
    conn, ev = env
    job = ev.queue("c", "dev", ["question_only", "hybrid"], False, None)
    ev.run(job)
    legacy = {"hit_ids": [], "hit_skills": [], "usd": 0.0}
    legacy["retrieval"] = {"recall@1": 0.0, "recall@2": 0.0, "solution_sim@1": 0.0}
    for r in run_items(conn, job, "question_only"):
        db.upsert_run_item(conn, job, "question_only", r["question_id"], legacy, complete=True)

    migrated = connect(tmp_path / "r.db")  # a second connection, as a restart would open
    rows = run_items(migrated, job, "question_only")
    assert len(rows) == 5 and all(r["retrieval"] == {} for r in rows)
    t = ev.table(job)
    assert t["methods"]["question_only"] == {"usd": 0.0, "n": 5}
    assert all(g["method"] != "question_only" for g in t["gaps"])
    assert ev.items(job, "question_only", wrong_only=True) == []
    before = [tuple(r) for r in migrated.execute("select * from run_items order by rowid")]
    again = connect(tmp_path / "r.db")
    assert [tuple(r) for r in again.execute("select * from run_items order by rowid")] == before


def test_question_only_answers_are_scored(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "Answer: B")
    job = ev.queue("c", "dev", ["question_only"], True, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert all(r["retrieval"] == {} and r["complete"] for r in run_items(conn, job))
    assert len(ev.items(job, "question_only", wrong_only=True)) == 5


def test_queue_refuses_an_empty_split(env):
    conn, ev = env
    ev.retriever.indexer.create("e", "mcq", False)
    db.create_eval_set(conn, "e", "mcq")
    with pytest.raises(ValueError, match="eval set 'e' has no questions"):
        ev.queue("e", "dev", ["hybrid"], False, None)
    assert list_jobs(conn) == []


def test_table_with_question_only_and_two_answered_methods(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["question_only", "hybrid", "vector"], True, None)
    ev.run(job)
    t = ev.table(job)
    q_only = t["methods"]["question_only"]
    assert "accuracy" in q_only and not {"solution_sim@1", "same_skill@1"} & set(q_only)
    assert t["proxy"] is None  # two methods with both metrics are too few to rank


def test_a_resume_claimed_after_a_rekey_fails_without_spending(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["kw_individual", "vector"], True, None)

    def stop_after_first_method():
        first = run_items(conn, job, "kw_individual")
        if first and all(r["complete"] for r in first):
            raise Cancelled()

    ev.run(job, check=stop_after_first_method)
    old = ev.retriever.spec("c").keyword_version
    new = ev.retriever.indexer.set_lists(ev.retriever.spec("c"), OTHER_LISTS)
    rows = run_items(conn, job)
    ledger = conn.execute("select count(*) from ai_ledger")[0][0]
    requeue(conn, job)  # as if the re-key was queued after the resume was accepted
    ev.run(job)
    after = get_job(conn, job)
    assert after["status"] == "failed"
    assert after["error"] == (
        f"keyword lists changed since this run ({old} → {new}); start a new run"
    )
    assert run_items(conn, job) == rows
    assert conn.execute("select count(*) from ai_ledger")[0][0] == ledger


def test_resume_refused_when_judge_model_changed(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    db.update_job(conn, job, status="interrupted")
    db.set_job_setup(conn, job, {**get_job(conn, job)["setup"], "judge_model": "old/judge"})
    assert ev.resume_refusal(job) == (
        "the judge model changed since this run (old/judge → fake/judge); start a new run"
    )


def legacy_job(conn, ev):
    """An eval from before question ids were recorded: versions, rows, no ids."""
    setup = {
        **setup_for(["kw_individual"]),
        "dense_model": "fake/dense",
        "judge_model": "fake/judge",
        "keyword_version": ev.retriever.spec("c").keyword_version,
        "seed": 7,
        "dataset_filter": "an older filter",
    }
    job = create_job(conn, "eval", "c", setup)
    for qid in ("q-market", "q-enzyme"):
        db.upsert_run_item(conn, job, "kw_individual", qid, {"retrieval": {}}, complete=False)
    db.update_job(conn, job, status="interrupted")
    return job, setup


def test_a_legacy_run_keeps_its_setup_and_its_own_questions(env):
    conn, ev = env
    job, setup = legacy_job(conn, ev)
    assert ev.estimate_resume(job)["questions"] == 2
    requeue(conn, job)
    ev.run(job)
    after = get_job(conn, job)
    assert after["status"] == "done"
    assert after["setup"]["question_ids"] == ["q-enzyme", "q-market"]
    assert {k: after["setup"][k] for k in setup} == setup
    assert sorted(r["question_id"] for r in run_items(conn, job)) == ["q-enzyme", "q-market"]


def test_a_legacy_run_is_refused_after_list_changes(env):
    conn, ev = env
    job, _ = legacy_job(conn, ev)
    ev.retriever.indexer.set_lists(ev.retriever.spec("c"), OTHER_LISTS)
    assert "keyword lists changed" in (ev.resume_refusal(job) or "")


def test_table_serves_gaps_and_answer(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid", "kw_individual", "random_example"], True, None)
    ev.run(job)
    t = ev.table(job)
    assert t["reference"] == "hybrid" and t["headline"] == "accuracy"
    assert {g["method"] for g in t["gaps"]} == {"kw_individual", "random_example"}
    assert t["answer"]["candidate"] == "kw_individual" and t["usd_total"] >= 0
    assert t["question_count"] == 5 and t["scale"] == {"unit": "pts", "factor": 100}
    assert t["best_of_each"]["candidate"] == "kw_individual"


def test_table_without_the_pair_has_no_answer_and_headlines_reasoning_match(env):
    _, ev = env
    job = ev.queue("c", "dev", ["vector", "random_example"], False, None)
    ev.run(job)
    t = ev.table(job)
    assert t["answer"] is None and t["headline"] == "solution_sim@1"
    assert t["scale"]["unit"] == "match"
    assert t["question_count"] == 5


def test_collection_answer_prefers_the_latest_answered_run(env):
    conn, ev = env
    assert ev.collection_answer("c") is None
    answered = ev.queue("c", "dev", ["hybrid", "kw_individual"], True, None)
    ev.run(answered)
    free = ev.queue("c", "dev", ["hybrid", "kw_individual"], False, None)
    ev.run(free)
    a = ev.collection_answer("c")
    assert a["run_id"] == answered and a["answered"] is True and a["runs"] == 2
    assert a["candidate"] == "kw_individual" and a["verdict"] in ("yes", "no", "cant_tell")
    assert a["at"] == get_job(conn, answered)["created_at"]


def test_collection_answer_falls_back_to_the_newest_free_run(env):
    _, ev = env
    older = ev.queue("c", "dev", ["hybrid", "kw_individual"], False, None)
    ev.run(older)
    newer = ev.queue("c", "dev", ["hybrid", "kw_individual"], False, None)
    ev.run(newer)
    a = ev.collection_answer("c")
    assert a["run_id"] == newer and a["answered"] is False and a["runs"] == 2
    assert a["scale"] == ev.table(newer)["scale"]  # for the raw reasoning-match note


def test_collection_answer_is_none_when_the_run_lacks_the_pair(env):
    _, ev = env
    job = ev.queue("c", "dev", ["hybrid", "vector"], True, None)
    ev.run(job)
    assert ev.collection_answer("c") is None


def test_summary_matches_the_table_without_building_it(env, monkeypatch):
    from ragqa_core import evaluator

    _, ev = env
    methods = ["hybrid", "kw_individual", "random_example"]
    answered = ev.queue("c", "dev", methods, True, None)
    ev.run(answered)
    free = ev.queue("c", "dev", methods, False, None)
    ev.run(free)
    trusted = ev.collection_proxy_trusted("c")
    tables = {job: ev.table(job) for job in (answered, free)}

    def no_intervals(*args, **kwargs):
        raise AssertionError("summary must not compute per-method intervals")

    monkeypatch.setattr(evaluator, "bootstrap_ci", no_intervals)
    ev.table = no_intervals
    for job, t in tables.items():
        s = ev.summary(job, trusted)
        assert s is not None and s["answer"] is not None
        assert s["answer"] == t["answer"]  # the same numbers
        keys = ("headline", "scale", "usd_total", "question_count")
        assert {k: s[k] for k in keys} == {k: t[k] for k in keys}
    assert tables[answered]["headline"] == "accuracy"
    assert tables[free]["headline"] != "accuracy"
    for job, t in tables.items():
        assert ev.summary(job, trusted)["proxy_trusted"] == t["proxy_trusted"]
    assert ev.summary("missing", None) is None


def finished_run(conn, scores):
    """A done answered run with given {method: (solution_sim, accuracy)} for one question."""
    job = create_job(conn, "eval", "c", {**setup_for(list(scores), answer=True), "eval_set": "c"})
    for method, (sim, acc) in scores.items():
        row = {"retrieval": {"solution_sim@1": sim}, "answer": {"accuracy": acc}, "usd": 0.0}
        db.upsert_run_item(conn, job, method, "q-enzyme", row, complete=True)
    db.update_job(conn, job, status="done")
    return job


def test_collection_proxy_trusted_uses_the_latest_real_check(env):
    conn, ev = env
    assert ev.collection_proxy_trusted("c") is None
    four_bad = {
        "vector": (0.1, 0.9),
        "keyword": (0.2, 0.8),
        "hybrid": (0.3, 0.7),
        "kw_all": (0.4, 0.6),
    }
    finished_run(conn, four_bad)
    assert ev.collection_proxy_trusted("c") is False
    finished_run(conn, {"hybrid": (0.5, 0.5), "vector": (0.4, 0.4)})  # too few for a check
    assert ev.collection_proxy_trusted("c") is False  # the older real check still stands


def test_a_run_with_no_methods_is_refused_and_an_empty_one_tabulates(env):
    conn, ev = env
    with pytest.raises(ValueError, match="choose at least one method"):
        ev.queue("c", "dev", [], True, None)
    job = create_job(conn, "eval", "c", setup_for([], answer=True))
    ev.run(job)  # a legacy empty run must not recurse
    t = ev.table(job)
    assert t["proxy"] is None and t["gaps"] == [] and t["answer"] is None


def test_headline_falls_back_when_items_have_no_explanations(env):
    conn, ev = env
    ix = ev.retriever.indexer
    spec = ix.create("e", "mcq", False)
    ix.add(spec, [Item(f"{t}-item", f"{t} things", "A", None, {"skill": t}) for t in TOPICS], "x")
    ix.set_lists(spec, LISTS)
    save_questions(conn, "e", QUESTIONS)
    job = ev.queue("e", "dev", ["hybrid", "random_example"], False, None)
    ev.run(job)
    t = ev.table(job)
    assert "solution_sim@1" not in t["methods"]["hybrid"]
    assert t["headline"] == "same_skill@1" and t["proxy_trusted"] is None


def test_unanswered_run_headline_follows_a_failed_proxy_check(env):
    conn, ev = env
    four_bad = {
        "vector": (0.1, 0.9),
        "keyword": (0.2, 0.8),
        "hybrid": (0.3, 0.7),
        "kw_all": (0.4, 0.6),
    }
    finished_run(conn, four_bad)
    job = ev.queue("c", "dev", ["hybrid", "random_example"], False, None)
    ev.run(job)
    t = ev.table(job)
    assert t["proxy_trusted"] is False and t["headline"] == "same_skill@1"


def test_rows_carry_the_matched_keywords_per_hit(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["kw_individual", "vector"]))
    ev.run(job)
    for row in run_items(conn, job, "kw_individual"):
        assert len(row["hit_matched"]) == len(row["hit_ids"])
        assert all(isinstance(m, list) for m in row["hit_matched"])
    assert any(m for row in run_items(conn, job, "kw_individual") for m in row["hit_matched"])
    assert all(not m for row in run_items(conn, job, "vector") for m in row["hit_matched"])


def test_collection_answer_keeps_the_pair_run_when_a_newer_run_lacks_the_pair(env):
    _, ev = env
    pair = ev.queue("c", "dev", ["kw_individual", "hybrid"], True, None)
    ev.run(pair)
    ev.run(ev.queue("c", "dev", ["vector"], True, None))
    a = ev.collection_answer("c")
    assert a is not None and a["run_id"] == pair and a["runs"] == 2
    # An answered pair run wins over a newer free one; the newest wins among equals.
    free = ev.queue("c", "dev", ["kw_individual", "hybrid"], False, None)
    ev.run(free)
    assert ev.collection_answer("c")["run_id"] == pair
    newer = ev.queue("c", "dev", ["kw_individual", "hybrid"], True, None)
    ev.run(newer)
    assert ev.collection_answer("c")["run_id"] == newer


def test_collection_answer_carries_proxy_trusted(env):
    _, ev = env
    ev.run(ev.queue("c", "dev", ["kw_individual", "hybrid"], False, None))
    a = ev.collection_answer("c")
    assert "proxy_trusted" in a and a["proxy_trusted"] == ev.collection_proxy_trusted("c")


def many_questions(conn, n=25):
    save_questions(
        conn,
        "c",
        [
            Question(f"m{i:02}", f"a {TOPICS[i % 5]} question", ["yes", "no"], ["A"], [], {})
            for i in range(n)
        ],
    )


def test_a_long_method_reports_progress_and_checks_inside(env):
    conn, ev = env
    many_questions(conn)
    job = ev.queue("c", "dev", ["vector"], False, None)
    seen = []

    def check():
        seen.append(get_job(conn, job)["progress"])

    ev.run(job, check=check)
    inside = [p for p in seen if 0 < p < 1]
    assert len(set(inside)) >= 2, seen  # beyond the one report between methods
    assert seen == sorted(seen) and get_job(conn, job)["progress"] == 1.0


def test_cancel_takes_effect_inside_a_method(env):
    conn, ev = env
    many_questions(conn)
    job = ev.queue("c", "dev", ["vector"], False, None)
    calls = []

    def check():
        calls.append(1)
        if len(calls) > 2 and get_job(conn, job)["progress"] > 0:
            raise Cancelled()

    ev.run(job, check=check)
    assert get_job(conn, job)["status"] == "cancelled"
    assert len(run_items(conn, job)) == 0  # stopped while retrieving, before any row


def test_an_absent_primary_metric_is_not_wrong():
    from ragqa_core.evaluator import is_wrong

    assert is_wrong({"retrieval": {"same_skill@1": 0.0}}, "same_skill@1") is True
    assert is_wrong({"retrieval": {"solution_sim@1": 0.4}}, "same_skill@1") is False
    assert is_wrong({"retrieval": {}}, "same_skill@1") is False


def test_pairs_match_questions_by_id_when_a_metric_is_absent_on_different_ones(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["vector", "keyword"]))
    sims = {
        "vector": {"q-enzyme": 0.9, "q-market": 0.5, "q-volcano": None, "q-optics": 0.2},
        "keyword": {"q-enzyme": 0.1, "q-market": None, "q-volcano": 0.7, "q-optics": 0.0},
    }
    for method, per_q in sims.items():
        for q, sim in per_q.items():
            retrieval = {"same_skill@1": 1.0} | ({} if sim is None else {"solution_sim@1": sim})
            row = {"retrieval": retrieval, "usd": 0.0}
            db.upsert_run_item(conn, job, method, q, row, complete=True)
    db.update_job(conn, job, status="done")
    pair = next(p for p in ev.table(job)["pairs"] if p["metric"] == "solution_sim@1")
    # shared: enzyme (0.9-0.1) and optics (0.2-0.0), not a positional 0.5-0.7 mix
    assert pair["a"] == "vector" and pair["diff"] == pytest.approx(0.5)


def test_setup_records_the_collections_dense_model_the_resume_check_compares(env):
    conn, ev = env
    conn.execute("update collections set dense_model = 'other/dense' where name = 'c'")
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    assert get_job(conn, job)["setup"]["dense_model"] == "other/dense"
    db.update_job(conn, job, status="interrupted")
    assert ev.resume_refusal(job) is None


def test_a_migrated_run_without_versions_cannot_be_resumed(env):
    conn, ev = env
    job, setup = legacy_job(conn, ev)
    unversioned = {k: v for k, v in setup.items() if k not in ("keyword_version", "dense_model")}
    db.set_job_setup(conn, job, {**unversioned, "question_ids": ["q-enzyme", "q-market"]})
    reason = "this run predates version tracking; start a new run"
    assert ev.resume_refusal(job) == reason
    requeue(conn, job)
    ev.run(job)
    after = get_job(conn, job)
    assert after["status"] == "failed" and after["error"] == reason


def test_an_answering_run_judges_only_answers(env):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["hybrid", "vector"], answer=True))
    right = {"retrieval": {"same_skill@1": 0.0}, "answer": {"accuracy": 1.0}, "usd": 0.0}
    db.upsert_run_item(conn, job, "hybrid", "q-enzyme", right, complete=True)
    # vector stopped before its answer (a budget stop): a miss on retrieval, but unmeasured
    missed = {"retrieval": {"same_skill@1": 0.0}, "usd": 0.0}
    db.upsert_run_item(conn, job, "vector", "q-enzyme", missed, complete=False)
    db.update_job(conn, job, status="stopped_budget")
    assert ev.questions(job, disagree=True) == []
    methods = ev.questions(job, disagree=False)[0]["methods"]
    assert methods["hybrid"]["outcome"] == "right" and "outcome" not in methods["vector"]
    assert ev.items(job, "vector", wrong_only=True) == []


def test_since_notes_lists_set_after_a_run_that_had_none(env):
    from ragqa_core.evaluator import changes_since

    spec = env[1].retriever.spec("c")
    assert spec.keyword_version is not None
    setup = {**setup_for(["kw_individual"]), "keyword_version": None, "dense_model": "fake/dense"}
    assert changes_since(setup, spec, []) == {"untracked": False, "changes": ["keyword lists"]}


def test_since_has_no_keyword_note_for_a_run_without_keyword_methods(env):
    from ragqa_core.evaluator import changes_since

    spec = env[1].retriever.spec("c")
    for version in (None, "older"):
        setup = {
            **setup_for(["vector", "keyword", "hybrid"]),
            "keyword_version": version,
            "dense_model": "fake/dense",
        }
        assert changes_since(setup, spec, []) == {"untracked": False, "changes": []}


def test_a_legacy_run_left_running_by_a_stopped_process_is_refused_on_resume(env, tmp_path):
    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True))
    db.upsert_run_item(conn, job, "hybrid", "q-enzyme", {"retrieval": {}}, complete=False)
    db.update_job(conn, job, status="running")  # an old process stopped mid-run
    connect(tmp_path / "r.db")  # the restart: its migration, then the lifespan's interrupt
    db.interrupt_running(conn)
    assert ev.resume_refusal(job) == "this run predates version tracking; start a new run"


def science_env(conn, ev, name, items, questions):
    ix = ev.retriever.indexer
    spec = ix.create(name, "science", False)
    ix.add(spec, items, "x")
    ix.set_lists(spec, LISTS)
    save_questions(conn, name, questions)


def test_science_run_scores_free_form_and_choice_questions(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "Answer: A" if "Choices:" in p else "Answer: 3.1 J")
    ev.retriever.ai = ev.ai
    questions = [
        Question("s-free", "an enzyme problem", None, ["3.0"], [], {"skill": "enzyme"}),
        Question("s-mc", "a market problem", ["yes", "no"], ["A"], [], {"skill": "market"}),
    ]
    science_env(conn, ev, "sci", ITEMS, questions)
    job = ev.queue("sci", "dev", ["hybrid"], True, None)
    ev.run(job)
    rows = run_items(conn, job)
    assert len(rows) == 2 and all(r["answer"] == {"accuracy": 1.0} for r in rows)
    assert ev.table(job)["headline"] == "accuracy"


def test_science_free_run_without_solutions_headlines_same_skill(env):
    conn, ev = env
    items = [Item(f"{t}-item", f"{t} things", "A", None, {"skill": t}) for t in TOPICS]
    questions = [
        Question(f"q-{t}", f"a {t} problem", None, ["3"], [], {"skill": t}) for t in TOPICS
    ]
    science_env(conn, ev, "sci", items, questions)
    job = ev.queue("sci", "dev", ["hybrid", "random_example"], False, None)
    ev.run(job)
    assert all("solution_sim@1" not in r["retrieval"] for r in run_items(conn, job))
    assert ev.table(job)["headline"] == "same_skill@1"


def test_no_hits_score_zero_only_where_measurable(env):
    conn, ev = env
    questions = [
        Question("q-sol", "a problem", None, ["3"], [], {"skill": "enzyme", "solution": "work"}),
        Question("q-nosol", "another problem", None, ["3"], [], {"skill": "market"}),
        Question("q-bare", "a third problem", None, ["3"], [], {}),
    ]
    science_env(conn, ev, "sci", ITEMS, questions)
    spec = ev.retriever.indexer.spec("sci")
    rows = ev._retrieval_rows(spec, questions, [[], [], []], set())
    assert rows[0]["retrieval"] == {"solution_sim@1": 0.0, "same_skill@1": 0.0}
    assert rows[1]["retrieval"] == {"same_skill@1": 0.0}
    assert rows[2]["retrieval"] == {}


def test_mcq_scoring_and_headline_unchanged(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], True, None)
    ev.run(job)
    rows = run_items(conn, job)
    assert all(r["answer"] == {"accuracy": 1.0} for r in rows)
    assert ev.table(job)["headline"] == "accuracy"


def multihop_env(tmp_path):
    conn = connect(tmp_path / "m.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "x")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("c", "multihop", False)
    ix.add(spec, [Item("g1", "enzyme rate rose", "A")], "x")
    ix.add(spec, [Item("g2", "market price fell", "B")], "y")
    save_questions(
        conn,
        "c",
        [
            Question("q1", "enzyme rate", None, ["A"], ["g1"], {}),
            Question("q2", "market price", None, ["B"], ["g2"], {}),
            Question("q3", "volcano lava", None, ["C"], [], {}),
        ],
    )
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    return conn, Evaluator(conn, retriever, models, ai)


def run_multihop(conn, ev, methods=("hybrid",)):
    job = create_job(conn, "eval", "c", setup_for(list(methods), split="full"))
    ev.run(job)
    return job


def test_run_records_sources_and_resume_is_refused_after_a_switch(env):
    from ragqa_core.evaluator import changes_since

    conn, ev = env
    job = create_job(conn, "eval", "c", setup_for(["hybrid"]))
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert setup["disabled_sources"] == [] and setup["evidence_off"] == 0
    db.update_job(conn, job, status="stopped_budget")
    assert ev.resume_refusal(job) is None
    db.set_source_enabled(conn, "c", "x", False)
    assert "sources switched" in (ev.resume_refusal(job) or "")
    spec = ev.retriever.spec("c")
    assert changes_since(setup, spec, ["x"])["changes"] == ["sources"]
    assert changes_since(setup, spec, [])["changes"] == []


def test_untracked_run_shows_no_source_change(env):
    from ragqa_core.evaluator import changes_since

    spec = env[1].retriever.spec("c")
    setup = {**setup_for(["hybrid"]), "question_ids": ["q-enzyme"]}
    assert changes_since(setup, spec, ["x"]) == {"untracked": True, "changes": []}


def test_recall_absent_without_gold(tmp_path):
    conn, ev = multihop_env(tmp_path)
    job = run_multihop(conn, ev)
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    assert "recall@2" in rows["q1"]["retrieval"]
    assert rows["q3"]["retrieval"] == {}
    assert ev.table(job)["methods"]["hybrid"]["recall@2"]["n"] == 2


def test_evidence_off_counts_and_drops_recall(tmp_path):
    conn, ev = multihop_env(tmp_path)
    db.set_source_enabled(conn, "c", "y", False)
    job = run_multihop(conn, ev)
    setup = get_job(conn, job)["setup"]
    assert setup["disabled_sources"] == ["y"] and setup["evidence_off"] == 1
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    assert rows["q2"]["retrieval"] == {}
    assert "g2" not in rows["q2"]["hit_ids"] + rows["q1"]["hit_ids"]
    assert rows["q1"]["retrieval"]["recall@2"] > 0


def test_recall_unchanged_for_questions_with_gold(tmp_path):
    from ragqa_core.metrics import recall_at

    conn, ev = multihop_env(tmp_path)
    job = run_multihop(conn, ev)
    for r in run_items(conn, job):
        gold = {"q1": ["g1"], "q2": ["g2"]}.get(r["question_id"])
        if gold:
            assert r["retrieval"]["recall@2"] == recall_at(r["hit_ids"], gold, 2)
            assert r["retrieval"]["recall@5"] == recall_at(r["hit_ids"], gold, 5)


LONG = "the enzyme rate rose sharply after the temperature was raised"


def test_leaked_questions_are_left_out_unless_included(env):
    conn, ev = env
    ix = ev.retriever.indexer
    ix.add(ix.spec("c"), [Item("long-item", LONG + " in the lab", "A", "x", {})], "x")
    leaked = replace(QUESTIONS[0], text=LONG, meta={**QUESTIONS[0].meta, "source": "hand"})
    save_questions(conn, "c", [leaked])
    assert ev.estimate("c", "full", ["hybrid"], False, None)["questions"] == 4
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], split="full"))
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert setup["leaked_excluded"] == 1 and len(setup["question_ids"]) == 4
    db.set_include_leaked(conn, "c", True)
    assert ev.estimate("c", "full", ["hybrid"], False, None)["questions"] == 5
    job = create_job(conn, "eval", "c", setup_for(["hybrid"], split="full"))
    ev.run(job)
    assert get_job(conn, job)["setup"]["leaked_excluded"] == 0


def fields(**over):
    base = {
        "questions": {"total": 200, "solution": 200, "skill": 200, "gold_ids": 200},
        "items": {"total": 50, "explanation": 50, "skill": 50},
    }
    for group, values in over.items():
        base[group] = {**base[group], **values}
    return base


def test_measurable_rules():
    from ragqa_core.evaluator import measurable

    ok = measurable("mcq", fields())
    assert ok["solution_sim@1"] == {"measurable": True, "note": None}
    assert ok["same_skill@1"]["measurable"] and ok["accuracy"] == {"measurable": True, "note": None}

    out = measurable("science", fields(questions={"solution": 0}))
    assert out["solution_sim@1"] == {
        "measurable": False,
        "note": "no question has a worked solution",
    }
    assert out["same_skill@1"]["measurable"]
    out = measurable("mcq", fields(items={"explanation": 0, "skill": 0}))
    assert out["solution_sim@1"]["note"] == "no item has an explanation"
    assert out["same_skill@1"]["measurable"] is False
    assert measurable("mcq", fields(questions={"skill": 0}))["same_skill@1"]["note"]

    out = measurable("multihop", fields(questions={"gold_ids": 0}))
    assert out["recall@2"] == {"measurable": False, "note": "no question names its evidence"}
    assert out["recall@5"]["measurable"] is False and out["em"]["measurable"]
    out = measurable("table", fields(questions={"gold_ids": 140}))
    assert out["recall@1"] == {"measurable": True, "note": "on 140 of 200 questions"}
    assert measurable("table", fields())["recall@3"] == {"measurable": True, "note": None}


# -- Keyword-RAG runs -----------------------------------------------------------------


def with_vocabulary(ev: Evaluator, terms: list[str]) -> str:
    """Put a vocabulary in force on collection c and tag every item with it."""
    builder = ev.retriever.builder
    spec = ev.retriever.spec("c")
    version = builder.save_vocabulary(spec, terms)
    db.set_vocabulary_version(ev.conn, "c", version)
    builder.tag(ev.retriever.spec("c"), builder.stored_items("c"))
    return version


def test_eval_records_search_and_query_tags(env):
    conn, ev = env
    version = with_vocabulary(ev, TOPICS)
    job = ev.queue(
        "c", "dev", ["keyword_rag", "hybrid"], False, None, search={"weights": {"keywords": 2.0}}
    )
    ev.run(job)
    assert get_job(conn, job)["status"] == "done", get_job(conn, job)["error"]
    setup = get_job(conn, job)["setup"]
    assert setup["search"]["weights"] == {"meaning": 1.0, "words": 1.0, "keywords": 2.0}
    assert setup["search"]["query_tags"] == "similarity"  # the collection's default
    assert setup["vocabulary_version"] == version
    rows = {(r["method"], r["question_id"]): r for r in run_items(conn, job)}
    tags = rows[("keyword_rag", "q-enzyme")]["query_tags"]
    assert tags and tags[0] == "enzyme"
    assert "query_tags" not in rows[("hybrid", "q-enzyme")]
    assert ev.table(job)["methods"]["keyword_rag"]["same_skill@1"]["mean"] == 1.0


def test_every_run_records_search_and_vocabulary(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert setup["search"] == {
        "weights": {"meaning": 1.0, "words": 1.0, "keywords": 1.0},
        "query_tags": "similarity",
        "example": "full",
        "program": False,
    }
    assert "vocabulary_version" in setup and setup["vocabulary_version"] is None


def test_queue_refuses_bad_search(env):
    _, ev = env
    with pytest.raises(ValueError, match="unknown channel"):
        ev.queue("c", "dev", ["keyword_rag"], False, None, search={"weights": {"vibes": 1}})
    with pytest.raises(ValueError, match="query_tags"):
        ev.queue("c", "dev", ["keyword_rag"], False, None, search={"query_tags": "psychic"})


def test_model_query_tags_are_priced_and_asked_once_per_question(env):
    conn, ev = env
    with_vocabulary(ev, TOPICS)
    asked = {"query_tags": "model"}
    est = ev.estimate("c", "dev", ["keyword_rag"], answer=False, model=None, search=asked)
    assert est["calls"] == 5 and est["est_usd"] > 0 and est["roles"] == ["query"]
    free = ev.estimate("c", "dev", ["hybrid"], answer=False, model=None, search=asked)
    assert free["calls"] == 0  # only keyword_rag asks for tags
    fake_replies(ev.ai, lambda p: json.dumps({"terms": [TOPICS[0]]}))
    job = ev.queue("c", "dev", ["keyword_rag"], False, None, search=asked)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done", get_job(conn, job)["error"]
    assert {r["query_tags"][0] for r in run_items(conn, job)} == {TOPICS[0]}
    # Sent once up front; each question's search then finds its tags cached.
    calls = conn.execute("select count(*) from ai_ledger where purpose='query_tags'")[0][0]
    assert calls == 5


def test_oracle_tags_from_gold_units(tmp_path):
    conn, ev = multihop_env(tmp_path)
    with_vocabulary(ev, ["enzyme", "market", "volcano"])
    spec = ev.retriever.spec("c")
    both = Question("qb", "x", None, ["A"], ["g2", "g1"], {})
    tags_g1 = ev.retriever.store.get("c", ["g1"])[0].tags
    tags_g2 = ev.retriever.store.get("c", ["g2"])[0].tags
    assert ev.oracle_tags(spec, both) == list(dict.fromkeys(tags_g2 + tags_g1))
    assert ev.oracle_tags(spec, both)[0] == "market"
    # No gold unit: the tags of the worked solution; neither: none.
    solved = Question("qs", "x", None, ["A"], [], {"solution": "volcano lava"})
    assert ev.oracle_tags(spec, solved)[0] == "volcano"
    assert ev.oracle_tags(spec, Question("qn", "x", None, ["A"], [], {})) == []


def test_oracle_run_searches_with_the_gold_tags(tmp_path):
    conn, ev = multihop_env(tmp_path)
    with_vocabulary(ev, ["enzyme", "market", "volcano"])
    job = ev.queue("c", "full", ["keyword_rag"], False, None, search={"query_tags": "oracle"})
    ev.run(job)
    assert get_job(conn, job)["status"] == "done", get_job(conn, job)["error"]
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    spec = ev.retriever.spec("c")
    assert rows["q2"]["query_tags"] == ev.oracle_tags(spec, db.get_questions(conn, "c")[1])
    assert rows["q2"]["query_tags"][0] == "market" and rows["q3"]["query_tags"] == []


def test_since_notes_a_new_vocabulary_for_keyword_rag_runs(env):
    from ragqa_core.evaluator import changes_since

    _, ev = env
    version = with_vocabulary(ev, TOPICS)
    spec = ev.retriever.spec("c")
    base = {"keyword_version": spec.keyword_version, "dense_model": "fake/dense"}
    old = {**setup_for(["keyword_rag"]), **base, "vocabulary_version": "older"}
    assert changes_since(old, spec, [])["changes"] == ["vocabulary"]
    same = {**old, "vocabulary_version": version}
    assert changes_since(same, spec, [])["changes"] == []
    other_methods = {**old, "methods": ["hybrid", "kw_individual"]}
    assert changes_since(other_methods, spec, [])["changes"] == []


def test_resume_is_refused_after_a_new_vocabulary(env):
    conn, ev = env
    with_vocabulary(ev, TOPICS)
    job = ev.queue("c", "dev", ["keyword_rag"], False, None)
    ev.run(job)
    db.update_job(conn, job, status="interrupted")
    assert ev.resume_refusal(job) is None
    with_vocabulary(ev, TOPICS[:2])
    assert "vocabulary changed" in (ev.resume_refusal(job) or "")
    hybrid = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(hybrid)
    db.update_job(conn, hybrid, status="interrupted")
    with_vocabulary(ev, TOPICS[:3])
    assert ev.resume_refusal(hybrid) is None  # the vocabulary is not what it searched with


PROGRAM_REPLY = "plan\n```python\nprint(6*7)\n```\nAnswer: 41"


def test_eval_uses_program_answer_and_records_it(env):
    conn, ev = env
    sent = []

    def reply(prompt):
        sent.append(prompt)
        return "Answer: A" if "Choices:" in prompt else PROGRAM_REPLY

    ev.ai = fake_ai(conn, reply)
    ev.retriever.ai = ev.ai
    questions = [
        Question("s-free", "an enzyme problem", None, ["42"], [], {"skill": "enzyme"}),
        Question("s-mc", "a market problem", ["yes", "no"], ["A"], [], {"skill": "market"}),
    ]
    science_env(conn, ev, "sci", ITEMS, questions)
    job = ev.queue("sci", "dev", ["hybrid"], True, None, search={"program": True})
    ev.run(job)
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    free, chosen = rows["s-free"], rows["s-mc"]
    assert free["pred"] == "42" and free["answer"] == {"accuracy": 1.0}
    assert free["program"] == {"ran": True, "output": "42\n", "error": None}
    assert chosen["pred"] == "A" and chosen["program"] is None
    assert free["example_form"] == chosen["example_form"] == "full"
    assert sum("```python" in p for p in sent) == 1


def test_cancel_stops_between_programs_and_resuming_pays_nothing(env, monkeypatch):
    conn, ev = env
    asked = []

    def reply(prompt):
        asked.append(prompt)
        return PROGRAM_REPLY

    ev.ai = fake_ai(conn, reply)
    ev.retriever.ai = ev.ai
    questions = [
        Question(f"s-{n}", f"an enzyme problem {n}", None, ["42"], [], {"skill": "enzyme"})
        for n in range(25)
    ]
    science_env(conn, ev, "sci", ITEMS, questions)
    job = ev.queue("sci", "dev", ["hybrid"], True, None, search={"program": True})
    ran = []

    def counted(text):
        ran.append(text)
        return "42", {"ran": True, "output": "42\n", "error": None}

    monkeypatch.setattr("ragqa_core.evaluator.program_answer", counted)

    def check():
        if ran:
            raise Cancelled()

    ev.run(job, check)
    assert db.get_job(conn, job)["status"] == "cancelled"
    assert len(ran) == 9  # stopped at the check before the 10th, not after all 25
    paid = len(asked)
    assert paid == 25
    ev.run(job)
    assert db.get_job(conn, job)["status"] == "done"
    assert len(asked) == paid  # the answers were cached
    assert len(ran) == 25


def test_eval_program_that_fails_falls_back_to_the_answer_line(env):
    conn, ev = env
    ev.ai = fake_ai(conn, lambda p: "```python\nraise ValueError('boom')\n```\nAnswer: 3.0")
    ev.retriever.ai = ev.ai
    questions = [Question("s-free", "an enzyme problem", None, ["3.0"], [], {"skill": "enzyme"})]
    science_env(conn, ev, "sci", ITEMS, questions)
    job = ev.queue("sci", "dev", ["hybrid"], True, None, search={"program": True})
    ev.run(job)
    (row,) = run_items(conn, job)
    assert row["pred"] == "3.0" and row["program"]["ran"] is False
    assert "boom" in row["program"]["error"]


def test_eval_example_off_sends_no_example(env):
    conn, ev = env
    sent = []
    fake_replies(ev.ai, lambda p: sent.append(p) or "Answer: A")
    job = ev.queue("c", "dev", ["hybrid"], True, None, search={"example": "off"})
    ev.run(job)
    assert len(sent) == 5 and not any("Example" in p for p in sent)
    assert all(r["example_form"] == "off" and r["program"] is None for r in run_items(conn, job))


def test_eval_abstracted_example_and_its_fallback(env):
    conn, ev = env
    sent = []
    fake_replies(ev.ai, lambda p: sent.append(p) or "Answer: A")
    abstracted = "Principle: P\nMethod: M\nAnswer: A"
    ev.retriever.store.set_payloads("c", {"enzyme-item": {"abstracted": abstracted}})
    job = ev.queue("c", "dev", ["hybrid"], True, None, search={"example": "abstracted"})
    ev.run(job)
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    assert rows["q-enzyme"]["example_form"] == "abstracted"
    assert rows["q-market"]["example_form"] == "full"
    assert sum(abstracted in p for p in sent) == 1


def test_program_raises_table_answer_tokens():
    from ragqa_core.evaluator import PROGRAM_TOKENS, answer_tokens

    assert PROGRAM_TOKENS == 600
    assert answer_tokens("table", True) == 600 and answer_tokens("table", False) == 100
    assert answer_tokens("science", True) == 1500


# -- Gold ids on worked collections (BRIGHT) -------------------------------------------

OWN = "Find the sum of the cubes of the roots of a cubic by Newton's sums."
WORKED = [
    Item("own", OWN, None, "The question's own solution, LEAKED-MARKER", {}),
    Item("g1", "Newton's sums relate power sums of roots to coefficients.", None, "sums", {}),
    Item("g2", "Power sums of the roots of a cubic follow a recurrence.", None, "roots", {}),
    *ITEMS,
]
GOLD_QUESTIONS = [
    Question("b1", OWN, None, [], ["g1", "g2"], {"excluded_ids": ["own"]}),
    Question("b2", "the roots of a cubic and their power sums", None, [], ["g2"], {}),
]


def test_worked_collection_with_gold_ids_scores_ndcg(env):
    from ragqa_core.metrics import ndcg_at, recall_at

    conn, ev = env
    science_env(conn, ev, "sci", WORKED, GOLD_QUESTIONS)
    job = ev.queue("sci", "full", ["hybrid"], False, None)
    ev.run(job)
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    for q in GOLD_QUESTIONS:
        ids = rows[q.id]["hit_ids"]
        assert 5 < len(ids) <= 10  # k is 5; gold ids retrieve to max(k, 10)
        assert rows[q.id]["retrieval"] == {
            "ndcg@10": ndcg_at(ids, q.gold_ids, 10),
            "recall@1": recall_at(ids, q.gold_ids, 1),
        }
    assert rows["b2"]["retrieval"]["ndcg@10"] > 0
    table = ev.table(job)
    assert table["methods"]["hybrid"]["ndcg@10"]["n"] == 2
    assert table["methods"]["hybrid"]["recall@1"]["n"] == 2
    assert table["headline"] == "ndcg@10"


def test_excluded_ids_never_scored_or_shown(env):
    conn, ev = env
    sent = []
    fake_replies(ev.ai, lambda p: sent.append(p) or "Answer: 3")
    science_env(conn, ev, "sci", WORKED, GOLD_QUESTIONS[:1])
    assert ev.retriever.retrieve("sci", OWN, "hybrid")[0].id == "own"  # it would come first
    methods = ["hybrid", "keyword", "vector", "random_example"]
    job = ev.queue("sci", "full", methods, True, None, search={"example": "full"})
    ev.run(job)
    rows = run_items(conn, job)
    assert len(rows) == 4
    assert all(r["hit_ids"] and "own" not in r["hit_ids"] for r in rows)
    assert sent and not any("LEAKED-MARKER" in p for p in sent)


def test_scienceqa_like_collection_lists_no_ndcg(env):
    from ragqa_core.evaluator import measurable

    out = measurable("mcq", fields(questions={"gold_ids": 0}))
    assert "ndcg@10" not in out and "recall@1" not in out
    out = measurable("science", fields(questions={"gold_ids": 0}))
    assert "ndcg@10" not in out and "recall@1" not in out
    out = measurable("science", fields(questions={"gold_ids": 111, "solution": 0, "skill": 0}))
    assert out["ndcg@10"] == {"measurable": True, "note": "on 111 of 200 questions"}
    assert out["recall@1"]["measurable"] and out["same_skill@1"]["measurable"] is False
    assert measurable("science", fields())["ndcg@10"] == {"measurable": True, "note": None}
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    rows = run_items(conn, job)
    assert all(len(r["hit_ids"]) == 5 for r in rows)  # no gold ids: k as asked
    assert not any({"ndcg@10", "recall@1"} & set(r["retrieval"]) for r in rows)


def test_bright_run_pairs_methods_on_ndcg(env, tmp_path):
    from pathlib import Path

    from ragqa_core.datasets import _bright

    conn, ev = env
    mini = Path(__file__).parent / "data" / "bright_mini"
    files = {
        part: mini / part / "aops-00000-of-00001.parquet" for part in ("examples", "documents")
    }
    items, questions = _bright("aops", "ba-", tmp_path, files)
    science_env(conn, ev, "sci", items, questions)
    job = ev.queue("sci", "full", ["hybrid", "keyword"], False, None)
    ev.run(job)
    table = ev.table(job)
    assert table["headline"] == "ndcg@10"
    (pair,) = table["pairs"]
    assert pair["metric"] == "ndcg@10" and {pair["a"], pair["b"]} == {"hybrid", "keyword"}
    assert pair["lo"] is not None and pair["hi"] is not None and pair["lo"] <= pair["hi"]


def test_run_asks_the_stores_default_set_when_it_has_another_name(env):
    conn, ev = env
    second = Question("lk", "second q", ["a"], ["A"], [], {"skill": "x"})
    other = [Question("o1", "other one", ["a"], ["A"], [], {"skill": "x"}), second]
    db.create_eval_set(conn, "other", "mcq")
    save_questions(conn, "other", other)
    db.set_include_leaked(conn, "other", True)
    db.set_default_eval_set(conn, "c", "other")
    job = ev.queue("c", "full", ["hybrid"], False, None)
    assert get_job(conn, job)["setup"]["eval_set"] == "other"
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert setup["eval_set"] == "other"
    assert setup["eval_set_version"] == db.get_eval_set(conn, "other")["version"]
    assert sorted(setup["question_ids"]) == ["lk", "o1"]
    assert setup["leaked_excluded"] == 0
    assert {r["question_id"] for r in run_items(conn, job)} == {"lk", "o1"}


def test_eval_refuses_an_eval_set_of_another_qa_type(env):
    conn, ev = env
    db.create_eval_set(conn, "s", "science")
    save_questions(conn, "s", [Question("s1", "how warm is it?", None, ["3"], [], {})])
    with pytest.raises(EvalSetMismatch, match="eval set 's' is science; store 'c' is mcq"):
        ev.queue("c", "full", ["hybrid"], False, None, eval_set="s")
    with pytest.raises(EvalSetMismatch):
        ev.estimate("c", "full", ["hybrid"], False, None, eval_set="s")
    assert list_jobs(conn, kind="eval") == []


def test_a_run_keeps_its_eval_set_version_after_an_edit(env):
    conn, ev = env
    done = ev.queue("c", "full", ["hybrid"], False, None)
    ev.run(done)
    ev.ai = fake_ai(conn, lambda p: "Answer: A", cap_usd=1e-9)
    stopped = ev.queue("c", "full", ["hybrid"], True, None)
    ev.run(stopped)
    assert get_job(conn, stopped)["status"] == "stopped_budget"
    queued = ev.queue("c", "full", ["hybrid"], False, None)
    old = db.get_eval_set(conn, "c")["version"]
    edited = [replace(q, text=f"edited {q.text}") for q in QUESTIONS]
    new = db.replace_questions(conn, "c", edited, old)
    texts = {q["question_id"]: q["question"] for q in ev.questions(done, False)}
    assert texts["q-enzyme"] == "a enzyme question"
    assert ev.table(done)["since"]["changes"] == ["eval set"]
    spec = ev.retriever.spec("c")
    assert changes_since(get_job(conn, done)["setup"], spec, [], new)["changes"] == ["eval set"]
    sent: list[str] = []
    ev.ai = fake_ai(conn, lambda p: sent.append(p) or "Answer: A")
    requeue(conn, stopped)
    ev.run(stopped)
    assert get_job(conn, stopped)["status"] == "done"
    assert sent and not any("edited" in p for p in sent)
    assert any("a enzyme question" in p for p in sent)
    ev.run(queued)
    assert get_job(conn, queued)["setup"]["eval_set_version"] == new
    texts = {q["question_id"]: q["question"] for q in ev.questions(queued, False)}
    assert texts["q-enzyme"] == "edited a enzyme question"
    assert ev.table(queued)["since"]["changes"] == []


def evidence_env(tmp_path):
    """Multihop store c holding g1 and g2 (set c), store far holding neither, store half
    holding g1 only, and set mq asking for g1 and g2."""
    conn, ev = multihop_env(tmp_path)
    ix = ev.retriever.indexer
    ix.add(ix.create("far", "multihop", False), [Item("f1", "glacier ice moved", "C")], "x")
    half = ix.create("half", "multihop", False)
    ix.add(half, [Item("g1", "enzyme rate rose", "A"), Item("f1", "glacier ice moved", "C")], "x")
    db.create_eval_set(conn, "mq", "multihop")
    save_questions(
        conn,
        "mq",
        [
            Question("m1", "enzyme rate", None, ["A"], ["g1"], {}),
            Question("m2", "market price", None, ["B"], ["g2"], {}),
        ],
    )
    return conn, ev


def test_evidence_counts_only_where_the_store_holds_it(tmp_path):
    conn, ev = evidence_env(tmp_path)
    est = ev.estimate("far", "full", ["hybrid"], True, None, eval_set="mq")
    assert (est["questions"], est["evidence_named"], est["evidence_held"]) == (2, 2, 0)
    far = ev.queue("far", "full", ["hybrid"], True, None, eval_set="mq")
    ev.run(far)
    setup = get_job(conn, far)["setup"]
    assert (setup["evidence_named"], setup["evidence_held"]) == (2, 0)
    assert all(r["retrieval"] == {} for r in run_items(conn, far))
    table = ev.table(far)
    assert "recall@2" not in table["methods"]["hybrid"] and table["headline"] == "f1"
    half = ev.queue("half", "full", ["hybrid"], False, None, eval_set="mq")
    ev.run(half)
    setup = get_job(conn, half)["setup"]
    assert (setup["evidence_named"], setup["evidence_held"]) == (2, 1)
    rows = {r["question_id"]: r for r in run_items(conn, half)}
    assert rows["m2"]["retrieval"] == {} and "recall@2" in rows["m1"]["retrieval"]
    table = ev.table(half)
    assert table["methods"]["hybrid"]["recall@2"]["n"] == 1 and table["question_count"] == 2
    held = ev.estimate("half", "full", ["hybrid"], False, None, eval_set="mq")
    assert (held["evidence_named"], held["evidence_held"]) == (2, 1)


def test_retrieval_only_run_that_measures_nothing_is_refused(tmp_path):
    conn, ev = evidence_env(tmp_path)
    nothing = "nothing to measure: this store holds none of the eval set's evidence"
    with pytest.raises(ValueError, match=nothing):
        ev.queue("far", "full", ["hybrid"], False, None, eval_set="mq")
    assert list_jobs(conn, kind="eval") == []
    assert ev.estimate("far", "full", ["hybrid"], False, None, eval_set="mq")["calls"] == 0
    assert ev.queue("far", "full", ["hybrid"], True, None, eval_set="mq")
    assert ev.queue("half", "full", ["hybrid"], False, None, eval_set="mq")


def test_leaked_is_worked_out_against_the_run_store_before_the_dev_slice(env):
    conn, ev = env
    ix = ev.retriever.indexer
    a, b = ix.create("a", "mcq", False), ix.create("b", "mcq", False)
    ix.add(a, [*ITEMS, Item("long-item", LONG + " in the lab", "A", "x", {})], "x")
    ix.add(b, ITEMS, "x")
    others = [
        Question(f"o{n:03}", f"other question {n}", ["y", "n"], ["A"], [], {"skill": "s"})
        for n in range(DEV_SLICE)
    ]
    # A hand question that a slice taken before the leak check would hold.
    hand = next(
        h
        for n in range(1000)
        if (h := Question(f"h{n}", LONG, ["y", "n"], ["A"], [], {"source": "hand"}))
        in dev_slice([*others, h])
    )
    db.create_eval_set(conn, "e", "mcq")
    save_questions(conn, "e", [*others, hand])
    assert ev.leaked_ids(a, "e") == {hand.id} and ev.leaked_ids(b, "e") == set()
    on_a = ev.queue("a", "dev", ["hybrid"], False, None, eval_set="e")
    setup = ev._record_setup(on_a, a, get_job(conn, on_a)["setup"])
    assert setup["leaked_excluded"] == 1 and setup["leaked_ids"] == [hand.id]
    assert setup["question_ids"] == [q.id for q in others]
    assert ev.estimate("a", "dev", ["hybrid"], False, None, eval_set="e")["dev_questions"] == 200
    on_b = ev.queue("b", "dev", ["hybrid"], False, None, eval_set="e")
    setup = ev._record_setup(on_b, b, get_job(conn, on_b)["setup"])
    assert setup["leaked_excluded"] == 0 and hand.id in setup["question_ids"]


def test_leaked_ids_skips_the_store_scan_without_hand_or_uploaded_questions(env):
    _, ev = env

    def no_scan(*args, **kwargs):
        raise AssertionError("scanned the store")

    ev.retriever.store.scroll = no_scan
    assert ev.leaked_ids(ev.retriever.spec("c"), "c") == set()


def test_store_answer_counts_only_its_default_eval_set_runs(env):
    conn, ev = env
    mine = ev.queue("c", "dev", ["hybrid", "kw_individual"], True, None)
    ev.run(mine)
    db.create_eval_set(conn, "other", "mcq")
    save_questions(conn, "other", QUESTIONS[:2])
    theirs = ev.queue("c", "dev", ["hybrid", "kw_individual"], True, None, eval_set="other")
    ev.run(theirs)
    assert get_job(conn, theirs)["setup"]["eval_set"] == "other"
    a = ev.collection_answer("c")
    assert a is not None and a["run_id"] == mine and a["runs"] == 1


def test_leaked_ids_scan_once_until_the_store_or_the_set_changes(env):
    conn, ev = env
    ix, store = ev.retriever.indexer, ev.retriever.store
    spec = ix.spec("c")
    ix.add(spec, [Item("long-item", LONG + " in the lab", "A", "x", {})], "x")
    hand = Question("h1", LONG, ["y", "n"], ["A"], [], {"source": "hand"})
    save_questions(conn, "c", [hand])
    scans = []
    scroll = store.scroll
    store.scroll = lambda *a, **kw: scans.append(1) or scroll(*a, **kw)
    assert ev.leaked_ids(spec, "c") == {"h1"} and ev.leaked_ids(spec, "c") == {"h1"}
    assert len(scans) == 1
    ix.add(spec, [Item("more-item", "another item text", "A", "x", {})], "x")
    assert ev.leaked_ids(spec, "c") == {"h1"} and len(scans) == 2
    save_questions(conn, "c", [replace(hand, text="a text that sits in no item at all")])
    assert ev.leaked_ids(spec, "c") == set() and len(scans) == 3
    db.set_include_leaked(conn, "c", True)
    ev._leaked.clear()
    ev.queue("c", "full", ["hybrid"], False, None)  # included: nothing to drop, so no scan
    assert len(scans) == 3


def test_leaked_ids_reads_no_questions_while_the_scan_is_kept(env, monkeypatch):
    conn, ev = env
    ix = ev.retriever.indexer
    spec = ix.spec("c")
    ix.add(spec, [Item("long-item", LONG + " in the lab", "A", "x", {})], "x")
    save_questions(conn, "c", [Question("h1", LONG, ["y", "n"], ["A"], [], {"source": "hand"})])
    assert ev.leaked_ids(spec, "c") == {"h1"}
    reads = []
    get_questions = db.get_questions
    monkeypatch.setattr(db, "get_questions", lambda *a: reads.append(a) or get_questions(*a))
    assert ev.leaked_ids(spec, "c") == {"h1"} and reads == []
    ev.forget("c")
    assert ev._leaked == {}
    assert ev.leaked_ids(spec, "c") == {"h1"} and len(reads) == 1
