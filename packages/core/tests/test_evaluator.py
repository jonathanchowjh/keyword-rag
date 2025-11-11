import pytest

from ragqa_core import db
from ragqa_core.db import (
    connect,
    create_job,
    get_job,
    list_jobs,
    requeue,
    run_items,
    save_questions,
)
from ragqa_core.evaluator import Evaluator
from ragqa_core.fakes import fake_claude, fake_models
from ragqa_core.indexer import Indexer
from ragqa_core.llm import Cancelled
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
    claude = fake_claude(conn, lambda p: "Answer: A")
    ix = Indexer(conn, store, models, claude)
    spec = ix.create("c", "mcq", False)
    ix.add(spec, ITEMS)
    ix.set_lists(spec, LISTS)
    save_questions(conn, "c", QUESTIONS)
    retriever = Retriever(conn, store, models, claude, ix)
    return conn, Evaluator(conn, retriever, models, claude)


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
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
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
    modes = {r["batch"] for r in conn.execute("select batch from ledger where purpose='answer'")}
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
    ev.claude = fake_claude(conn, lambda p: "thinking\nAnswer: The answer is A")
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
    job = ev.queue("c", "dev", ["hybrid"], True, None, k=3, rrf_k=10)
    assert get_job(conn, job)["setup"] == {
        "collection": "c",
        "split": "dev",
        "methods": ["hybrid"],
        "answer": True,
        "model": ev.claude.default_model,
        "k": 3,
        "rrf_k": 10,
    }
    assert get_job(conn, job)["status"] == "queued"


def test_run_skips_recorded_ids_whose_question_is_gone(env):
    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    setup = get_job(conn, job)["setup"]
    db.set_job_setup(conn, job, {**setup, "question_ids": ["q-gone", "q-market", "q-enzyme"]})
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
    complete_many, judge = ev.claude.complete_many, ev.models.judge

    def counting_complete_many(prompts, **kw):
        sent.append(len(prompts))
        return complete_many(prompts, **kw)

    def counting_judge(texts):
        judged.append(len(texts))
        return judge(texts)

    ev.claude.complete_many = counting_complete_many
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
    assert ev.estimate_resume(job) == {"questions": 0, "calls": 0, "est_usd": 0.0}
    rows = run_items(conn, job)
    ledger = conn.execute("select count(*) from ledger")[0][0]
    db.update_job(conn, job, status="interrupted")
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert run_items(conn, job) == rows
    assert conn.execute("select count(*) from ledger")[0][0] == ledger


def test_resume_estimate_of_a_run_that_never_started_is_the_full_estimate(env):
    _, ev = env
    job = ev.queue("c", "dev", ["hybrid", "kw_individual"], True, None)
    full = ev.estimate("c", "dev", ["hybrid", "kw_individual"], answer=True, model=None)
    assert ev.estimate_resume(job) == full


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
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    db.update_job(conn, job, status="interrupted")
    assert ev.resume_refusal(job) is None
    old = ev.retriever.spec("c").keyword_version
    new = ev.retriever.indexer.set_lists(ev.retriever.spec("c"), OTHER_LISTS)
    assert ev.resume_refusal(job) == (
        f"keyword lists changed since this run ({old} → {new}); start a new run"
    )


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


def test_question_only_answers_are_scored(env):
    conn, ev = env
    ev.claude = fake_claude(conn, lambda p: "Answer: B")
    job = ev.queue("c", "dev", ["question_only"], True, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert all(r["retrieval"] == {} and r["complete"] for r in run_items(conn, job))
    assert len(ev.items(job, "question_only", wrong_only=True)) == 5


def test_queue_refuses_an_empty_split(env):
    conn, ev = env
    ev.retriever.indexer.create("e", "mcq", False)
    with pytest.raises(ValueError, match="collection 'e' has no questions"):
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
    job = ev.queue("c", "dev", ["hybrid", "vector"], True, None)

    def stop_after_first_method():
        hybrid = run_items(conn, job, "hybrid")
        if hybrid and all(r["complete"] for r in hybrid):
            raise Cancelled()

    ev.run(job, check=stop_after_first_method)
    old = ev.retriever.spec("c").keyword_version
    new = ev.retriever.indexer.set_lists(ev.retriever.spec("c"), OTHER_LISTS)
    rows = run_items(conn, job)
    ledger = conn.execute("select count(*) from ledger")[0][0]
    requeue(conn, job)  # as if the re-key was queued after the resume was accepted
    ev.run(job)
    after = get_job(conn, job)
    assert after["status"] == "failed"
    assert after["error"] == (
        f"keyword lists changed since this run ({old} → {new}); start a new run"
    )
    assert run_items(conn, job) == rows
    assert conn.execute("select count(*) from ledger")[0][0] == ledger


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
        **setup_for(["hybrid"]),
        "dense_model": "fake/dense",
        "judge_model": "fake/judge",
        "keyword_version": ev.retriever.spec("c").keyword_version,
        "seed": 7,
        "dataset_filter": "an older filter",
    }
    job = create_job(conn, "eval", "c", setup)
    for qid in ("q-market", "q-enzyme"):
        db.upsert_run_item(conn, job, "hybrid", qid, {"retrieval": {}}, complete=False)
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


def test_collection_answer_is_none_when_the_run_lacks_the_pair(env):
    _, ev = env
    job = ev.queue("c", "dev", ["hybrid", "vector"], True, None)
    ev.run(job)
    assert ev.collection_answer("c") is None


def finished_run(conn, scores):
    """A done answered run with given {method: (solution_sim, accuracy)} for one question."""
    job = create_job(conn, "eval", "c", {**setup_for(list(scores), answer=True)})
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
    ix.add(spec, [Item(f"{t}-item", f"{t} things", "A", None, {"skill": t}) for t in TOPICS])
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
