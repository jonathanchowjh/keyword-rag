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
    ix.add(spec, ITEMS, "x")
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
    ev.claude = fake_claude(conn, lambda p: "Answer: A" if "Choices:" in p else "Answer: 3.1 J")
    ev.retriever.claude = ev.claude
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
    claude = fake_claude(conn, lambda p: "x")
    ix = Indexer(conn, store, models, claude)
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
    return conn, Evaluator(conn, Retriever(conn, store, models, claude, ix), models, claude)


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
