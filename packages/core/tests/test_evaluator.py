import json
from dataclasses import replace

import pytest

from aiconnect.ai import Cancelled
from ragqa_core import db, search_defaults
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
from ragqa_core.types import CollectionSpec, Hit, Item, KeywordLists, Question, SearchSettings

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


def test_a_run_records_the_stores_model_and_its_query_instruction(env):
    conn, ev = env
    qwen = "Qwen/Qwen3-Embedding-0.6B-Q"  # the fake gives any model a dimension
    ix = ev.retriever.indexer
    ix.add(ix.create("q", "mcq", False, dense_model=qwen), ITEMS, "x")
    save_questions(conn, "q", QUESTIONS)
    job = ev.queue("q", "dev", ["hybrid"], False, None)
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert setup["dense_model"] == qwen and setup["query_instruction"].startswith("Instruct: ")
    plain = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(plain)
    assert get_job(conn, plain)["setup"]["query_instruction"] == ""


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


def _questions(start, count):
    return [
        Question(f"u{i}", f"enzyme question {i}", ["yes", "no"], ["A"], [], {"skill": "enzyme"})
        for i in range(start, start + count)
    ]


def test_estimate_is_never_below_a_run_whose_misses_fall_under_the_batch_threshold(env):
    conn, ev = env
    save_questions(conn, "c", _questions(0, 5))  # 10 questions
    ev.run(create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True, split="full")))
    save_questions(conn, "c", _questions(5, 41))  # 51 in all, 41 of them misses
    est = ev.estimate("c", "full", ["hybrid"], answer=True, model=None)
    sent = []
    real = ev.ai._usd

    def record(rates, system, prompts, max_tokens, batch):
        usd = real(rates, system, prompts, max_tokens, batch)
        sent.append((batch, usd))
        return usd

    ev.ai._usd = record
    ev.run(create_job(conn, "eval", "c", setup_for(["hybrid"], answer=True, split="full")))
    assert sent and all(not batch for batch, _ in sent)
    assert est["est_usd"] >= sum(usd for _, usd in sent)


def test_a_part_of_fifty_calls_is_priced_real_time_and_fifty_one_at_its_worst_case(env):
    conn, ev = env
    save_questions(conn, "c", _questions(0, 45))  # 50 questions
    est50 = ev.estimate("c", "full", ["hybrid"], answer=True, model=None)
    save_questions(conn, "c", _questions(45, 1))  # 51
    est51 = ev.estimate("c", "full", ["hybrid"], answer=True, model=None)
    assert est50["calls"] == 50 and est51["calls"] == 51
    assert est51["est_usd"] == pytest.approx(est50["est_usd"])


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


def test_queue_refuses_an_unknown_rerank_model(env):
    conn, ev = env
    with pytest.raises(ValueError, match="unknown reranker nope/x"):
        ev.queue("c", "dev", ["hybrid_rerank"], False, None, rerank_model="nope/x")
    assert list_jobs(conn, kind="eval") == []
    for model in (None, ev.models.rerank_model):  # the fakes' reranker is not on fastembed's list
        job = ev.queue("c", "dev", ["hybrid_rerank"], False, None, rerank_model=model)
        assert get_job(conn, job)["setup"]["rerank_model"] == ev.models.rerank_model


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
        "rerank_pool": 50,
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


def test_run_views_say_which_evidence_the_store_holds(tmp_path):
    conn, ev = evidence_env(tmp_path)
    half = ev.queue("half", "full", ["hybrid"], False, None, eval_set="mq")
    ev.run(half)
    store, reads = ev.retriever.store, []
    held = store.held
    store.held = lambda *a, **kw: reads.append(1) or held(*a, **kw)
    store.get = lambda *a, **kw: pytest.fail("the views read full payloads")
    items = {r["question_id"]: r for r in ev.items(half, "hybrid", wrong_only=False)}
    asked = {q["question_id"]: q for q in ev.questions(half, disagree=False)}
    for view in (items, asked):
        assert (view["m1"]["gold_ids"], view["m1"]["gold_held"]) == (["g1"], ["g1"])
        assert (view["m2"]["gold_ids"], view["m2"]["gold_held"]) == (["g2"], [])
    assert len(reads) == 2  # one ids-only read of the store per view


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


# -- Documents stores -----------------------------------------------------------------

PASSAGES = [
    Item("p1", "alpha beta gamma"),
    Item("p2", "alpha beta zeta"),
    Item("p3", "alpha beta omicron theta"),
    *(Item(f"p{n}", f"filler{n} padding{n} other{n}") for n in range(4, 13)),
]


def documents_env(tmp_path, questions: list[Question]):
    """A Documents store of 12 passages, and its eval set imported from a file: questions
    with gold ids, and judged ids and reference answers where known."""
    from ragqa_core import imports

    conn = connect(tmp_path / "d.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "Answer: x")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("d", "documents", False, "documents")
    ix.add(spec, PASSAGES, "upload:docs")
    rows = [
        {"id": q.id, "text": q.text, "gold_ids": q.gold_ids}
        | ({} if q.judged_ids is None else {"judged_ids": q.judged_ids})
        | ({"gold_answers": q.gold_answers} if q.gold_answers else {})
        for q in questions
    ]
    mapping = imports.guess_mapping(imports.columns(rows), "questions")
    built, _ = imports.build_questions(rows, mapping, "documents")
    save_questions(conn, "d", [replace(q, meta={"source": "upload:set"}) for q in built])
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    return conn, Evaluator(conn, retriever, models, ai)


def test_a_documents_run_scores_recall_mrr_and_ndcg_at_10(tmp_path):
    import math

    asked = [
        Question("d1", "alpha beta gamma", None, [], ["p2", "p3"], {}),
        Question("d2", "filler5 padding5", None, [], ["p5"], {}),
    ]
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector", "hybrid"], False, None)  # k is 5
    ev.run(job)
    rows = {(r["method"], r["question_id"]): r for r in run_items(conn, job)}
    assert all(len(r["hit_ids"]) == 10 for r in rows.values())  # gold ids keep 10
    assert all(set(r["retrieval"]) == {"ndcg@10", "recall@10", "mrr@10"} for r in rows.values())
    # By meaning d1 ranks p1 (all its words), p2 (two of three), p3 (two of four): gold 2nd, 3rd.
    vector = rows[("vector", "d1")]
    assert vector["hit_ids"][:3] == ["p1", "p2", "p3"]
    assert vector["retrieval"]["recall@10"] == 1.0
    assert vector["retrieval"]["mrr@10"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert vector["retrieval"]["ndcg@10"] == pytest.approx((1 / math.log2(3) + 1 / 2) / ideal)
    table = ev.table(job)
    assert table["headline"] == "ndcg@10"
    assert {m for m in table["methods"]["vector"] if "@" in m} == {"ndcg@10", "recall@10", "mrr@10"}
    assert table["gaps"] and {p["metric"] for p in table["pairs"]} == {"ndcg@10"}
    # Best of each asks "do keywords help?": the best keyword-list method against the best
    # other. A Documents store has no keyword lists and a Documents run doesn't lead with that
    # question (spec section 1), so with no lists there is nothing to set against.
    assert table["best_of_each"] is None
    # Best of each sets the best keyword-list method against the best other: lists let one run.
    ev.retriever.indexer.set_lists(ev.retriever.spec("d"), LISTS)
    pair = ev.queue("d", "full", ["kw_individual", "hybrid"], False, None)
    ev.run(pair)
    table = ev.table(pair)
    assert table["headline"] == "ndcg@10" and table["gaps"] and table["best_of_each"]
    assert table["answer"] is not None


def test_a_documents_run_that_measures_nothing_only_says_choose_another_store(tmp_path):
    conn, ev = documents_env(tmp_path, [Question("d1", "alpha beta", None, [], ["gone"], {})])
    with pytest.raises(ValueError) as refused:
        ev.queue("d", "full", ["hybrid"], False, None)
    assert str(refused.value) == (
        "nothing to measure: this store holds none of the eval set's evidence; choose another store"
    )
    assert db.list_jobs(conn, kind="eval") == []


def test_documents_questions_are_never_leaked(tmp_path):
    words = "alpha beta gamma alpha beta gamma alpha"  # every word is inside p1, 7 of them
    conn, ev = documents_env(tmp_path, [Question("d1", words, None, [], ["p1"], {})])
    from ragqa_core import imports

    held = Item("p0", f"{words} and more")  # the question's whole text inside a passage
    ev.retriever.indexer.add(ev.retriever.spec("d"), [held], "upload:docs")
    assert imports.leaked_ids(db.get_questions(conn, "d"), [held.text]) == {"d1"}  # it would be
    assert ev.leaked_ids(ev.retriever.spec("d"), "d") == set()
    job = ev.queue("d", "full", ["hybrid"], False, None)
    ev.run(job)
    setup = get_job(conn, job)["setup"]
    assert setup["leaked_excluded"] == 0 and setup["question_ids"] == ["d1"]


# -- Documents run labels -------------------------------------------------------------

JUDGED = [f"p{n}" for n in range(1, 11)]


def unjudged_hits(n: int) -> list[str]:
    """Ten hits, the last n of them never judged."""
    return JUDGED[: 10 - n] + [f"x{i}" for i in range(n)]


def test_unjudged_count_reads_the_top_10_only():
    from ragqa_core.evaluator import unjudged_count

    ids = [f"a{n}" for n in range(1, 13)]
    assert unjudged_count(ids, ["a1", "a11"]) == 9  # a11 is past the top 10, so a12 isn't read
    assert unjudged_count(["a1", "b", "c"], ["a1"]) == 2


def test_a_run_flags_a_method_whose_top_10_is_mostly_unjudged(tmp_path):
    asked = [Question(f"q{n}", "alpha beta", None, [], ["p1"], {}, JUDGED) for n in range(1, 6)]
    asked.append(Question("q6", "alpha beta", None, [], ["p1"], {}))  # judged_ids unknown
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector", "hybrid", "keyword", "random_example"], False, None)
    ev.run(job)
    counts = {
        "hybrid": [7, 7, 7, 7, 7],
        "vector": [2, 3, 2, 3, 3],
        "keyword": [3, 3, 3, 4, 4],
        "random_example": [9, 9, 9, 9, 9],
    }
    for method, per_q in counts.items():
        for n, unjudged in enumerate([*per_q, 10]):  # q6's judged ids are unknown: left out
            qid = f"q{n + 1}"
            row = {"hit_ids": unjudged_hits(unjudged), "retrieval": {"ndcg@10": 0.1 * n}}
            db.upsert_run_item(conn, job, method, qid, row, complete=True)
    labels = ev.table(job)["labels"]
    assert labels["tracked"] is True
    unjudged = labels["unjudged"]
    assert unjudged["hybrid"] == {"shown": 7, "questions": 5, "flag": True}
    assert unjudged["vector"] == {"shown": 3, "questions": 5, "flag": False}  # 2.6 on average
    assert unjudged["keyword"] == {"shown": 4, "questions": 5, "flag": True}  # 3.4 on average
    assert set(unjudged) == {"hybrid", "vector", "keyword"}  # the control is left out
    assert labels["unmeasured"] == 0 and labels["no_right_passage"] == 0


def test_a_run_on_a_set_without_judged_ids_says_unjudged_is_not_tracked(tmp_path):
    asked = [Question(f"q{n}", "alpha beta", None, [], ["p1"], {}) for n in range(1, 4)]
    _, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector", "hybrid"], False, None)
    ev.run(job)
    labels = ev.table(job)["labels"]
    assert labels["tracked"] is False and labels["unjudged"] == {}


def test_a_run_counts_questions_whose_right_passages_are_gone(tmp_path):
    asked = [
        Question("q1", "alpha beta zeta", None, [], ["p2", "gone1"], {}),
        Question("q2", "alpha beta gamma", None, [], ["gone2"], {}),
        Question("q3", "alpha beta omicron", None, [], [], {}),
    ]
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector"], False, None)
    ev.run(job)
    labels = ev.table(job)["labels"]
    assert labels["unmeasured"] == 1 and labels["no_right_passage"] == 1
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    assert "ndcg@10" in rows["q1"]["retrieval"]
    assert rows["q1"]["retrieval"]["recall@10"] == 1.0  # measured on p2 alone: gone1 is not held


def test_old_runs_get_labels_from_their_pinned_version(tmp_path):
    conn, ev = documents_env(tmp_path, [Question("q1", "alpha beta", None, [], ["p1"], {}, JUDGED)])
    job = ev.queue("d", "full", ["vector"], False, None)
    ev.run(job)
    stored = db.get_questions(conn, "d")[0]
    db.upsert_question(conn, "d", replace(stored, judged_ids=["p1"]), stored)
    assert db.get_questions(conn, "d")[0].judged_ids == ["p1"]  # relabelled since the run
    from ragqa_core.evaluator import unjudged_count

    hits = run_items(conn, job)[0]["hit_ids"]
    assert unjudged_count(hits, JUDGED) < unjudged_count(hits, ["p1"])
    shown = ev.table(job)["labels"]["unjudged"]["vector"]["shown"]
    assert shown == unjudged_count(hits, JUDGED)


def test_documents_run_questions_carry_judged_ids(tmp_path):
    asked = [
        Question("q1", "alpha beta", None, [], ["p1"], {}, JUDGED),
        Question("q2", "alpha beta zeta", None, [], ["p2"], {}),
    ]
    _, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector"], False, None)
    ev.run(job)
    rows = ev.questions(job, False)
    assert {r["question_id"]: r["judged_ids"] for r in rows} == {"q1": JUDGED, "q2": None}


def test_labels_are_none_for_other_types(env):
    _, ev = env
    job = ev.queue("c", "dev", ["hybrid"], False, None)
    ev.run(job)
    assert ev.table(job)["labels"] is None
    assert all("judged_ids" not in r for r in ev.questions(job, False))


def test_a_runs_unmeasured_count_is_the_one_it_was_scored_with(tmp_path):
    asked = [
        Question("q1", "alpha beta zeta", None, [], ["p2"], {}),
        Question("q2", "alpha beta gamma", None, [], ["gone2"], {}),
    ]
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector"], False, None)
    ev.run(job)
    assert ev.table(job)["labels"]["unmeasured"] == 1
    ev.retriever.store.delete_items("d", ["p2"])  # q1's right passage is gone since the run
    labels = ev.table(job)["labels"]
    assert labels["unmeasured"] == 1 and labels["no_right_passage"] == 0
    rows = {r["question_id"]: r for r in run_items(conn, job)}
    assert "ndcg@10" in rows["q1"]["retrieval"]  # it was measured, so it is not unmeasured


def test_the_unjudged_mean_reads_only_the_questions_a_method_scored(tmp_path):
    asked = [Question(f"q{n}", "alpha beta", None, [], ["p1"], {}, JUDGED) for n in range(1, 3)]
    asked.append(Question("q3", "alpha beta", None, [], ["gone"], {}, JUDGED))  # nothing held
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector"], False, None)
    ev.run(job)
    assert run_items(conn, job)[2]["retrieval"] == {}  # q3 had nothing to find, so no score
    for qid, unjudged in (("q1", 2), ("q2", 4)):
        row = {"hit_ids": unjudged_hits(unjudged), "retrieval": {"ndcg@10": 0.5}}
        db.upsert_run_item(conn, job, "vector", qid, row, complete=True)
    labelled = {"hit_ids": unjudged_hits(0)[:5], "retrieval": {}}  # 5 deep, all labelled
    db.upsert_run_item(conn, job, "vector", "q3", labelled, complete=True)
    assert ev.table(job)["labels"]["unjudged"]["vector"] == {
        "shown": 3,
        "questions": 2,
        "flag": False,
    }


def test_the_controls_and_question_only_get_no_unjudged_entry(tmp_path):
    asked = [Question("q1", "alpha beta", None, [], ["p1"], {}, JUDGED)]
    _, ev = documents_env(tmp_path, asked)
    methods = ["vector", "random_example", "fixed_example", "question_only"]
    job = ev.queue("d", "full", methods, False, None)
    ev.run(job)
    assert set(ev.table(job)["labels"]["unjudged"]) == {"vector"}


def test_mid_run_a_method_with_no_scored_row_has_no_unjudged_entry(tmp_path):
    asked = [Question(f"q{n}", "alpha beta", None, [], ["p1"], {}, JUDGED) for n in range(1, 3)]
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector", "hybrid"], False, None)  # queued, not run yet
    scored = {"hit_ids": unjudged_hits(1), "retrieval": {"ndcg@10": 0.5}}
    db.upsert_run_item(conn, job, "vector", "q1", scored, complete=True)
    db.upsert_run_item(conn, job, "hybrid", "q1", {"retrieval": {}}, complete=False)
    labels = ev.table(job)["labels"]
    assert labels["tracked"] is True
    assert labels["unjudged"] == {"vector": {"shown": 1, "questions": 1, "flag": False}}
    assert labels["unmeasured"] == 0  # read from the store until the run records its counts


# -- Documents answers ----------------------------------------------------------------


def test_documents_answers_are_never_scored_by_token_overlap():
    from ragqa_core.evaluator import score

    spec = CollectionSpec("d", "documents", False, "fake/dense", 4)
    q = Question("d1", "t", None, ["Ten days"], ["p1"], {})
    assert score(spec, q, "Ten days") == {}


def test_documents_measurable_leaves_out_the_judged_metric():
    from ragqa_core.evaluator import measurable

    out = measurable("documents", fields(questions={"gold_ids": 200}))
    assert set(out) == {"ndcg@10", "recall@10", "mrr@10"}


def ten_hits() -> list[Hit]:
    return [Hit(f"h{n}", f"passage {n}", 1.0, {}, [], None, None, {}) for n in range(10)]


def test_answer_hits_gives_documents_the_evidence_size():
    from ragqa_core.evaluator import answer_hits

    hits = ten_hits()
    assert answer_hits("documents", hits, "full", evidence_size=8) == hits[:8]
    assert answer_hits("documents", hits, "full", None) == hits
    assert answer_hits("documents", hits, "full") == hits
    assert answer_hits("multihop", hits, "full", evidence_size=8) == hits
    assert answer_hits("mcq", hits, "full", evidence_size=8) == hits[:1]


def test_retrieval_depth_reaches_the_evidence_size():
    from ragqa_core.evaluator import retrieval_depth

    gold = Question("d1", "t", None, [], ["p1"], {})
    none = Question("d2", "t", None, [], [], {})
    assert retrieval_depth("documents", gold, 5) == 10
    assert retrieval_depth("documents", gold, 5, answer_depth=8) == 10
    assert retrieval_depth("documents", gold, 5, answer_depth=20) == 20
    assert retrieval_depth("documents", none, 5, answer_depth=8) == 8
    assert retrieval_depth("documents", none, 5) == 5
    assert retrieval_depth("table", gold, 5) == 5


def test_run_prompt_gives_one_system_per_documents_method():
    from ragqa_core.evaluator import run_prompt
    from ragqa_core.prompts import CHAT_SYSTEM, answer_prompt, chat_prompt

    q = Question("d1", "How long is leave?", None, ["Ten days"], ["h1"], {})
    hits, search = ten_hits()[:3], SearchSettings()
    assert run_prompt("documents", q, hits, search) == chat_prompt(q.text, [], hits)
    system, user = run_prompt("documents", q, [], search, retrieves=True)
    assert system == CHAT_SYSTEM and user == chat_prompt(q.text, [], [])[1]
    assert run_prompt("documents", q, [], search, retrieves=False) == chat_prompt(q.text, [], [])
    assert run_prompt("multihop", q, hits, search) == answer_prompt(
        "multihop", q.text, None, hits, "full", False
    )


def test_answered_rows_on_a_documents_table_keep_ndcg_as_headline(tmp_path):
    asked = [
        Question("d1", "alpha beta gamma", None, [], ["p2", "p3"], {}),
        Question("d2", "filler5 padding5", None, [], ["p5"], {}),
    ]
    conn, ev = documents_env(tmp_path, asked)
    job = ev.queue("d", "full", ["vector", "hybrid"], False, None)
    ev.run(job)
    before = ev.table(job)
    assert not any("accuracy" in entry for entry in before["methods"].values())
    assert {p["metric"] for p in before["pairs"]} == {"ndcg@10"}
    judged = {("hybrid", "d1"): {"accuracy": 1.0}, ("vector", "d1"): {"accuracy": 0.0}}
    for r in run_items(conn, job):
        key = (r.pop("method"), r.pop("question_id"))
        r.pop("complete")
        r["answer"] = judged.get(key, {})
        db.upsert_run_item(conn, job, *key, r, complete=True)
    table = ev.table(job)
    assert table["headline"] == "ndcg@10"
    assert table["methods"]["hybrid"]["accuracy"]["n"] == 1
    assert {p["metric"] for p in table["pairs"]} == {"ndcg@10", "accuracy"}


# -- Documents runs that answer -------------------------------------------------------

D1 = Question("d1", "alpha beta gamma", None, [], ["p2", "p3"], {})
D2 = Question("d2", "filler5 padding5", None, [], ["p5"], {})


def record_calls(ev) -> list[tuple[str, list[str], str]]:
    """Every complete_many from here on, as (purpose, prompts, system); the calls go through.
    A complete is one complete_many of one prompt."""
    calls, complete_many = [], ev.ai.complete_many

    def recording(prompts, **kw):
        calls.append((kw["purpose"], list(prompts), kw.get("system", "")))
        return complete_many(prompts, **kw)

    ev.ai.complete_many = recording
    return calls


def row_of(conn, job, method, qid) -> dict:
    return next(r for r in run_items(conn, job, method) if r["question_id"] == qid)


def test_a_documents_run_answers_from_the_evidence_size_with_chats_prompt(tmp_path):
    from ragqa_core import search_defaults
    from ragqa_core.prompts import chat_prompt

    conn, ev = documents_env(tmp_path, [D1, D2])
    fake_replies(ev.ai, lambda p: "Ten days [1].")
    search_defaults.save(conn, {"evidence_size": 3})
    job = ev.queue("d", "full", ["hybrid"], True, None)
    search_defaults.save(conn, {"evidence_size": 5})  # a change after queueing is not used
    calls = record_calls(ev)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert get_job(conn, job)["setup"]["evidence_size"] == 3
    rows = {q.id: row_of(conn, job, "hybrid", q.id) for q in (D1, D2)}
    store = ev.retriever.store
    expected = [
        chat_prompt(q.text, [], store.get("d", rows[q.id]["hit_ids"][:3]))[1] for q in (D1, D2)
    ]
    assert [c[1] for c in calls] == [expected]
    for row in rows.values():
        assert len(row["hit_ids"]) == 10
        assert set(row["retrieval"]) == {"ndcg@10", "recall@10", "mrr@10"}
        assert row["pred"] == "Ten days." and row["raw"] == "Ten days [1]."
        assert row["answer"] == {} and row["complete"]


def test_queue_records_the_evidence_size_it_is_given(tmp_path):
    from ragqa_core import search_defaults

    conn, ev = documents_env(tmp_path, [D1, D2])
    search_defaults.save(conn, {"evidence_size": 6})
    job = ev.queue("d", "full", ["hybrid"], True, None, evidence_size=4)
    assert get_job(conn, job)["setup"]["evidence_size"] == 4
    free = ev.queue("d", "full", ["hybrid"], False, None, evidence_size=4)
    assert "evidence_size" not in get_job(conn, free)["setup"]


def test_a_documents_answer_run_makes_one_call_per_method_and_its_estimate_none(tmp_path):
    from ragqa_core.prompts import CHAT_SYSTEM

    conn, ev = documents_env(tmp_path, [D1, D2])
    calls = record_calls(ev)
    estimate = ev.estimate("d", "full", ["vector", "hybrid"], True, None)
    assert calls == []
    assert estimate["calls"] == 4 and estimate["roles"] == ["answer"]
    assert estimate["evidence_size"] == 8 and estimate["est_usd"] is not None
    job = ev.queue("d", "full", ["vector", "hybrid"], True, None, evidence_size=8)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [(p, len(prompts), s) for p, prompts, s in calls] == [("answer", 2, CHAT_SYSTEM)] * 2


def test_question_only_answers_with_chats_no_passage_prompt(tmp_path):
    from ragqa_core.prompts import CHAT_NO_PASSAGES, CHAT_SYSTEM, chat_prompt

    conn, ev = documents_env(tmp_path, [D1, D2])
    calls = record_calls(ev)
    job = ev.queue("d", "full", ["question_only", "hybrid"], True, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [(p, len(prompts), s) for p, prompts, s in calls] == [
        ("answer", 2, CHAT_SYSTEM + CHAT_NO_PASSAGES),
        ("answer", 2, CHAT_SYSTEM),
    ]
    assert calls[0][1] == [chat_prompt(q.text, [], [])[1] for q in (D1, D2)]


def test_a_method_that_finds_nothing_for_a_question_still_makes_one_call(tmp_path):
    from ragqa_core.prompts import CHAT_SYSTEM, chat_prompt

    conn, ev = documents_env(tmp_path, [D1, D2])
    real = ev.retriever.retrieve

    def retrieve(collection, question, method, *args, **kw):
        if method == "keyword" and question == D2.text:
            return []
        return real(collection, question, method, *args, **kw)

    ev.retriever.retrieve = retrieve
    calls = record_calls(ev)
    job = ev.queue("d", "full", ["keyword"], True, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [(p, len(prompts), s) for p, prompts, s in calls] == [("answer", 2, CHAT_SYSTEM)]
    assert calls[0][1][1] == chat_prompt(D2.text, [], [])[1]
    assert row_of(conn, job, "keyword", "d2")["hit_ids"] == []


def test_unjudged_documents_answers_keep_the_retrieval_outcome(tmp_path):
    conn, ev = documents_env(tmp_path, [D1, D2])
    job = ev.queue("d", "full", ["vector", "hybrid"], True, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    for question in ev.questions(job, False):
        for m in question["methods"].values():
            assert m["outcome"] == ("right" if m["retrieval"]["recall@10"] else "wrong")
    wrong = [r["question_id"] for r in ev.items(job, "hybrid", True)]
    assert wrong == [
        r["question_id"] for r in run_items(conn, job, "hybrid") if r["retrieval"]["recall@10"] == 0
    ]


def test_answering_outcome():
    from ragqa_core.evaluator import answering_outcome

    assert answering_outcome("documents", {"answer": True}) is False
    assert answering_outcome("documents", {"answer": True, "judge": True}) is True
    assert answering_outcome("mcq", {"answer": True}) is True
    assert answering_outcome("mcq", {}) is False


def test_a_documents_answer_run_still_refuses_a_store_without_its_evidence(tmp_path):
    from ragqa_core.evaluator import NOTHING_TO_MEASURE_DOCUMENTS

    conn, ev = documents_env(tmp_path, [Question("d1", "alpha beta", None, [], ["gone"], {})])
    with pytest.raises(ValueError) as refused:
        ev.queue("d", "full", ["hybrid"], True, None)
    assert str(refused.value) == NOTHING_TO_MEASURE_DOCUMENTS
    assert db.list_jobs(conn, kind="eval") == []


def test_a_resume_prices_only_answers_not_cached(tmp_path):
    from aiconnect.ai import BudgetExceeded
    from ragqa_core.prompts import CHAT_SYSTEM, CHAT_TOKENS, chat_prompt

    conn, ev = documents_env(tmp_path, [D1, D2])
    job = ev.queue("d", "full", ["hybrid"], True, None)

    def over_budget(prompts, **kw):
        raise BudgetExceeded("cap")

    ev.ai.complete_many = over_budget
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_budget"
    del ev.ai.complete_many
    calls = record_calls(ev)
    assert ev.estimate_resume(job)["calls"] == 2
    assert calls == []
    ids = row_of(conn, job, "hybrid", "d1")["hit_ids"][:8]
    prompt = chat_prompt(D1.text, [], ev.retriever.store.get("d", ids))[1]
    ev.ai.complete(
        prompt, role="answer", system=CHAT_SYSTEM, max_tokens=CHAT_TOKENS, purpose="answer"
    )
    calls.clear()
    estimate = ev.estimate_resume(job)
    assert estimate["calls"] == 1 and calls == []
    before = answers_sent(conn)
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert answers_sent(conn) - before == estimate["calls"]  # d2 only: d1's is cached


def answers_sent(conn) -> int:
    """The answers paid for so far: a cached one is not sent, so it is not in the ledger."""
    return len(conn.execute("select 1 from ai_ledger where purpose = 'answer'"))


def test_a_documents_resume_answers_from_the_passages_it_priced(tmp_path):
    from aiconnect.ai import BudgetExceeded
    from ragqa_core.prompts import chat_prompt

    conn, ev = documents_env(tmp_path, [D1, D2])
    job = ev.queue("d", "full", ["hybrid"], True, None)
    complete_many = ev.ai.complete_many

    def one_then_stop(prompts, **kw):  # d1's answer is sent and cached, then the cap
        complete_many(prompts[:1], **kw)
        raise BudgetExceeded("cap")

    ev.ai.complete_many = one_then_stop
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_budget"
    del ev.ai.complete_many
    stored = {q.id: row_of(conn, job, "hybrid", q.id) for q in (D1, D2)}
    # A passage that now outranks the stored hits of both questions.
    ev.retriever.indexer.add(
        ev.retriever.spec("d"), [Item("p0", "alpha beta gamma filler5 padding5")], "upload:docs"
    )
    for q in (D1, D2):
        fresh = [h.id for h in ev.retriever.retrieve("d", q.text, "hybrid", 10)]
        assert "p0" in fresh[:8] and "p0" not in stored[q.id]["hit_ids"]
    estimate = ev.estimate_resume(job)
    assert estimate["calls"] == 1
    calls, before = record_calls(ev), answers_sent(conn)
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert answers_sent(conn) - before == estimate["calls"]
    store = ev.retriever.store
    priced = [
        chat_prompt(q.text, [], store.get("d", stored[q.id]["hit_ids"][:8]))[1] for q in (D1, D2)
    ]
    assert [c[1] for c in calls] == [priced]
    for q in (D1, D2):  # the rows keep the hits and metrics they were scored with
        row = row_of(conn, job, "hybrid", q.id)
        assert row["hit_ids"] == stored[q.id]["hit_ids"] and row["complete"]
        assert row["retrieval"] == stored[q.id]["retrieval"] and row["pred"]


def test_a_store_answer_skips_an_unjudged_documents_answer_run(tmp_path):
    conn, ev = documents_env(tmp_path, [D1, D2])
    answered = ev.queue("d", "full", ["hybrid_rerank", "vector"], True, None)
    ev.run(answered)
    free = ev.queue("d", "full", ["hybrid_rerank", "vector"], False, None)
    ev.run(free)
    assert {get_job(conn, j)["status"] for j in (answered, free)} == {"done"}
    a = ev.collection_answer("d")
    assert a["run_id"] == free and a["answered"] is False and a["runs"] == 2
    assert a["kind"] == "score"


def test_a_store_answer_takes_the_newest_documents_run_judged_or_not(tmp_path):
    conn, ev = documents_env(tmp_path, [J1, J2])
    fake_replies(ev.ai, judging)
    judged = ev.queue("d", "full", ["hybrid_rerank", "vector"], True, None, judge=True)
    ev.run(judged)
    a = ev.collection_answer("d")
    assert a["run_id"] == judged and a["answered"] is False  # its score is nDCG@10
    assert a["kind"] == "score" and a["metric"] == "ndcg@10"
    free = ev.queue("d", "full", ["hybrid_rerank", "vector"], False, None)
    ev.run(free)
    assert {get_job(conn, j)["status"] for j in (judged, free)} == {"done"}
    a = ev.collection_answer("d")
    assert a["run_id"] == free and a["answered"] is False and a["runs"] == 2


def test_a_store_answer_passes_over_a_judged_run_that_measured_no_ndcg(tmp_path):
    conn, ev = documents_env(tmp_path, [J1])
    fake_replies(ev.ai, judging)
    free = ev.queue("d", "full", ["hybrid_rerank", "vector"], False, None)
    ev.run(free)
    stored = db.get_questions(conn, "d")[0]
    db.upsert_question(conn, "d", replace(stored, gold_ids=["gone"]), stored)
    judged = ev.queue("d", "full", ["hybrid_rerank", "vector"], True, None, judge=True)
    ev.run(judged)
    assert {get_job(conn, j)["status"] for j in (judged, free)} == {"done"}
    assert ev.summary(judged, None)["answer"] is None  # the store holds no right passage
    a = ev.collection_answer("d")
    assert a["run_id"] == free and a["answered"] is False
    assert a["newer_without"] == 0  # the judged run has the method; it measured nothing


# -- A Documents store's answer: the score of the search its chats use -------------------


def documents_runs(tmp_path, *methods: list[str], asked=(D1, D2)):
    """A Documents store and a done free run of each list of methods, oldest first."""
    conn, ev = documents_env(tmp_path, list(asked))
    jobs = []
    for m in methods:
        jobs.append(ev.queue("d", "full", m, False, None))
        ev.run(jobs[-1])
        assert get_job(conn, jobs[-1])["status"] == "done"
    return conn, ev, jobs


def test_documents_answer_is_chat_search_score(tmp_path):
    conn, ev, (run,) = documents_runs(tmp_path, ["hybrid_rerank", "vector"])
    a = ev.collection_answer("d")
    assert a["kind"] == "score" and a["method"] == "hybrid_rerank" and a["newer_without"] == 0
    assert a["run_id"] == run and a["answered"] is False and a["runs"] == 1
    assert a["n"] == 2 and a["metric"] == "ndcg@10" and a["before_rerank"] is False
    assert a["proxy_trusted"] is None and a["unreliable"] is False
    assert a["unfinished_newer"] is False
    assert {"at", "scale", "since", "switch", "mean", "lo", "hi"} <= set(a)
    later = ev.queue("d", "full", ["hybrid_rerank"], False, None)
    db.request_cancel(conn, later)  # queued, so cancelled at once
    assert ev.collection_answer("d")["unfinished_newer"] is True
    other = ev.queue("d", "full", ["vector"], False, None)
    db.request_cancel(conn, other)  # lacks the method: not counted
    assert ev.collection_answer("d")["unfinished_newer"] is True


def test_documents_score_uses_older_run_with_the_method(tmp_path):
    _, ev, (older, _) = documents_runs(tmp_path, ["hybrid_rerank", "vector"], ["vector"])
    a = ev.collection_answer("d")
    assert a["run_id"] == older and a["method"] == "hybrid_rerank"
    assert ev.collection_answer("d")["newer_without"] == 1


def test_documents_score_follows_settings_method(tmp_path):
    conn, ev, _ = documents_runs(tmp_path, ["hybrid_rerank", "hybrid"])
    search_defaults.save(conn, {"method": "vector"})
    assert ev.collection_answer("d") is None


def test_documents_run_keeps_frozen_method(tmp_path):
    conn, ev, (run_id,) = documents_runs(tmp_path, ["hybrid_rerank", "hybrid", "vector"])
    frozen = get_job(conn, run_id)["setup"]["chat_search"]
    assert frozen == {
        "method": "hybrid",
        "rerank": True,
        "run_method": "hybrid_rerank",
        "before_rerank": False,
    }
    search_defaults.save(conn, {"method": "vector"})
    assert ev.summary(run_id, None)["answer"]["method"] == "hybrid_rerank"
    table = ev.table(run_id)
    assert table["reference"] == "hybrid_rerank" and table["answer"]["kind"] == "score"
    assert {g["method"] for g in table["gaps"]} == {"hybrid", "vector"}
    assert ev.summary(run_id, None, "vector")["answer"]["method"] == "vector"


def test_a_documents_run_without_a_frozen_choice_falls_back(tmp_path):
    conn, ev, (run_id,) = documents_runs(tmp_path, ["hybrid", "vector"])
    setup = get_job(conn, run_id)["setup"]
    del setup["chat_search"]  # a run from before the choice was recorded
    db.set_job_setup(conn, run_id, setup)
    assert ev.summary(run_id, None)["answer"]["method"] == "hybrid"  # no hybrid_rerank ran
    search_defaults.save(conn, {"method": "vector"})
    assert ev.summary(run_id, None)["answer"]["method"] == "vector"  # the current choice


def test_a_documents_run_without_the_chat_search_or_hybrid_has_no_score(tmp_path):
    _, ev, (job,) = documents_runs(tmp_path, ["keyword"])
    t = ev.table(job)
    # The reference is the search the chats use, which this run lacks: no gaps.
    assert t["answer"] is None and t["reference"] == "hybrid_rerank" and t["gaps"] == []
    assert ev.summary(job, None)["answer"] is None


def test_a_documents_run_falls_back_to_hybrid_only_for_hybrid_chats(tmp_path):
    conn, ev = documents_env(tmp_path, [D1, D2])
    search_defaults.save(conn, {"method": "vector"})
    job = ev.queue("d", "full", ["hybrid"], False, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    # The chats use Vector search, which this run lacks: Hybrid is no stand-in for it.
    t = ev.table(job)
    assert t["answer"] is None and t["reference"] == "vector" and t["gaps"] == []
    assert ev.summary(job, None)["answer"] is None


def overwrite_rows(conn, job, rows: dict[str, tuple[int, float]], n: int = 5) -> None:
    """Each method's rows for q1..qn: `unjudged` never-judged hits in its top 10, one score."""
    for method, (unjudged, value) in rows.items():
        for i in range(1, n + 1):
            row = {"hit_ids": unjudged_hits(unjudged), "retrieval": {"ndcg@10": value}}
            db.upsert_run_item(conn, job, method, f"q{i}", row, complete=True)


def test_documents_score_withheld_when_unjudged(tmp_path):
    asked = [Question(f"q{n}", "alpha beta", None, [], ["p1"], {}, JUDGED) for n in range(1, 6)]
    conn, ev, (job,) = documents_runs(tmp_path, ["hybrid_rerank", "vector"], asked=asked)
    overwrite_rows(conn, job, {"hybrid_rerank": (7, 0.1), "vector": (0, 0.9)})
    assert ev.summary(job, None)["answer"]["switch"]["method"] == "vector"  # the run says it
    a = ev.collection_answer("d")
    assert a["unreliable"] is True and a["switch"] is None
    # The switch's own score unreliable: no switch, though the score itself stands.
    overwrite_rows(conn, job, {"hybrid_rerank": (0, 0.1), "vector": (7, 0.9)})
    a = ev.collection_answer("d")
    assert a["unreliable"] is False and a["switch"] is None and a["mean"] == pytest.approx(10)


def test_mcq_answer_keeps_the_pair(env):
    _, ev = env
    ev.run(ev.queue("c", "dev", ["hybrid", "kw_individual"], False, None))
    assert ev.collection_answer("c")["kind"] == "pair"


# -- Judge answers ----------------------------------------------------------------------

REF = ["Ten days"]
J1 = replace(D1, gold_answers=REF)
J2 = replace(D2, gold_answers=REF)
J3 = Question("d3", "alpha beta omicron theta", None, [], ["p3"], {})
J4 = Question("d4", "filler7 padding7", None, [], ["p7"], {})
TABLE_KEYS = {
    "setup",
    "status",
    "progress",
    "error",
    "methods",
    "pairs",
    "proxy",
    "headline",
    "proxy_trusted",
    "scale",
    "reference",
    "gaps",
    "answer",
    "best_of_each",
    "labels",
    "usd_total",
    "question_count",
    "since",
}


def judging(p: str) -> str:
    return "correct" if "Answer to grade" in p else "Ten days [1]."


def judged_run(tmp_path, reply=judging, methods=("vector", "hybrid"), asked=(J1, J2, J3, J4)):
    conn, ev = documents_env(tmp_path, list(asked))
    fake_replies(ev.ai, reply)
    job = ev.queue("d", "full", list(methods), True, None, judge=True)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    return conn, ev, job


def test_a_judged_run_makes_one_answer_and_one_judge_call_per_method(tmp_path):
    from ragqa_core.prompts import JUDGE_SYSTEM, judge_prompt

    conn, ev = documents_env(tmp_path, [J1, J2, J3, J4])
    fake_replies(ev.ai, judging)
    calls = record_calls(ev)
    estimate = ev.estimate("d", "full", ["vector", "hybrid"], True, None, judge=True)
    plain = ev.estimate("d", "full", ["vector", "hybrid"], True, None)
    assert calls == []
    assert estimate["calls"] == 12 and estimate["references"] == 2
    assert plain["calls"] == 8 and plain["references"] == 2
    assert estimate["est_usd"] > plain["est_usd"]
    job = ev.queue("d", "full", ["vector", "hybrid"], True, None, judge=True)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [(p, len(prompts)) for p, prompts, _ in calls] == [
        ("answer", 4),
        ("judge", 2),
        ("answer", 4),
        ("judge", 2),
    ]
    graded = [judge_prompt(q.text, REF, "Ten days.")[1] for q in (J1, J2)]
    assert [(prompts, s) for p, prompts, s in calls if p == "judge"] == [(graded, JUDGE_SYSTEM)] * 2
    setup = get_job(conn, job)["setup"]
    assert setup["judge"] is True and setup["answer_judge_model"] == setup["model"]
    assert setup["model"] is not None


def test_judge_skips_questions_without_a_reference_and_counts_them(tmp_path):
    conn, ev, job = judged_run(tmp_path)
    for method in ("vector", "hybrid"):
        for qid in ("d1", "d2"):
            row = row_of(conn, job, method, qid)
            assert row["judge"] == "correct" and row["answer"] == {"accuracy": 1.0}
            assert row["complete"] and row["pred"] == "Ten days." and row["judge_raw"] == "correct"
        for qid in ("d3", "d4"):
            row = row_of(conn, job, method, qid)
            assert row["judge"] == "no_reference" and row["answer"] == {}
            assert "judge_raw" not in row
            assert row["complete"] and row["pred"] == "Ten days."
    outcomes = {
        q["question_id"]: {m: r.get("outcome") for m, r in q["methods"].items()}
        for q in ev.questions(job, False)
    }
    assert outcomes["d3"] == outcomes["d4"] == {"vector": None, "hybrid": None}
    assert outcomes["d1"] == outcomes["d2"] == {"vector": "right", "hybrid": "right"}
    assert not {"d3", "d4"} & {r["question_id"] for r in ev.items(job, "hybrid", True)}
    assert ev.table(job)["answers"] == {
        "judged": True,
        "no_reference": 2,
        "methods": {
            m: {"answered": 4, "judged": 2, "correct": 2, "unreadable": 0}
            for m in ("vector", "hybrid")
        },
    }


def test_an_unreadable_verdict_is_not_judged_and_is_counted(tmp_path):
    def reply(p: str) -> str:
        if "Answer to grade" in p:
            return "maybe" if D2.text in p else "correct"
        return "Ten days [1]."

    conn, ev, job = judged_run(tmp_path, reply, ("hybrid",), (J1, J2))
    row = row_of(conn, job, "hybrid", "d2")
    assert row["judge"] == "unreadable" and row["answer"] == {} and row["complete"]
    assert row["judge_raw"] == "maybe"  # why it was not judged, for the run page
    table = ev.table(job)
    assert table["answers"]["methods"]["hybrid"] == {
        "answered": 2,
        "judged": 1,
        "correct": 1,
        "unreadable": 1,
    }
    assert table["methods"]["hybrid"]["accuracy"]["n"] == 1
    (d2,) = [q for q in ev.questions(job, False) if q["question_id"] == "d2"]
    assert d2["methods"]["hybrid"].get("outcome") is None


def test_an_incorrect_verdict_scores_zero_and_marks_the_question_wrong(tmp_path):
    def reply(p: str) -> str:
        return "Incorrect." if "Answer to grade" in p else "Ten days [1]."

    conn, ev, job = judged_run(tmp_path, reply, ("hybrid",), (J1, J2))
    for qid in ("d1", "d2"):
        row = row_of(conn, job, "hybrid", qid)
        assert row["judge"] == "incorrect" and row["answer"] == {"accuracy": 0.0}
        assert row["judge_raw"] == "Incorrect."
    assert {q["methods"]["hybrid"]["outcome"] for q in ev.questions(job, False)} == {"wrong"}
    assert {r["question_id"] for r in ev.items(job, "hybrid", True)} == {"d1", "d2"}
    assert ev.table(job)["answers"]["methods"]["hybrid"]["correct"] == 0


def refused(ev, message: str, *args, **kw) -> None:
    """Both the estimate and queue refuse these arguments with `message`."""
    for call in (ev.estimate, ev.queue):
        with pytest.raises(ValueError) as e:
            call(*args, **kw)
        assert str(e.value) == message, call


def test_judge_answers_refuses_what_it_cannot_judge(env, tmp_path):
    from ragqa_core.evaluator import JUDGE_NEEDS_ANSWERS, JUDGE_NO_REFERENCE, JUDGE_NOT_DOCUMENTS

    mcq_conn, mcq = env
    for answer in (True, False):  # not a Documents store is said first
        refused(mcq, JUDGE_NOT_DOCUMENTS, "c", "dev", ["hybrid"], answer, None, judge=True)
    assert db.list_jobs(mcq_conn, kind="eval") == []
    (tmp_path / "a").mkdir()
    conn, ev = documents_env(tmp_path / "a", [J1, D2])
    refused(ev, JUDGE_NEEDS_ANSWERS, "d", "full", ["hybrid"], False, None, judge=True)
    assert db.list_jobs(conn, kind="eval") == []
    (tmp_path / "b").mkdir()
    conn, ev = documents_env(tmp_path / "b", [D1, D2])
    refused(ev, JUDGE_NO_REFERENCE, "d", "full", ["hybrid"], True, None, judge=True)
    assert db.list_jobs(conn, kind="eval") == []
    many = [Question(f"q{n:03}", f"alpha beta {n}", None, [], ["p1"], {}) for n in range(201)]
    dev = {q.id for q in dev_slice(many)}
    outside = next(q for q in many if q.id not in dev)
    asked = [replace(q, gold_answers=REF) if q is outside else q for q in many]
    (tmp_path / "c").mkdir()
    conn, ev = documents_env(tmp_path / "c", asked)
    refused(ev, JUDGE_NO_REFERENCE, "d", "dev", ["hybrid"], True, None, judge=True)
    assert db.list_jobs(conn, kind="eval") == []
    assert ev.estimate("d", "full", ["hybrid"], True, None, judge=True)["references"] == 1
    job = ev.queue("d", "full", ["hybrid"], True, None, judge=True)
    assert get_job(conn, job)["setup"]["judge"] is True


def test_a_judged_run_needs_an_answer_model(tmp_path):
    from aiconnect.ai import NoConnection

    conn, ev = documents_env(tmp_path, [J1, J2])
    ev.ai.set_role("default", None)
    with pytest.raises(NoConnection):
        ev.queue("d", "full", ["hybrid"], True, None, judge=True)
    assert db.list_jobs(conn, kind="eval") == []


def test_a_judged_run_may_measure_only_its_answers(tmp_path):
    from ragqa_core.evaluator import (
        JUDGE_NO_REFERENCE,
        NOTHING_TO_MEASURE_DOCUMENTS,
        NOTHING_TO_MEASURE_JUDGE,
    )

    gone = Question("d1", "alpha beta gamma", None, REF, ["gone"], {})
    (tmp_path / "a").mkdir()
    conn, ev, job = judged_run(tmp_path / "a", methods=("hybrid",), asked=(gone,))
    table = ev.table(job)
    assert table["headline"] == "ndcg@10" and "ndcg@10" not in table["methods"]["hybrid"]
    assert table["labels"]["unmeasured"] == 1
    assert table["answers"]["methods"]["hybrid"]["judged"] == 1
    for answer in (True, False):
        with pytest.raises(ValueError) as e:
            ev.queue("d", "full", ["hybrid"], answer, None)
        assert str(e.value) == NOTHING_TO_MEASURE_JUDGE
    assert len(db.list_jobs(conn, kind="eval")) == 1
    (tmp_path / "b").mkdir()
    conn, ev = documents_env(tmp_path / "b", [replace(gone, gold_answers=[])])
    for answer in (True, False):
        with pytest.raises(ValueError) as e:
            ev.queue("d", "full", ["hybrid"], answer, None)
        assert str(e.value) == NOTHING_TO_MEASURE_DOCUMENTS
    with pytest.raises(ValueError) as e:
        ev.queue("d", "full", ["hybrid"], True, None, judge=True)
    assert str(e.value) == JUDGE_NO_REFERENCE
    assert db.list_jobs(conn, kind="eval") == []


def test_a_run_stopped_while_judging_resumes_without_paying_for_its_answers_again(tmp_path):
    from aiconnect.ai import BudgetExceeded

    conn, ev = documents_env(tmp_path, [J1, J2])
    fake_replies(ev.ai, judging)
    job = ev.queue("d", "full", ["hybrid"], True, None, judge=True)
    complete_many, judged = ev.ai.complete_many, []

    def stop_at_judging(prompts, **kw):
        if kw["purpose"] == "judge" and not judged:
            judged.append(1)
            raise BudgetExceeded("cap")
        return complete_many(prompts, **kw)

    ev.ai.complete_many = stop_at_judging
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_budget"
    for qid in ("d1", "d2"):
        row = row_of(conn, job, "hybrid", qid)
        assert row["pred"] == "Ten days." and not row["complete"] and "judge" not in row
    del ev.ai.complete_many
    calls = record_calls(ev)
    estimate = ev.estimate_resume(job)
    assert calls == [] and estimate["calls"] == 2  # the verdicts only
    reached: list[str] = []

    def counting(p: str) -> str:
        reached.append(p)
        return judging(p)

    fake_replies(ev.ai, counting)
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [p for p in reached if "Answer to grade" not in p] == []
    assert len(reached) == 2
    for qid in ("d1", "d2"):
        row = row_of(conn, job, "hybrid", qid)
        assert row["judge"] == "correct" and row["complete"]


def typed_env(tmp_path, qa_type):
    """A store t of `qa_type` holding ITEMS, and three questions with gold answers."""
    conn = connect(tmp_path / "t.db")
    store, models = Store(None, None), fake_models()
    ai = fake_ai(conn, lambda p: "Answer: A")
    ix = Indexer(conn, store, models, ai)
    spec = ix.create("t", qa_type, False)
    ix.add(spec, ITEMS, "x")
    choices = ["yes", "no"] if qa_type == "mcq" else None
    asked = [Question(f"q-{t}", f"a {t} question", choices, ["A"], [], {}) for t in TOPICS[:3]]
    save_questions(conn, "t", asked)
    retriever = Retriever(conn, store, models, ai, ix, Builder(conn, store, models, ai, ix))
    return conn, Evaluator(conn, retriever, models, ai)


@pytest.mark.parametrize("qa_type", ["mcq", "science", "multihop", "table"])
def test_answered_runs_of_other_types_are_unchanged(tmp_path, qa_type):
    from ragqa_core.evaluator import answer_hits, run_prompt

    conn, ev = typed_env(tmp_path, qa_type)
    calls = record_calls(ev)
    job = ev.queue("t", "full", ["vector", "hybrid"], True, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert [(p, len(prompts)) for p, prompts, _ in calls] == [("answer", 3)] * 2
    setup = get_job(conn, job)["setup"]
    search = ev._search(ev.retriever.spec("t"), setup)  # the store's own: science asks programs
    by_id = {q.id: q for q in db.get_questions(conn, "t")}
    for (_, prompts, _), method in zip(calls, ("vector", "hybrid"), strict=True):
        expected = []
        for qid in setup["question_ids"]:
            q, row = by_id[qid], row_of(conn, job, method, qid)
            hits = ev.retriever.store.get("t", row["hit_ids"])
            shown = answer_hits(qa_type, hits, search.example)
            expected.append(run_prompt(qa_type, q, shown, search, True)[1])
        assert prompts == expected, method
    assert not {"judge", "answer_judge_model", "evidence_size"} & set(setup)
    table = ev.table(job)
    assert table["answers"] is None
    assert table["headline"] == ("accuracy" if qa_type in ("mcq", "science") else "f1")
    keys = {"accuracy"} if qa_type in ("mcq", "science") else {"em", "f1"}
    assert all(set(r["answer"]) == keys for r in run_items(conn, job))
    assert set(table) == TABLE_KEYS | {"answers"}


def test_an_old_documents_run_reads_as_before(tmp_path):
    conn, ev = documents_env(tmp_path, [D1, D2])
    job = ev.queue("d", "full", ["vector", "hybrid"], False, None)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    table = ev.table(job)
    assert table["answers"] is None
    assert not {"judge", "answer_judge_model", "evidence_size"} & set(table["setup"])
    assert table["headline"] == "ndcg@10"
    for entry in table["methods"].values():
        assert set(entry) - {"usd", "n"} <= {"ndcg@10", "recall@10", "mrr@10"}
    assert {p["metric"] for p in table["pairs"]} == {"ndcg@10"}
    assert set(table) == TABLE_KEYS | {"answers"}


@pytest.mark.parametrize("unanswered", [False, True])
def test_a_judge_only_resume_keeps_the_answers_it_paid_for(tmp_path, unanswered):
    """Stopped between answering and judging: a resume judges the stored answers and sends
    no answer again, keeping each answer's cost. With `unanswered`, d2's row holds only its
    hits (as a stop part-way through storing the answers leaves it): only d2 is answered."""
    from aiconnect.ai import BudgetExceeded

    conn, ev = documents_env(tmp_path, [J1, J2])
    fake_replies(ev.ai, judging)
    job = ev.queue("d", "full", ["hybrid"], True, None, judge=True)
    complete_many = ev.ai.complete_many

    def stop_at_judging(prompts, **kw):
        if kw["purpose"] == "judge":
            raise BudgetExceeded("cap")
        return complete_many(prompts, **kw)

    ev.ai.complete_many = stop_at_judging
    ev.run(job)
    assert get_job(conn, job)["status"] == "stopped_budget"
    del ev.ai.complete_many
    stored = {q.id: row_of(conn, job, "hybrid", q.id) for q in (J1, J2)}
    assert all(r["usd"] > 0 and not r["complete"] for r in stored.values())
    if unanswered:
        bare = {k: v for k, v in stored["d2"].items() if k not in ("pred", "raw", "answer")}
        for k in ("method", "question_id", "complete"):
            bare.pop(k)
        db.upsert_run_item(conn, job, "hybrid", "d2", {**bare, "usd": 0.0}, complete=False)
    sent, replies = [], []

    def recording(prompts, **kw):
        sent.append((kw["purpose"], len(prompts)))
        out = complete_many(prompts, **kw)
        replies.append(out)
        return out

    ev.ai.complete_many = recording
    estimate = ev.estimate_resume(job)
    assert sent == []
    before = len(conn.execute("select 1 from ai_ledger"))
    requeue(conn, job)
    ev.run(job)
    assert get_job(conn, job)["status"] == "done"
    assert sent == ([("answer", 1)] if unanswered else []) + [("judge", 2)]
    assert len(conn.execute("select 1 from ai_ledger")) - before == estimate["calls"] == 2
    verdicts = dict(zip(("d1", "d2"), replies[-1], strict=True))
    for q in (J1, J2):
        row = row_of(conn, job, "hybrid", q.id)
        assert row["complete"] and row["judge"] == "correct"
        assert row["pred"] == stored[q.id]["pred"] and row["raw"] == stored[q.id]["raw"]
        paid = 0.0 if unanswered and q.id == "d2" else stored[q.id]["usd"]
        assert row["usd"] == pytest.approx(paid + verdicts[q.id].usd)
        assert verdicts[q.id].usd > 0


def test_every_recorded_setup_key_has_a_label(env):
    from ragqa_core.catalog import SETUP_LABELS

    conn, ev = env
    job = ev.queue("c", "dev", ["hybrid", "hybrid_rerank"], False, None)
    ev.run(job)
    recorded = set(get_job(conn, job)["setup"])
    # keys only an answered, judged or Documents run adds
    recorded |= {"evidence_size", "judge", "answer_judge_model", "chat_search", "rerank_model"}
    assert recorded <= set(SETUP_LABELS), recorded - set(SETUP_LABELS)


def record_answers(ev) -> tuple[list[tuple[str, list[str], int]], list[tuple[str, str, list]]]:
    """Every answer call from here on as (system, prompts, max_tokens), and every retrieval
    as (method, question text, hits); both go through."""
    calls, found = [], []
    complete_many, retrieve = ev.ai.complete_many, ev.retriever.retrieve

    def recording(prompts, **kw):
        if kw["purpose"] == "answer":
            calls.append((kw.get("system", ""), list(prompts), kw["max_tokens"]))
        return complete_many(prompts, **kw)

    def retrieving(collection, question, method, *args, **kw):
        hits = retrieve(collection, question, method, *args, **kw)
        found.append((method, question, hits))
        return hits

    ev.ai.complete_many = recording
    ev.retriever.retrieve = retrieving
    return calls, found


def test_answer_parts_match_the_run(env, tmp_path):
    """A run's answer call for each question is the run prompt over the hits it is shown,
    with the type's reply size, byte for byte, and answer_parts gives that same call: a
    Documents store's, retrieving or not, and an mcq one's."""
    from ragqa_core.evaluator import answer_hits, answer_parts, answer_tokens, run_prompt
    from ragqa_core.prompts import CHAT_TOKENS

    doc_conn, doc_ev = documents_env(tmp_path, [D1, D2])
    search_defaults.save(doc_conn, {"evidence_size": 3})
    doc_job = doc_ev.queue("d", "full", ["hybrid", "question_only"], True, None)
    mcq_conn, mcq_ev = env
    mcq_job = create_job(mcq_conn, "eval", "c", setup_for(["hybrid"], answer=True))
    cases = [
        (doc_ev, doc_job, "d", ["hybrid", "question_only"], [D1, D2], 3),
        (mcq_ev, mcq_job, "c", ["hybrid"], QUESTIONS, None),
    ]
    for ev, job, name, methods, questions, size in cases:
        calls, found = record_answers(ev)
        ev.run(job)
        assert get_job(ev.conn, job)["status"] == "done"
        spec = ev.retriever.spec(name)
        search = spec.search or SearchSettings()
        by_text = {q.text: q for q in questions}
        assert len(calls) == len(methods)
        for method, (system, prompts, max_tokens) in zip(methods, calls, strict=True):
            asked = [(by_text[text], hits) for m, text, hits in found if m == method]
            assert len(asked) == len(prompts) > 0
            for (q, hits), user in zip(asked, prompts, strict=True):
                retrieves = method != "question_only"
                shown = answer_hits(spec.qa_type, hits, search.example, size)
                expected = run_prompt(spec.qa_type, q, shown, search, retrieves)
                assert spec.qa_type == "documents" or q.choices  # a choice is never a program
                tokens = CHAT_TOKENS if spec.qa_type == "documents" else answer_tokens("mcq", False)
                assert (system, user, max_tokens) == (*expected, tokens)
                parts = answer_parts(spec, q, hits, search, retrieves=retrieves, evidence_size=size)
                assert parts == (*expected, tokens, False)
