"""The run form and the run pages, Documents runs included."""

import json
import re

import httpx
import pytest
from page_helpers import (
    ai_state,
    documents_run,
    documents_store,
    interrupt,
    jobs_done,
    labelling_set,
    mcq_set,
    put_questions,
    replies,
    tag_jobs,
    worker_stopped,
)
from playwright.sync_api import expect

from ragqa_core import db
from ragqa_core.fakes import fake_replies
from ragqa_core.types import Question

pytestmark = pytest.mark.pages


def test_new_run_estimates_then_starts(page, seeded):
    seeded.build("runs", run=True)
    page.goto(seeded.url + "/c/runs")
    expect(page.locator("#runs tbody tr")).to_have_count(1)
    page.get_by_role("button", name="New run").click()
    expect(page.get_by_label("kw_individual")).to_be_checked()
    expect(page.get_by_label("random_example")).to_be_checked()
    page.get_by_role("button", name="Estimate").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    page.get_by_role("button", name="Start").click()
    expect(page).to_have_url(re.compile(r"/c/runs/runs/[^/]+$"))


def test_a_refused_start_stays_on_the_run_form(page, seeded):
    seeded.build("nostart", run=True)
    refusal = "the price changed since you saw it: check it again"

    def refuse(route):
        if route.request.method == "POST" and route.request.post_data_json.get("confirm"):
            route.fulfill(status=409, json={"detail": refusal})
        else:
            route.fallback()

    page.route("**/eval", refuse)
    page.expected = ["409"]
    page.goto(seeded.url + "/c/nostart")
    page.get_by_role("button", name="New run").click()
    page.get_by_role("button", name="Estimate").click()
    page.get_by_role("button", name="Start").click()
    expect(page.locator("#rf-msg")).to_have_text(refusal)
    expect(page).to_have_url(seeded.url + "/c/nostart")
    expect(page.locator("#run-form")).to_be_visible()


def test_run_page_answer_and_gaps(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.get_by_text("Do keywords help?")).to_be_visible()
    expect(page.get_by_text("chance level")).to_be_visible()
    expect(page.locator("[data-gap-row]").first).to_be_visible()
    assert page.locator("[data-gap-row]").count() == 2  # kw_individual, random_example
    expect(page.locator("[data-ref-row]")).to_have_count(1)


def test_run_without_the_pair_says_so(page, seeded):
    page.goto(f"{seeded.url}/c/vo/runs/{seeded.vector_only_run_id}")
    expect(page.get_by_text("does not include Keywords per list and Hybrid")).to_be_visible()


def test_run_page_shows_na_without_retrieval(page, seeded):
    seeded.build("na")
    run_id = seeded.run("na", ["question_only", "hybrid"])
    page.goto(f"{seeded.url}/c/na/runs/{run_id}")
    page.get_by_text("Each method's own score").click()
    expect(page.locator("#scores td", has_text="n/a").first).to_be_visible()


def test_run_page_questions_open_with_gold_and_answer(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details").first).to_be_visible()
    page.locator("#questions details > summary").first.click()
    expect(page.locator("#questions details[open]").first).to_contain_text("Gold")
    expect(page.locator("#questions details[open]").first).to_contain_text("Parsed answer")
    # kw_individual is the first method; the matched words are rendered per hit
    expect(page.locator("#questions details[open] [data-matched]").first).to_be_visible()


def test_run_page_mutes_evidence_the_store_does_not_hold(page, seeded):
    url = f"{seeded.url}/eval/{seeded.run_id}/questions?disagree=false"
    rows = httpx.get(url).json()
    rows[0] = {**rows[0], "gold_answers": [], "gold_ids": ["enzyme-item", "gone"]}
    rows[0]["gold_held"] = ["enzyme-item"]
    page.route("**/questions?disagree=false", lambda route: route.fulfill(json=rows))
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Questions", exact=True).click()
    opened = page.locator("#questions details", has_text=rows[0]["question"]).first
    opened.locator("summary").click()
    expect(opened).to_contain_text("Gold: enzyme-item · gone (not in this store)")
    expect(opened.locator("[data-unheld].muted")).to_have_text("gone (not in this store)")


def test_run_head_says_what_changed_since_the_run(page, seeded):
    run_id = seeded.build("since", run=True)
    page.goto(f"{seeded.url}/c/since/runs/{run_id}")
    expect(page.locator("#head")).to_contain_text("Run " + run_id[:8])
    expect(page.get_by_text("changed after this run")).to_have_count(0)
    lists = {"topic": ["x", "y"], "method": ["experiment"], "common": ["rate"]}
    r = httpx.put(seeded.url + "/collections/since/keyword-lists", json=lists)
    seeded.wait(r.json()["job_id"])
    page.goto(f"{seeded.url}/c/since/runs/{run_id}")
    expect(page.get_by_text("keyword lists changed after this run")).to_be_visible()


def test_a_stopped_runs_answer_is_marked_partial(page, seeded):
    run_id = seeded.build("partial", run=True)
    page.goto(f"{seeded.url}/c/partial/runs/{run_id}")
    expect(page.locator(".verdict")).to_be_visible()
    expect(page.locator("[data-partial]")).to_have_count(0)
    db.update_job(seeded.app.state.conn, run_id, status="stopped_budget")
    page.goto(f"{seeded.url}/c/partial/runs/{run_id}")
    expect(page.locator("[data-partial]")).to_contain_text(
        "Partial: the run stopped before it finished"
    )
    expect(page.locator("[data-partial]")).to_contain_text("resume it for the rest")


def test_run_head_names_a_changed_dense_model(page, seeded):
    run_id = seeded.build("dense", run=True)
    conn = seeded.app.state.conn
    conn.execute("update collections set dense_model = 'other/model' where name = 'dense'")
    page.goto(f"{seeded.url}/c/dense/runs/{run_id}")
    expect(page.get_by_text("dense model changed after this run")).to_be_visible()


def test_run_head_says_when_settings_were_not_tracked(page, seeded):
    run_id = seeded.build("untracked", run=True)
    conn = seeded.app.state.conn
    setup = db.get_job(conn, run_id)["setup"]
    for key in ("keyword_version", "dense_model"):
        del setup[key]
    db.set_job_setup(conn, run_id, setup)
    page.goto(f"{seeded.url}/c/untracked/runs/{run_id}")
    expect(page.get_by_text("recorded before settings were tracked")).to_be_visible()
    expect(page.get_by_text("changed after this run")).to_have_count(0)


def disagreeing_run(seeded, name):
    """A done run where hybrid gets q0 wrong and every other cell right."""
    run_id = seeded.build(name, run=True)
    conn = seeded.app.state.conn
    row = next(r for r in db.run_items(conn, run_id, "hybrid") if r["question_id"] == "q0")
    row = {k: v for k, v in row.items() if k not in ("method", "question_id", "complete")}
    row["answer"] = {"accuracy": 0.0}
    db.upsert_run_item(conn, run_id, "hybrid", "q0", row, complete=True)
    return run_id


def test_an_opened_question_shows_every_methods_hits(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Questions", exact=True).click()
    page.locator("#questions details > summary").first.click()
    opened = page.locator("#questions details[open]").first
    expect(opened.locator("[data-method-hits]")).to_have_count(3)  # kw_individual, hybrid, random
    expect(opened.locator("[data-method-hits]").nth(1)).to_contain_text("Hybrid")
    expect(opened.locator("[data-method-hits]").nth(1)).to_contain_text("matched keywords")


def test_run_picker_defaults_to_reference(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details").first).to_be_visible()
    expect(page.locator("#q-method")).to_have_value("hybrid")  # not the first method, kw_individual
    # A run without the reference method gets the first method that is not a control.
    seeded.build("nohy")
    run_id = seeded.run("nohy", ["random_example", "kw_individual", "vector"])
    page.goto(f"{seeded.url}/c/nohy/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details").first).to_be_visible()
    expect(page.locator("#q-method")).to_have_value("kw_individual")
    # question_only is a baseline, not a control, so on a Documents run it is what gets picked.
    run_id, _ = documents_run(seeded, "nohyd", ["question_only", "vector"])
    page.goto(f"{seeded.url}/c/nohyd/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#q-method")).to_have_value("question_only")


def test_outcome_is_a_badge(page, seeded):
    run_id = disagreeing_run(seeded, "badge")
    page.goto(f"{seeded.url}/c/badge/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details")).to_have_count(40)
    wrong = page.locator("#questions summary .badge.bad")
    expect(wrong).to_have_count(1)
    expect(wrong).to_have_text("Wrong")
    right = page.locator("#questions summary .badge.ok")
    expect(right).to_have_count(39)
    expect(right.first).to_have_text("Right")


def test_free_score_is_explained(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Can the free score be trusted here?").click()
    expect(page.locator("#proxy-box")).to_contain_text(
        "The free score is measured without the AI answering, so it costs nothing; "
        "this checks it against the answered score."
    )


def test_where_methods_disagree_filters_the_questions(page, seeded):
    run_id = disagreeing_run(seeded, "dis")
    page.goto(f"{seeded.url}/c/dis/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details")).to_have_count(40)
    page.get_by_label("Where methods disagree").check()
    expect(page.locator("#questions details")).to_have_count(1)
    expect(page.locator("#questions details")).to_contain_text("question 0")
    page.get_by_label("Where methods disagree").uncheck()
    expect(page.locator("#questions details")).to_have_count(40)


def test_wrong_only_still_filters_the_picked_method(page, seeded):
    run_id = disagreeing_run(seeded, "wrongonly")
    page.goto(f"{seeded.url}/c/wrongonly/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    page.locator("#q-method").select_option("hybrid")
    page.get_by_label("Wrong only").check()
    expect(page.locator("#questions details")).to_have_count(1)


def test_run_rerun_estimates_then_starts(page, seeded):
    seeded.build("rerun", run=True)
    runs = httpx.get(f"{seeded.url}/eval", params={"collection": "rerun"}).json()
    page.goto(f"{seeded.url}/c/rerun/runs/{runs[0]['id']}")
    page.get_by_role("button", name="Re-run").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    assert len(httpx.get(f"{seeded.url}/eval", params={"collection": "rerun"}).json()) == 1
    page.get_by_role("button", name="Start").click()
    page.wait_for_url(lambda u: "/runs/" in u and runs[0]["id"] not in u)
    assert len(httpx.get(f"{seeded.url}/eval", params={"collection": "rerun"}).json()) == 2


def test_run_form_and_rerun_send_the_price_they_showed(page, seeded):
    run_id = seeded.build("rsp", run=True)
    shown, confirms = [], []

    def record(route):
        if route.request.method != "POST":
            return route.fallback()
        sent = route.request.post_data_json
        if sent.get("confirm"):
            confirms.append((sent, shown[-1]))  # the estimate the page drew last
            return route.continue_()
        reply = route.fetch()
        shown.append(reply.json()["estimate"])
        route.fulfill(response=reply)

    page.route("**/eval", record)
    page.goto(seeded.url + "/c/rsp")
    page.get_by_role("button", name="New run").click()
    page.get_by_role("button", name="Estimate").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    page.get_by_role("button", name="Start").click()
    page.wait_for_url(re.compile(r"/c/rsp/runs/[^/]+$"))
    page.goto(f"{seeded.url}/c/rsp/runs/{run_id}")
    page.get_by_role("button", name="Re-run").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    page.get_by_role("button", name="Start").click()
    page.wait_for_url(lambda u: "/runs/" in u and run_id not in u)
    assert len(confirms) == 2
    for sent, estimate in confirms:
        assert (sent["calls"], sent["est_usd"]) == (estimate["calls"], estimate["est_usd"])
    assert confirms[1][1]["calls"] > 0  # the Re-run answers, so it was priced


def test_run_delete_asks_inline_then_deletes(page, seeded):
    run_id = seeded.build("gone", run=True)
    page.goto(f"{seeded.url}/c/gone/runs/{run_id}")
    page.get_by_role("button", name="Delete run…").click()
    page.get_by_role("button", name="Cancel").click()
    expect(page.get_by_role("button", name="Yes, delete this run")).to_have_count(0)
    page.get_by_role("button", name="Delete run…").click()
    page.get_by_role("button", name="Yes, delete this run").click()
    page.wait_for_url(f"{seeded.url}/c/gone")
    assert httpx.get(f"{seeded.url}/eval", params={"collection": "gone"}).json() == []


def test_resume_refusal_shows_instead_of_the_button(page, seeded):
    run_id = seeded.build("refused", run=True)
    lists = {"topic": ["x", "y"], "method": ["experiment"], "common": ["rate"]}
    r = httpx.put(f"{seeded.url}/collections/refused/keyword-lists", json=lists)
    assert r.status_code == 202
    seeded.wait(r.json()["job_id"])
    interrupt(seeded, run_id)
    page.expected = ["409"]  # the probe is refused, which is the point
    page.goto(f"{seeded.url}/c/refused/runs/{run_id}")
    expect(page.get_by_text("keyword lists changed since this run")).to_be_visible()
    expect(page.get_by_role("button", name="Resume")).to_have_count(0)


def test_resume_without_refusal_shows_the_button(page, seeded):
    run_id = seeded.build("resumable", run=True)
    interrupt(seeded, run_id)
    page.goto(f"{seeded.url}/c/resumable/runs/{run_id}")
    expect(page.get_by_role("button", name="Resume")).to_be_visible()
    page.get_by_role("button", name="Resume").click()
    expect(page.get_by_text("AI call", exact=False)).to_be_visible()
    assert httpx.get(f"{seeded.url}/jobs/{run_id}").json()["status"] == "interrupted"


def test_resume_proposal_and_tags_send_the_price_they_showed(page, seeded):
    seeded.build("rps")
    worker = seeded.app.state.worker
    worker.stop()
    try:
        run_id = seeded.queue_eval("rps")
        interrupt(seeded, run_id)  # stopped before it answered: its resume is priced
    finally:
        worker.start()
    documents_store(seeded, "dps")
    shown, confirms = [], []

    def record(route):
        if route.request.method != "POST":
            return route.fallback()
        sent = route.request.post_data_json or {}
        if sent.get("confirm"):
            confirms.append((sent, shown[-1]))  # the estimate the page drew last
            return route.continue_()
        reply = route.fetch()
        shown.append(reply.json()["estimate"])
        route.fulfill(response=reply)

    page.route(f"**/jobs/{run_id}/resume", record)
    page.route("**/collections/rps/keyword-lists/propose", record)
    page.route("**/collections/dps/keyword-tags", record)
    ai = seeded.app.state.ai
    try:
        page.goto(f"{seeded.url}/c/rps/runs/{run_id}")
        page.get_by_role("button", name="Resume").click()
        expect(page.get_by_text("AI call", exact=False)).to_be_visible()
        page.get_by_role("button", name="Start").click()
        expect(page.get_by_role("button", name="Start")).to_have_count(0)
        assert seeded.wait(run_id)["status"] == "done"
        lists = '{"topic": ["enzymes"], "method": ["assay"], "common": ["rate"]}'
        fake_replies(ai, lambda p: lists)
        page.goto(seeded.url + "/c/rps")
        page.get_by_role("button", name="Propose with AI").click()
        expect(page.get_by_text("AI call", exact=False)).to_be_visible()
        page.get_by_role("button", name="Start").click()
        expect(page.locator("#kw-topic")).to_have_value("enzymes")
        fake_replies(ai, lambda p: '{"terms": ["alpha", "brine", "cocoa"]}')
        page.goto(seeded.url + "/c/dps")
        page.get_by_role("button", name="Add keyword tags").click()
        confirm = page.locator("#tags-confirm")
        expect(confirm).to_contain_text("AI call")
        confirm.get_by_role("button", name="Start").click()
        expect(confirm.get_by_role("button", name="Start")).to_have_count(0)
        (job,) = tag_jobs(seeded, "dps")
        assert seeded.wait(job["id"])["status"] == "done"
    finally:
        fake_replies(ai, lambda p: "Answer: A")
    assert len(confirms) == 3
    for sent, estimate in confirms:
        assert (sent["calls"], sent["est_usd"]) == (estimate["calls"], estimate["est_usd"])
        assert estimate["calls"] > 0  # each was priced


def free_run(seeded, name):
    """A retrieval-only mcq run with the pair; its headline is the free reasoning match."""
    seeded.build(name)
    body = {
        "collection": name,
        "methods": ["kw_individual", "hybrid"],
        "answer": False,
        "confirm": True,
    }
    r = httpx.post(f"{seeded.url}/eval", json=body)
    assert r.status_code == 202, r.text
    assert seeded.wait(r.json()["job_id"])["status"] == "done"
    return r.json()["job_id"]


CAVEAT = "free score, not yet checked against answers"


def test_free_score_verdict_carries_the_caveat_on_every_page(page, seeded):
    run_id = free_run(seeded, "fs")
    answer = httpx.get(f"{seeded.url}/collections/fs").json()["answer"]
    assert answer["metric"] == "solution_sim@1" and answer["proxy_trusted"] is not True
    page.goto(seeded.url + "/")
    expect(page.locator('main a.row-link[href="/c/fs"]')).to_contain_text(CAVEAT)
    expect(page.locator('main a.row-link[href="/c/c"]')).not_to_contain_text(CAVEAT)
    page.goto(seeded.url + "/c/fs")
    expect(page.locator(".verdict")).to_contain_text(CAVEAT)
    page.goto(f"{seeded.url}/c/fs/runs/{run_id}")
    # on the run page the caveat is its own sentence under the verdict, not mid-sentence
    expect(page.locator(".verdict")).to_contain_text("on the same 40 questions.")
    expect(page.locator("#answer")).to_contain_text(CAVEAT, ignore_case=True)
    expect(page.locator(".verdict")).not_to_contain_text(CAVEAT, ignore_case=True)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator(".verdict")).to_be_visible()
    expect(page.locator("#answer")).not_to_contain_text(CAVEAT, ignore_case=True)


def test_pairs_show_a_dash_for_a_nan_cleaned_gap(page, seeded):
    def nan_pair(route):
        data = route.fetch().json()
        data["pairs"][0].update(diff=None, lo=None, hi=None)
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", nan_pair)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Every pair compared").click()
    first = page.locator("#pairs tbody tr").first
    expect(first).to_contain_text("–")
    expect(first).not_to_contain_text("+0 pts")


def test_reasoning_match_is_rescaled_in_own_scores_and_pairs(page, seeded):
    sim = "solution_sim@1"

    def rescaled(route):
        data = route.fetch().json()
        row = lambda mean: {"mean": mean, "lo": mean - 0.05, "hi": mean + 0.05}  # noqa: E731
        data["methods"]["random_example"][sim] = row(0.5)
        data["methods"]["hybrid"][sim] = row(0.75)
        data["pairs"] = [
            {
                "metric": sim,
                "a": "hybrid",
                "b": "random_example",
                "diff": 0.25,
                "lo": 0.2,
                "hi": 0.3,
                "separated": True,
            }
        ]
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", rescaled)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Each method's own score").click()
    hybrid = page.locator("#scores tbody tr", has_text="hybrid").first
    expect(hybrid).to_contain_text("50 match")  # (0.75 - 0.5) / (1 - 0.5) * 100
    expect(hybrid).not_to_contain_text("75%")
    page.get_by_text("Every pair compared").click()
    expect(page.locator("#pairs tbody tr").first).to_contain_text("+50 pts")


def cost_label(page, name):
    return page.locator(f"label.check:has(input[value='{name}'])").locator(".cost")


def test_paid_methods_carry_a_costs_money_mark(page, seeded):
    seeded.build("cost")
    page.goto(seeded.url + "/c/cost")
    page.get_by_role("button", name="New run").click()
    expect(cost_label(page, "iterative")).to_have_text("costs money")
    expect(cost_label(page, "hybrid")).to_have_count(0)
    # a keyword method spends only with summaries, and this collection has none
    expect(cost_label(page, "kw_individual")).to_have_count(0)
    expect(
        page.locator("label.check", has=page.locator("#rf-answer")).locator(".cost")
    ).to_have_text("costs money")


def test_keyword_methods_say_costs_money_on_a_summarising_collection(page, seeded):
    seeded.build("sumcol", summarize=True)
    page.goto(seeded.url + "/c/sumcol")
    page.get_by_role("button", name="New run").click()
    expect(cost_label(page, "kw_individual")).to_have_text("costs money")
    expect(cost_label(page, "iterative")).to_have_text("costs money")


def test_run_with_overlapping_own_scores_but_a_separated_pair_says_ahead(page, seeded):
    sim = "solution_sim@1"
    run_id = seeded.build("ovl", run=True)
    conn = seeded.app.state.conn
    for r in db.run_items(conn, run_id):
        if r["method"] not in ("kw_individual", "hybrid"):
            continue
        n = int(r["question_id"][1:])
        base = 0.8 * (n % 2)  # 0 or 0.8: a wide spread within each method
        r["retrieval"][sim] = base + (0.1 if r["method"] == "kw_individual" else 0.0)
        row = {k: v for k, v in r.items() if k not in ("method", "question_id", "complete")}
        db.upsert_run_item(conn, run_id, r["method"], r["question_id"], row, True)
    data = httpx.get(f"{seeded.url}/eval/{run_id}").json()
    kw, hy = data["methods"]["kw_individual"][sim], data["methods"]["hybrid"][sim]
    assert kw["lo"] < hy["hi"] and hy["lo"] < kw["hi"]  # the own-score ranges overlap
    pair = next(
        p
        for p in data["pairs"]
        if p["metric"] == sim and {p["a"], p["b"]} == {"kw_individual", "hybrid"}
    )
    assert pair["separated"] and pair["diff"] != 0
    page.goto(f"{seeded.url}/c/ovl/runs/{run_id}")
    page.get_by_text("Every pair compared").click()
    row = page.locator("#pairs tbody tr", has_text="Reasoning match").first
    expect(row).to_contain_text("ahead")
    expect(row).not_to_contain_text("can't tell")


def test_a_queued_run_waits_for_its_answer_and_offers_only_stop_run(page, seeded):
    with worker_stopped(seeded, "qr") as run_id:
        page.goto(f"{seeded.url}/c/qr/runs/{run_id}")
        expect(page.locator("#answer")).to_contain_text("The answer appears when the run finishes")
        expect(page.locator("#gaps")).to_contain_text("The gaps appear when the run finishes")
        expect(page.locator("#head")).to_contain_text("– questions")
        expect(page.locator("#head")).not_to_contain_text("0 questions")
        expect(page.get_by_text("does not include")).to_have_count(0)
        expect(page.get_by_role("button", name="Stop run")).to_be_visible()
        expect(page.get_by_role("button", name="Delete run…")).to_have_count(0)
        expect(page.get_by_role("button", name="Re-run")).to_have_count(0)


def test_a_finished_run_names_only_the_method_it_lacks(page, seeded):
    seeded.build("half")
    run_id = seeded.run("half", ["hybrid"])
    page.goto(f"{seeded.url}/c/half/runs/{run_id}")
    expect(page.locator("#answer")).to_contain_text(
        "This run does not include Keywords per list, so it gives no answer."
    )
    expect(page.locator("#answer")).not_to_contain_text("and Hybrid")
    expect(page.locator("#gaps")).to_contain_text("no other method has results")
    expect(page.get_by_role("button", name="Delete run…")).to_be_visible()


def test_verdicts_explain_the_range_and_their_words(page, seeded):
    page.goto(seeded.url + "/")
    cell = page.locator('main a.row-link[href="/c/c"] [title*="spans 0"]')
    expect(cell).to_have_count(1)
    expect(cell.locator("strong.word")).to_have_count(1)  # the verdict word has weight
    page.goto(seeded.url + "/c/c")
    expect(page.locator("#answer .legend")).to_contain_text("Can't tell: it spans 0")
    expect(page.locator("#answer .verdict strong.word")).to_have_count(1)
    expect(page.locator("#answer .axis")).to_contain_text("pts")
    # the verdict says the names and numbers; nothing under the bar repeats them
    expect(page.locator("#answer")).not_to_contain_text("against Hybrid")
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#answer .legend")).to_contain_text("spans 0")
    expect(page.locator("#answer .legend")).to_contain_text("Ahead, behind and can't tell")


def test_run_page_says_what_the_metric_means_and_draws_the_axis(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#gaps")).to_contain_text("Share of questions answered correctly")
    expect(page.locator("#gaps thead th")).to_have_count(4)
    axis = page.locator("#gaps thead .axis")
    expect(axis).to_contain_text("−")
    expect(axis).to_contain_text("pts")
    page.get_by_text("Each method's own score").click()
    says = "Share of questions answered correctly"
    expect(page.locator(f'#scores th[title="{says}"]')).to_have_count(1)


def test_raw_reasoning_match_is_marked_no_chance_level(page, seeded):
    def raw(route):
        data = route.fetch().json()
        data["scale"] = {"unit": "raw", "factor": 100}
        data["headline"] = "solution_sim@1"
        data["answer"]["metric"] = "solution_sim@1"
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", raw)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#answer")).to_contain_text("no chance level")
    expect(page.locator("#gaps > p").first).to_contain_text("no chance level")
    for row in page.locator("[data-gap-row]").all():
        expect(row).to_contain_text("no chance level")


def test_money_below_a_cent_never_shows_as_zero(page, seeded):
    def cheap_rows(route):
        rows = route.fetch().json()
        for r in rows:
            r["usd_total"] = 0.004
        route.fulfill(json=rows)

    def cheap_run(route):
        data = route.fetch().json()
        data["usd_total"] = 0.004
        route.fulfill(json=data)

    page.route(lambda url: "/eval?collection=c" in url, cheap_rows)
    page.route(f"**/eval/{seeded.run_id}", cheap_run)
    page.goto(seeded.url + "/c/c")
    expect(page.locator("[data-run]").first).to_contain_text("< $0.01")
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#head")).to_contain_text("< $0.01")
    expect(page.locator("#head")).not_to_contain_text("$0.00")


def test_own_scores_use_a_true_minus_and_never_print_nan(page, seeded):
    sim = "solution_sim@1"

    def scores(route):
        data = route.fetch().json()
        data["methods"]["random_example"][sim] = {"mean": 0.5, "lo": 0.45, "hi": 0.55}
        data["methods"]["hybrid"][sim] = {"mean": 0.3, "lo": 0.25, "hi": 0.35}
        data["methods"]["kw_individual"]["accuracy"] = {"mean": None, "lo": None, "hi": None}
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", scores)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Each method's own score").click()
    hybrid = page.locator("#scores tbody tr", has_text="hybrid").first
    expect(hybrid).to_contain_text("−40 match")  # (0.3 - 0.5) / (1 - 0.5) * 100
    expect(hybrid).to_contain_text("−50 to −30")
    kw = page.locator("#scores tbody tr", has_text="kw_individual").first
    expect(kw).to_contain_text("n/a")
    expect(page.locator("#scores")).not_to_contain_text("NaN")


def questions_counter(page, run_id):
    """Counts GET .../questions per disagree value."""
    seen = {"true": 0, "false": 0}

    def count(route):
        seen["true" if "disagree=true" in route.request.url else "false"] += 1
        route.continue_()

    page.route(f"**/eval/{run_id}/questions*", count)
    return seen


def test_questions_filter_without_refetching_and_sort_naturally(page, seeded):
    run_id = disagreeing_run(seeded, "qcache")
    seen = questions_counter(page, run_id)
    page.goto(f"{seeded.url}/c/qcache/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    summaries = page.locator("#questions details > summary")
    expect(summaries).to_have_count(40)
    texts = summaries.all_inner_texts()
    order = [int(re.search(r"question (\d+)", t).group(1)) for t in texts]
    assert order == sorted(order)  # q2 before q10
    page.locator("#q-method").select_option("hybrid")
    page.get_by_label("Wrong only").check()
    expect(page.locator("#questions details")).to_have_count(1)
    page.get_by_label("Wrong only").uncheck()
    page.get_by_label("Where methods disagree").check()
    expect(page.locator("#questions details")).to_have_count(1)
    page.get_by_label("Where methods disagree").uncheck()
    expect(page.locator("#questions details")).to_have_count(40)
    assert seen == {"false": 1, "true": 1}


def test_opening_questions_during_the_first_load_fetches_them_once(page, seeded):
    run_id = disagreeing_run(seeded, "qfirst")
    seen = questions_counter(page, run_id)
    held = []

    def running(route):  # a busy run keeps nothing, so a second load would fetch again
        response = route.fetch()
        route.fulfill(response=response, json={**response.json(), "status": "running"})

    page.route(f"**/eval/{run_id}", running)
    page.route("**/collections/qfirst", lambda route: held.append(route))
    page.goto(f"{seeded.url}/c/qfirst/runs/{run_id}")
    for _ in range(200):  # the run is in; the first load waits on its store
        if held:
            break
        page.wait_for_timeout(50)
    assert held
    page.get_by_text("Questions", exact=True).click()
    page.wait_for_timeout(300)  # the toggle's own load, if it starts one, reaches the route
    page.unroute("**/collections/qfirst")  # a later refetch goes straight through
    for route in held:
        route.continue_()
    expect(page.locator("#questions details")).to_have_count(40)
    page.wait_for_timeout(300)  # a second fetch, if one was started, has arrived by now
    assert seen["false"] == 1


def test_questions_on_a_busy_run_refetch_until_it_finishes(page, seeded):
    run_id = disagreeing_run(seeded, "qbusy")
    seen = questions_counter(page, run_id)
    busy = {"on": True}

    def status(route):  # the run reads as running until the test lets it finish
        response = route.fetch()
        body = response.json()
        if busy["on"]:
            body["status"] = "running"
        route.fulfill(response=response, json=body)

    page.route(f"**/eval/{run_id}", status)
    page.goto(f"{seeded.url}/c/qbusy/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details")).to_have_count(40)
    with page.expect_response(f"**/eval/{run_id}"):  # one poll seen: the filters must still work
        pass
    page.locator("#q-method").select_option("hybrid")
    page.get_by_label("Wrong only").check()
    expect(page.locator("#questions details")).to_have_count(1)
    assert seen["false"] == 3  # rows still coming: fetched each time, never kept
    # finishes: the next poll sees it done, and the open view reloads once, then filters from
    # what it kept. A response wait: `seen` is bumped in the route handler.
    with page.expect_response(lambda r: "/questions?disagree=false" in r.url, timeout=10000):
        busy["on"] = False
    assert seen["false"] == 4
    page.get_by_label("Wrong only").uncheck()
    expect(page.locator("#questions details")).to_have_count(40)
    assert seen["false"] == 4


def test_a_late_questions_response_does_not_overwrite_a_newer_choice(page, seeded):
    run_id = disagreeing_run(seeded, "qrace")
    held = []

    def hold_disagree(route):
        if "disagree=true" in route.request.url:
            held.append(route)
        else:
            route.continue_()

    page.route(f"**/eval/{run_id}/questions*", hold_disagree)
    page.goto(f"{seeded.url}/c/qrace/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    expect(page.locator("#questions details")).to_have_count(40)
    page.get_by_label("Where methods disagree").check()
    page.wait_for_timeout(300)  # let the held request reach the route
    page.get_by_label("Where methods disagree").uncheck()
    expect(page.locator("#questions details")).to_have_count(40)
    for route in held:
        route.continue_()
    page.wait_for_timeout(500)
    assert held, "the disagree request was never made"
    expect(page.locator("#questions details")).to_have_count(40)


def test_run_form_lists_what_is_not_measurable(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "nm", "qa_type": "science"})
    plain = {"id": "p", "text": "enzyme rate", "answer": "3 J", "explanation": "", "meta": {}}
    r = httpx.post(
        f"{seeded.url}/collections/nm/documents", json={"items": [plain], "confirm": True}
    )
    assert r.status_code == 200, r.text
    db.save_questions(
        seeded.app.state.conn,  # type: ignore[attr-defined]
        "nm",
        [Question("p1", "enzyme rate", None, ["3 J"], [], {})],
    )
    page.goto(seeded.url + "/c/nm")
    expect(page.locator("#runs")).not_to_be_empty()  # the page has loaded its collection
    page.get_by_role("button", name="New run").click()
    form = page.locator("#run-form")
    expect(form).to_contain_text("This run will measure:")
    expect(form).to_contain_text("Not measurable here:")
    expect(form).to_contain_text("(no question has a worked solution)")
    page.goto(seeded.url + "/c/c")
    expect(page.locator("#runs")).not_to_be_empty()  # the page has loaded its collection
    page.get_by_role("button", name="New run").click()
    expect(page.locator("#run-form")).to_contain_text("This run will measure:")


def test_run_page_shows_partial_gold(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "pg", "qa_type": "multihop"})
    items = [
        {"id": f"d{i}", "text": f"passage {i} about {t}", "meta": {}}
        for i, t in enumerate(["reefs", "tides", "volcanoes", "enzymes"])
    ]
    r = httpx.post(f"{seeded.url}/collections/pg/documents", json={"items": items, "confirm": True})
    assert r.status_code == 200, r.text
    jobs_done(seeded, "pg")
    db.save_questions(
        seeded.app.state.conn,  # type: ignore[attr-defined]
        "pg",
        [
            Question("q0", "what about reefs", None, ["x"], ["d0"], {}),
            Question("q1", "what about tides", None, ["x"], ["d1"], {}),
            Question("q2", "what about volcanoes", None, ["x"], [], {}),
            Question("q3", "what about enzymes", None, ["x"], [], {}),
        ],
    )
    body = {"collection": "pg", "methods": ["hybrid"], "answer": False, "confirm": True}
    r = httpx.post(f"{seeded.url}/eval", json=body)
    assert r.status_code == 202, r.text
    assert seeded.wait(r.json()["job_id"])["status"] == "done"
    page.goto(f"{seeded.url}/c/pg/runs/{r.json()['job_id']}")
    page.get_by_text("Each method's own score").click()
    expect(page.locator("#scores tbody tr", has_text="hybrid")).to_contain_text(
        "on 2 of 4 questions"
    )


def test_run_page_says_one_question_s_evidence_in_the_singular(page, seeded):
    def left_out(route):
        data = route.fetch().json()
        data["setup"].update(evidence_off=1)
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", left_out)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Setup this ran with").click()
    expect(page.locator("#setup")).to_contain_text("1 question's evidence is switched off")


def test_run_page_shows_the_search_settings_and_rerun_repeats_them(page, seeded):
    search = {
        "weights": {"meaning": 1, "words": 1, "keywords": 2},
        "query_tags": "classifier",
        "example": "abstracted",
        "program": False,
    }
    sent: list[dict] = []

    def with_search(route):
        data = route.fetch().json()
        data["setup"]["search"] = search
        route.fulfill(json=data)

    def capture(route):
        if route.request.method == "POST":
            sent.append(route.request.post_data_json)
        route.continue_()

    page.route(f"**/eval/{seeded.run_id}", with_search)
    page.route("**/eval", capture)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Setup this ran with").click()
    setup = page.locator("#setup")
    expect(setup).to_contain_text("weights (meaning 1 · words 1 · keywords 2)")
    expect(setup).to_contain_text("query_tags classifier")
    expect(setup).not_to_contain_text("[object Object]")
    page.get_by_role("button", name="Re-run").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    assert sent and sent[0]["search"] == search


def rerun_bodies(page, seeded, setup: dict) -> list[dict]:
    """Opens the seeded run with `setup` merged into its own, and collects what Re-run POSTs."""
    sent: list[dict] = []

    def with_setup(route):
        data = route.fetch().json()
        data["setup"].update(setup)
        for key in [k for k, v in setup.items() if v is None]:
            del data["setup"][key]  # None: a setup recorded before the field
        route.fulfill(json=data)

    def capture(route):
        if route.request.method == "POST":
            sent.append(route.request.post_data_json)
        route.continue_()

    page.route(f"**/eval/{seeded.run_id}", with_setup)
    page.route("**/eval", capture)
    return sent


def test_rerun_repeats_the_run_s_reranker_and_pool(page, seeded):
    tiny = "jinaai/jina-reranker-v1-tiny-en"
    setup = {"methods": ["hybrid", "hybrid_rerank"], "rerank_pool": 20, "rerank_model": tiny}
    sent = rerun_bodies(page, seeded, setup)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#head")).to_contain_text("reranker jina-reranker-v1-tiny-en, pool 20")
    page.get_by_role("button", name="Re-run").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    assert sent and sent[0]["rerank_model"] == tiny and sent[0]["rerank_pool"] == 20


def test_rerun_of_a_run_from_before_the_reranker_was_recorded_uses_bge_at_50(page, seeded):
    setup = {"methods": ["hybrid", "hybrid_rerank"], "rerank_pool": None, "rerank_model": None}
    sent = rerun_bodies(page, seeded, setup)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#head")).to_contain_text("reranker bge-reranker-v2-m3-int8, pool 50")
    page.get_by_role("button", name="Re-run").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    assert sent and sent[0]["rerank_model"] == "BAAI/bge-reranker-v2-m3-int8"
    assert sent[0]["rerank_pool"] == 50


def test_a_run_without_hybrid_rerank_names_no_reranker(page, seeded):
    sent = rerun_bodies(page, seeded, {})
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator("#head")).to_contain_text("question")
    expect(page.locator("#head")).not_to_contain_text("reranker")
    page.get_by_role("button", name="Re-run").click()
    expect(page.get_by_text("AI calls", exact=False)).to_be_visible()
    assert sent and "rerank_model" not in sent[0] and "rerank_pool" not in sent[0]


def test_run_page_lists_what_the_run_left_out(page, seeded):
    def left_out(route):
        data = route.fetch().json()
        data["setup"].update(disabled_sources=["upload:notes"], evidence_off=2, leaked_excluded=3)
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", left_out)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Setup this ran with").click()
    left = page.locator("#setup")
    expect(left).to_contain_text("Retrieval left out: notes (uploaded)")
    expect(left).to_contain_text("2 questions' evidence is switched off")
    expect(left).to_contain_text("3 leaked questions left out")


def test_run_form_models_come_from_settings(page, seeded):
    seeded.build("pickers")
    state = ai_state(seeded)
    want = [
        m["label"]
        for m in state["models"]
        if m["enabled"] and m["price_in"] is not None and m["price_out"] is not None
    ]
    assert want
    page.goto(seeded.url + "/c/pickers")
    page.get_by_role("button", name="New run").click()
    expect(page.locator("#rf-model option").first).to_be_attached()
    assert page.locator("#rf-model option").all_inner_texts() == want
    refs = page.locator("#rf-model option").evaluate_all("os => os.map(o => o.value)")
    assert refs == [m["ref"] for m in state["models"] if m["label"] in want]


NEEDS_TAGS = "needs keyword tags: Add keyword tags on the store page"


def test_run_form_says_why_keyword_rag_is_off(page, seeded):
    documents_store(seeded, "dwhy", passages=False)
    httpx.post(
        f"{seeded.url}/eval-sets", json={"name": "dwhy", "qa_type": "documents"}
    )  # New run waits for a set
    page.goto(seeded.url + "/c/dwhy")
    page.get_by_role("button", name="New run").click()
    box = page.locator("input[name=method][value=keyword_rag]")
    expect(box).to_be_disabled()
    label = page.locator("#run-form label.check", has=box)
    expect(label).to_contain_text(f"({NEEDS_TAGS})")
    expect(
        page.locator("#run-form label.check", has=page.locator("[value=hybrid]"))
    ).not_to_contain_text("(")


def test_run_form_picks_an_eval_set_of_the_store_type_and_shows_its_counts(page, seeded):
    seeded.build("rf1")
    mcq_set(seeded, "rf2", 3)
    httpx.post(f"{seeded.url}/eval-sets", json={"name": "rfsci", "qa_type": "science"})
    page.goto(seeded.url + "/c/rf1")
    expect(page.locator("#runs")).not_to_be_empty()
    page.get_by_role("button", name="New run").click()
    pick = page.locator("#rf-set")
    expect(pick).to_have_value("rf1")
    values = pick.locator("option").evaluate_all("o => o.map(x => x.value)")
    assert "rf2" in values and "rfsci" not in values
    first = httpx.post(
        f"{seeded.url}/eval",
        json={"collection": "rf1", "methods": ["hybrid"], "answer": True, "eval_set": "rf2"},
    ).json()["estimate"]
    assert first["questions"] == 3
    pick.select_option("rf2")
    dev = first["dev_questions"]
    expect(page.locator("#rf-split option[value=dev]")).to_have_text(f"Dev slice ({dev})")
    expect(page.locator("#rf-split option[value=full]")).to_have_text("All (3)")
    page.get_by_role("button", name="Estimate").click()
    page.get_by_role("button", name="Start").click()
    expect(page).to_have_url(re.compile(r"/c/rf1/runs/[^/]+$"))
    runs = httpx.get(f"{seeded.url}/eval?collection=rf1").json()
    assert [r["eval_set"] for r in runs] == ["rf2"]


def test_run_form_counts_follow_the_newest_pick(page, seeded):
    seeded.build("rc")
    mcq_set(seeded, "rc3", 3)
    mcq_set(seeded, "rc5", 5)
    held, failing = [], []

    def estimate(route):
        if route.request.method != "POST":
            route.fallback()
        elif failing:
            route.fulfill(status=500, json={"detail": "estimate broke"})
        elif json.loads(route.request.post_data or "{}").get("eval_set") == "rc3":
            held.append(route)  # answered late, after the newer pick
        else:
            route.fallback()

    page.route("**/eval", estimate)
    page.goto(seeded.url + "/c/rc")
    expect(page.locator("#runs")).not_to_be_empty()
    page.get_by_role("button", name="New run").click()
    full = page.locator("#rf-split option[value=full]")
    page.locator("#rf-set").select_option("rc3")
    page.locator("#rf-set").select_option("rc5")
    expect(full).to_have_text("All (5)")
    assert len(held) == 1
    with page.expect_response(lambda r: '"rc3"' in (r.request.post_data or "")):
        held[0].fallback()
    page.wait_for_timeout(200)  # the older answer is in, and must not draw
    expect(full).to_have_text("All (5)")
    # A split change re-reads too; a failed read shows no counts rather than the last ones.
    page.expected = ["500"]
    failing.append(True)
    page.locator("#rf-split").select_option("full")
    expect(full).to_have_text("All (–)")
    expect(page.locator("#rf-split option[value=dev]")).to_have_text("Dev slice (–)")
    expect(page.locator("#rf-msg")).to_contain_text("estimate broke")


def test_run_form_says_when_the_estimate_refuses_a_set(page, seeded):
    seeded.build("rf3")
    httpx.post(f"{seeded.url}/eval-sets", json={"name": "rf3e", "qa_type": "mcq"})
    page.goto(seeded.url + "/c/rf3")
    expect(page.locator("#runs")).not_to_be_empty()
    page.get_by_role("button", name="New run").click()
    page.expected = ["409", "422"]
    page.locator("#rf-set").select_option("rf3e")
    expect(page.locator("#rf-split option[value=full]")).to_have_text("All (0)")


def test_run_page_names_its_eval_set_and_says_when_retrieval_is_not_measured(page, seeded):
    detail = httpx.get(f"{seeded.url}/eval/{seeded.run_id}").json()
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    version = detail["eval_set_version"][:8]
    expect(page.locator("#head")).to_contain_text(f"eval set c · version {version}")
    expect(page.locator("#gaps")).not_to_contain_text("holds none of the eval set's evidence")

    def none_held(route):
        body = route.fetch().json()
        body.update(headline="recall@5", evidence_named=3, evidence_held=0)
        route.fulfill(json=body)

    page.route(f"**/eval/{seeded.run_id}", none_held)
    page.reload()
    expect(page.locator("#gaps")).to_contain_text(
        "This store holds none of the eval set's evidence, "
        "so retrieval isn't measured; the answers are."
    )


def test_run_page_says_the_eval_set_changed_after_the_run(page, seeded):
    run_id = seeded.build("chg", run=True)
    now = httpx.get(f"{seeded.url}/eval-sets/chg").json()
    qs = [q for q in now["questions"] if q["id"] != "q3"]
    r = httpx.put(
        f"{seeded.url}/eval-sets/chg/questions",
        json={"questions": qs, "base_version": now["version"]},
    )
    assert r.status_code == 200, r.text
    page.goto(f"{seeded.url}/c/chg/runs/{run_id}")
    expect(page.locator("#head")).to_contain_text("eval set changed after this run")


def test_run_against_its_store_opens_the_run_form_with_the_set(page, seeded):
    documents_store(seeded, "run1")
    for name in ("run1a", "run1b"):
        body = {"name": name, "qa_type": "documents", "store": "run1"}
        r = httpx.post(f"{seeded.url}/eval-sets", json=body)
        assert r.status_code == 201, r.text
        r = httpx.post(
            f"{seeded.url}/eval-sets/{name}/questions", json={"question": {"text": "alpha?"}}
        )
        assert r.status_code == 201, r.text
    assert httpx.get(f"{seeded.url}/collections/run1").json()["eval_set"] == "run1a"
    page.goto(seeded.url + "/e/run1b")
    link = page.get_by_role("link", name="Run against its store")
    expect(link).to_have_attribute("href", "/c/run1?eval_set=run1b")
    link.click()
    expect(page).to_have_url(seeded.url + "/c/run1?eval_set=run1b")
    expect(page.locator("#run-form")).to_be_visible()
    expect(page.locator("#rf-set")).to_have_value("run1b")
    expect(page.locator("#rf-split option[value=full]")).to_have_text("All (1)")


def test_run_form_warns_when_most_right_passages_are_missing(page, seeded):
    seeded.build("few1")
    for name, ids in (
        ("few1a", ["gone-1", "gone-2", "gone-3", "enzyme-item"]),
        ("few1b", ["enzyme-item", "market-item"]),
    ):
        r = httpx.post(f"{seeded.url}/eval-sets", json={"name": name, "qa_type": "mcq"})
        assert r.status_code == 201, r.text
        qs = [
            {
                "id": f"s{i}",
                "text": f"question {i}",
                "choices": ["x", "y"],
                "gold_answers": ["A"],
                "gold_ids": [g],
            }
            for i, g in enumerate(ids)
        ]
        put_questions(seeded, name, qs)
    page.goto(seeded.url + "/c/few1")
    expect(page.locator("#runs")).not_to_be_empty()
    page.get_by_role("button", name="New run").click()
    page.locator("#rf-split").select_option("full")
    warn = page.locator("#rf-measure [data-few-evidence]")
    page.locator("#rf-set").select_option("few1a")
    expect(warn).to_have_text(
        "Most of this eval set's questions have no right passage in this store: only 1 of 4 do."
    )
    page.locator("#rf-set").select_option("few1b")
    expect(page.locator("#rf-split option[value=full]")).to_have_text("All (2)")
    expect(warn).to_have_count(0)


RUN_LEAD = "How well does this store find the right passages?"

NOT_TRACKED = (
    "This eval set doesn't record which passages were judged, so score reliability can't be checked"
)


def test_documents_run_page_leads_with_finding_the_right_passages(page, seeded):
    run_id, ids = documents_run(seeded, "rp1", ["hybrid", "vector"])
    conn = seeded.app.state.conn
    for r in db.run_items(conn, run_id, "hybrid"):
        row = {k: v for k, v in r.items() if k not in ("method", "question_id", "complete")}
        row["hit_ids"] = ids + [f"rp1-never-{i}" for i in range(7)]  # 7 of its top 10 unjudged
        db.upsert_run_item(conn, run_id, "hybrid", r["question_id"], row, True)
    labels = httpx.get(f"{seeded.url}/eval/{run_id}").json()["labels"]
    assert labels["unjudged"]["hybrid"] == {"shown": 7, "questions": 2, "flag": True}
    assert not labels["unjudged"]["vector"]["flag"]
    page.goto(f"{seeded.url}/c/rp1/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer.locator("p.verdict strong")).to_have_text(RUN_LEAD)
    expect(
        answer.get_by_text(
            "Hybrid: score unreliable: on average 7 of its top 10 were never judged", exact=True
        )
    ).to_be_visible()
    expect(answer.get_by_text("Vector search:")).to_have_count(0)
    expect(answer).not_to_contain_text("score unreliable: on average 0")
    expect(page.get_by_text("Do keywords help?")).to_have_count(0)
    expect(answer).not_to_contain_text(NOT_TRACKED)
    expect(page.locator("#scores-box")).to_have_attribute("open", "")
    expect(page.locator("#scores table")).to_be_visible()
    expect(page.locator("#proxy")).to_be_hidden()  # the free score is not a Documents idea
    # Gaps now draw for a Documents run: against the run's scored method, not the keyword pair.
    expect(page.locator("#gaps table")).to_be_visible()
    expect(page.locator("[data-gap-row]")).to_have_count(1)
    expect(page.locator("#note")).to_have_text(
        "Leading is not winning: the order above is this run's alone."
    )


def test_documents_run_form_ticks_chat_method_not_controls(page, seeded):
    run_form_set(seeded, "dd2")
    page.goto(seeded.url + "/c/dd2")
    page.get_by_role("button", name="New run").click()
    box = lambda m: page.locator(f"input[name=method][value={m}]")  # noqa: E731
    expect(box("hybrid_rerank")).to_be_checked()
    expect(box("hybrid")).to_be_checked()
    expect(box("random_example")).not_to_be_checked()
    expect(box("fixed_example")).not_to_be_checked()
    with page.expect_request(lambda r: r.url.endswith("/eval") and r.method == "POST") as sent:
        page.get_by_role("button", name="Estimate").click()
    posted = json.loads(sent.value.post_data)
    assert posted["rerank_pool"] == 20


def test_documents_partial_run_says_partial(page, seeded):
    run_id, _ = documents_run(seeded, "dd5", ["hybrid_rerank", "vector"])

    def inject(route):
        data = route.fetch().json()
        data["status"] = "cancelled"
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.expected = ["409"]  # the page asks to resume the run, which is really done
    page.goto(f"{seeded.url}/c/dd5/runs/{run_id}")
    expect(page.locator("#answer [data-partial]")).to_contain_text("Partial")
    expect(page.locator("#answer")).to_contain_text("the search your chats use")


def test_documents_run_without_the_chats_vector_search_has_no_score(page, seeded):
    r = httpx.put(f"{seeded.url}/search-defaults", json={"method": "vector"})
    assert r.status_code == 200, r.text
    try:
        run_id, _ = documents_run(seeded, "dd8", ["hybrid"])  # frozen: the chats use Vector
    finally:
        httpx.put(f"{seeded.url}/search-defaults", json={"method": "hybrid"})
    page.goto(f"{seeded.url}/c/dd8/runs/{run_id}")
    expect(page.locator("#answer")).to_contain_text(
        "This run doesn't include Vector search, the search your chats use."
    )


def test_documents_run_page_draws_gaps_against_chat_method(page, seeded):
    run_id, _ = documents_run(seeded, "dd7", ["hybrid_rerank", "vector"])
    page.goto(f"{seeded.url}/c/dd7/runs/{run_id}")
    ref = page.locator("[data-ref-row]")
    expect(ref).to_contain_text("Hybrid + reranker")
    expect(ref).to_contain_text("(reference)")
    expect(page.locator("#answer")).to_contain_text("the search your chats use")


def test_documents_run_page_says_what_it_could_not_measure(page, seeded):
    run_id, _ = documents_run(seeded, "rp2", ["vector"])
    labels = {"tracked": False, "unjudged": {}, "unmeasured": 1, "no_right_passage": 2}

    def inject(route):
        data = route.fetch().json()
        data["labels"] = labels
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.goto(f"{seeded.url}/c/rp2/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer.locator("p.verdict strong")).to_have_text(RUN_LEAD)
    for line in (
        "1 question unmeasured: its right passages aren't in this store",
        "2 questions have no right passage; they are not scored",
        NOT_TRACKED,
    ):
        expect(answer.get_by_text(line, exact=True)).to_be_visible()
    expect(page.get_by_text("labelled passages are no longer")).to_have_count(0)
    expect(page.get_by_text("labelled passage is no longer")).to_have_count(0)
    expect(page.locator("[data-no-evidence]")).to_have_count(0)


def test_documents_run_page_shows_the_unjudged_flag_only_once_the_run_is_done(page, seeded):
    run_id, _ = documents_run(seeded, "rp4", ["hybrid"])
    status = {"now": "running"}
    flagged = {"shown": 5, "questions": 2, "flag": True}

    def inject(route):
        data = route.fetch().json()
        data["status"] = status["now"]
        data["progress"] = 0.5 if status["now"] == "running" else 1.0
        data["labels"] = {**data["labels"], "unjudged": {"hybrid": flagged}}
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.goto(f"{seeded.url}/c/rp4/runs/{run_id}")
    answer = page.locator("#answer")
    flag = "Hybrid: score unreliable: on average 5 of its top 10 were never judged"
    expect(answer.locator("p.verdict")).to_have_text(
        f"{RUN_LEAD} The scores appear when the run finishes."
    )
    expect(answer.get_by_text(flag)).to_have_count(0)
    expect(page.get_by_role("button", name="Stop run")).to_be_visible()
    status["now"] = "done"
    page.reload()
    expect(answer.get_by_text(flag, exact=True)).to_be_visible()
    expect(answer).not_to_contain_text("The scores appear when the run finishes.")


def test_documents_run_questions_mark_hits_never_judged(page, seeded):
    run_id, (right, other) = documents_run(seeded, "rp3", ["vector"], judged_all=False)
    rows = httpx.get(f"{seeded.url}/eval/{run_id}/questions?disagree=false").json()
    assert all(r["judged_ids"] == [right] for r in rows)
    assert all(set(r["methods"]["vector"]["hit_ids"]) == {right, other} for r in rows)
    page.goto(f"{seeded.url}/c/rp3/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    opened = page.locator("#questions details").first
    opened.locator("summary").click()
    hits = opened.locator("[data-method-hits] li")
    expect(hits).to_have_count(2)
    expect(hits.filter(has_text=other).locator("[data-unjudged]")).to_have_text("(never judged)")
    expect(hits.filter(has_text=right).locator("[data-unjudged]")).to_have_count(0)


def test_documents_run_page_never_says_the_answers_are_measured(page, seeded):
    run_id, _ = documents_run(seeded, "rp5", ["vector"])

    def inject(route):  # the store lost every right passage; one question never had one
        data = route.fetch().json()
        data["evidence_named"], data["evidence_held"] = 1, 0
        data["labels"] = {**data["labels"], "unmeasured": 1, "no_right_passage": 1}
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.goto(f"{seeded.url}/c/rp5/runs/{run_id}")
    answer = page.locator("#answer")
    unmeasured = "1 question unmeasured: its right passages aren't in this store"
    expect(answer.get_by_text(unmeasured, exact=True)).to_be_visible()
    expect(page.locator("[data-no-evidence]")).to_have_count(0)
    expect(page.locator("main")).not_to_contain_text("the answers are")


def test_documents_run_page_drops_the_keyword_comparison_lines(page, seeded):
    run_id, _ = documents_run(seeded, "rp6", ["hybrid", "vector"])
    best = {"candidate": "vector", "reference": "hybrid", "diff": 3.0, "lo": 1.0, "hi": 5.0}

    def inject(route):  # as if keyword_rag ran, with a best-of-each pair
        data = route.fetch().json()
        data["setup"]["methods"] = [*data["setup"]["methods"], "keyword_rag"]
        data["best_of_each"] = {**(data["best_of_each"] or {}), **best}
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.goto(f"{seeded.url}/c/rp6/runs/{run_id}")
    expect(page.locator("#gaps table")).to_be_visible()
    expect(page.locator("#note")).to_have_text(
        "Leading is not winning: the order above is this run's alone."
    )
    expect(page.locator("#best")).to_be_empty()
    expect(page.locator("main")).not_to_contain_text("Best keyword method")


def run_form_set(seeded, name, reference=True):
    """A Documents store `name` and its linked set `<name>q` of two questions right on the
    store's first passage; the first has the reference answer "Ten days" (with `reference`)."""
    labelling_set(seeded, name)
    pool = httpx.post(f"{seeded.url}/eval-sets/{name}q/pool", json={"question": "alpha"})
    ids = [p["id"] for p in pool.json()["passages"]]
    texts = (f"{name} how long does alpha last?", f"{name} what does alpha hold?")
    for i, text in enumerate(texts):
        question = {"text": text, "gold_ids": ids[:1], "judged_ids": ids}
        if reference and i == 0:
            question["gold_answers"] = ["Ten days"]
        r = httpx.post(f"{seeded.url}/eval-sets/{name}q/questions", json={"question": question})
        assert r.status_code == 201, r.text


def hold_first_store_get(page, name):
    """Hold the store's first GET; later ones go through. Returns the held routes."""
    held = []

    def hold(route):
        if held:
            route.continue_()
        else:
            held.append(route)

    page.route(f"**/collections/{name}", hold)
    return held


def wait_held(page, held):
    for _ in range(100):
        if held:
            return
        page.wait_for_timeout(50)
    raise AssertionError("the store GET was never made")


def test_new_run_waits_for_the_store(page, seeded):
    run_form_set(seeded, "nrw1")
    held = hold_first_store_get(page, "nrw1")
    page.goto(seeded.url + "/c/nrw1")
    wait_held(page, held)
    expect(page.locator("#new-run")).to_be_disabled()
    held[0].continue_()
    expect(page.locator("#new-run")).to_be_enabled()
    page.locator("#new-run").click()
    expect(page.locator("#rf-split")).to_be_visible()


def test_run_against_its_store_picks_the_set_without_toggling_an_open_form_shut(page, seeded):
    run_form_set(seeded, "nrw2")
    held = hold_first_store_get(page, "nrw2")
    page.goto(seeded.url + "/c/nrw2?eval_set=nrw2q")
    wait_held(page, held)
    # A click can't open the form before the store has loaded (col is null), so set the state
    # it would leave by hand: the form open when the pick runs.
    page.evaluate(
        "const f = document.getElementById('run-form');"
        'f.innerHTML = \'<select id="rf-set"><option value=""></option>'
        '<option value="nrw2q">nrw2q</option></select>\'; f.hidden = false'
    )
    held[0].continue_()
    expect(page.locator("#new-run")).to_be_enabled()
    expect(page.locator("#run-form")).to_be_visible()  # the pick must not toggle it shut
    expect(page.locator("#rf-set")).to_have_value("nrw2q")
    page.locator("#new-run").click()
    expect(page.locator("#run-form")).to_be_hidden()


def test_documents_run_form_offers_answers_and_judging(page, seeded):
    run_form_set(seeded, "rfj1")
    page.goto(seeded.url + "/c/rfj1")
    page.get_by_role("button", name="New run").click()
    answer, judge = page.locator("#rf-answer"), page.locator("#rf-judge")
    expect(answer).not_to_be_checked()
    expect(judge).to_be_disabled()
    answer.check()
    expect(judge).to_be_enabled()
    judge.check()
    answer.uncheck()
    expect(judge).to_be_disabled()
    expect(judge).not_to_be_checked()
    answer.check()
    judge.check()

    def estimate_of_judged(r):
        if not (r.url.endswith("/eval") and r.method == "POST"):
            return False
        sent = json.loads(r.post_data)
        return not sent["confirm"] and sent["judge"]

    with page.expect_request(estimate_of_judged) as sent:
        page.get_by_role("button", name="Estimate").click()
    assert json.loads(sent.value.post_data)["answer"] is True
    page.get_by_role("button", name="Start").click()
    expect(page).to_have_url(re.compile(r"/c/rfj1/runs/[^/]+$"))
    runs = httpx.get(f"{seeded.url}/eval?collection=rfj1").json()
    assert len(runs) == 1
    assert seeded.wait(runs[0]["id"])
    assert db.get_job(seeded.app.state.conn, runs[0]["id"])["setup"]["judge"] is True


def test_documents_run_form_says_how_many_questions_have_a_reference_answer(page, seeded):
    run_form_set(seeded, "rfj2")
    page.goto(seeded.url + "/c/rfj2")
    page.get_by_role("button", name="New run").click()
    page.locator("#rf-split").select_option("full")
    refs = page.locator("[data-references]")
    expect(refs).to_have_count(0)
    page.locator("#rf-answer").check()
    expect(refs).to_have_text("1 of 2 questions have a reference answer.")
    page.locator("#rf-judge").check()
    expect(refs).to_have_text("1 of 2 questions have a reference answer; the rest are not judged.")
    page.locator("#rf-answer").uncheck()
    expect(refs).to_have_count(0)


def test_documents_run_form_shows_why_judging_is_refused(page, seeded):
    run_form_set(seeded, "rfj3", reference=False)
    page.expected = ["422"]
    page.goto(seeded.url + "/c/rfj3")
    page.get_by_role("button", name="New run").click()
    page.locator("#rf-split").select_option("full")
    page.locator("#rf-answer").check()
    page.locator("#rf-judge").check()
    expect(page.locator("#rf-msg")).to_have_text(
        "none of the questions in this run has a reference answer to judge against; "
        "add reference answers on the eval set page, or choose All"
    )
    expect(page.locator("#rf-split option[value=full]")).to_have_text("All (2)")
    expect(page.locator("[data-references]")).to_have_count(0)


def test_documents_run_form_drops_an_estimate_that_returns_after_a_change(page, seeded):
    run_form_set(seeded, "rfj4")
    page.goto(seeded.url + "/c/rfj4")
    page.get_by_role("button", name="New run").click()
    page.locator("#rf-answer").check()
    expect(page.locator("[data-references]")).to_be_visible()  # the toggle's recount is in
    held = []

    def hold(route):
        if route.request.method == "POST" and not held:
            held.append(route)
        else:
            route.continue_()

    page.route("**/eval", hold)
    page.get_by_role("button", name="Estimate").click()
    for _ in range(50):
        if held:
            break
        page.wait_for_timeout(20)
    assert len(held) == 1 and json.loads(held[0].request.post_data)["answer"] is True
    page.locator("#rf-answer").uncheck()
    expect(page.locator("[data-references]")).to_have_count(0)
    held[0].continue_()
    page.wait_for_timeout(500)  # the held estimate's answer is back, and dropped
    expect(page.locator("#rf-confirm")).to_be_empty()
    expect(page.get_by_role("button", name="Start")).to_have_count(0)
    assert httpx.get(f"{seeded.url}/eval?collection=rfj4").json() == []


def test_run_form_answer_toggle_makes_no_request_on_other_store_types(page, seeded):
    seeded.build("rfj5")
    page.goto(seeded.url + "/c/rfj5")
    page.get_by_role("button", name="New run").click()
    expect(page.locator("#rf-answer")).to_be_checked()
    page.wait_for_timeout(300)  # the open recount, if any, is done
    sent = []
    page.on("request", lambda r: sent.append(r.url) if r.url.endswith("/eval") else None)
    page.locator("#rf-answer").uncheck()
    page.locator("#rf-answer").check()
    page.wait_for_timeout(300)
    assert sent == []
    expect(page.locator("#rf-msg")).to_be_empty()


JUDGING = lambda p: "correct" if "Answer to grade" in p else "Ten days [1]."  # noqa: E731

JUDGED_LEAD = "How often are its answers right?"


def judged_run(seeded, name, methods=("hybrid", "vector"), **kw):
    with replies(seeded, JUDGING):
        return documents_run(
            seeded, name, list(methods), answer=True, judge=True, references=1, **kw
        )


def test_documents_run_page_shows_judged_answers(page, seeded):
    run_id, _ = judged_run(seeded, "ja1")
    got = httpx.get(f"{seeded.url}/eval/{run_id}").json()
    assert got["headline"] == "ndcg@10"
    page.goto(f"{seeded.url}/c/ja1/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer).to_contain_text(JUDGED_LEAD)
    expect(answer.get_by_text("Hybrid: 1 of 1 answer judged right", exact=True)).to_be_visible()
    expect(
        answer.get_by_text("Vector search: 1 of 1 answer judged right", exact=True)
    ).to_be_visible()
    expect(
        answer.get_by_text("1 question not judged: it has no reference answer", exact=True)
    ).to_be_visible()
    expect(answer.locator("p.verdict strong").first).to_have_text(RUN_LEAD)


def test_documents_run_page_shows_the_answers_only_once_the_run_is_done(page, seeded):
    run_id, _ = judged_run(seeded, "ja2")

    def inject(route):
        data = route.fetch().json()
        data["status"], data["progress"] = "running", 0.5
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.goto(f"{seeded.url}/c/ja2/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer).to_contain_text("The scores appear when the run finishes.")
    expect(answer).not_to_contain_text(JUDGED_LEAD)
    expect(answer).not_to_contain_text("answer judged right")


def test_documents_run_page_counts_unreadable_verdicts(page, seeded):
    run_id, _ = documents_run(seeded, "ja3", ["hybrid"], answer=True, judge=True, references=2)
    page.goto(f"{seeded.url}/c/ja3/runs/{run_id}")
    expect(
        page.locator("#answer").get_by_text(
            "2 answers not judged: the judge's reply couldn't be read", exact=True
        )
    ).to_be_visible()
    page.get_by_text("Questions", exact=True).click()
    page.locator("#questions details").first.locator("summary").click()
    answer = page.locator("[data-ai-answer]").first
    expect(answer.locator("[data-judge]")).to_have_text(
        "(not judged: the judge's reply couldn't be read)"
    )
    expect(answer.locator("[data-judge-raw]")).to_have_text("Judge's reply: Answer: A")


def test_documents_run_page_says_unscored_answers_are_not_scored(page, seeded):
    run_id, _ = documents_run(seeded, "ja4", ["hybrid"], answer=True)
    page.goto(f"{seeded.url}/c/ja4/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer).to_contain_text("The AI's answers are not scored.")
    expect(answer).to_contain_text(
        "To score them, start a new run from the store page with Judge answers ticked. It "
        "judges the questions that have a reference answer; add them on the eval set page."
    )
    expect(answer).not_to_contain_text(JUDGED_LEAD)


def test_documents_run_page_says_judged_answers_of_a_cancelled_run_are_partial(page, seeded):
    run_id, _ = judged_run(seeded, "ja9", methods=("hybrid",))

    def inject(route):
        data = route.fetch().json()
        data["status"] = "cancelled"
        route.fulfill(json=data)

    page.route(f"**/eval/{run_id}", inject)
    page.expected = ["409"]  # the page asks to resume the run, which is really done
    page.goto(f"{seeded.url}/c/ja9/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer.get_by_text("Hybrid: 1 of 1 answer judged right", exact=True)).to_be_visible()
    expect(answer.locator("[data-partial]")).to_have_text(
        "Partial: the run was cancelled before it finished, so this is from the questions it "
        "reached; resume it for the rest."
    )


def test_documents_run_questions_show_each_answer_and_its_verdict(page, seeded):
    run_id, _ = judged_run(seeded, "ja5", methods=("hybrid",))
    page.goto(f"{seeded.url}/c/ja5/runs/{run_id}")
    page.get_by_text("Questions", exact=True).click()
    rows = page.locator("#questions details")
    rows.first.locator("summary").click()
    rows.last.locator("summary").click()
    answers = page.locator("[data-ai-answer]")
    expect(answers).to_have_count(2)
    expect(answers.filter(has_text="(judged right)")).to_have_text(
        "The AI's answer: Ten days. (judged right)"
    )
    expect(answers.filter(has_text="not judged: no reference answer")).to_have_count(1)
    expect(page.locator("[data-judge]")).to_have_count(2)


def test_documents_run_page_with_only_judged_answers(page, seeded):
    with replies(seeded, JUDGING):
        run_id, _ = documents_run(
            seeded, "ja6", ["hybrid"], answer=True, judge=True, references=1, held=False
        )
    page.goto(f"{seeded.url}/c/ja6/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer).to_contain_text(JUDGED_LEAD)
    expect(answer.get_by_text("Hybrid: 1 of 1 answer judged right", exact=True)).to_be_visible()
    expect(
        answer.get_by_text(
            "2 questions unmeasured: their right passages aren't in this store", exact=True
        )
    ).to_be_visible()


def test_documents_rerun_keeps_judge_answers(page, seeded):
    run_id, _ = judged_run(seeded, "ja7", methods=("hybrid",))
    sent = []

    def watch(route):
        if route.request.method == "POST":
            sent.append(route.request.post_data_json)
        route.continue_()

    page.route("**/eval", watch)
    page.goto(f"{seeded.url}/c/ja7/runs/{run_id}")
    page.get_by_role("button", name="Re-run").click()
    expect(page.locator("#act-confirm")).not_to_be_empty()
    assert sent and sent[0]["judge"] is True and sent[0]["confirm"] is False


def test_documents_run_page_without_answers_is_unchanged(page, seeded):
    run_id, _ = documents_run(seeded, "ja8", ["hybrid"])
    assert httpx.get(f"{seeded.url}/eval/{run_id}").json()["answers"] is None
    page.goto(f"{seeded.url}/c/ja8/runs/{run_id}")
    answer = page.locator("#answer")
    expect(answer.locator("p.verdict")).to_have_count(1)
    expect(answer.locator("p.verdict strong")).to_have_text(RUN_LEAD)
    page.get_by_text("Questions", exact=True).click()
    page.locator("#questions details").first.locator("summary").click()
    expect(page.locator("[data-ai-answer]")).to_have_count(0)
    expect(page.locator("[data-judge]")).to_have_count(0)


def test_run_setup_uses_labels(page, seeded):
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Setup this ran with").click()
    setup = page.locator("#setup")
    expect(setup).to_contain_text("Embedding model")
    expect(setup).not_to_contain_text("dense_model")
    expect(setup).not_to_contain_text("leaked_ids")  # an empty list has no row


def test_new_run_disabled_without_eval_set_of_type(page, seeded):
    # A table store: no other test makes a table eval set, so none exists in the shared `seeded`.
    r = httpx.post(f"{seeded.url}/collections", json={"name": "noeval", "qa_type": "table"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/c/noeval")  # only sets of other types exist
    button = page.get_by_role("button", name="New run")
    expect(button).to_be_disabled()
    expect(button).to_have_attribute("title", "Create an eval set first")
    why = page.locator("#new-run-why")
    expect(why).to_have_text("Create an eval set first")
    expect(why.get_by_role("link")).to_have_attribute("href", "/eval-sets?store=noeval")
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": "other-table", "qa_type": "table"})
    assert r.status_code == 201, r.text
    try:
        page.goto(seeded.url + "/c/noeval")  # a set of its type, though not its default
        expect(button).to_be_enabled()
        expect(why).to_have_text("")
    finally:
        httpx.delete(f"{seeded.url}/eval-sets/other-table", params={"confirm": True})
        httpx.delete(f"{seeded.url}/collections/noeval", params={"confirm": True})
