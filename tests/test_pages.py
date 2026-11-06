"""The console pages, driven in a real browser against the app on fakes."""

import json
import re
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from playwright.sync_api import expect

from ragqa_core import db
from ragqa_core.fakes import fake_replies
from ragqa_core.types import Question

pytestmark = pytest.mark.pages

ORIGINAL_TOPICS = "enzyme\nmarket\nvolcano\noptics"  # the topic list `seeded.build` saves


def test_collections_page_shows_the_answer(page, seeded):
    page.goto(seeded.url + "/")
    row = page.locator('main a.row-link[href="/c/c"]')
    expect(row).to_be_visible()
    expect(row).to_contain_text("Keywords per list")
    expect(row).to_contain_text("run " + seeded.run_id[:8])
    expect(row).to_contain_text("Ready")


def test_collections_page_new_collection_form(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("fresh")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("main a.row-link", has_text="fresh")).to_be_visible()


def test_collection_page_sections_in_order(page, seeded):
    page.goto(seeded.url + "/c/c")
    expect(page.locator("main h2").first).to_be_visible()
    titles = page.locator("main h2").all_inner_texts()
    assert titles[:5] == [
        "What's the answer?",
        "Runs",
        "What's in it?",
        "Eval set",
        "How are items described?",
    ]
    assert titles[-1] == "Delete this store"


def test_collection_page_shows_verdict_gap_bar_and_runs(page, seeded):
    page.goto(seeded.url + "/c/c")
    expect(page.locator("main svg[role=img]").first).to_be_visible()
    expect(page.locator("#runs tbody tr")).to_have_count(1)
    expect(page.get_by_text("run 1 on this store", exact=False)).to_be_visible()


def test_lists_save_shows_rekey_count_and_keeps_edits(page, seeded):
    seeded.build("lists")
    page.goto(seeded.url + "/c/lists")
    expect(page.get_by_text("Saving re-keys 3 items, free")).to_be_visible()
    page.locator("#kw-topic").fill("alpha\nbeta")
    page.get_by_role("button", name="Save lists").click()
    expect(page.get_by_text("re-keys 3 items", exact=False).first).to_be_visible()
    page.wait_for_timeout(3500)  # past one fast refresh
    assert page.locator("#kw-topic").input_value() == "alpha\nbeta"


def test_stale_lists_show_in_the_head(page, seeded):
    seeded.build("stale", run=True)
    lists = {"topic": ["x", "y"], "method": ["experiment"], "common": ["rate"]}
    r = httpx.put(seeded.url + "/collections/stale/keyword-lists", json=lists)
    assert r.status_code == 202
    page.goto(seeded.url + "/c/stale")
    expect(page.get_by_text("keyword lists changed after run", exact=False)).to_be_visible(
        timeout=15000
    )


def test_a_run_without_recorded_versions_does_not_warn_of_stale_lists(page, seeded):
    run_id = seeded.build("migrated", run=True)
    conn = seeded.app.state.conn
    setup = db.get_job(conn, run_id)["setup"]
    for key in ("keyword_version", "dense_model"):
        del setup[key]
    db.set_job_setup(conn, run_id, setup)
    lists = {"topic": ["x", "y"], "method": ["experiment"], "common": ["rate"]}
    r = httpx.put(seeded.url + "/collections/migrated/keyword-lists", json=lists)
    seeded.wait(r.json()["job_id"])
    page.goto(seeded.url + "/c/migrated")
    expect(page.get_by_text("run " + run_id[:8]).first).to_be_visible(timeout=15000)
    expect(page.get_by_text("changed after run")).to_have_count(0)


def test_propose_shows_its_estimate_first(page, seeded):
    seeded.build("prop")
    ai = seeded.app.state.ai
    fake_replies(ai, lambda p: '{"topic": ["enzymes"], "method": ["assay"], "common": ["rate"]}')
    try:
        page.goto(seeded.url + "/c/prop")
        expect(page.locator("#kw-topic")).to_have_value("enzyme\nmarket\nvolcano\noptics")
        page.get_by_role("button", name="Propose with AI").click()
        expect(page.get_by_text("AI call", exact=False)).to_be_visible()
        assert page.locator("#kw-topic").input_value().startswith("enzyme\n")  # not proposed yet
        page.get_by_role("button", name="Start").click()
        expect(page.locator("#kw-topic")).to_have_value("enzymes")
    finally:
        fake_replies(ai, lambda p: "Answer: A")


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
    expect(page.locator("#runs tbody tr")).to_have_count(2)


def test_delete_asks_inline_first(page, seeded):
    page.goto(seeded.url + "/c/c")
    page.get_by_role("button", name="Delete…").click()
    expect(page.get_by_role("button", name="Yes, delete c")).to_be_visible()
    page.get_by_role("button", name="Keep it").click()
    expect(page.get_by_role("button", name="Yes, delete c")).to_have_count(0)


def test_gap_bar_is_accessible_and_clamped(page, seeded):
    page.goto(seeded.url + "/")
    svg = page.evaluate("import('/static/ui.js').then(m => m.gapBar(-30, 50, 80, 10))")
    assert 'role="img"' in svg and 'aria-label="Gap +50 pts (−30 to 80)"' in svg
    cx = float(svg.split('<circle cx="')[1].split('"')[0])
    assert cx <= 220  # clamped to the axis, not drawn at 50/10 of the way out


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


def test_pts_and_clock_format(page, seeded):
    page.goto(seeded.url + "/")
    out = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
      return [m.pts(6.2, 1.8, 10.4), m.pts(-7, -12, -2),
              m.clock('2026-10-21T06:02:00Z', new Date('2026-10-23T00:00:00Z'))] }"""
    )
    assert out[0] == "+6 pts (2–10)" and out[1] == "−7 pts (−12 to −2)" and "21 Oct" in out[2]


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


def test_run_delete_asks_inline_then_deletes(page, seeded):
    run_id = seeded.build("gone", run=True)
    page.goto(f"{seeded.url}/c/gone/runs/{run_id}")
    page.get_by_role("button", name="Delete…").click()
    page.get_by_role("button", name="Keep it").click()
    expect(page.get_by_role("button", name="Yes, delete this run")).to_have_count(0)
    page.get_by_role("button", name="Delete…").click()
    page.get_by_role("button", name="Yes, delete this run").click()
    page.wait_for_url(f"{seeded.url}/c/gone")
    assert httpx.get(f"{seeded.url}/eval", params={"collection": "gone"}).json() == []


def interrupt(seeded, run_id):
    db.update_job(seeded.app.state.conn, run_id, status="interrupted")


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


@contextmanager
def worker_stopped(seeded, name):
    """Stop the shared worker and queue an eval on its own collection `name`. Always leaves the
    worker running again with that job no longer queued (R18)."""
    worker = seeded.app.state.worker
    seeded.build(name)  # its lists are keyed by the worker, so before stopping it
    worker.stop()
    try:
        yield seeded.queue_eval(name)
    finally:
        try:
            for job in httpx.get(f"{seeded.url}/jobs", params={"status": "queued"}).json():
                httpx.post(f"{seeded.url}/jobs/{job['id']}/cancel")
        finally:
            worker.start()


def test_worker_page_shows_problems_and_queue(page, seeded):
    with worker_stopped(seeded, "wq"):
        page.goto(seeded.url + "/worker")
        expect(
            page.locator("[data-problem]", has_text="The worker thread has stopped")
        ).to_be_visible()
        expect(page.locator("[data-queued-job]")).to_have_count(1)
        expect(page.locator("[data-queued-job]")).to_contain_text("Evaluating wq")


def test_worker_page_cancels_a_queued_job(page, seeded):
    with worker_stopped(seeded, "wc") as job_id:
        page.goto(seeded.url + "/worker")
        page.locator("[data-queued-job] button", has_text="Cancel").click()
        expect(page.locator("[data-queued-job]")).to_have_count(0)
        expect(page.locator(f"[data-finished-job='{job_id}']")).to_contain_text("Cancelled")
        assert httpx.get(f"{seeded.url}/jobs/{job_id}").json()["status"] == "cancelled"


def test_worker_page_resume_estimates_first_and_shows_a_refusal(page, seeded):
    ok = seeded.build("wr", run=True)
    interrupt(seeded, ok)
    page.goto(seeded.url + "/worker")
    page.locator(f"[data-finished-job='{ok}'] button", has_text="Resume").click()
    expect(page.get_by_text("AI call", exact=False)).to_be_visible()
    assert httpx.get(f"{seeded.url}/jobs/{ok}").json()["status"] == "interrupted"
    page.expected = ["409"]
    r = httpx.put(
        f"{seeded.url}/collections/wr/keyword-lists",
        json={"topic": ["x", "y"], "method": ["experiment"], "common": ["rate"]},
    )
    seeded.wait(r.json()["job_id"])
    page.locator(f"[data-finished-job='{ok}'] button", has_text="Resume").click()
    expect(page.locator("[data-refusal]")).to_be_visible()
    assert httpx.get(f"{seeded.url}/jobs/{ok}").json()["status"] == "interrupted"


def test_worker_page_poll_error_clears_when_it_recovers(page, seeded):
    page.expected = ["503"]
    failed = []

    def fail_once(route):
        if len(failed) >= 2:  # the sidebar dot and the page both ask on load
            route.continue_()
        else:
            failed.append(1)
            route.fulfill(status=503, json={"detail": "busy"})

    page.route("**/status", fail_once)
    page.goto(seeded.url + "/worker")
    expect(page.locator("#poll-err")).to_contain_text("Couldn't refresh at")
    expect(page.locator("#poll-err")).to_have_text("", timeout=8000)


def test_one_path_end_to_end(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("e2e")
    page.get_by_role("button", name="Create").click()
    page.locator("main a.row-link", has_text="e2e").click()
    expect(page).to_have_url(seeded.url + "/c/e2e")
    expect(page.get_by_text("No answer yet").first).to_be_visible()
    page.goto(seeded.url + "/worker")
    expect(page.locator("#main")).to_be_visible()


# -- final fix round, batch B ------------------------------------------------------


def route_load(page, name, reply):
    """Answer POST .../{name}/load with `reply(body)` -> (status, json); returns the bodies seen."""
    seen = []

    def handle(route):
        if route.request.method != "POST":
            return route.continue_()
        body = route.request.post_data_json
        seen.append(body)
        status, payload = reply(body)
        route.fulfill(status=status, json=payload)

    page.route(f"**/collections/{name}/load", handle)
    return seen


def test_collection_load_shows_its_estimate_and_queues_only_on_start(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "est", "qa_type": "mcq"})
    estimate = {"est_usd": 0.12, "items": 50, "calls": 51}
    seen = route_load(
        page,
        "est",
        lambda b: (
            (200, {"estimate": estimate}) if not b.get("confirm") else (202, {"job_id": "ab12cd"})
        ),
    )
    page.goto(seeded.url + "/c/est")
    page.get_by_role("button", name="Load scienceqa").click()
    expect(page.get_by_text("51 AI calls")).to_be_visible()
    assert [b.get("confirm") for b in seen] == [False]
    page.get_by_role("button", name="Start").click()
    expect(page.get_by_text("Queued (job ab12cd)")).to_be_visible()
    assert [b.get("confirm") for b in seen] == [False, True]


def test_collection_load_that_is_free_queues_at_once(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "freeload", "qa_type": "mcq"})
    seen = route_load(page, "freeload", lambda b: (202, {"job_id": "ee99ff"}))
    page.goto(seeded.url + "/c/freeload")
    page.get_by_role("button", name="Load scienceqa").click()
    expect(page.get_by_text("Queued (job ee99ff)")).to_be_visible()
    assert len(seen) == 1


def test_setup_benchmarks_sums_one_estimate_and_confirms_the_paying_loads(page, seeded):
    for n in ("sqa", "mq", "tq", "sb"):
        assert n not in [c["name"] for c in httpx.get(f"{seeded.url}/collections").json()]
    paying = {"sqa": 0.10, "tq": 0.25}
    seen = {n: route_load(page, n, _reply(paying, n)) for n in ("sqa", "mq", "tq", "sb")}
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="Set up the four benchmark stores").click()
    expect(page.get_by_text("5 AI calls")).to_be_visible()
    expect(page.get_by_text("$0.35")).to_be_visible()
    expect(page.get_by_role("button", name="Start")).to_have_count(1)
    assert [b.get("confirm") for b in seen["sqa"]] == [None] and len(seen["tq"]) == 1
    page.get_by_role("button", name="Start").click()
    expect(page.get_by_text("sqa: loading scienceqa (queued, job sqa01234)")).to_be_visible()
    expect(page.get_by_text("tq: loading tatqa (queued, job tq012345)")).to_be_visible()
    assert [b.get("confirm") for b in seen["sqa"]] == [None, True]
    assert [b.get("confirm") for b in seen["tq"]] == [None, True]
    assert len(seen["mq"]) == 1  # already queued by its first answer, never posted twice


def test_new_collection_form_defaults_to_science(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    select = page.locator("#f-type")
    expect(select).to_have_value("science")
    expect(select.locator("option:checked")).to_contain_text("Science problems")


def test_science_collection_offers_scibench_and_runs(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "sci0", "qa_type": "science"})
    page.goto(seeded.url + "/c/sci0")
    expect(page.get_by_role("button", name="Load scibench")).to_be_visible()
    run_id = seeded.build("sci", run=True, qa_type="science")
    page.goto(f"{seeded.url}/c/sci/runs/{run_id}")
    page.get_by_text("Each method's own score").click()
    expect(page.get_by_text("Example tests the same skill").first).to_be_visible()


def _reply(paying, name):
    def reply(body):
        if name in paying and not body.get("confirm"):
            return 200, {
                "estimate": {
                    "est_usd": paying[name],
                    "items": 1,
                    "calls": 2 if name == "sqa" else 3,
                }
            }
        return 202, {"job_id": f"{name}012345"}

    return reply


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


def test_collection_offers_finish_loading_after_an_interrupted_load(page, seeded):
    seeded.build("fl")
    job = [{"id": "dead0000", "kind": "load", "collection": "fl", "status": "interrupted"}]
    page.route("**/jobs?*kind=load*", lambda route: route.fulfill(json=job))
    page.goto(seeded.url + "/c/fl")
    expect(page.get_by_role("button", name="Finish loading")).to_be_visible()


def test_collection_without_questions_is_not_offered_the_benchmark(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "noq", "qa_type": "mcq"})
    r = httpx.post(
        f"{seeded.url}/collections/noq/documents", json={"items": ITEMS_ONE, "confirm": True}
    )
    assert r.status_code == 200, r.text
    page.goto(seeded.url + "/c/noq")
    # holding items, a collection with no questions is not offered the benchmark again
    expect(page.get_by_role("button", name="Finish loading")).to_have_count(0)
    page.goto(seeded.url + "/c/c")
    expect(page.get_by_role("button", name="Finish loading")).to_have_count(0)


ITEMS_ONE = [
    {"id": "one", "text": "enzyme kinetics", "answer": "A", "explanation": "x", "meta": {}}
]


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


def test_confirm_spend_prints_sub_cent_and_zero(page, seeded):
    page.goto(seeded.url + "/")
    out = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
      return [0.004, 0, 0.41].map((usd) => { const el = document.createElement('div');
        m.confirmSpend(el, { est_usd: usd, calls: 2 }, () => {}); return el.textContent; }) }"""
    )
    assert "< $0.01" in out[0] and "≈ $0.00" in out[1] and "< $0.01" not in out[1]
    assert "$0.41" in out[2]


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


def test_keyword_rag_costs_money_only_when_the_ai_picks_its_tags(page, seeded):
    page.goto(seeded.url + "/")
    marks = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
        const cat = await (await fetch('/catalog')).json();
        return ['model', 'classifier'].map(t =>
          m.costMark('keyword_rag', {summarize: false, search: {query_tags: t}}, cat)); }"""
    )
    assert "costs money" in marks[0] and marks[1] == ""


def test_keyword_methods_say_costs_money_on_a_summarising_collection(page, seeded):
    seeded.build("sumcol", summarize=True)
    page.goto(seeded.url + "/c/sumcol")
    page.get_by_role("button", name="New run").click()
    expect(cost_label(page, "kw_individual")).to_have_text("costs money")
    expect(cost_label(page, "iterative")).to_have_text("costs money")


def test_lab_marks_paid_methods_and_the_answer_box(page, seeded):
    page.goto(seeded.url + "/lab")
    expect(page.locator("#methods label", has_text="iterative").locator(".cost")).to_have_text(
        "costs money"
    )
    expect(page.locator("#methods label", has_text="hybrid").first.locator(".cost")).to_have_count(
        0
    )
    expect(page.locator("#methods label", has_text="kw_individual").locator(".cost")).to_have_count(
        0
    )  # `c` has no summaries
    expect(page.locator("label", has=page.locator("#answer")).locator(".cost")).to_have_text(
        "costs money"
    )


def test_lab_shows_keyword_rags_question_tags_hit_tags_and_channels(page, seeded):
    hit = {
        "id": "i1", "text": "A ball falls.", "score": 0.5, "keywords": {}, "matched": ["drag"],
        "answer": None, "explanation": None, "meta": {}, "tags": ["drag", "energy"],
        "channels": ["meaning", "keywords"],
    }  # fmt: skip
    empty = {"hits": [], "answer": None, "usd": 0.0, "error": None, "query_tags": []}
    results = {"keyword_rag": {**empty, "hits": [hit], "query_tags": ["drag"]}}
    page.route("**/query", lambda route: route.fulfill(json={"results": results}))
    page.goto(seeded.url + "/lab")
    page.locator("#question").fill("how fast does it fall")
    page.locator("#run").click()
    tags = page.locator("[data-query-tags]")
    expect(tags).to_contain_text("Question tags")
    expect(tags.locator(".chip.on")).to_have_text(["drag"])
    card = page.locator(".hit")
    expect(card.locator(".chip")).to_have_text(["drag", "energy"])
    expect(card.locator(".chip.on")).to_have_text(["drag"])  # the tag the question shares
    expect(card.locator("[data-channels]")).to_have_text("found by meaning, keywords")
    results["keyword_rag"] = empty  # a question with no tags says what that means
    page.locator("#run").click()
    expect(tags).to_contain_text("none, so the keywords channel found nothing")


def test_lab_marks_keyword_methods_on_a_summarising_collection(page, seeded):
    seeded.build("labsum", summarize=True)
    page.goto(seeded.url + "/lab")
    page.locator("#collection").select_option("labsum")
    expect(page.locator("#methods label", has_text="kw_individual").locator(".cost")).to_have_text(
        "costs money"
    )


def test_lab_renders_its_method_picker_for_a_collection(page, seeded):
    page.goto(seeded.url + "/lab")
    expect(page.locator("#collection option", has_text="c")).not_to_have_count(0)
    picker = page.locator("#methods input[type=checkbox]")
    expect(picker.first).to_be_visible()
    expect(page.locator("#methods input[value=hybrid]")).to_be_checked()
    expect(page.locator("#methods input[value=iterative]")).not_to_be_checked()


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


def test_lists_history_loads_an_old_version_and_keeps_it(page, seeded):
    seeded.build("hist")
    page.goto(seeded.url + "/c/hist")
    expect(page.locator("#lists-history li")).to_have_count(1)
    expect(page.locator("#lists-history li").first).to_contain_text("active")
    page.locator("#kw-topic").fill("alpha\nbeta")
    page.get_by_role("button", name="Save lists").click()
    expect(page.locator("#lists-history li")).to_have_count(2)
    page.locator("#lists-history li").nth(1).get_by_role("button", name="Load into editor").click()
    expect(page.locator("#kw-topic")).to_have_value(ORIGINAL_TOPICS)
    page.wait_for_timeout(3500)  # past one fast refresh
    assert page.locator("#kw-topic").input_value() == ORIGINAL_TOPICS


def test_sidebar_shows_the_first_problem_under_the_worker_dot(page, seeded):
    with worker_stopped(seeded, "sp"):
        page.goto(seeded.url + "/")
        problem = page.locator("#worker-problem")
        expect(problem).to_contain_text("The worker thread has stopped")
        expect(problem).to_have_attribute("href", "/worker")


def test_sidebar_counts_further_problems_and_escapes_the_text(page, seeded):
    def problems(route):
        data = route.fetch().json()
        data["problems"] = ["<b>first</b> trouble", "second", "third"]
        route.fulfill(json=data)

    page.route("**/status", problems)
    page.goto(seeded.url + "/")
    box = page.locator("#worker-problem")
    expect(box).to_contain_text("<b>first</b> trouble")
    expect(box).to_contain_text("and 2 more")
    expect(box.locator("b")).to_have_count(0)


def test_sidebar_shows_no_problem_text_when_there_are_none(page, seeded):
    def none(route):
        data = route.fetch().json()
        data["problems"] = []
        route.fulfill(json=data)

    page.route("**/status", none)
    page.goto(seeded.url + "/")
    expect(page.locator("#worker-dot")).to_have_class("dot ok")
    expect(page.locator("#worker-problem")).to_be_empty()


# -- console polish, batch E (R58-R60) ---------------------------------------------


def test_a_queued_run_waits_for_its_answer_and_offers_only_cancel(page, seeded):
    with worker_stopped(seeded, "qr") as run_id:
        page.goto(f"{seeded.url}/c/qr/runs/{run_id}")
        expect(page.locator("#answer")).to_contain_text("The answer appears when the run finishes")
        expect(page.locator("#gaps")).to_contain_text("The gaps appear when the run finishes")
        expect(page.locator("#head")).to_contain_text("– questions")
        expect(page.locator("#head")).not_to_contain_text("0 questions")
        expect(page.get_by_text("does not include")).to_have_count(0)
        expect(page.get_by_role("button", name="Cancel")).to_be_visible()
        expect(page.get_by_role("button", name="Delete…")).to_have_count(0)
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
    expect(page.get_by_role("button", name="Delete…")).to_be_visible()


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


def test_runs_table_says_what_each_run_said(page, seeded):
    page.goto(seeded.url + "/c/c")
    said = page.locator("[data-said]").first
    expect(said).to_have_text(re.compile(r"^(Yes|No|Can't tell) [+−]\d+ pts?"))
    expect(said.locator("strong.word")).to_have_count(1)
    page.goto(seeded.url + "/c/vo")
    expect(page.locator("[data-said]").first).to_have_text("–")


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


def test_worker_page_links_jobs_and_problems_to_their_pages(page, seeded):
    with worker_stopped(seeded, "wl") as run_id:

        def problem(route):
            data = route.fetch().json()
            data["problems"].append(f"Eval {run_id[:8]} on wl failed")
            route.fulfill(json=data)

        page.route("**/status", problem)
        page.goto(seeded.url + "/worker")
        queued = page.locator(f"[data-queued-job='{run_id}'] a")
        expect(queued).to_have_attribute("href", f"/c/wl/runs/{run_id}")
        expect(queued).to_contain_text("Evaluating wl")
        expect(page.locator("[data-finished-job] a[href='/c/wl']").first).to_be_visible()  # re-key
        expect(page.locator("[data-problem] a", has_text="on wl failed")).to_have_attribute(
            "href", f"/c/wl/runs/{run_id}"
        )
        expect(page.locator("#queued thead th")).to_have_count(3)
        expect(page.locator("#finished thead th")).to_have_count(4)
        expect(page.locator("#worker-dot")).to_have_attribute("aria-label", "needs attention")


def test_worker_resume_estimate_shows_in_the_clicked_row(page, seeded):
    ok = seeded.build("wrow", run=True)
    interrupt(seeded, ok)
    polls = {"n": 0}

    def changing(route):  # each poll changes a finished row, so the table really repaints
        response = route.fetch()
        jobs = response.json()
        polls["n"] += 1
        for job in jobs:
            if job["id"] == ok:
                job["updated_at"] = f"2026-10-27T10:{polls['n'] % 60:02d}:00"
        route.fulfill(response=response, json=jobs)

    page.route("**/jobs?limit=200", changing)
    page.goto(seeded.url + "/worker")
    page.locator(f"[data-finished-job='{ok}'] button", has_text="Resume").click()
    box = page.locator(f"[data-finished-job='{ok}'] + tr[data-resume-row]")
    expect(box).to_contain_text("AI call")
    expect(box.get_by_role("button", name="Start")).to_be_in_viewport()
    seen = polls["n"]
    page.wait_for_timeout(5500)  # past a poll that repaints: the estimate and its Start stay put
    assert polls["n"] > seen
    expect(box.get_by_role("button", name="Start")).to_be_visible()
    assert httpx.get(f"{seeded.url}/jobs/{ok}").json()["status"] == "interrupted"


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
    page.wait_for_timeout(3500)  # past a poll: the filters must still work on a running run
    page.locator("#q-method").select_option("hybrid")
    page.get_by_label("Wrong only").check()
    expect(page.locator("#questions details")).to_have_count(1)
    assert seen["false"] == 3  # rows still coming: fetched each time, never kept
    busy["on"] = False  # finishes: the open view reloads once, then filters from what it kept
    page.wait_for_timeout(3500)  # the next poll sees it done
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


def lab_methods(page):
    return page.locator("#methods input[type=checkbox]")


def test_lab_method_list_follows_the_last_pick_when_responses_race(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "labbare", "qa_type": "mcq"})
    held = []
    holding = {"on": False}

    def hold_c(route):
        if holding["on"]:
            held.append(route)
        else:
            route.continue_()

    page.route("**/collections/c", hold_c)
    page.goto(seeded.url + "/lab")
    page.locator("#collection").select_option("c")
    expect(page.locator("#methods input[value=kw_individual]")).to_have_count(1)
    expect(page.locator("fieldset legend", has_text="Methods")).to_be_visible()
    # A -> B -> A: c's answer is held while the pick moves to labbare, back to c, then labbare.
    holding["on"] = True
    page.locator("#collection").select_option("labbare")
    expect(page.locator("#methods input[value=kw_individual]")).to_have_count(0)  # no lists
    page.locator("#collection").select_option("c")
    page.locator("#collection").select_option("labbare")
    expect(lab_methods(page).first).to_be_visible()
    page.wait_for_timeout(300)
    assert held, "the request for c was never held"
    for route in held:
        route.continue_()
    page.wait_for_timeout(500)
    expect(page.locator("#methods input[value=kw_individual]")).to_have_count(0)
    # and the other way: labbare is picked last but answered first, c's late answer loses
    holding["on"] = True
    held.clear()
    page.locator("#collection").select_option("c")
    page.locator("#collection").select_option("labbare")
    page.wait_for_timeout(300)
    for route in held:
        route.continue_()
    page.wait_for_timeout(500)
    expect(page.locator("#methods input[value=kw_individual]")).to_have_count(0)
    # A -> B -> A with A's first answer arriving last: the newest pick of c draws, not the old one
    held.clear()
    page.locator("#collection").select_option("c")
    page.locator("#collection").select_option("labbare")
    page.locator("#collection").select_option("c")
    page.wait_for_timeout(300)
    assert len(held) == 2
    held[1].continue_()
    expect(page.locator("#methods input[value=kw_individual]")).to_have_count(1)
    holding["on"] = False
    held[0].continue_()
    page.wait_for_timeout(500)
    expect(page.locator("#methods input[value=kw_individual]")).to_have_count(1)
    expect(page.locator("#collection")).to_have_value("c")


# -- your own corpus ---------------------------------------------------------------------------


def jobs_done(seeded, name):
    for j in httpx.get(f"{seeded.url}/jobs", params={"collection": name, "kind": "import"}).json():
        assert seeded.wait(j["id"])["status"] == "done"


def upload_and_preview(page, kind, path):
    page.locator(f"#imp-{kind}-file").set_input_files(str(path))
    page.locator(f"#imp-{kind}-go").click()


def sources(seeded, name):
    return httpx.get(f"{seeded.url}/collections/{name}/sources").json()


def test_import_items_from_a_file_then_switch_the_source_off(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "imp", "qa_type": "science"})
    r = httpx.post(
        f"{seeded.url}/collections/imp/documents", json={"items": ITEMS_ONE, "confirm": True}
    )
    assert r.status_code == 200, r.text
    rows = [{"id": f"n{i}", "text": f"note {i} about <b>reefs</b>"} for i in range(3)]
    notes = tmp_path / "notes.jsonl"
    notes.write_text("\n".join(json.dumps(x) for x in rows))
    page.goto(seeded.url + "/c/imp")
    expect(page.locator("#imp-items-source")).to_have_value("")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("3 rows")
    expect(page.locator("#imp-items-source")).to_have_value("notes")  # named by the upload
    expect(preview.locator("table tbody tr")).to_have_count(3)
    expect(preview.locator("b")).to_have_count(0)  # file text is escaped
    page.locator("#imp-items-import").click()
    # no keyword lists yet, so the import pays to make them: it asks before it queues
    page.locator("#imp-items-confirm").get_by_role("button", name="Start").click()
    expect(page.get_by_text(re.compile(r"Queued \(job \w{8}\)"))).to_be_visible()
    jobs_done(seeded, "imp")
    page.reload()
    row = page.locator("#sources tbody tr", has_text="upload:notes")
    expect(row).to_contain_text("3")
    box = row.get_by_role("checkbox")
    expect(box).to_be_checked()
    box.uncheck()
    expect(box).not_to_be_checked()
    got = {s["source"]: s["enabled"] for s in sources(seeded, "imp")}
    assert got["upload:notes"] is False
    # the last source on cannot be switched off, and the API's words show
    page.expected = ["409"]
    page.locator("#sources tbody tr", has_text="hand").get_by_role("checkbox").uncheck()
    expect(page.locator("#sources-msg")).to_contain_text("at least one source must stay on")
    # delete asks first, with the question in the brief's words
    page.locator("#sources tbody tr", has_text="upload:notes").get_by_role("button").click()
    expect(page.locator("#sources-confirm")).to_contain_text(
        "Delete source upload:notes: its 3 items?"
    )
    page.get_by_role("button", name="Yes, delete upload:notes").click()
    expect(page.locator("#sources tbody tr", has_text="upload:notes")).to_have_count(0)


def test_an_unpriced_items_import_asks_before_it_queues(page, seeded, tmp_path):
    seeded.build("unp")  # keyword lists made: the import itself is free on the server

    def unpriced(route):
        body = route.fetch().json()
        body["estimate"] = {**body["estimate"], "est_usd": None}
        route.fulfill(json=body)

    page.route("**/collections/unp/uploads?*", unpriced)
    imports = []
    page.on("request", lambda r: imports.append(r) if r.url.endswith("/imports") else None)
    notes = tmp_path / "notes.jsonl"
    notes.write_text(json.dumps({"id": "u1", "text": "a note about reefs"}))
    page.goto(seeded.url + "/c/unp")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    expect(page.locator("#imp-items-preview")).to_contain_text("1 row")
    page.locator("#imp-items-import").click()
    confirm = page.locator("#imp-items-confirm")
    expect(confirm).to_contain_text("cost unknown: no AI model is set up")
    page.wait_for_timeout(200)
    assert imports == []  # nothing queued until Start
    confirm.get_by_role("button", name="Start").click()
    expect(page.get_by_text(re.compile(r"Queued \(job \w{8}\)"))).to_be_visible()
    assert len(imports) == 1 and json.loads(imports[0].post_data)["confirm"] is True
    jobs_done(seeded, "unp")


def test_import_questions_on_the_eval_set_page_then_include_leaked(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "qimp", "qa_type": "mcq"})
    httpx.post(f"{seeded.url}/eval-sets", json={"name": "qimp", "qa_type": "mcq"})
    rows = [
        {"id": "u1", "text": "tides follow the moon", "choices": ["x", "y"], "answer": "A"},
        {"id": "u2", "text": "an unrelated one", "choices": ["x", "y"], "answer": "B"},
    ]
    qs = tmp_path / "qs.jsonl"
    qs.write_text("\n".join(json.dumps(x) for x in rows))
    page.goto(seeded.url + "/e/qimp")
    upload_and_preview(page, "questions", qs)
    expect(page.locator("#imp-questions-preview")).to_contain_text("2 rows")
    page.locator("#imp-questions-import").click()
    expect(page.locator("#imp-questions-preview")).to_contain_text("2 new")
    expect(page.locator("#questions tbody tr")).to_have_count(2)
    expect(page.locator("#fields")).to_contain_text("Worked solutions: 0 of 2")
    expect(page.locator("#fields")).to_contain_text("Evidence ids: —")
    page.get_by_label("Include leaked questions").check()
    expect(page.locator("#leaked-msg")).to_have_text("")
    expect(page.get_by_label("Include leaked questions")).to_be_checked()
    assert httpx.get(f"{seeded.url}/eval-sets/qimp").json()["include_leaked"] is True


def test_own_science_collection_offers_import_and_benchmark(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "own", "qa_type": "science"})
    page.goto(seeded.url + "/c/own")
    expect(page.locator("#imp-items-file")).to_be_visible()
    expect(page.locator("#imp-questions-file")).to_have_count(0)
    expect(page.get_by_text("Or load the built-in benchmark, scibench")).to_be_visible()
    expect(page.get_by_role("button", name="Load scibench")).to_be_visible()
    notes = tmp_path / "own notes!.csv"
    notes.write_text("text\nfirst thing about reefs\nsecond thing about tides\n")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    expect(page.locator("#imp-items-source")).to_have_value("own notes")
    page.locator("#imp-items-import").click()
    # no keyword lists yet, so the import pays to make them: it asks before it queues
    page.locator("#imp-items-confirm").get_by_role("button", name="Start").click()
    expect(page.get_by_text(re.compile(r"Queued \(job \w{8}\)"))).to_be_visible()
    jobs_done(seeded, "own")
    page.reload()
    expect(page.locator("#imp-items-file")).to_be_visible()
    expect(page.get_by_text("built-in benchmark")).to_have_count(0)
    expect(page.get_by_role("button", name="Load scibench")).to_have_count(0)


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


def test_the_store_upload_names_the_source_as_the_chat_does(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "named", "qa_type": "science"})
    f = tmp_path / "Q3 (final).md"
    f.write_text("# Q3\nrevenue grew")
    page.goto(seeded.url + "/c/named")
    upload_and_preview(page, "items", f)
    expect(page.locator("#imp-items-source")).to_have_value("Q3 -final")
    # a name typed before the upload is kept, and priced under that name
    other = tmp_path / "other.md"
    other.write_text("more text")
    page.locator("#imp-items-file").set_input_files(str(other))
    expect(page.locator("#imp-items-source")).to_have_value("")
    page.locator("#imp-items-source").fill("mine")
    with page.expect_request(
        lambda r: r.url.endswith("/collections/named/imports") and r.method == "POST"
    ) as sent:
        page.locator("#imp-items-go").click()
    assert json.loads(sent.value.post_data)["source_name"] == "mine"
    expect(page.locator("#imp-items-source")).to_have_value("mine")


def test_a_new_file_clears_the_old_preview(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "swap", "qa_type": "science"})
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "a1", "text": "first file text"}))
    b.write_text(json.dumps({"id": "b1", "text": "second file text"}))
    page.goto(seeded.url + "/c/swap")
    upload_and_preview(page, "items", a)
    expect(page.locator("#imp-items-import")).to_be_visible()
    page.locator("#imp-items-file").set_input_files(str(b))
    expect(page.locator("#imp-items-preview")).to_be_empty()
    expect(page.locator("#imp-items-import")).to_have_count(0)
    page.locator("#imp-items-go").click()
    expect(page.locator("#imp-items-preview")).to_contain_text("second file text")
    expect(page.locator("#imp-items-import")).to_be_visible()


def test_items_without_a_text_column_cannot_be_imported(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "notext", "qa_type": "science"})
    f = tmp_path / "t.csv"
    f.write_text("title,answer\nreefs,3\n")
    page.goto(seeded.url + "/c/notext")
    upload_and_preview(page, "items", f)
    expect(page.locator("#imp-items-preview")).to_contain_text("Map these fields first")
    expect(page.locator("#imp-items-import")).to_be_disabled()
    expect(page.get_by_text("text (required)")).to_be_visible()


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


def test_a_new_source_name_gets_its_own_estimate_before_import_is_allowed(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "ren", "qa_type": "science"})
    notes = tmp_path / "notes.csv"
    notes.write_text("text\nfirst thing about reefs\nsecond thing about tides\n")
    page.goto(seeded.url + "/c/ren")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    expect(page.locator("#imp-items-import")).to_be_enabled()
    held = []
    page.route("**/collections/ren/imports", lambda route: held.append(route))
    page.locator("#imp-items-source").fill("renamed")
    expect(page.locator("#imp-items-import")).to_be_disabled()
    for _ in range(50):
        if held:
            break
        page.wait_for_timeout(20)
    assert len(held) == 1
    sent = json.loads(held[0].request.post_data)
    assert sent["source_name"] == "renamed" and sent["confirm"] is False
    data = held[0].fetch().json()
    data["estimate"]["calls"] = 7
    held[0].fulfill(json=data)
    expect(page.locator("#imp-items-import")).to_be_enabled()
    expect(page.locator("#imp-items-preview")).to_contain_text("7 AI calls")


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


def test_run_page_lists_what_the_run_left_out(page, seeded):
    def left_out(route):
        data = route.fetch().json()
        data["setup"].update(disabled_sources=["upload:notes"], evidence_off=2, leaked_excluded=3)
        route.fulfill(json=data)

    page.route(f"**/eval/{seeded.run_id}", left_out)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    page.get_by_text("Setup this ran with").click()
    left = page.locator("#setup")
    expect(left).to_contain_text("Retrieval left out: upload:notes")
    expect(left).to_contain_text("2 questions' evidence is switched off")
    expect(left).to_contain_text("3 leaked questions left out")


# -- Settings page ------------------------------------------------------------------------


def ai_state(seeded):
    return httpx.get(seeded.url + "/ai").json()


def add_connection(page, seeded, preset, id, key=""):
    page.goto(seeded.url + "/settings")
    page.locator("#add-preset").select_option(label=preset)
    page.locator("#add-id").fill(id)
    if key:
        page.locator("#add-key").fill(key)
    page.get_by_role("button", name="Add connection").click()
    row = page.locator(f'#connections [data-connection="{id}"]')
    expect(row).to_be_visible()
    return row


def test_settings_page_adds_a_connection_whose_models_need_a_price(page, seeded, monkeypatch):
    ai = seeded.app.state.ai
    monkeypatch.setattr(ai, "_list_models", lambda connection: ["llama3"])
    monkeypatch.setattr(
        ai, "_send", lambda connection, model_id, system, prompt, max_tokens: ("OK", 3, 1)
    )
    row = add_connection(page, seeded, "OpenRouter", "router", key="sk-or-TEST123456")
    expect(row).to_contain_text("OpenRouter")
    row.get_by_role("button", name="Fetch models").click()
    expect(page.locator('#models [data-model="router/llama3"]')).to_be_visible()
    # Fetched at $0, so the Test call can go at once.
    row.get_by_role("button", name="Test").click()
    expect(row.locator(".result")).to_contain_text("Worked with llama3")


def test_settings_page_shows_a_connection_answered_by_hand_without_key_or_test(page, seeded):
    row = add_connection(page, seeded, "Answered by hand", "hand")
    expect(row).to_contain_text("ragqa_core.replies")
    for name in ("Test", "Fetch models", "Save key"):
        expect(row.get_by_role("button", name=name)).to_have_count(0)


def test_settings_page_never_shows_key(page, seeded):
    row = add_connection(page, seeded, "OpenAI", "keyed", key="sk-proj-SECRET123456")
    expect(row).to_contain_text("…3456")
    assert "SECRET" not in page.content()
    assert page.locator("#add-key").input_value() == ""
    page.reload()
    expect(page.locator('#connections [data-connection="keyed"]')).to_be_visible()
    assert "SECRET" not in page.content()


def test_a_fetched_model_starts_at_zero_and_takes_a_listed_typed_or_no_price(
    page, seeded, monkeypatch
):
    monkeypatch.setattr(seeded.app.state.ai, "_list_models", lambda connection: ["gpt-new"])
    row = add_connection(page, seeded, "OpenAI", "remote", key="sk-remote-key-0000")
    row.get_by_role("button", name="Fetch models").click()
    model = page.locator('#models [data-model="remote/gpt-new"]')
    price = model.locator("select[name=price]")
    expect(price).to_have_value("0/0")  # $0 by default, and usable at once
    expect(model.locator("[data-free]")).to_contain_text("the budget does not limit it")
    expect(model.locator("input[name=price_in]")).to_be_hidden()
    roles = page.locator("#roles select").first
    expect(roles.locator("option", has_text="gpt-new")).to_have_count(1)

    def save():
        with page.expect_response(
            lambda r: r.request.method == "PUT" and "remote/gpt-new" in r.url
        ):
            model.get_by_role("button", name="Save").click()
        return next(m for m in ai_state(seeded)["models"] if m["ref"] == "remote/gpt-new")

    price.select_option(label="Haiku 5.5 rates: $0.1 in, $0.5 out")
    stored = save()
    assert (stored["price_in"], stored["price_out"]) == (0.1, 0.5)
    expect(model.locator("[data-free]")).to_have_count(0)
    price.select_option(label="Enter by hand…")
    expect(model.locator("input[name=price_in]")).to_be_visible()
    model.locator("input[name=price_in]").fill("1")
    model.locator("input[name=price_out]").fill("2")
    stored = save()
    assert (stored["price_in"], stored["price_out"]) == (1, 2)
    expect(price).to_have_value("hand")  # a typed price that is not listed stays typed
    price.select_option(label="No price (can't be used)")
    stored = save()
    assert stored["price_in"] is None
    expect(model).to_contain_text("needs a price before use")
    expect(roles.locator("option", has_text="gpt-new")).to_have_count(0)


def test_half_a_price_is_not_sent_and_typing_is_kept(page, seeded, monkeypatch):
    monkeypatch.setattr(seeded.app.state.ai, "_list_models", lambda connection: ["gpt-half"])
    row = add_connection(page, seeded, "OpenAI", "half", key="sk-half-key-00000")
    row.get_by_role("button", name="Fetch models").click()
    model = page.locator('#models [data-model="half/gpt-half"]')
    expect(model).to_be_visible()
    page.locator('#models [data-connection="anthropic"] input[name=new-id]').fill("typed-id")
    model.locator("select[name=price]").select_option(label="Enter by hand…")
    model.locator("input[name=price_in]").fill("1")
    model.locator("input[name=price_out]").fill("")  # the boxes start at the price in force
    model.get_by_role("button", name="Save").click()
    expect(page.locator("#msg")).to_contain_text("Give both prices")
    assert model.locator("input[name=price_in]").input_value() == "1"
    assert (
        page.locator('#models [data-connection="anthropic"] input[name=new-id]').input_value()
        == "typed-id"
    )
    stored = next(m for m in ai_state(seeded)["models"] if m["ref"] == "half/gpt-half")
    assert stored["price_in"] == 0  # still the default


def test_env_connection_has_no_remove(page, seeded):
    page.goto(seeded.url + "/settings")
    env = page.locator('#connections [data-connection="anthropic"]')
    expect(env).to_be_visible()
    expect(env.get_by_role("button", name="Remove")).to_have_count(0)
    add_connection(page, seeded, "OpenRouter", "removable", key="sk-or-TEST123456")
    expect(
        page.locator('#connections [data-connection="removable"]').get_by_role(
            "button", name="Remove"
        )
    ).to_have_count(1)


def test_empty_save_key_leaves_the_key_alone(page, seeded):
    row = add_connection(page, seeded, "OpenAI", "keep", key="sk-proj-KEEPME123456")
    expect(row.get_by_role("button", name="Save key")).to_be_disabled()
    before = next(c for c in ai_state(seeded)["connections"] if c["id"] == "keep")["key"]
    assert before
    row.locator("input[name=key]").fill("x")
    row.locator("input[name=key]").fill("")
    expect(row.get_by_role("button", name="Save key")).to_be_disabled()
    after = next(c for c in ai_state(seeded)["connections"] if c["id"] == "keep")["key"]
    assert after == before


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


def test_a_job_on_a_model_it_can_no_longer_use_shows_it_and_can_be_cleared(page, seeded):
    ai = seeded.app.state.ai
    ai.add_model("anthropic", "pinned-off", label="Pinned off", price_in=1, price_out=1)
    ai.add_model(
        "anthropic", "pinned-unpriced", label="Pinned unpriced", price_in=None, price_out=None
    )
    ai.set_role("answer", "anthropic/pinned-off")
    ai.set_role("summary", "anthropic/pinned-unpriced")
    ai.update_model("anthropic", "pinned-off", enabled=False)
    try:
        page.goto(seeded.url + "/settings")
        answer = page.locator('#roles select[data-role="answer"]')
        expect(answer.locator("option:checked")).to_have_text("Pinned off (disabled)")
        summary = page.locator('#roles select[data-role="summary"]')
        expect(summary.locator("option:checked")).to_have_text("Pinned unpriced (needs a price)")
        with page.expect_response(
            lambda r: r.request.method == "PUT" and r.url.endswith("/ai/roles/answer")
        ):
            answer.select_option(label="same as default")
        assert "answer" not in ai_state(seeded)["roles"]
        expect(answer.locator("option:checked")).to_have_text("same as default")
        expect(answer.locator("option", has_text="Pinned off")).to_have_count(0)
    finally:
        ai.set_role("answer", None)
        ai.set_role("summary", None)


def test_default_job_set_by_the_environment_is_read_only(page, seeded):
    def env_default(route):
        response = route.fetch()
        route.fulfill(response=response, json={**response.json(), "default_from_env": True})

    page.route("**/ai", env_default)
    page.goto(seeded.url + "/settings")
    expect(page.locator('#roles select[data-role="default"]')).to_be_disabled()
    expect(page.locator("#roles")).to_contain_text("set by environment")
    expect(page.locator('#roles select[data-role="answer"]')).to_be_enabled()


def test_add_form_id_follows_the_server_rule(page, seeded):
    page.goto(seeded.url + "/settings")
    box = page.locator("#add-id")

    def valid(value):
        box.fill(value)
        return box.evaluate("e => e.checkValidity()")

    assert valid("gateway-2") and valid("a" * 32)
    assert not valid("a_b") and not valid("-a") and not valid("a" * 33)


# -- stores, eval sets and the run form -----------------------------------------------------


def mcq_set(seeded, name, n):
    """An mcq eval set of `n` questions (ids s0..), made through the API."""
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": name, "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    qs = [
        {"id": f"s{i}", "text": f"question {i}", "choices": ["x", "y"], "gold_answers": ["A"]}
        for i in range(n)
    ]
    r = httpx.put(f"{seeded.url}/eval-sets/{name}/questions", json={"questions": qs})
    assert r.status_code == 200, r.text


def test_nav_says_stores_and_eval_sets(page, seeded):
    page.goto(seeded.url + "/")
    links = page.locator("#side a:not(.problem)")
    expect(links.first).to_have_text("Stores")
    texts = links.all_inner_texts()
    assert "Collections" not in texts
    assert texts.index("Stores") < texts.index("c") < texts.index("Eval sets")
    assert texts.index("Eval sets") < texts.index("Lab") < texts.index("Settings")
    expect(page.get_by_role("link", name="Eval sets")).to_have_attribute("href", "/eval-sets")
    expect(page.locator("main h1")).to_have_text("Stores")


def test_create_store_form_offers_the_recipes(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    kind = page.locator("#f-recipe")
    options = kind.locator("option")
    expect(options.first).to_have_attribute("value", "paper_2024")
    expect(options.first).to_contain_text("Paper 2024")
    expect(options.first).to_contain_text("your three keyword lists")
    expect(kind.locator("option[value=worked_examples]")).to_contain_text("Worked examples")
    kind.select_option("worked_examples")
    page.get_by_label("Name").fill("recipe-store")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("main a.row-link", has_text="recipe-store")).to_be_visible()
    assert httpx.get(f"{seeded.url}/collections/recipe-store").json()["recipe"] == "worked_examples"


def test_add_a_default_store_lists_every_dataset_and_loads_one(page, seeded):
    datasets = httpx.get(f"{seeded.url}/catalog").json()["datasets"]
    seen = route_load(page, "gpqa", lambda b: (202, {"job_id": "dd11ee22"}))
    page.goto(seeded.url + "/")
    expect(page.get_by_role("heading", name="Add a default store")).to_be_visible()
    buttons = page.locator("#defaults button")
    expect(buttons).to_have_count(len(datasets))
    expect(page.locator("#defaults")).to_contain_text("Exam multiple choice")
    page.get_by_role("button", name="Add gpqa").click()
    expect(page.get_by_text("Queued (job dd11ee22)")).to_be_visible()
    assert [b.get("dataset") for b in seen] == ["gpqa"]
    store = httpx.get(f"{seeded.url}/collections/gpqa").json()
    assert store["qa_type"] == "mcq"


def test_add_a_default_store_asks_before_it_spends(page, seeded):
    estimate = {"est_usd": 0.2, "items": 5, "calls": 7, "hub": {}}
    seen = route_load(
        page,
        "tatqa",
        lambda b: (
            (202, {"job_id": "ff00aa11"}) if b.get("confirm") else (200, {"estimate": estimate})
        ),
    )
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="Add tatqa").click()
    expect(page.get_by_text("7 AI calls")).to_be_visible()
    assert [b.get("confirm") for b in seen] == [False]
    page.locator("#defaults-confirm").get_by_role("button", name="Start").click()
    expect(page.get_by_text("Queued (job ff00aa11)")).to_be_visible()


def test_add_a_default_store_shows_a_hub_build_as_free(page, seeded):
    estimate = {"est_usd": 0, "items": 5, "calls": 0, "hub": {"entry": "x/y"}}
    route_load(page, "musique", lambda b: (200, {"estimate": estimate}))
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="Add musique").click()
    expect(page.locator("#defaults-confirm")).to_contain_text("Free")


def test_store_page_links_its_eval_set_or_offers_to_make_one(page, seeded):
    page.goto(seeded.url + "/c/c")
    link = page.locator('#asked a[href="/e/c"]')
    expect(link).to_be_visible()
    expect(page.locator("#asked")).to_contain_text("40 questions")
    expect(page.locator("#imp-questions")).to_have_count(0)
    httpx.post(f"{seeded.url}/collections", json={"name": "noset", "qa_type": "mcq"})
    page.goto(seeded.url + "/c/noset")
    expect(page.locator("#asked")).to_contain_text("No eval set yet")
    expect(page.locator('#asked a[href="/eval-sets"]')).to_be_visible()
    expect(page.get_by_label("Default eval set")).to_have_value("")  # mcq sets exist to pick


def test_eval_sets_page_lists_and_creates(page, seeded):
    page.goto(seeded.url + "/eval-sets")
    expect(page.locator("main h1")).to_have_text("Eval sets")
    expect(page.locator('main a.row-link[href="/e/c"]')).to_contain_text("40 questions")
    page.get_by_role("button", name="+ New eval set").click()
    page.get_by_label("Name").fill("made-here")
    page.locator("#f-type").select_option("science")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("main a.row-link", has_text="made-here")).to_be_visible()
    got = httpx.get(f"{seeded.url}/eval-sets/made-here").json()
    assert got["qa_type"] == "science"


def question_map(seeded, name):
    return {q["id"]: q for q in httpx.get(f"{seeded.url}/eval-sets/{name}").json()["questions"]}


def test_eval_set_page_edits_and_saves_a_version_keeping_meta(page, seeded):
    seeded.build("edit")
    before = httpx.get(f"{seeded.url}/eval-sets/edit").json()
    page.goto(seeded.url + "/e/edit")
    rows = page.locator("#questions tbody tr")
    expect(rows).to_have_count(40)
    row = rows.filter(has=page.locator("td", has_text="q0")).first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("enzyme question zero, reworded")
    row.get_by_role("button", name="Done").click()
    q1 = rows.filter(has=page.locator("td", has_text="q1")).first
    q1.get_by_role("button", name="Delete").click()
    page.get_by_role("button", name="Add question").click()
    fresh = rows.last
    fresh.locator("[data-field=text]").fill("a brand new question")
    fresh.locator("[data-field=choices]").fill("one\ntwo")
    fresh.locator("[data-field=gold_answers]").fill("B")
    fresh.get_by_role("button", name="Done").click()
    assert httpx.get(f"{seeded.url}/eval-sets/edit").json() == before  # nothing written yet
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/edit").json()
    by_id = {q["id"]: q for q in got["questions"]}
    assert len(got["questions"]) == 40
    assert "q1" not in by_id
    assert by_id["q0"]["text"] == "enzyme question zero, reworded"
    assert by_id["q0"]["meta"] == next(q for q in before["questions"] if q["id"] == "q0")["meta"]
    added = [q for q in got["questions"] if q["text"] == "a brand new question"]
    assert [q["gold_answers"] for q in added] == [["B"]]
    assert [q["meta"] for q in added] == [{"source": "hand"}]  # checked for leaks
    assert len(got["versions"]) == len(before["versions"]) + 1
    expect(page.locator("#versions li").first).to_contain_text(got["version"][:8])


def test_eval_set_page_keeps_edits_on_a_stale_save(page, seeded):
    seeded.build("stalee")
    page.goto(seeded.url + "/e/stalee")
    expect(page.locator("#questions tbody tr")).to_have_count(40)
    now = httpx.get(f"{seeded.url}/eval-sets/stalee").json()
    qs = [q for q in now["questions"] if q["id"] != "q5"]
    r = httpx.put(
        f"{seeded.url}/eval-sets/stalee/questions",
        json={"questions": qs, "base_version": now["version"]},
    )
    assert r.status_code == 200, r.text
    row = page.locator("#questions tbody tr").first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("my edit survives")
    page.expected = ["409"]
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("changed")
    expect(page.locator("[data-field=text]").first).to_have_value("my edit survives")
    assert question_map(seeded, "stalee")["q0"]["text"] != "my edit survives"
    page.get_by_role("button", name="Reload (discard my edits)").click()
    expect(page.locator("#questions tbody tr")).to_have_count(39)
    expect(page.locator("[data-field=text]")).to_have_count(0)
    expect(page.locator("#save-msg")).to_have_text("")
    expect(page.locator("#dirty")).to_have_text("")


def test_eval_set_page_pages_fifty_at_a_time(page, seeded):
    mcq_set(seeded, "many", 60)
    page.goto(seeded.url + "/e/many")
    expect(page.locator("#questions tbody tr")).to_have_count(50)
    page.get_by_role("button", name="Next").click()
    expect(page.locator("#questions tbody tr")).to_have_count(10)


def test_eval_set_page_deletes_after_saying_how_many_runs(page, seeded):
    run_id = seeded.build("gonee", run=True)
    assert run_id
    page.goto(seeded.url + "/e/gonee")
    page.get_by_role("button", name="Delete eval set…").click()
    expect(page.locator("#del-confirm")).to_contain_text("Delete eval set gonee and its 1 run?")
    page.get_by_role("button", name="Yes, delete gonee").click()
    expect(page).to_have_url(seeded.url + "/eval-sets")
    assert httpx.get(f"{seeded.url}/eval-sets/gonee").status_code == 404


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
    expect(page.locator("#runs tbody tr")).to_have_count(1)
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


def test_store_page_picks_its_default_eval_set(page, seeded):
    seeded.build("pk")
    mcq_set(seeded, "pk2", 3)
    httpx.post(f"{seeded.url}/eval-sets", json={"name": "pksci", "qa_type": "science"})
    page.goto(seeded.url + "/c/pk")
    pick = page.get_by_label("Default eval set")
    expect(pick).to_have_value("pk")
    values = pick.locator("option").evaluate_all("o => o.map(x => x.value)")
    assert "" in values and "pk2" in values and "pksci" not in values
    page.get_by_role("button", name="New run").click()
    expect(page.locator("#rf-set")).to_have_value("pk")
    pick.select_option("pk2")
    expect(page.locator('#asked a[href="/e/pk2"]')).to_be_visible()
    expect(page.locator("#asked")).to_contain_text("3 questions")
    expect(page.locator("#rf-set")).to_have_value("pk2")
    expect(page.locator("#rf-split option[value=full]")).to_have_text("All (3)")
    assert httpx.get(f"{seeded.url}/collections/pk").json()["eval_set"] == "pk2"
    pick.select_option("")  # none: the set named after the store again
    expect(page.locator('#asked a[href="/e/pk"]')).to_be_visible()
    expect(pick).to_have_value("pk")
    assert httpx.get(f"{seeded.url}/collections/pk").json()["eval_set"] == "pk"


def test_store_page_says_how_many_of_its_questions_leaked(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "lk", "qa_type": "mcq"})
    tides = [{"id": "t1", "text": "so the tides follow the moon each day we saw"}]
    r = httpx.post(f"{seeded.url}/collections/lk/documents", json={"items": tides, "confirm": True})
    assert r.status_code == 200, r.text
    httpx.post(f"{seeded.url}/eval-sets", json={"name": "lk", "qa_type": "mcq"})
    qs = [
        {"id": "h1", "text": "the tides follow the moon each day", "choices": ["x", "y"]},
        {"id": "h2", "text": "an unrelated one", "choices": ["x", "y"]},
    ]
    r = httpx.put(f"{seeded.url}/eval-sets/lk/questions", json={"questions": qs})
    assert r.status_code == 200, r.text
    page.goto(seeded.url + "/c/lk")
    asked = page.locator("#asked")
    expect(asked).to_contain_text(
        "1 question appears among the items (leaked): it is left out of runs."
    )
    expect(asked.locator("input[type=checkbox]")).to_have_count(
        0
    )  # that lives on the eval set page
    httpx.patch(f"{seeded.url}/eval-sets/lk", json={"include_leaked": True})
    page.reload()
    expect(asked).to_contain_text("the eval set includes them in runs")


def test_eval_set_page_will_not_import_over_unsaved_edits(page, seeded, tmp_path):
    seeded.build("dirt")
    before = httpx.get(f"{seeded.url}/eval-sets/dirt").json()
    qs = tmp_path / "more.jsonl"
    qs.write_text(
        json.dumps({"id": "u1", "text": "one more", "choices": ["x", "y"], "answer": "A"})
    )
    page.goto(seeded.url + "/e/dirt")
    expect(page.locator("#questions tbody tr")).to_have_count(40)
    expect(page.locator("#imp-questions-go")).to_be_enabled()
    row = page.locator("#questions tbody tr").first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("my unsaved edit")
    expect(page.locator("#imp-questions-go")).to_be_disabled()  # typing is an edit already
    row.get_by_role("button", name="Done").click()
    expect(page.locator("#imp-note")).to_have_text("Save or discard your changes before importing")
    expect(page.locator("#imp-questions-file")).to_be_disabled()
    expect(page.locator("#imp-questions-go")).to_be_disabled()
    expect(page.locator("#questions tbody tr").first).to_contain_text("my unsaved edit")
    assert httpx.get(f"{seeded.url}/eval-sets/dirt").json() == before  # nothing imported
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    expect(page.locator("#imp-note")).to_have_text("")
    upload_and_preview(page, "questions", qs)
    page.locator("#imp-questions-import").click()
    expect(page.locator("#imp-questions-preview")).to_contain_text("1 new")
    expect(page.locator("#questions tbody tr")).to_have_count(41)
    assert question_map(seeded, "dirt")["q0"]["text"] == "my unsaved edit"


def test_eval_set_versions_mark_the_current_one_after_a_revert(page, seeded):
    mcq_set(seeded, "rev", 2)
    first = httpx.get(f"{seeded.url}/eval-sets/rev").json()
    r = httpx.put(
        f"{seeded.url}/eval-sets/rev/questions", json={"questions": first["questions"][:1]}
    )
    second = r.json()["version"]
    r = httpx.put(f"{seeded.url}/eval-sets/rev/questions", json={"questions": first["questions"]})
    assert r.json()["version"] == first["version"]  # the same content: the same version
    page.goto(seeded.url + "/e/rev")
    items = page.locator("#versions li")
    expect(items).to_have_count(2)
    expect(items.filter(has_text=first["version"][:8])).to_contain_text("current")
    expect(items.filter(has_text=second[:8])).not_to_contain_text("current")


def test_a_default_store_that_exists_is_opened_not_added_again(page, seeded):
    r = httpx.post(f"{seeded.url}/collections", json={"name": "scienceqa", "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/")
    expect(page.locator("#defaults").get_by_role("link", name="Open scienceqa")).to_have_attribute(
        "href", "/c/scienceqa"
    )
    expect(page.get_by_role("button", name="Add scienceqa")).to_have_count(0)
    expect(page.get_by_role("button", name="Add scibench")).to_be_visible()


# -- Chat page ---------------------------------------------------------------------------

CHAT_ATTEMPT = {
    "n": 1,
    "attempt": 1,
    "search_text": "asked",
    "decisions": [],
    "evidence": [],
    "answer": "Rates rise [1] with <b>heat</b> [2].",
    "citations": [
        {
            "n": 1,
            "id": "enzyme-item",
            "text": "enzyme " + "kinetics " * 60,
            "source": "hand",
            "part": None,
            "parts": None,
            "store": "c",
        },
        {
            "n": 2,
            "id": "other",
            "text": "short <i>passage</i>",
            "source": None,
            "part": None,
            "parts": None,
            "store": "c",
        },
    ],
    "usd": 0.0123,
    "error": None,
    "model": "anthropic/fake",
}
# The passages the answer was given, each with its number in the prompt: what [n] links to.
CHAT_ATTEMPT["evidence"] = [
    {**c, "rank": c["n"], "score": 0.016, "channels": ["meaning", "words"]}
    for c in CHAT_ATTEMPT["citations"]
]


def make_chat(seeded, stores=("c",), title=None) -> str:
    body = {"stores": list(stores), **({"title": title} if title else {})}
    r = httpx.post(f"{seeded.url}/chats", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def open_chat(page, seeded, stores=("c",), title=None) -> str:
    """A new saved chat on `stores`, opened on the page."""
    chat_id = make_chat(seeded, stores, title)
    page.goto(f"{seeded.url}/chat#{chat_id}")
    expect(page.locator("#chips .chip")).to_have_count(len(stores))
    return chat_id


def chat_route(page, handler):
    """Answer each POST of a message to a chat."""
    page.route("**/chats/*/messages", lambda route: handler(route))


def chat_message(route, n=1, **attempt):
    """What the server sends back for the message POSTed at `route`: its attempt is
    CHAT_ATTEMPT with `attempt`'s changes."""
    content = json.loads(route.request.post_data)["content"]
    return {"n": n, "content": content, "attempts": [{**CHAT_ATTEMPT, "n": n, **attempt}]}


def chat_ask(page, text):
    page.locator("#box").fill(text)
    page.locator("#send").click()


def chat_link(page, chat_id):
    return page.locator(f'#chat-list a[href="#{chat_id}"]')


def test_new_chat_picks_stores_and_lists_saved_chats(page, seeded):
    page.goto(seeded.url + "/chat")
    expect(page.locator("#send")).to_be_disabled()  # no chat open yet
    page.locator("#new").click()
    picker = page.locator("#picking")
    expect(picker.locator("label", has=page.locator('input[value="c"]'))).to_contain_text("mcq")
    picker.locator('input[value="c"]').check()
    picker.locator('input[value="vo"]').check()
    page.get_by_role("button", name="Start chat").click()
    expect(page).to_have_url(re.compile(r"/chat#[0-9a-f]{32}$"))
    first = page.url.split("#")[1]
    expect(page.locator("#chips .chip")).to_have_text([re.compile("^c"), re.compile("^vo")])
    expect(picker).to_be_empty()
    chat_ask(page, "the first of two saved chats")
    expect(page.locator("#log .reply")).to_have_count(1)
    expect(chat_link(page, first)).to_contain_text("the first of two saved chats")
    expect(chat_link(page, first)).to_contain_text("c, vo")
    expect(page.locator("#chat-title")).to_have_text("the first of two saved chats")

    # A second chat, asked later, is listed above the first.
    page.locator("#new").click()
    picker.locator('input[value="c"]').check()
    page.get_by_role("button", name="Start chat").click()
    expect(page).not_to_have_url(re.compile(first))
    second = page.url.split("#")[1]
    expect(page.locator("#log .msg")).to_have_count(0)
    chat_ask(page, "the second of two saved chats")
    expect(chat_link(page, second)).to_contain_text("the second of two saved chats")
    hrefs = page.locator("#chat-list a").evaluate_all("els => els.map(e => e.getAttribute('href'))")
    assert hrefs.index(f"#{second}") < hrefs.index(f"#{first}")

    # The header's edits are saved on the chat.
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.locator("#add-store").select_option("vo")
    expect(page.locator("#chips .chip")).to_have_count(2)
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.get_by_role("button", name="Remove c").click()
    expect(page.locator("#chips .chip")).to_have_text([re.compile("^vo")])
    expect(page.get_by_role("button", name="Remove vo")).to_be_disabled()  # a chat needs one
    page.locator("#evidence").fill("3")
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.locator("#evidence").blur()
    expect(page.locator("#chat-head")).to_contain_text("costs one AI call per follow-up")
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.get_by_label("Rewrite follow-ups with AI").check()
    saved = httpx.get(f"{seeded.url}/chats/{second}").json()
    assert saved["stores"] == ["vo"]
    assert saved["settings"]["evidence_size"] == 3 and saved["settings"]["rewrite"] is True

    # Clicking a chat opens it.
    chat_link(page, first).click()
    expect(page).to_have_url(re.compile(f"#{first}$"))
    expect(page.locator("#log .user")).to_have_text("the first of two saved chats")
    expect(page.locator("#evidence")).to_have_value("8")

    # Delete asks first.
    row = page.locator("#chat-list li", has=page.locator(f'a[href="#{second}"]'))
    row.get_by_role("button", name="Delete").click()
    expect(page.get_by_text("Delete chat the second of two saved chats?")).to_be_visible()
    page.get_by_role("button", name="Keep it").click()
    expect(page.get_by_text("Delete chat the second of two saved chats?")).to_have_count(0)
    row.get_by_role("button", name="Delete").click()
    page.get_by_role("button", name="Yes, delete").click()
    expect(chat_link(page, second)).to_have_count(0)
    assert httpx.get(f"{seeded.url}/chats/{second}").status_code == 404
    expect(page.locator("#log .user")).to_have_text("the first of two saved chats")  # still open
    # Deleting the open chat closes it.
    open_row = page.locator("#chat-list li", has=page.locator(f'a[href="#{first}"]'))
    open_row.get_by_role("button", name="Delete").click()
    page.get_by_role("button", name="Yes, delete").click()
    expect(chat_link(page, first)).to_have_count(0)
    expect(page.locator("#log .msg")).to_have_count(0)
    expect(page).not_to_have_url(re.compile("#"))
    expect(page.locator("#send")).to_be_disabled()


def test_new_chat_needs_a_store(page, seeded):
    posts = []
    page.on(
        "request",
        lambda r: posts.append(r.url) if r.method == "POST" and r.url.endswith("/chats") else None,
    )
    page.goto(seeded.url + "/chat")
    expect(page.locator("#send")).to_be_disabled()
    expect(page.locator("#add-doc")).to_be_disabled()
    page.locator("#new").click()
    page.get_by_role("button", name="Start chat").click()
    expect(page.locator("#pick-note")).to_have_text("Pick at least one store.")
    assert posts == []
    page.route(
        "**/collections",
        lambda r: r.fulfill(json=[]) if r.request.method == "GET" else r.fallback(),
    )
    page.reload()
    page.locator("#new").click()
    picker = page.locator("#picking")
    expect(picker).to_contain_text("No stores yet")
    expect(picker.get_by_role("link", name="Stores")).to_have_attribute("href", "/")
    expect(page.get_by_role("button", name="Start chat")).to_have_count(0)
    expect(page.locator("#send")).to_be_disabled()


def test_chat_reopens_after_reload_from_the_server(page, seeded):
    ai = seeded.app.state.ai
    fake_replies(ai, lambda p: "Rates rise [1].")
    # A conversation the old page kept in the tab: ignored.
    page.add_init_script(
        "sessionStorage.setItem('chat:c', JSON.stringify([{role: 'user', content: 'old tab'}]))"
    )
    try:
        chat_id = open_chat(page, seeded)
        chat_ask(page, "enzyme kinetics rate experiment")  # close enough to c to clear its bar
        expect(page.locator("#log .reply")).to_have_count(1)
        page.reload()
        expect(page).to_have_url(re.compile(f"#{chat_id}$"))
        expect(page.locator("#log .user")).to_have_text("enzyme kinetics rate experiment")
        expect(page.locator("#log .reply .md")).to_have_text("Rates rise [1].")
        sources_ = page.locator("#log .reply .sources li")
        expect(sources_).to_have_count(1)
        expect(sources_.locator(".what")).to_contain_text(re.compile(r" · c$"))
        expect(page.locator("#log")).not_to_contain_text("old tab")
        assert page.evaluate("Object.keys(sessionStorage)") == ["chat:c"]  # nothing written
        expect(chat_link(page, chat_id)).to_contain_text("enzyme kinetics rate experiment")
    finally:
        fake_replies(ai, lambda p: "Answer: A")


def test_switching_chats_while_a_reply_is_on_its_way_shows_it_in_its_own_chat(page, seeded):
    held = []
    chat_route(page, held.append)
    first = make_chat(seeded, ["c"], "slow chat one")
    second = make_chat(seeded, ["vo"], "slow chat two")
    page.goto(f"{seeded.url}/chat#{first}")
    expect(page.locator("#chips .chip")).to_have_count(1)

    def ask_and_hold(question):
        count = len(held)
        chat_ask(page, question)
        for _ in range(100):
            if len(held) > count:
                break
            page.wait_for_timeout(20)
        assert len(held) == count + 1

    # The reply comes while the page is on the other chat: not drawn there, kept in its own.
    ask_and_hold("asked in the first chat")
    expect(page.locator("#log .user")).to_have_text("asked in the first chat")
    expect(page.locator("#status")).to_have_text("thinking…")
    chat_link(page, second).click()
    expect(page).to_have_url(re.compile(f"#{second}$"))
    expect(page.locator("#log .msg")).to_have_count(0)
    expect(page.locator("#status")).to_have_text("")
    expect(page.locator("#send")).to_be_enabled()  # this chat has nothing on its way
    with page.expect_response(lambda r: r.url.endswith(f"/chats/{first}/messages")):
        held[-1].continue_()
    expect(page.locator("#log .msg")).to_have_count(0)
    expect(page.locator("#spoken")).to_have_text("")
    chat_link(page, first).click()
    expect(page.locator("#log .user")).to_have_text("asked in the first chat")
    expect(page.locator("#log .reply")).to_have_count(1)

    # Back on its own chat before the reply comes: drawn there, and read out.
    ask_and_hold("asked again")
    chat_link(page, second).click()
    expect(page.locator("#log .msg")).to_have_count(0)
    chat_link(page, first).click()
    expect(page.locator("#log .user")).to_have_text(["asked in the first chat", "asked again"])
    expect(page.locator("#send")).to_be_disabled()  # its reply is still on its way
    held[-1].continue_()
    expect(page.locator("#log .reply")).to_have_count(2)
    expect(page.locator("#send")).to_be_enabled()
    # Only the newest reply is read out; the log itself is not live.
    expect(page.locator("#spoken")).to_have_text("Answer: A")
    expect(page.locator("#log")).not_to_have_attribute("aria-live", re.compile(".*"))


@contextmanager
def held_answers(seeded, text):
    """The fake AI answers `text`, but only once the yielded event is set: the server has
    saved the message and is mid-attempt until then."""
    ai = seeded.app.state.ai
    go = threading.Event()
    fake_replies(ai, lambda p: text if go.wait(20) else "too late")
    try:
        yield go
    finally:
        go.set()
        fake_replies(ai, lambda p: "Answer: A")


def saved_messages(seeded, chat_id, count):
    """Waits until the server has saved `count` messages on the chat."""
    for _ in range(200):
        if len(httpx.get(f"{seeded.url}/chats/{chat_id}").json()["messages"]) == count:
            return
        time.sleep(0.02)
    raise AssertionError(f"the server never saved {count} messages")


def test_back_on_a_chat_whose_reply_the_server_is_making_shows_the_question_once(page, seeded):
    chat_id = make_chat(seeded, ["c"], "mid-attempt chat")
    other = make_chat(seeded, ["vo"], "mid-attempt other")
    with held_answers(seeded, "The held reply.") as go:
        page.goto(f"{seeded.url}/chat#{chat_id}")
        expect(page.locator("#chips .chip")).to_have_count(1)
        chat_ask(page, "asked while the server works")
        saved_messages(seeded, chat_id, 1)  # saved, with no attempt yet
        chat_link(page, other).click()
        expect(page.locator("#log .msg")).to_have_count(0)
        chat_link(page, chat_id).click()
        expect(page.locator("#chat-title")).to_have_text("mid-attempt chat")
        expect(page.locator("#log .user")).to_have_text(["asked while the server works"])
        expect(page.locator("#send")).to_be_disabled()
        with page.expect_response(lambda r: r.url.endswith(f"/chats/{chat_id}/messages")):
            go.set()
        expect(page.locator("#log .reply")).to_have_count(1)
        expect(page.locator("#log .user")).to_have_text(["asked while the server works"])
        expect(page.locator("#log .turn")).to_have_count(1)
        expect(page.locator("#spoken")).to_have_text("The held reply.")
        expect(page.locator("#send")).to_be_enabled()


def test_a_reply_landing_while_its_chat_reopens_is_kept(page, seeded):
    chat_id = make_chat(seeded, ["c"], "reopening chat")
    other = make_chat(seeded, ["vo"], "reopening other")
    reopen = {"hold": False, "held": []}

    def get(route):
        if not reopen["hold"] or route.request.method != "GET":
            route.fallback()
            return
        reopen["held"].append((route, route.fetch()))  # read now, while the reply is pending

    page.route(f"**/chats/{chat_id}", get)
    with held_answers(seeded, "The late reply.") as go:
        page.goto(f"{seeded.url}/chat#{chat_id}")
        expect(page.locator("#chips .chip")).to_have_count(1)
        chat_ask(page, "asked before reopening")
        saved_messages(seeded, chat_id, 1)
        chat_link(page, other).click()
        expect(page.locator("#log .msg")).to_have_count(0)
        reopen["hold"] = True
        chat_link(page, chat_id).click()
        for _ in range(200):
            if reopen["held"]:
                break
            page.wait_for_timeout(20)
        route, stale = reopen["held"][0]
        assert stale.json()["messages"][0]["attempts"] == []  # the server was mid-attempt
        with page.expect_response(lambda r: r.url.endswith(f"/chats/{chat_id}/messages")):
            go.set()
        expect(page.locator("#spoken")).to_have_text("The late reply.")
        route.fulfill(response=stale)  # the reopen lands after the reply, and is older
        expect(page.locator("#log .reply")).to_have_count(1)
        expect(page.locator("#log .user")).to_have_text(["asked before reopening"])
        expect(page.locator("#log .reply .md")).to_have_text("The late reply.")
        expect(page.locator("#send")).to_be_enabled()


def test_a_deleted_store_shows_as_deleted(page, seeded):
    free_store(seeded, "doomed")
    chat_id = make_chat(seeded, ["c", "doomed"], "a chat that outlives a store")
    r = httpx.post(f"{seeded.url}/chats/{chat_id}/messages", json={"content": "before it goes"})
    assert r.status_code == 200, r.text
    assert httpx.delete(f"{seeded.url}/collections/doomed").status_code == 204
    page.goto(f"{seeded.url}/chat#{chat_id}")  # no request for the deleted store's detail
    chips = page.locator("#chips .chip")
    expect(chips).to_have_count(2)
    expect(chips.nth(1)).to_have_text(re.compile("^doomed · deleted"))
    expect(chips.first).not_to_contain_text("deleted")
    expect(page.locator("#doc-store option")).to_have_text(["c"])
    expect(page.locator("#log .user")).to_have_text("before it goes")  # the old chat reads whole
    expect(page.locator("#log .reply")).to_have_count(1)
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.get_by_role("button", name="Remove doomed").click()
    expect(chips).to_have_count(1)
    assert httpx.get(f"{seeded.url}/chats/{chat_id}").json()["stores"] == ["c"]


def test_chat_page_sends_shows_citations_and_cost(page, seeded):
    sent = []

    def reply(route):
        sent.append(json.loads(route.request.post_data))
        route.fulfill(json=chat_message(route))

    chat_route(page, reply)
    chat_id = open_chat(page, seeded)
    with page.expect_request(lambda r: r.url.endswith(f"/chats/{chat_id}/messages")):
        chat_ask(page, "how fast <u>are</u> enzymes?")
    reply_box = page.locator("#log .reply")
    expect(reply_box).to_be_visible()
    assert sent[0]["content"] == "how fast <u>are</u> enzymes?"
    assert sent[0]["model"]
    expect(page.locator("#log .user")).to_have_text("how fast <u>are</u> enzymes?")
    expect(reply_box).to_contain_text("Rates rise [1] with <b>heat</b> [2].")
    assert page.locator("#log b, #log i, #log u").count() == 0  # nothing the model said is markup
    expect(reply_box.locator("a.cite")).to_have_count(2)
    expect(reply_box.locator(".sources li")).to_have_count(2)
    first = reply_box.locator(".sources li").first
    expect(first).to_contain_text("enzyme-item")
    expect(first).to_contain_text("hand")
    expect(first.locator(".what")).to_have_text("[1] enzyme-item · hand · c")
    expect(first.locator("details")).to_have_count(1)  # longer than 300 characters
    assert len(first.locator("summary").inner_text()) <= 305
    first.locator("summary").click()
    expect(first.locator("details .full")).to_contain_text("kinetics " * 50)
    expect(reply_box.locator(".sources li").nth(1).locator("details")).to_have_count(0)
    expect(reply_box.locator(".sources li").nth(1)).to_contain_text("short <i>passage</i>")
    expect(reply_box.locator(".meta")).to_contain_text("$0.01")
    expect(page.locator("#box")).to_have_value("")
    reply_box.locator("a.cite").first.click()
    expect(page.locator("#log li:target")).to_have_count(1)
    expect(page.locator("#log li:target")).to_have_attribute("id", "ev-1-1-1")
    expect(page.locator("#msg-1 details.retrieval")).to_have_attribute("open", "")
    expect(page).to_have_url(re.compile(f"#{chat_id}$"))  # the address still names the chat


def test_chat_links_each_number_of_a_grouped_citation(page, seeded):
    answer = "Rates rise [1, 2], again [02]."
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, answer=answer)))
    open_chat(page, seeded)
    chat_ask(page, "grouped?")
    reply_box = page.locator("#log .reply")
    expect(reply_box).to_contain_text(answer)
    expect(reply_box.locator("a.cite")).to_have_text(["1", "2", "[02]"])
    expect(reply_box.locator(".sources li")).to_have_count(2)
    reply_box.locator("a.cite").nth(1).click()
    expect(page.locator("#log li:target")).to_contain_text("other")
    reply_box.locator("a.cite").first.click()  # off source 2, so the next click must set it
    expect(page.locator("#log li:target")).to_contain_text("enzyme-item")
    reply_box.locator("a.cite").nth(2).click()  # [02] is source 2, not a missing "02"
    expect(page.locator("#log li:target")).to_contain_text("other")


def test_chat_sources_name_the_part_of_a_split_file(page, seeded):
    part = {"n": 1, "id": "x.md#2", "text": "x.md\n\nbrine", "source": "upload:x", "store": "c"}
    citations = [{**part, "part": 2, "parts": 3}, CHAT_ATTEMPT["citations"][1]]
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, citations=citations)))
    open_chat(page, seeded)
    chat_ask(page, "parts?")
    sources_ = page.locator("#log .reply .sources li .what")
    expect(sources_.first).to_have_text("[1] x.md#2 · upload:x · part 2 of 3 · c")
    expect(sources_.nth(1)).not_to_contain_text("part")


def test_chat_renders_a_safe_subset_of_markdown(page, seeded):
    answer = "**Rates** rise [1].\n\n- one\n- two\n\n```\n<b>x</b>\n```\n"
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, answer=answer)))
    open_chat(page, seeded)
    chat_ask(page, "markdown?")
    body = page.locator("#log .reply .md")
    expect(body.locator("strong")).to_have_text("Rates")
    expect(body.locator("ul li")).to_have_text(["one", "two"])
    expect(body.locator("pre")).to_have_text("<b>x</b>")
    assert body.locator("b").count() == 0
    expect(body.locator("a.cite")).to_have_text(["[1]"])
    expect(body).to_have_attribute("data-text", answer)


def test_chat_markdown_makes_no_link_and_keeps_snake_case(page, seeded):
    answer = "See [a](javascript:alert(1)) and snake_case_name [1]."
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, answer=answer)))
    open_chat(page, seeded)
    chat_ask(page, "links?")
    body = page.locator("#log .reply .md")
    expect(body).to_have_text(answer)
    expect(body.locator("a")).to_have_count(1)  # the citation only
    expect(body.locator("a.cite")).to_have_text(["[1]"])
    assert body.locator("em, strong").count() == 0


def test_chat_markdown_headings_ordered_lists_italic_and_code(page, seeded):
    answer = "## Plan\n1. *first* [1]\n2. __second__\n\nUse `x [1] **y**` here,\nthen _go_."
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, answer=answer)))
    open_chat(page, seeded)
    chat_ask(page, "more?")
    body = page.locator("#log .reply .md")
    expect(body.locator("p").first.locator("strong")).to_have_text("Plan")
    assert body.locator("h1, h2, h3").count() == 0
    expect(body.locator("ol li")).to_have_text(["first [1]", "second"])
    expect(body.locator("ol li em")).to_have_text("first")
    expect(body.locator("ol li strong")).to_have_text("second")
    expect(body.locator("code")).to_have_text("x [1] **y**")
    expect(body.locator("a.cite")).to_have_count(1)  # none inside code
    expect(body.locator("p br")).to_have_count(1)
    expect(body.locator("p em")).to_have_text("go")


def test_chat_markdown_lists_run_across_blank_lines_resume_and_continue(page, seeded):
    answer = (
        "1. a\n\n2. b\n\n3. c\n\nThen:\n\n4. d\n5. e\n\n- one\n  more of one [1]\n- two\n\nAfter."
    )
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, answer=answer)))
    open_chat(page, seeded)
    chat_ask(page, "lists?")
    body = page.locator("#log .reply .md")
    lists = body.locator("ol")
    expect(lists).to_have_count(2)
    expect(lists.first.locator("li")).to_have_text(["a", "b", "c"])
    assert lists.first.get_attribute("start") is None
    expect(lists.nth(1)).to_have_attribute("start", "4")
    expect(lists.nth(1).locator("li")).to_have_text(["d", "e"])
    items = body.locator("ul li")
    expect(items).to_have_count(2)
    expect(items.first).to_have_text("onemore of one [1]")
    expect(items.first.locator("br")).to_have_count(1)
    expect(items.first.locator("a.cite")).to_have_count(1)
    expect(body.locator("p").last).to_have_text("After.")


def test_chat_store_detail_failing_still_enables_send_and_add(page, seeded):
    page.expected = ["500"]
    page.route(
        "**/collections/c",
        lambda r: (
            r.fulfill(status=500, json={"detail": "store detail broke"})
            if r.request.method == "GET"
            else r.fallback()
        ),
    )
    open_chat(page, seeded)
    expect(page.locator("#status")).to_contain_text("store detail broke")
    expect(page.locator("#send")).to_be_enabled()
    expect(page.locator("#add-doc")).to_be_enabled()


def test_chat_without_an_answer_model_says_what_to_set_up(page, seeded):
    ai = httpx.get(seeded.url + "/ai").json()
    page.route("**/ai", lambda route: route.fulfill(json={**ai, "models": []}))
    open_chat(page, seeded)
    note = page.locator("#note")
    expect(note).to_contain_text("Chat needs an answer model: choose one on Settings")
    expect(note.get_by_role("link", name="Settings")).to_have_attribute("href", "/settings")
    expect(page.locator("#send")).to_be_disabled()


def test_chat_reply_error_keeps_the_conversation(page, seeded):
    replies = [
        {"detail": "hybrid is down"},
        {"answer": None, "citations": [], "error": "budget used up"},
    ]

    def reply(route):
        body = replies.pop(0)
        if "detail" in body:
            route.fulfill(status=409, json=body)
        else:
            route.fulfill(json=chat_message(route, **body))

    page.expected = ["409"]  # the first reply is a refusal, which is the point
    chat_route(page, reply)
    other = make_chat(seeded, ["vo"], "the other chat")
    chat_id = open_chat(page, seeded)
    chat_ask(page, "will this fail?")
    expect(page.locator("#log .err")).to_contain_text("hybrid is down")
    expect(page.locator("#spoken")).to_have_text("Error: hybrid is down")
    expect(page.locator("#log .user")).to_have_text("will this fail?")
    expect(page.locator("#send")).to_be_enabled()
    chat_ask(page, "and again?")
    expect(page.locator("#log .err").nth(1)).to_contain_text("budget used up")
    expect(page.locator("#log [role=alert]")).to_have_count(0)
    expect(page.locator("#spoken")).to_have_text("Error: budget used up")
    expect(page.locator("#log .user")).to_have_count(2)
    chat_link(page, other).click()  # another chat's conversation: nothing spoken
    expect(page.locator("#spoken")).to_have_text("")
    expect(page.locator("#log .msg")).to_have_count(0)
    chat_link(page, chat_id).click()
    # The refusal saved nothing on the server; the page keeps it in its chat until a reload.
    expect(page.locator("#log .user")).to_have_text("will this fail?")
    expect(page.locator("#log .err")).to_contain_text("hybrid is down")
    expect(page.locator("#spoken")).to_have_text("")


def test_chat_error_line_shows_what_it_cost(page, seeded):
    spent = {"answer": None, "citations": [], "error": "budget used up", "usd": 0.5}
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, **spent)))
    open_chat(page, seeded)
    chat_ask(page, "pricey")
    expect(page.locator("#log .err")).to_contain_text("budget used up")
    expect(page.locator("#log .err")).to_contain_text("$0.50")


DOC_LISTS = {"topic": ["volcano", "coral"], "method": ["survey"], "common": ["rate"]}


def free_store(seeded, name):
    """A store with keyword lists whose recipe tags by similarity, so adding to it is free."""
    r = httpx.post(
        f"{seeded.url}/collections",
        json={"name": name, "qa_type": "science", "recipe": "paper_2024"},
    )
    assert r.status_code == 201, r.text
    base = [
        {"id": "base", "text": "volcano magma lava pressure"},
        {"id": "tax", "text": "income tax filing deadline"},
        {"id": "bread", "text": "sourdough bread flour hydration"},
        {"id": "star", "text": "neutron star gravity collapse"},
    ]
    body = {"items": base, "confirm": True}
    assert httpx.post(f"{seeded.url}/collections/{name}/documents", json=body).status_code == 200
    r = httpx.put(f"{seeded.url}/collections/{name}/keyword-lists", json=DOC_LISTS)
    assert r.status_code == 202, r.text
    assert seeded.wait(r.json()["job_id"])["status"] == "done"


def chat_upload(page, seeded, store, tmp_path, filename, text):
    """Opens a chat on c and `store`, and adds the file to `store`."""
    path = tmp_path / filename
    path.write_text(text)
    chat_id = open_chat(page, seeded, ("c", store))
    page.locator("#doc-store").select_option(store)
    page.locator("#doc-file").set_input_files(str(path))
    return chat_id


def test_add_document_asks_which_store(page, seeded, tmp_path):
    free_store(seeded, "pick-a")
    free_store(seeded, "pick-b")
    open_chat(page, seeded, ("pick-a", "pick-b"))
    expect(page.locator("#doc-store option")).to_have_text(["pick-a", "pick-b"])
    page.locator("#doc-store").select_option("pick-b")
    path = tmp_path / "tides.txt"
    path.write_text("tides follow the moon")
    page.locator("#doc-file").set_input_files(str(path))
    expect(page.locator("#log")).to_contain_text("Added tides.txt to pick-b")
    assert "upload:tides" in [s["source"] for s in sources(seeded, "pick-b")]
    assert "upload:tides" not in [s["source"] for s in sources(seeded, "pick-a")]
    expect(page.locator("#doc-store")).to_have_value("pick-b")  # kept for the next file


def test_chat_upload_adds_a_document_and_the_next_answer_can_cite_it(page, seeded, tmp_path):
    free_store(seeded, "docs")
    db.set_chat_method(seeded.app.state.conn, "docs", "hybrid")
    chat_id = chat_upload(
        page, seeded, "docs", tmp_path, "reefs.txt", "coral reefs bleach when oceans warm"
    )
    expect(page.locator("#add-doc")).to_be_disabled()  # while the file is being added
    expect(page.locator("#log")).to_contain_text("Added reefs.txt")
    expect(page.locator("#add-doc")).to_be_enabled()
    assert [s["source"] for s in sources(seeded, "docs")] == ["hand", "upload:reefs"]
    store = seeded.app.state.store
    owners = store.sources_of("docs", store.ids("docs"))
    hit_id = next(i for i, source in owners.items() if source == "upload:reefs")
    with page.expect_response(lambda r: r.url.endswith(f"/chats/{chat_id}/messages")) as got:
        chat_ask(page, "why do coral reefs bleach?")
    evidence = got.value.json()["attempts"][0]["evidence"]
    assert [e["id"] for e in evidence if e["store"] == "docs"][0] == hit_id
    expect(page.locator("#log .reply")).to_be_visible()


def test_chat_upload_cleans_the_file_name(page, seeded, tmp_path):
    free_store(seeded, "cleaned")
    chat_upload(page, seeded, "cleaned", tmp_path, "Q3 (final).md", "# Q3\nrevenue grew")
    expect(page.locator("#log")).to_contain_text("Added Q3 (final).md")
    assert "upload:Q3 -final" in [s["source"] for s in sources(seeded, "cleaned")]


def test_chat_upload_shows_the_estimate_and_cancel_adds_nothing(page, seeded, tmp_path):
    ai = seeded.app.state.ai
    ai.add_model("anthropic", "kw-priced", label="Priced", price_in=1, price_out=1)
    ai.set_role("keywords", "anthropic/kw-priced")
    try:
        r = httpx.post(f"{seeded.url}/collections", json={"name": "paid", "qa_type": "science"})
        assert r.status_code == 201, r.text
        chat_upload(page, seeded, "paid", tmp_path, "pay.txt", "some words about <b>tides</b>")
        box = page.locator("#adding")
        expect(box).to_contain_text("AI call")
        expect(box.get_by_role("button", name="Add", exact=True)).to_be_visible()
        assert sources(seeded, "paid") == []
        box.get_by_role("button", name="Cancel").click()
        expect(box).to_be_empty()
        expect(page.locator("#log")).not_to_contain_text("Added")
        assert sources(seeded, "paid") == []
        assert httpx.get(f"{seeded.url}/jobs", params={"collection": "paid"}).json() == []
    finally:
        ai.set_role("keywords", None)


def test_chat_upload_that_spends_adds_after_add_and_refreshes_the_store(page, seeded, tmp_path):
    ai = seeded.app.state.ai
    ai.add_model("anthropic", "kw-paid", label="Paid", price_in=1, price_out=1)
    ai.set_role("keywords", "anthropic/kw-paid")
    try:
        r = httpx.post(f"{seeded.url}/collections", json={"name": "paid2", "qa_type": "science"})
        assert r.status_code == 201, r.text
        details = []
        page.on(
            "request",
            lambda r: details.append(r.url) if r.url.endswith("/collections/paid2") else None,
        )
        open_chat(page, seeded, ("c", "paid2"))
        chip = page.locator('#chips .chip[data-store="paid2"]')
        expect(chip).to_contain_text("0 items")
        page.locator("#doc-store").select_option("paid2")
        path = tmp_path / "pay.txt"
        path.write_text("coral reefs bleach when oceans warm")
        page.locator("#doc-file").set_input_files(str(path))
        box = page.locator("#adding")
        expect(box).to_contain_text("AI call")
        with page.expect_response(
            lambda r: r.url.endswith("/collections/paid2/imports") and r.request.method == "POST"
        ) as queued:
            box.get_by_role("button", name="Add", exact=True).click()
        assert queued.value.status == 202 and queued.value.json()["job_id"]
        expect(page.locator("#log")).to_contain_text("Added pay.txt")
        assert [s["source"] for s in sources(seeded, "paid2")] == ["upload:pay"]
        # the store is looked up again after the add, so its chip shows the new item
        expect(chip).to_contain_text("1 item")
        assert len(details) == 2
    finally:
        ai.set_role("keywords", None)


def test_chat_upload_queued_behind_another_job_says_it_is_waiting(page, seeded, tmp_path):
    free_store(seeded, "behind")
    polls = []

    def job(route):
        polls.append(route.request.url)
        if len(polls) <= 3:
            route.fulfill(json={"id": "x", "status": "queued", "progress": 0})
        else:
            route.fallback()

    page.route("**/jobs/*", job)
    chat_upload(page, seeded, "behind", tmp_path, "later.txt", "tides follow the moon")
    expect(page.locator("#adding")).to_have_text(
        "Adding later.txt: waiting for the store's other job…"
    )
    expect(page.locator("#log")).to_contain_text("Added later.txt")


def test_chat_upload_refuses_other_files_without_uploading(page, seeded, tmp_path):
    free_store(seeded, "refuse")
    posts = []
    page.on("request", lambda r: posts.append(r.url) if "/uploads" in r.url else None)
    chat_upload(page, seeded, "refuse", tmp_path, "notes.pdf", "not text")
    expect(page.locator("#log .err")).to_contain_text("only .txt and .md")
    assert posts == []
    expect(page.locator("#add-doc")).to_be_enabled()


# -- The retrieval step under each answer --------------------------------------------------


def retrieval_step(page, n=1):
    """Message n's Retrieval step, opened."""
    step = page.locator(f"#msg-{n} details.retrieval")
    if step.get_attribute("open") is None:
        step.locator(":scope > summary").click()
    expect(step).to_have_attribute("open", "")
    return step


def store_row(step, store):
    return step.locator(f'table.decisions tr[data-store="{store}"]')


def edit_box(step, store):
    return step.locator(f'fieldset[data-edit="{store}"]')


def attempt_posts(page):
    """The bodies the page POSTs to re-run a message's retrieval."""
    sent = []
    page.on(
        "request",
        lambda r: (
            sent.append(json.loads(r.post_data))
            if r.method == "POST" and r.url.endswith("/attempts")
            else None
        ),
    )
    return sent


def test_retrieval_step_shows_search_text_store_decisions_and_evidence(page, seeded):
    def tagged(route):
        # The real reply, with vo run as keyword_rag (no store has a vocabulary here) and
        # its first passage longer than 300 characters.
        got = route.fetch()
        message = got.json()
        attempt = message["attempts"][0]
        vo = next(d for d in attempt["decisions"] if d["store"] == "vo")
        vo.update(method="keyword_rag", search={**vo["search"], "query_tags": "classifier"})
        first = next(e for e in attempt["evidence"] if e["n"] == 1)
        first["text"] += " and more" * 40
        route.fulfill(response=got, json=message)

    ai = seeded.app.state.ai
    fake_replies(ai, lambda p: "Rates rise [1].")
    try:
        chat_route(page, tagged)
        chat_id = open_chat(page, seeded, ("c", "vo"))
        chat_ask(page, "enzyme kinetics rate experiment")
        expect(page.locator("#msg-1 .reply")).to_have_count(1)
        step = page.locator("#msg-1 details.retrieval")
        expect(step.locator(":scope > summary")).to_have_text("Retrieval step")
        expect(step).not_to_have_attribute("open", "")  # collapsed until opened
        step = retrieval_step(page)
        expect(step.locator(".searched")).to_have_text("enzyme kinetics rate experiment")

        c = store_row(step, "c")
        expect(c.locator(".method")).to_have_text("hybrid")
        expect(c.locator(".best")).to_have_text("1.00")
        expect(c.locator(".bar")).to_have_text("0.29")
        expect(c.locator(".kept")).to_have_text("kept · clears its bar")
        vo = store_row(step, "vo")
        expect(vo.locator(".method")).to_have_text("keyword_rag · query tags: classifier")
        expect(vo.locator(".kept")).to_have_text("kept · clears its bar")

        # Every passage the answer was given, by its number in the prompt.
        expect(step.locator("ol.evidence li")).to_have_count(6)
        first = step.locator("#ev-1-1-1")
        expect(first.locator(".what")).to_have_text("[1] enzyme-item · c · hand · rank 1")
        expect(first.locator("details")).to_have_count(1)  # longer than 300 characters
        assert len(first.locator("summary").inner_text()) <= 305
        expect(step.locator("#ev-1-1-6 .what")).to_have_text("[6] enzyme-item · vo · hand · rank 1")
        expect(step.locator("#ev-1-1-2")).to_contain_text("volcano magma lava pressure")

        # Closed again, a citation opens it at the passage it cites.
        step.locator(":scope > summary").click()
        expect(step).not_to_have_attribute("open", "")
        page.locator("#msg-1 .reply a.cite").click()
        expect(step).to_have_attribute("open", "")
        expect(page.locator("#log li:target")).to_have_attribute("id", "ev-1-1-1")
        expect(page).to_have_url(re.compile(f"#{chat_id}$"))

        # What is open survives the log's redraw on the next reply.
        first.locator("summary").click()
        expect(first.locator("details")).to_have_attribute("open", "")
        chat_ask(page, "and the volcano?")
        expect(page.locator("#log .turn")).to_have_count(2)
        expect(page.locator("#msg-2 .reply")).to_have_count(1)
        expect(step).to_have_attribute("open", "")
        expect(first.locator("details")).to_have_attribute("open", "")
        expect(page.locator("#msg-2 details.retrieval")).not_to_have_attribute("open", "")
    finally:
        fake_replies(ai, lambda p: "Answer: A")


def test_rerun_with_an_override_adds_an_attempt_tab_and_answer_from_it(page, seeded):
    sent = attempt_posts(page)
    chat_id = open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "enzyme kinetics rate experiment")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    expect(turn.get_by_role("tab")).to_have_count(0)  # one attempt: no tabs
    step = retrieval_step(page)
    vo = edit_box(step, "vo")
    expect(vo.get_by_label("Method")).to_have_value("hybrid")
    expect(vo.get_by_label("k", exact=True)).to_have_value("8")
    vo.get_by_label("k", exact=True).fill("2")
    vo.get_by_label("Search anyway").check()
    with page.expect_response(lambda r: r.url.endswith(f"/chats/{chat_id}/messages/1/attempts")):
        vo.get_by_role("button", name="Re-run retrieval").click()
    assert sent == [{"overrides": {"vo": {"k": 2, "keep": True}}}]  # only what was changed

    second = turn.get_by_role("tab", name="Attempt 2")
    expect(second).to_have_attribute("aria-selected", "true")
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "false")
    expect(step).to_have_attribute("open", "")
    expect(store_row(step, "vo").locator(".kept")).to_have_text("kept · search anyway")
    expect(step.locator('ol.evidence li[data-store="vo"]')).to_have_count(2)  # k = 2
    expect(turn.locator(".reply")).to_have_count(0)  # a re-run is not answered
    expect(edit_box(step, "vo").get_by_label("k", exact=True)).to_have_value("2")

    answer = turn.get_by_role("button", name="Answer from this attempt")
    with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts/2/answer")):
        answer.click()
    expect(turn.locator(".reply .md")).to_have_text("Answer: A")
    expect(answer).to_have_count(0)
    saved = httpx.get(f"{seeded.url}/chats/{chat_id}").json()["messages"][0]["attempts"]
    assert saved[0]["answer"] and saved[1]["answer"] == "Answer: A"

    # The first attempt is still there, as it was.
    turn.get_by_role("tab", name="Attempt 1").click()
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "true")
    expect(store_row(step, "vo").locator(".kept")).to_have_text("kept · clears its bar")
    expect(step.locator('ol.evidence li[data-store="vo"]')).to_have_count(3)

    expect(step.locator("fieldset[data-edit]")).to_have_count(0)  # edits build on the latest
    expect(step).to_contain_text("To edit and re-run, open the latest attempt.")

    # The chosen tab and an edit not yet re-run survive the log's redraw on the next reply.
    chat_ask(page, "and the volcano?")
    expect(page.locator("#msg-2 .reply")).to_have_count(1)
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "true")
    expect(step).to_have_attribute("open", "")
    second.click()
    edit_box(step, "c").get_by_label("k", exact=True).fill("5")
    chat_ask(page, "and the market?")
    expect(page.locator("#msg-3 .reply")).to_have_count(1)
    expect(second).to_have_attribute("aria-selected", "true")
    expect(edit_box(step, "c").get_by_label("k", exact=True)).to_have_value("5")


def test_rerun_refused_shows_the_reason_and_keeps_the_edit(page, seeded):
    page.expected = ["422", "409"]  # refusals are the point
    sent = attempt_posts(page)
    open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "enzyme kinetics rate experiment")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    step = retrieval_step(page)
    c = edit_box(step, "c")
    for channel in ("Meaning", "Words", "Keywords"):
        c.get_by_label(f"{channel} weight", exact=True).fill("0")
    with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts")) as got:
        c.get_by_role("button", name="Re-run retrieval").click()
    assert got.value.status == 422
    expect(c.locator(".edit-note")).to_have_text("at least one channel weight must be above 0")
    expect(edit_box(step, "vo").locator(".edit-note")).to_have_text("")  # beside c only
    expect(c.get_by_label("Meaning weight", exact=True)).to_have_value("0")
    expect(turn.get_by_role("tab")).to_have_count(0)  # no attempt was added

    # A method the store can't run is refused with its reason (409), and the edits stay.
    page.route(
        "**/messages/1/attempts",
        lambda r: r.fulfill(status=409, json={"detail": "vector can't run on 'c' just now"}),
    )
    c.get_by_label("Method").select_option("vector")
    c.get_by_role("button", name="Re-run retrieval").click()
    expect(c.locator(".edit-note")).to_have_text("vector can't run on 'c' just now")
    expect(c.get_by_label("Method")).to_have_value("vector")
    expect(c.get_by_label("Words weight", exact=True)).to_have_value("0")

    # A k the route would refuse is said on the page, and nothing is sent.
    count = len(sent)
    c.get_by_label("k", exact=True).fill("30")
    c.get_by_role("button", name="Re-run retrieval").click()
    expect(c.locator(".edit-note")).to_have_text("k is a whole number from 1 to 20.")
    assert len(sent) == count
    expect(turn.get_by_role("tab")).to_have_count(0)


def test_set_as_default_asks_then_saves(page, seeded):
    free_store(seeded, "deflt")  # no eval set, so no bar
    conn = seeded.app.state.conn
    defaults = []
    page.on(
        "request",
        lambda r: defaults.append(r.url) if r.url.endswith("/stores/deflt/default") else None,
    )
    sent = attempt_posts(page)
    open_chat(page, seeded, ("deflt",))
    chat_ask(page, "volcano magma")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    step = retrieval_step(page)
    expect(store_row(step, "deflt").locator(".bar")).to_have_text("none")
    expect(store_row(step, "deflt").locator(".kept")).to_have_text("kept · no bar")
    box = edit_box(step, "deflt")
    box.get_by_label("k", exact=True).fill("3")
    box.get_by_label("Use the store's bar").uncheck()
    box.get_by_label("Bar", exact=True).fill("0.25")
    box.get_by_label("Search anyway").check()

    box.get_by_role("button", name="Set as default").click()
    question = "Save these settings as deflt's default? Search anyway is not saved."
    expect(box).to_contain_text(question)
    box.get_by_role("button", name="Cancel").click()
    expect(box).not_to_contain_text(question)
    assert defaults == [] and sent == []
    expect(box.get_by_label("k", exact=True)).to_have_value("3")  # the edit is kept

    box.get_by_role("button", name="Set as default").click()
    with page.expect_response(lambda r: r.url.endswith("/stores/deflt/default")):
        box.get_by_role("button", name="Yes, save").click()
    expect(box.locator(".edit-note")).to_contain_text("Saved as deflt's default")
    # The edit is re-run first, so the server holds it as the chat's override, then saved.
    assert sent == [{"overrides": {"deflt": {"k": 3, "bar": 0.25, "keep": True}}}]
    assert db.get_chat_k(conn, "deflt") == 3
    assert db.get_chat_bar(conn, "deflt") == 0.25
    expect(turn.get_by_role("tab", name="Attempt 2")).to_have_attribute("aria-selected", "true")
    expect(store_row(step, "deflt").locator(".bar")).to_have_text("0.25 edited")
    expect(box).not_to_contain_text(question)


DROPPED = {
    "store": "c",
    "method": "hybrid",
    "k": 8,
    "search": {"weights": {"meaning": 1, "words": 1, "keywords": 1}, "query_tags": "similarity"},
    "best": 0.123,
    "bar": 0.3,
    "bar_source": "set",
    "kept": False,
    "reason": "below its bar",
    "usd": 0.0,
    "error": None,
}
GONE = {
    **DROPPED,
    "store": "gone",
    "method": None,
    "k": None,
    "search": None,
    "best": None,
    "bar": None,
    "bar_source": None,
    "reason": "deleted",
}


def test_dropped_store_shows_its_reason_and_no_store_kept_note(page, seeded):
    decisions = [
        DROPPED,
        {**DROPPED, "store": "vo", "bar": None, "bar_source": "few", "best": 0.5},
        {**DROPPED, "store": "hand-bar", "bar": 0.4, "bar_source": "manual"},
        {**DROPPED, "store": "edited-bar", "bar": 0.35, "bar_source": "override"},
        {**DROPPED, "store": "no-set", "bar": None, "bar_source": "none"},
        {**DROPPED, "store": "broke", "best": None, "bar": None, "bar_source": None}
        | {"reason": "error", "error": "budget used up"},
        GONE,
    ]
    answer = {"answer": "No store passages were used.", "citations": [], "evidence": []}
    chat_route(
        page,
        lambda route: route.fulfill(json=chat_message(route, decisions=decisions, **answer)),
    )
    open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "something no store holds")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply .md")).to_have_text("No store passages were used.")
    # Said under the answer, with the step still closed.
    note = turn.locator(".reply .none-kept")
    expect(note).to_have_text("No store was kept, so this answer used no store passages.")
    expect(note).to_be_visible()
    step = retrieval_step(page)
    c = store_row(step, "c")
    expect(c.locator(".best")).to_have_text("0.12")
    expect(c.locator(".bar")).to_have_text("0.30")
    expect(c.locator(".kept")).to_have_text("dropped · below its bar")
    expect(store_row(step, "vo").locator(".bar")).to_have_text("too few questions")
    expect(store_row(step, "hand-bar").locator(".bar")).to_have_text("0.40 manual")
    expect(store_row(step, "edited-bar").locator(".bar")).to_have_text("0.35 edited")
    expect(store_row(step, "no-set").locator(".bar")).to_have_text("none")
    broke = store_row(step, "broke")
    expect(broke.locator(".kept")).to_have_text("dropped · error: budget used up")
    expect(broke.locator(".best")).to_have_text("–")
    gone = store_row(step, "gone")
    expect(gone.locator(".kept")).to_have_text("deleted")
    expect(gone.locator(".method")).to_have_text("–")
    expect(step.locator("ol.evidence li")).to_have_count(0)
    # Only the chat's own stores can be edited.
    expect(step.locator("fieldset[data-edit]")).to_have_count(2)
    expect(edit_box(step, "c")).to_be_visible()
    expect(edit_box(step, "vo")).to_be_visible()
