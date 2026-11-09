"""The console pages, driven in a real browser against the app on fakes."""

import json
import re
import shlex
import threading
import time
from contextlib import contextmanager

import httpx
import pytest
from playwright.sync_api import expect

from ragqa_core import db, search_defaults
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
    expect(page).to_have_url(seeded.url + "/c/fresh")


def start_page(page, seeded, models=(), connections=(), chats=(), jobs=(), stores=None):
    """Open Stores with /ai, /chats and the eval-jobs query fixed (stores: seeded unless given)."""

    def ai(route):
        if route.request.method != "GET":
            return route.continue_()
        reply = route.fetch().json()
        reply["models"], reply["connections"] = list(models), list(connections)
        route.fulfill(json=reply)

    def eval_jobs(route):
        if "kind=eval" in route.request.url:
            route.fulfill(json=list(jobs))
        else:
            route.continue_()

    page.route("**/ai", ai)
    page.route("**/chats", lambda route: route.fulfill(json=list(chats)))
    page.route("**/jobs?*", eval_jobs)
    if stores is not None:
        page.route("**/collections", lambda route: route.fulfill(json=stores))
    page.goto(seeded.url + "/")


def start_state(page):
    return {
        li.get_attribute("data-step"): li.get_attribute("data-done")
        for li in page.locator("#start-here li").all()
    }


GOOD_MODEL = {"connection_id": "k", "enabled": True, "price_in": 1.0, "price_out": 2.0}


def test_start_here_shows_each_step_and_its_state(page, seeded):
    start_page(page, seeded)
    items = page.locator("#start-here li")
    expect(items).to_have_count(4)
    labels = [li.locator("a").inner_text() for li in items.all()]
    assert labels == ["Connect AI", "Pick or make a store", "Chat", "Test with an eval set"]
    assert start_state(page) == {"ai": "false", "store": "true", "chat": "false", "run": "false"}
    hrefs = [li.locator("a").get_attribute("href") for li in items.all()]
    assert hrefs == ["/settings", "#new", "/chat", "/eval-sets"]
    expect(page.locator("#start-here")).to_contain_text(
        "Optional for Documents: without it, chat gives passages only."
    )


def test_start_here_hides_once_the_required_steps_are_done(page, seeded):
    chats = []  # the routes read these lists on every request, so the test can change them
    jobs = [{"job_id": "j"}]
    page.clock.install()
    start_page(page, seeded, chats=chats, jobs=jobs)
    expect(page.locator("#start-here")).to_be_visible()
    assert start_state(page)["chat"] == "false"
    chats.append({"id": "x", "messages": 1})
    for _ in range(50):  # the next 15 s refresh redraws the strip (it re-arms after painting)
        if page.locator("#start-here").count() == 0:
            break
        page.clock.fast_forward(16000)
        page.wait_for_timeout(100)
    expect(page.locator("#start-here")).to_have_count(0)


def test_start_here_counts_an_empty_chat_and_an_unfinished_run_as_not_done(page, seeded):
    start_page(page, seeded, chats=[{"id": "x", "messages": 0}])
    expect(page.locator("#start-here")).to_be_visible()
    assert start_state(page)["chat"] == "false"
    assert start_state(page)["run"] == "false"
    page.unroute_all()
    start_page(page, seeded, chats=[{"id": "x", "messages": 0}], stores=[])
    expect(page.locator("#start-here")).to_be_visible()
    assert start_state(page)["store"] == "false"


def test_start_here_counts_only_a_usable_model(page, seeded):
    keyed = [{"id": "k", "kind": "anthropic", "key": "sk-...abcd"}]
    unkeyed = [{"id": "k", "kind": "anthropic", "key": None}]
    for models, connections, done in [
        ([{**GOOD_MODEL, "enabled": False}], keyed, "false"),
        ([GOOD_MODEL], unkeyed, "false"),
        ([GOOD_MODEL], keyed, "true"),
    ]:
        page.unroute_all()
        start_page(page, seeded, models=models, connections=connections)
        expect(page.locator("#start-here")).to_be_visible()
        assert start_state(page)["ai"] == done


def test_start_here_survives_a_failed_step(page, seeded):
    page.expected = ["500"]
    page.route("**/ai", lambda route: route.fulfill(status=500, json={"detail": "down"}))
    page.route("**/chats", lambda route: route.fulfill(json=[]))  # not the module's chats
    page.goto(seeded.url + "/")
    expect(page.locator('main a.row-link[href="/c/c"]')).to_be_visible()
    expect(page.locator("#start-here")).to_be_visible()
    assert start_state(page)["ai"] == "false"


def test_start_here_store_step_opens_the_form(page, seeded):
    start_page(page, seeded)
    page.get_by_role("link", name="Pick or make a store").click()
    expect(page.locator("#form")).to_be_visible()
    expect(page.locator("#f-name")).to_be_focused()  # which scrolls the form into view
    page.locator("#f-cancel").click()
    page.get_by_role("button", name="+ New store").click()
    expect(page.locator("#f-name")).to_be_focused()


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


def test_new_store_form_defaults_to_documents(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    select = page.locator("#f-type")
    expect(select).to_have_value("documents")
    expect(select.locator("option:checked")).to_contain_text("Documents")
    expect(page.locator("#f-recipe option")).to_have_count(1)
    assert [o.get_attribute("value") for o in page.locator("#f-recipe option").all()] == [
        "documents"
    ]
    expect(page.locator("#f-sum")).to_be_hidden()  # summaries are for the keyword lists
    select.select_option("science")
    expect(page.locator("#f-sum")).to_be_visible()


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
    # no keyword lists yet, so the import pays to make them: its price sits beside Import,
    # and the one click on Import queues it
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text("Added 3 worked examples.")
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

    # Unpriced means no AI model to run it: the server, pricing it again on confirm, refuses.
    def refused(route):
        if not json.loads(route.request.post_data)["confirm"]:
            return route.fallback()
        no_ai = "no AI model is set up for the 'keywords' job; choose one on the Settings page"
        route.fulfill(status=503, json={"detail": no_ai})

    page.route("**/collections/unp/uploads?*", unpriced)
    page.route("**/collections/unp/imports", refused)
    imports = []
    page.on("request", lambda r: imports.append(r) if r.url.endswith("/imports") else None)
    notes = tmp_path / "notes.jsonl"
    notes.write_text(json.dumps({"id": "u1", "text": "a note about reefs"}))
    page.goto(seeded.url + "/c/unp")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("1 row")
    # the cost is shown beside Import before anything is sent
    expect(preview).to_contain_text("cost unknown: no AI model is set up")
    page.wait_for_timeout(200)
    assert imports == []  # nothing queued until Import
    page.expected = ["503"]
    page.locator("#imp-items-import").click()
    expect(page.locator("#imp-items-msg")).to_have_text(
        "no AI model is set up for the 'keywords' job; choose one on the Settings page"
    )
    assert len(imports) == 1 and json.loads(imports[0].post_data)["confirm"] is True
    expect(page.locator("#imp-items-import")).to_be_enabled()  # a refusal can be tried again
    assert (
        httpx.get(f"{seeded.url}/jobs", params={"collection": "unp", "kind": "import"}).json() == []
    )


def import_posts(page, name):
    """The import POSTs that confirm (queue) on store `name`, as they are sent."""
    sent = []
    path = f"/collections/{name}/imports"
    page.on(
        "request",
        lambda r: (
            sent.append(r)
            if r.url.endswith(path) and r.method == "POST" and json.loads(r.post_data)["confirm"]
            else None
        ),
    )
    return sent


def import_jobs(seeded, name):
    params = {"collection": name, "kind": "import"}
    return httpx.get(f"{seeded.url}/jobs", params=params).json()


def test_store_import_queues_on_one_click_and_says_what_it_added(page, seeded, tmp_path):
    documents_store(seeded, "oneclick", passages=False)
    handbook = tmp_path / "handbook.md"
    handbook.write_text(two_sections())
    queued = import_posts(page, "oneclick")
    page.goto(seeded.url + "/c/oneclick")
    upload_and_preview(page, "items", handbook)
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("2 passages")
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text("Added 2 passages.")
    expect(page.get_by_role("button", name="Start")).to_have_count(0)
    assert len(queued) == 1
    assert len(import_jobs(seeded, "oneclick")) == 1
    expect(page.locator("#head")).to_contain_text("2 passages")


def test_paid_store_import_queues_once_on_a_double_click(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "dbl", "qa_type": "science"})
    r = httpx.post(
        f"{seeded.url}/collections/dbl/documents", json={"items": ITEMS_ONE, "confirm": True}
    )
    assert r.status_code == 200, r.text
    rows = [{"id": f"n{i}", "text": f"note {i} about reefs"} for i in range(3)]
    notes = tmp_path / "notes.jsonl"
    notes.write_text("\n".join(json.dumps(x) for x in rows))
    queued = import_posts(page, "dbl")
    uploaded = []

    def keep(route):
        response = route.fetch()
        uploaded.append(response.json())
        route.fulfill(response=response)

    page.route("**/collections/dbl/uploads?*", keep)
    page.goto(seeded.url + "/c/dbl")
    upload_and_preview(page, "items", notes)
    expect(page.locator("#imp-items-import")).to_be_enabled()
    estimate = uploaded[0]["estimate"]
    assert estimate["est_usd"] > 0  # no keyword lists yet: the import pays to make them
    price = "< $0.01" if estimate["est_usd"] < 0.01 else f"${estimate['est_usd']:.2f}"
    calls = estimate["calls"]
    preview = page.locator("#imp-items-preview")
    # before the click, the price and the AI calls sit beside Import
    expect(preview).to_contain_text(f"{price} · {calls} AI call{'' if calls == 1 else 's'}")
    page.locator("#imp-items-import").dblclick()
    expect(preview).to_have_text("Added 3 worked examples.")
    assert len(queued) == 1
    assert len(import_jobs(seeded, "dbl")) == 1


@pytest.mark.parametrize("edit", ["source", "mapping"])
def test_an_edit_while_import_is_sent_does_not_arm_it_again(page, seeded, tmp_path, edit):
    name = f"mid-{edit}"
    seeded.build(name)  # keyword lists made: the import is free
    notes = tmp_path / "notes.jsonl"
    notes.write_text(json.dumps({"id": "m1", "text": "a note about reefs", "note": "reefs"}))
    queued = import_posts(page, name)
    page.goto(f"{seeded.url}/c/{name}")
    upload_and_preview(page, "items", notes)
    button = page.locator("#imp-items-import")
    expect(button).to_be_enabled()
    held = []
    page.route(
        f"**/collections/{name}/imports",
        lambda r: held.append(r) if json.loads(r.request.post_data)["confirm"] else r.fallback(),
    )
    button.click()
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, "Import was never sent"
    if edit == "source":
        page.locator("#imp-items-source").fill("renamed")
    else:
        page.locator('#imp-items-preview select[data-field="topic"]').select_option("note")
    page.wait_for_timeout(300)  # a fresh estimate, had one been asked for, is in
    expect(button).to_be_disabled()
    held[0].continue_()
    expect(page.locator("#imp-items-preview")).to_contain_text("Added 1 ")
    expect(button).to_have_count(0)
    assert len(queued) == 1
    assert len(import_jobs(seeded, name)) == 1


def test_an_import_refused_after_an_edit_is_priced_again_before_it_can_be_retried(
    page, seeded, tmp_path
):
    seeded.build("midrefused")
    notes = tmp_path / "notes.jsonl"
    notes.write_text(json.dumps({"id": "m1", "text": "a note about reefs"}))
    page.goto(f"{seeded.url}/c/midrefused")
    upload_and_preview(page, "items", notes)
    button = page.locator("#imp-items-import")
    expect(button).to_be_enabled()
    held, previews = [], []

    def route(r):
        sent = json.loads(r.request.post_data)
        if sent["confirm"]:
            return held.append(r)
        previews.append(sent["source_name"])
        r.fallback()

    page.route("**/collections/midrefused/imports", route)
    button.click()
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, "Import was never sent"
    page.locator("#imp-items-source").fill("renamed")
    page.expected = ["503"]
    held[0].fulfill(status=503, json={"detail": "busy"})
    expect(page.locator("#imp-items-msg")).to_have_text("busy")
    expect(button).to_be_enabled()  # once the fresh estimate is in
    assert previews == ["renamed"]


def test_store_import_says_it_waits_behind_the_stores_other_job(page, seeded, tmp_path):
    documents_store(seeded, "impwait", passages=False)
    handbook = tmp_path / "handbook.md"
    handbook.write_text(two_sections())
    polls = []

    def job(route):
        polls.append(route.request.url)
        if len(polls) <= 3:
            route.fulfill(json={"id": "x", "status": "queued", "progress": 0})
        else:
            route.fallback()

    page.route("**/jobs/*", job)
    page.goto(seeded.url + "/c/impwait")
    upload_and_preview(page, "items", handbook)
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("2 passages")
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text("Waiting for the store's other job…")
    expect(preview).to_have_text("Added 2 passages.")


def test_a_stopped_store_import_says_its_status_in_words(page, seeded, tmp_path):
    documents_store(seeded, "impstop", passages=False)
    handbook = tmp_path / "handbook.md"
    handbook.write_text(two_sections())
    stopped = {"id": "x", "status": "stopped_budget", "error": "the monthly cap is reached"}
    page.route("**/jobs/*", lambda route: route.fulfill(json=stopped))
    page.goto(seeded.url + "/c/impstop")
    upload_and_preview(page, "items", handbook)
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("2 passages")
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text("The import stopped budget: the monthly cap is reached")


def test_store_import_reimport_counts_what_was_already_there(page, seeded, tmp_path):
    documents_store(seeded, "reimp", passages=False)
    handbook = tmp_path / "handbook.md"
    handbook.write_text(two_sections())
    page.goto(seeded.url + "/c/reimp")
    preview = page.locator("#imp-items-preview")
    for said in ("Added 2 passages.", "Added 0 passages. 2 already in the store."):
        upload_and_preview(page, "items", handbook)
        expect(page.locator("#imp-items-import")).to_be_enabled()
        page.locator("#imp-items-import").click()
        expect(preview).to_have_text(said)


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
    # no keyword lists yet, so the import pays to make them: its price sits beside Import,
    # and the one click on Import queues it
    page.locator("#imp-items-import").click()
    expect(page.locator("#imp-items-preview")).to_have_text("Added 2 worked examples.")
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
    expect(left).to_contain_text("Retrieval left out: upload:notes")
    expect(left).to_contain_text("2 questions' evidence is switched off")
    expect(left).to_contain_text("3 leaked questions left out")


# -- Settings page ------------------------------------------------------------------------


@contextmanager
def search_defaults_restored(seeded):
    """Puts what a test changes back: store c's candidates, and every default but the fixture's."""
    try:
        yield
    finally:
        httpx.patch(seeded.url + "/collections/c", json={"chat_k": None})
        httpx.put(
            seeded.url + "/search-defaults", json={f: None for f in search_defaults.RECOMMENDED}
        )
        httpx.put(seeded.url + "/search-defaults", json={"rewrite": False})


def test_store_page_shows_its_chat_settings_and_resets_to_settings(page, seeded):
    with search_defaults_restored(seeded):
        assert httpx.patch(seeded.url + "/collections/c", json={"chat_k": 12}).status_code == 200
        page.goto(seeded.url + "/c/c")
        row = page.locator('#chat-settings [data-chat-field="candidates"]')
        expect(row.locator("[data-value]")).to_have_text("12")
        expect(row.locator("[data-from]")).to_have_text("saved on this store")
        row.get_by_role("button", name="Reset to Settings").click()
        expect(row.locator("[data-value]")).to_have_text("20")
        expect(row.locator("[data-from]")).to_have_text("recommended")
        expect(row.get_by_role("button", name="Reset to Settings")).to_have_count(0)
        assert httpx.get(seeded.url + "/collections/c").json()["chat_k"] is None


def test_settings_page_search_defaults_save_and_reset_to_recommended(page, seeded):
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        box = page.locator('#search [name="candidates"]')
        box.fill("30")
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["candidates"] == 30
        page.get_by_role("button", name="Reset to recommended").click()
        expect(page.locator("#search-msg")).to_have_text("Back to recommended")
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["candidates"] == 20
        expect(box).to_have_value("20")
        expect(box.locator("..").locator("[data-recommended]")).to_have_text("Recommended: 20")


def test_settings_page_says_what_each_job_does_and_what_works_without_ai(page, seeded):
    page.goto(seeded.url + "/settings")
    expect(page.locator("#head .sub")).to_have_text(
        "AI writes chat answers, rewrites follow-ups, and builds keyword tags. "
        "Search and eval runs on retrieval work without it."
    )
    expect(page.locator('#roles label:has([data-role="rewrite"]) .muted')).to_have_text(
        "Rewrites a chat follow-up as one standalone question"
    )
    expect(
        page.get_by_text("Used by chat and POST /chat, and by POST /query when no method is given.")
    ).to_be_visible()


def test_settings_says_why_rerank_is_off_by_default(page, seeded):
    def with_note(route):
        body = route.fetch().json()
        route.fulfill(
            json={**body, "notes": {"rerank": "Reranking is off until the model is ready."}}
        )

    page.route("**/search-defaults", with_note)
    page.goto(seeded.url + "/settings")
    expect(page.locator("#search [data-rerank-note]")).to_have_text(
        "Reranking is off until the model is ready."
    )


def test_settings_page_picks_a_reranker_and_a_pool(page, seeded):
    listed = httpx.get(seeded.url + "/search-defaults").json()
    names = [r["name"] for r in listed["rerankers"]]
    own = listed["recommended"]["rerank_model"]  # the fixture's own, not fastembed's
    turbo = "jinaai/jina-reranker-v1-turbo-en"
    tiny_name = "jinaai/jina-reranker-v1-tiny-en"
    assert own not in names and turbo in names and tiny_name in names
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        select = page.locator('#search [name="rerank_model"]')
        expect(select.locator("option")).to_have_count(len(names) + 1)
        for name in [*names, own]:
            expect(select.locator(f'option[value="{name}"]')).to_have_count(1)
        expect(select.locator(f'option[value="{own}"]')).to_contain_text("(recommended)")
        tiny = select.locator(f'option[value="{tiny_name}"]')
        expect(tiny).to_contain_text("about 1.5 s for 20 passages on an M4 Pro")
        bge = select.locator('option[value^="BAAI/bge-reranker-"]').first
        expect(bge).to_contain_text("slower; for a faster machine")
        select.select_option(turbo)
        page.locator('#search [name="rerank_pool"]').fill("30")
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        expect(page.locator("#search [data-downloading]")).to_have_count(0)  # the fake's is ready
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["rerank_model"] == turbo
        assert got["in_force"]["rerank_pool"] == 30


def test_settings_shows_the_reranker_warm_up_the_service_reports(page, seeded):
    state = {"rerank_status": "loading", "rerank_error": None}
    puts: list[dict] = []

    def reported(route):
        if route.request.method == "PUT":
            puts.append(route.request.post_data_json)
        route.fulfill(json={**route.fetch().json(), **state})

    page.route("**/search-defaults", reported)
    page.goto(seeded.url + "/settings")
    hint, error = (
        page.locator("#search [data-downloading]"),
        page.locator("#search [data-rerank-error]"),
    )
    expect(hint).to_have_text(
        "Downloading in the background; chat merges by rank until it is ready."
    )
    expect(error).to_have_count(0)

    state.update(rerank_status="failed", rerank_error="hub down")
    page.reload()
    expect(error).to_have_text(
        "Could not load it: hub down. Chat merges by rank; Save tries again."
    )
    expect(hint).to_have_count(0)
    with search_defaults_restored(seeded):
        page.locator("#search-go").click()  # nothing edited: Save sends the reranker again
        expect(page.locator("#search-msg")).to_have_text("Saved")
        assert puts == [{"rerank_model": None}]  # as saved: the service's own

    state.update(rerank_status="ready", rerank_error=None)
    page.reload()
    expect(page.locator('#search [name="rerank_model"]')).to_be_visible()
    expect(hint).to_have_count(0)
    expect(error).to_have_count(0)


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
    expect(options).to_have_count(1)  # Documents, the default type: its tagged recipe comes later
    expect(options.first).to_have_attribute("value", "documents")
    expect(options.first).to_contain_text("Your passages as uploaded")
    page.locator("#f-type").select_option("science")
    expect(options).to_have_count(3)
    expect(options.first).to_have_attribute("value", "paper_2024")
    expect(options.first).to_contain_text("Paper 2024")
    expect(options.first).to_contain_text("your three keyword lists")
    values = [o.get_attribute("value") for o in options.all()]
    assert values == ["paper_2024", "worked_examples", "reference"]
    expect(kind.locator("option[value=worked_examples]")).to_contain_text("Worked examples")
    kind.select_option("worked_examples")
    page.get_by_label("Name").fill("recipe-store")
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/c/recipe-store")
    assert httpx.get(f"{seeded.url}/collections/recipe-store").json()["recipe"] == "worked_examples"


def test_an_empty_documents_store_offers_no_benchmark(page, seeded):
    r = httpx.post(f"{seeded.url}/collections", json={"name": "docs0", "qa_type": "documents"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/c/docs0")
    expect(page.get_by_text("0 passages.", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name=re.compile(r"^(Load|Finish loading)"))).to_have_count(0)
    expect(page.get_by_text("built-in benchmark")).to_have_count(0)
    assert "undefined" not in page.locator("main").inner_text()


# -- a Documents store -------------------------------------------------------------------------

NEEDS_TAGS = "needs keyword tags: Add keyword tags on the store page"
NOT_BUILT = (
    "Not built. Keyword tags let keyword_rag search this store: an AI proposes a vocabulary "
    "from your passages, then tags each passage by similarity."
)


def two_sections() -> str:
    """A .md of two sections, each too long to share a chunk: two passages."""
    return "# Alpha\n\n" + "alpha " * 160 + "\n\n# Brine\n\n" + "brine " * 160


def documents_store(seeded, name, passages=True):
    """A Documents store `name`; with `passages`, holding two_sections() as x.md."""
    r = httpx.post(f"{seeded.url}/collections", json={"name": name, "qa_type": "documents"})
    assert r.status_code == 201, r.text
    if not passages:
        return
    up = httpx.post(
        f"{seeded.url}/collections/{name}/uploads",
        params={"filename": "x.md", "kind": "items"},
        content=two_sections().encode(),
    ).json()
    body = {
        "upload_id": up["upload_id"],
        "kind": "items",
        "source_name": "x",
        "mapping": up["preview"]["mapping"],
        "confirm": True,
    }
    r = httpx.post(f"{seeded.url}/collections/{name}/imports", json=body)
    assert r.status_code == 202, r.text
    assert seeded.wait(r.json()["job_id"])["status"] == "done"


def test_documents_store_imports_md_without_a_mapping_table(page, seeded, tmp_path):
    documents_store(seeded, "dmd", passages=False)
    handbook = tmp_path / "handbook.md"
    handbook.write_text(two_sections())
    page.goto(seeded.url + "/c/dmd")
    expect(page.locator("#imp-items-file")).to_have_attribute("accept", ".md,.txt")
    upload_and_preview(page, "items", handbook)
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("2 usable")
    expect(preview).to_contain_text("2 passages")
    expect(preview).to_contain_text("alpha alpha")  # a sample of the passages
    expect(preview.locator("select[data-field]")).to_have_count(0)
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text("Added 2 passages.")
    jobs_done(seeded, "dmd")
    assert httpx.get(f"{seeded.url}/collections/dmd").json()["count"] == 2
    page.reload()
    expect(page.get_by_text("2 passages.", exact=True)).to_be_visible()
    expect(page.locator("#head .pill")).to_have_text("Ready")
    expect(page.get_by_text("Or load the built-in benchmark")).to_have_count(0)
    expect(page.get_by_text("How are items described?")).to_be_hidden()
    expect(page.locator("#kw-topic")).to_be_hidden()
    expect(page.get_by_role("button", name="Propose with AI")).to_be_hidden()
    page.get_by_role("button", name="New run").click()
    expect(page.locator("input[name=method][value=hybrid]")).to_be_checked()
    expect(page.locator("#rf-answer")).not_to_be_checked()
    expect(page.locator("input[name=method][value=kw_individual]")).to_have_count(0)


def tag_jobs(seeded, name):
    return httpx.get(f"{seeded.url}/jobs", params={"collection": name, "kind": "tag_build"}).json()


def test_documents_store_adds_keyword_tags_after_a_confirm(page, seeded):
    documents_store(seeded, "dtag")
    ai = seeded.app.state.ai
    fake_replies(ai, lambda p: '{"terms": ["alpha", "brine", "cocoa"]}')
    try:
        page.goto(seeded.url + "/c/dtag")
        tags = page.locator("#tags")
        expect(tags.locator("h2")).to_have_text("Keyword tags")
        expect(tags).to_contain_text(NOT_BUILT)
        page.get_by_role("button", name="Add keyword tags").click()
        confirm = page.locator("#tags-confirm")
        expect(confirm).to_contain_text("AI call")
        assert tag_jobs(seeded, "dtag") == []  # nothing queued before Start
        confirm.get_by_role("button", name="Start").click()
        deadline = time.time() + 10
        while not tag_jobs(seeded, "dtag") and time.time() < deadline:
            time.sleep(0.05)
        (job,) = tag_jobs(seeded, "dtag")
        assert seeded.wait(job["id"])["status"] == "done"
        terms = httpx.get(f"{seeded.url}/collections/dtag").json()["keyword_tags"]["terms"]
        expect(tags).to_contain_text(f"Built: {terms} terms.", timeout=15000)
        expect(page.get_by_role("button", name="Add keyword tags")).to_have_count(0)
    finally:
        fake_replies(ai, lambda p: "Answer: A")


def route_keyword_tags(page, name, shown):
    """GET /collections/{name} answers with keyword_tags = shown["tags"]."""

    def handle(route):
        got = route.fetch().json()
        got["keyword_tags"] = shown["tags"]
        route.fulfill(json=got)

    page.route(f"**/collections/{name}", handle)


def test_keyword_tags_section_shows_progress_and_resume(page, seeded):
    documents_store(seeded, "dprog", passages=False)
    shown = {"tags": {"state": "building", "job_id": "tagjob01", "progress": 0.5}}
    route_keyword_tags(page, "dprog", shown)
    resumes = []

    def resume(route):
        body = route.request.post_data_json
        resumes.append(body)
        reply = (
            {"job_id": "tagjob01"}
            if body.get("confirm")
            else {"estimate": {"est_usd": 0.02, "calls": 3}}
        )
        route.fulfill(json=reply)

    page.route("**/jobs/tagjob01/resume", resume)
    page.goto(seeded.url + "/c/dprog")
    tags = page.locator("#tags")
    expect(tags).to_contain_text("Adding keyword tags: 50%")
    expect(page.get_by_role("button", name="Add keyword tags")).to_have_count(0)
    shown["tags"] = {"state": "stopped", "job_id": "tagjob01"}
    page.reload()
    tags.get_by_role("button", name="Resume").click()
    confirm = page.locator("#tags-confirm")
    expect(confirm).to_contain_text("3 AI calls")
    assert resumes == [{"confirm": False}]  # the click only asks for the estimate
    confirm.get_by_role("button", name="Start").click()
    expect(confirm.get_by_role("button", name="Start")).to_have_count(0)
    assert resumes == [{"confirm": False}, {"confirm": True}]


def test_keyword_tags_message_goes_when_the_state_changes(page, seeded):
    documents_store(seeded, "dmsg", passages=False)
    shown = {"tags": {"state": "none"}}
    route_keyword_tags(page, "dmsg", shown)
    refusal = {"detail": "connect a model for keywords first"}
    page.route("**/collections/dmsg/keyword-tags", lambda r: r.fulfill(status=409, json=refusal))
    page.expected = ["409"]  # the refusal is the point
    page.clock.install()
    page.goto(seeded.url + "/c/dmsg")
    page.get_by_role("button", name="Add keyword tags").click()
    msg = page.locator("#tags-msg")
    expect(msg).to_have_text(refusal["detail"])
    shown["tags"] = {"state": "building", "job_id": "tagjob03", "progress": 0}
    page.clock.run_for(16000)  # the next poll repaints the section into the new state
    expect(page.locator("#tags")).to_contain_text("Adding keyword tags")
    expect(msg).to_have_text("")


def test_run_form_says_why_keyword_rag_is_off(page, seeded):
    documents_store(seeded, "dwhy", passages=False)
    page.goto(seeded.url + "/c/dwhy")
    page.get_by_role("button", name="New run").click()
    box = page.locator("input[name=method][value=keyword_rag]")
    expect(box).to_be_disabled()
    label = page.locator("#run-form label.check", has=box)
    expect(label).to_contain_text(f"({NEEDS_TAGS})")
    expect(
        page.locator("#run-form label.check", has=page.locator("[value=hybrid]"))
    ).not_to_contain_text("(")


def test_worker_page_names_a_tag_build_and_resumes_it(page, seeded):
    stopped = {
        "id": "tagjob02aaaa",
        "kind": "tag_build",
        "collection": "docs",
        "status": "failed",
        "error": None,
        "progress": 0.4,
        "created_at": "2026-11-08T10:00:00",
        "updated_at": "2026-11-08T10:05:00",
    }

    def jobs(route):
        route.fulfill(json=[stopped, *route.fetch().json()])

    def status(route):
        got = route.fetch().json()
        got["problems"] = ["Tag_build tagjob02 on docs failed"]
        route.fulfill(json=got)

    resumes = []

    def resume(route):
        resumes.append(route.request.post_data_json)
        route.fulfill(json={"estimate": {"est_usd": 0.02, "calls": 3}})

    page.route(lambda url: url.endswith("/jobs?limit=200"), jobs)
    page.route("**/status", status)
    page.route("**/jobs/tagjob02aaaa/resume", resume)
    page.goto(seeded.url + "/worker")
    row = page.locator("[data-finished-job='tagjob02aaaa']")
    expect(row).to_contain_text("Adding keyword tags docs")
    expect(page.locator("[data-problem]")).to_have_text(
        "Adding keyword tags tagjob02 on docs failed"
    )
    expect(page.locator("#worker-problem")).to_have_text(
        "Adding keyword tags tagjob02 on docs failed"
    )
    row.get_by_role("button", name="Resume").click()
    expect(page.locator("[data-resume-row]")).to_contain_text("3 AI calls")
    assert resumes == [{"confirm": False}]


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
    expect(page.locator('#asked a[href="/eval-sets?store=noset"]')).to_have_text("Create one")
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
    "pipeline": {
        "search_from": "message",
        "rerank_text": "asked",
        "merged_by": "rerank",
        "rerank": True,
        "rewrite": False,
        "follow_up_turns": 2,
        "evidence_size": 8,
        "rrf_k": 60,
        "pool": 2,
        "notes": [],
        "passages_only": False,
        "rerank_model": "jinaai/jina-reranker-v1-tiny-en",
        "rerank_pool": 20,
    },
}
# The passages the answer was given, each with its number in the prompt: what [n] links to.
CHAT_ATTEMPT["evidence"] = [
    {**c, "rank": c["n"], "score": 1.2, "relevance": 0.77, "channels": ["meaning", "words"]}
    for c in CHAT_ATTEMPT["citations"]
]
# A store's decision as the pipeline saves it.
CHAT_DECISION = {
    "store": "c",
    "method": "hybrid",
    "candidates": 20,
    "search": {"weights": {"meaning": 1, "words": 1, "keywords": 1}, "query_tags": "similarity"},
    "tags": None,
    "found": 20,
    "used": 3,
    "note": None,
    "usd": 0.0,
    "error": None,
}


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
    expect(picker.locator("label", has=page.locator('input[value="c"]'))).to_contain_text(
        "Exam multiple choice"
    )
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

    # The header's edits are saved on the chat. Evidence size and rewrite are in the panel.
    expect(page.locator("#chat-head").get_by_label("Evidence size")).to_have_count(0)
    expect(page.locator("#chat-head").get_by_label(re.compile("Rewrite"))).to_have_count(0)
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.locator("#add-store").select_option("vo")
    expect(page.locator("#chips .chip")).to_have_count(2)
    expect(edit_box(settings_panel(page), "vo")).to_be_visible()  # the panel has it too
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.get_by_role("button", name="Remove c").click()
    expect(page.locator("#chips .chip")).to_have_text([re.compile("^vo")])
    expect(page.get_by_role("button", name="Remove vo")).to_be_disabled()  # a chat needs one
    expect(edit_box(settings_panel(page), "c")).to_have_count(0)
    saved = httpx.get(f"{seeded.url}/chats/{second}").json()
    assert saved["stores"] == ["vo"]

    # Clicking a chat opens it.
    chat_link(page, first).click()
    expect(page).to_have_url(re.compile(f"#{first}$"))
    expect(page.locator("#log .user")).to_have_text("the first of two saved chats")

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


def test_chat_shows_a_cached_answer(page, seeded):
    question = "which catalyst speeds the cached enzyme assay?"
    open_chat(page, seeded)
    chat_ask(page, question)
    expect(page.locator("#log .reply .meta.muted")).to_have_text(re.compile(r"^Cost [^·]+$"))
    open_chat(page, seeded)  # the same store, so the same passages
    chat_ask(page, question)
    expect(page.locator("#log .reply .meta.muted")).to_have_text("Cost $0.00 · answer cached")


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


def test_a_rerun_and_an_answer_landing_while_their_chat_reopens_are_kept(page, seeded):
    chat_id = make_chat(seeded, ["c"], "re-run reopening chat")
    other = make_chat(seeded, ["vo"], "re-run reopening other")
    page.goto(f"{seeded.url}/chat#{chat_id}")
    expect(page.locator("#chips .chip")).to_have_count(1)
    chat_ask(page, "enzyme kinetics rate experiment")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    held = {"posts": [], "reopen": None, "hold": False}

    def post(route):
        held["posts"].append(route)  # sent on from the test

    def get(route):
        if not held["hold"] or route.request.method != "GET" or held["reopen"]:
            route.fallback()
            return
        held["reopen"] = (route, route.fetch())  # read now, before the reply is saved

    def until(ready):
        for _ in range(200):
            if ready():
                return
            page.wait_for_timeout(20)
        raise AssertionError("never happened")

    def reopen_while(check):
        """With the clicked request held, reopens the chat, its read made before the reply is
        saved (`check` says so), then lets the reply land, then the older read."""
        until(lambda: len(held["posts"]) == 1)
        chat_link(page, other).click()
        expect(page.locator("#chat-title")).to_have_text("re-run reopening other")
        held["hold"] = True
        chat_link(page, chat_id).click()
        until(lambda: held["reopen"] is not None)
        route = held["posts"].pop()
        with page.expect_request(lambda r: r.url.endswith("/budget")):  # the reply was taken
            route.fulfill(response=route.fetch())
        reopen, stale = held["reopen"]
        check(stale.json())
        held.update(hold=False, reopen=None)
        reopen.fulfill(response=stale)

    page.route(f"**/chats/{chat_id}/messages/1/attempts", post)
    page.route(f"**/chats/{chat_id}/messages/1/attempts/2/answer", post)
    page.route(f"**/chats/{chat_id}", get)
    panel = settings_panel(page)
    edit_box(panel, "c").get_by_label("Candidates").fill("2")
    panel.get_by_role("button", name="Re-run", exact=True).click()

    def no_rerun(read):
        assert len(read["messages"][0]["attempts"]) == 1

    reopen_while(no_rerun)
    second = turn.get_by_role("tab", name="Attempt 2")
    expect(second).to_have_attribute("aria-selected", "true")
    expect(turn.locator(".reply")).to_have_count(0)
    # The settings the re-run saved on the chat, read again after the older read.
    expect(edit_box(settings_panel(page), "c").get_by_label("Candidates")).to_have_value("2")

    turn.get_by_role("button", name="Answer from this attempt").click()

    def no_answer(read):
        assert read["messages"][0]["attempts"][1]["answer"] is None

    reopen_while(no_answer)
    expect(second).to_have_attribute("aria-selected", "true")
    expect(turn.locator(".reply .md")).to_have_text("Answer: A")


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
    step = retrieval_step(page)
    expect(count_line(step, "doomed").locator(".count")).to_have_text(
        re.compile(r"^doomed · deleted: \d+ candidates?, \d+ kept$")
    )
    expect(count_line(step, "c").locator(".count")).to_have_text(re.compile(r"^c: "))
    passages = step.locator('ol.evidence li[data-store="doomed"] .what')
    expect(passages.first).to_contain_text(" · doomed · deleted · ")
    expect(edit_box(settings_panel(page), "doomed")).to_have_count(0)  # nothing to set there
    with page.expect_response(lambda r: r.request.method == "PATCH"):
        page.get_by_role("button", name="Remove doomed").click()
    expect(chips).to_have_count(1)
    assert httpx.get(f"{seeded.url}/chats/{chat_id}").json()["stores"] == ["c"]
    # Off the chat, its old rows still name it, and still say it is deleted.
    expect(count_line(step, "doomed")).to_contain_text("doomed · deleted")
    page.reload()
    step = retrieval_step(page)
    expect(count_line(step, "doomed")).to_contain_text("doomed · deleted")


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


def test_chat_without_an_answer_model_sends_and_shows_passages_only(page, seeded):
    ai = seeded.app.state.ai
    listed = httpx.get(seeded.url + "/ai").json()
    page.route("**/ai", lambda route: route.fulfill(json={**listed, "models": []}))
    ai.role_model = lambda role: None  # the server has no model for any role
    try:
        open_chat(page, seeded)
        note = page.locator("#compose #note")
        expect(note).to_have_text("Connect AI in Settings to get written answers")
        expect(note.get_by_role("link", name="Settings")).to_have_attribute("href", "/settings")
        expect(page.locator("#model")).to_be_hidden()
        expect(page.locator("#send")).to_be_enabled()
        chat_ask(page, "enzyme kinetics rate experiment")
        turn = page.locator("#msg-1")
        passages = turn.locator(".msg.passages")
        expect(passages.locator(".said")).to_have_text("Passages only: no answer model")
        expect(passages.locator("ol.evidence li")).to_have_count(3)  # open, under the message
        expect(passages.locator("ol.evidence")).to_contain_text("enzyme kinetics rate experiment")
        expect(turn.locator(".reply, .err")).to_have_count(0)
        expect(turn.get_by_role("button", name="Answer from this attempt")).to_have_count(0)
        expect(page.locator("#spoken")).to_have_text("Passages only: no answer model")
        step = retrieval_step(page)
        expect(step.locator("ol.evidence")).to_have_count(0)  # listed once, above
        expect(step).to_contain_text("The passages are listed above.")
        # A re-run without AI is passages only too.
        with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts")):
            settings_panel(page).get_by_role("button", name="Re-run", exact=True).click()
        expect(turn.get_by_role("tab", name="Attempt 2")).to_have_attribute("aria-selected", "true")
        expect(turn.locator(".msg.passages .said")).to_have_text("Passages only: no answer model")
    finally:
        del ai.role_model


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


def test_added_file_note_survives_a_reload(page, seeded, tmp_path):
    free_store(seeded, "kept-notes")
    chat_id = chat_upload(page, seeded, "kept-notes", tmp_path, "kept.md", "# Kept\ncoral reefs")
    note = page.locator("#log .note")
    expect(note).to_have_text("Added kept.md to kept-notes")
    page.evaluate("localStorage.setItem('chat-notes:gone', '[]')")  # a chat deleted elsewhere
    page.reload()
    expect(page.locator("#chips .chip")).to_have_count(2)
    expect(note).to_have_text("Added kept.md to kept-notes")
    page.wait_for_function("localStorage.getItem('chat-notes:gone') === null")
    row = page.locator("#chat-list li", has=page.locator(f'a[href="#{chat_id}"]'))
    row.get_by_role("button", name="Delete").click()
    page.get_by_role("button", name="Yes, delete").click()
    expect(chat_link(page, chat_id)).to_have_count(0)
    assert page.evaluate(f"localStorage.getItem('chat-notes:{chat_id}')") is None


def test_chat_works_when_storage_throws(page, seeded, tmp_path):
    page.add_init_script(
        """for (const m of ["getItem", "setItem", "removeItem"]) {
          Storage.prototype[m] = () => { throw new Error("storage is blocked"); };
        }"""
    )
    free_store(seeded, "no-storage")
    chat_upload(page, seeded, "no-storage", tmp_path, "blocked.md", "# Blocked\ncoral reefs")
    expect(page.locator("#log .note")).to_have_text("Added blocked.md to no-storage")
    expect(page.locator("#add-doc")).to_be_enabled()
    open_chat(page, seeded, ("no-storage",))  # an existing chat opens too
    expect(page.locator("#add-doc")).to_be_enabled()


def test_a_chat_list_without_the_open_chat_keeps_its_notes(page, seeded, tmp_path):
    free_store(seeded, "stale-list")
    chat_id = chat_upload(page, seeded, "stale-list", tmp_path, "kept.md", "# Kept\ncoral reefs")
    note = page.locator("#log .note")
    expect(note).to_have_text("Added kept.md to stale-list")
    # a list read before this chat was made (as from another tab), landing after its note
    page.route(
        "**/chats", lambda r: r.fulfill(json=[]) if r.request.method == "GET" else r.fallback()
    )
    with page.expect_response(lambda r: r.url.endswith("/chats") and r.request.method == "GET"):
        chat_ask(page, "What is alpha?")
    expect(page.locator("#log .reply")).to_have_count(1)
    page.wait_for_timeout(100)  # past the list's prune
    assert page.evaluate(f"localStorage.getItem('chat-notes:{chat_id}')") is not None
    page.unroute_all()
    page.reload()
    expect(page.locator("#log .msg.note")).to_have_text("Added kept.md to stale-list")


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


# -- The retrieval step under each answer, and the chat's settings panel --------------------


def retrieval_step(page, n=1):
    """Message n's Retrieval step, opened."""
    step = page.locator(f"#msg-{n} details.retrieval")
    if step.get_attribute("open") is None:
        step.locator(":scope > summary").click()
    expect(step).to_have_attribute("open", "")
    return step


def settings_panel(page):
    """The chat's one settings panel, under its latest message, opened."""
    panel = page.locator("#log details.panel")
    expect(panel).to_have_count(1)
    if panel.get_attribute("open") is None:
        panel.locator(":scope > summary").click()
    expect(panel).to_have_attribute("open", "")
    return panel


def count_line(step, store):
    """A store's line in an attempt's Retrieval step: what it found and what was kept."""
    return step.locator(f'ul.counts li[data-store="{store}"]')


def store_row(step, store):
    """A store's row in an old attempt's Retrieval step, as it was recorded."""
    return step.locator(f'table.decisions tr[data-store="{store}"]')


def edit_box(panel, store):
    return panel.locator(f'fieldset[data-edit="{store}"]')


def rerun_button(panel):
    return panel.get_by_role("button", name="Re-run", exact=True)


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


def test_retrieval_step_shows_counts_rerank_scores_and_notes(page, seeded):
    unavailable = "reranker unavailable: merged by rank"
    by_rank = {**CHAT_ATTEMPT["pipeline"], "merged_by": "rrf", "notes": [unavailable], "pool": 20}
    vo = {**CHAT_DECISION, "store": "vo", "method": "keyword_rag", "found": 3, "used": 0}
    vo.update(search={**vo["search"], "query_tags": "classifier"}, tags=["enzyme", "rate"])
    broke = {**CHAT_DECISION, "store": "broke", "found": 0, "used": 0}
    broke["error"] = "budget used up"
    first = {
        "search_text": "enzyme kinetics",
        "decisions": [CHAT_DECISION, vo, broke],
        "evidence": [{**e, "relevance": None} for e in CHAT_ATTEMPT["evidence"]],  # by RRF
        "pipeline": by_rank,
    }
    follow_up = {**CHAT_ATTEMPT["pipeline"], "search_from": "joined", "rerank_text": "the rate?"}
    second = {
        "n": 2,
        "search_text": "enzyme kinetics\nthe rate?",
        "decisions": [CHAT_DECISION],
        "pipeline": follow_up,
    }
    unnamed = {**CHAT_ATTEMPT["pipeline"], "rerank_model": None}  # no reranker recorded
    third = {"n": 3, "search_text": "asked", "decisions": [CHAT_DECISION], "pipeline": unnamed}
    replies = [first, second, third]
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, **replies.pop(0))))
    chat_id = open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "enzyme kinetics")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    step = turn.locator("details.retrieval")
    expect(step.locator(":scope > summary")).to_have_text("Retrieval step")
    expect(step).not_to_have_attribute("open", "")  # collapsed until opened
    step = retrieval_step(page)
    expect(step.locator(".searched")).to_have_text("enzyme kinetics")
    expect(step.locator(".from")).to_have_text("as typed")
    expect(count_line(step, "c").locator(".count")).to_have_text("c: 20 candidates, 3 kept")
    expect(count_line(step, "vo").locator(".count")).to_have_text("vo: 3 candidates, 0 kept")
    expect(count_line(step, "vo").locator(".method")).to_have_text(
        " · Keyword-RAG · query tags: classifier · enzyme, rate"
    )
    broken = count_line(step, "broke")
    expect(broken.locator(".note")).to_have_text(" · error: budget used up")
    expect(broken).to_have_class(re.compile("dropped"))
    expect(count_line(step, "c")).not_to_have_class(re.compile("dropped"))
    expect(step.locator(".notes li")).to_have_text([unavailable])
    expect(step.locator(".reranked")).to_have_count(0)  # merged by rank: no reranker named
    expect(step.locator(".pooled")).to_have_text("Pooled 20 of 23 (Rerank up to 20)")
    passage = step.locator("#ev-1-1-1")
    expect(passage.locator(".what")).to_have_text("[1] enzyme-item · c · hand · rank 1 in c")
    expect(passage.locator("details")).to_have_count(1)  # longer than 300 characters
    assert len(passage.locator("summary").inner_text()) <= 305

    # Closed again, a citation opens it at the passage it cites.
    step.locator(":scope > summary").click()
    expect(step).not_to_have_attribute("open", "")
    turn.locator(".reply a.cite").first.click()
    expect(step).to_have_attribute("open", "")
    expect(page.locator("#log li:target")).to_have_attribute("id", "ev-1-1-1")
    expect(page).to_have_url(re.compile(f"#{chat_id}$"))

    # A follow-up, reranked against the message alone by the reranker it names; what is
    # open survives the log's redraw.
    passage.locator("summary").click()
    expect(passage.locator("details")).to_have_attribute("open", "")
    chat_ask(page, "the rate?")
    expect(page.locator("#msg-2 .reply")).to_have_count(1)
    expect(step).to_have_attribute("open", "")
    expect(passage.locator("details")).to_have_attribute("open", "")
    expect(page.locator("#msg-2 details.retrieval")).not_to_have_attribute("open", "")
    two = retrieval_step(page, 2)
    expect(two.locator(".from")).to_have_text("with the previous message")
    reranked = two.locator(".reranked")
    expect(reranked).to_contain_text("Reranked against: the rate?")
    expect(reranked).to_contain_text("by jina-reranker-v1-tiny-en")
    expect(two.locator("#ev-2-1-1 .what")).to_have_text(
        "[1] enzyme-item · c · hand · rank 1 in c · rerank 0.77"
    )
    expect(two.locator(".notes")).to_have_count(0)
    chat_ask(page, "asked")
    expect(page.locator("#msg-3 .reply")).to_have_count(1)
    expect(retrieval_step(page, 3).locator(".reranked")).to_have_text("Reranked")  # no "by "


def test_an_old_attempt_shows_its_recorded_reasons_read_only(page, seeded):
    chat_id = make_chat(seeded, ["c", "vo"], "a chat from before the pipeline")
    r = httpx.post(f"{seeded.url}/chats/{chat_id}/messages", json={"content": "an old question"})
    assert r.status_code == 200, r.text
    old = {"method": "hybrid", "k": 8, "best": 0.6, "bar": 0.4, "usd": 0.0, "error": None}
    decisions = [
        {**old, "store": "c", "kept": True, "reason": "cleared its bar"},
        {**old, "store": "vo", "kept": False, "reason": "below its bar"},
        {**old, "store": "far", "method": None, "k": None, "kept": False, "reason": "not routed"},
        {**old, "store": "broke", "kept": False, "reason": "error", "error": "budget used up"},
    ]

    def old_shape(route):
        got = route.fetch()
        chat = got.json()
        chat["messages"][0]["attempts"][0].update(pipeline=None, decisions=decisions)
        route.fulfill(response=got, json=chat)

    page.route(f"**/chats/{chat_id}", old_shape)
    page.goto(f"{seeded.url}/chat#{chat_id}")
    step = retrieval_step(page)
    expect(step.locator(".searched")).to_have_text("an old question")
    expect(step.locator("table.decisions thead th")).to_have_text(["Store", "Method", "Result"])
    expect(store_row(step, "c").locator(".kept")).to_have_text("kept · cleared its bar")
    expect(store_row(step, "c")).not_to_have_class(re.compile("dropped"))
    expect(store_row(step, "vo").locator(".method")).to_have_text("hybrid")
    expect(store_row(step, "vo").locator(".kept")).to_have_text("dropped · below its bar")
    expect(store_row(step, "vo")).to_have_class(re.compile("dropped"))
    expect(store_row(step, "far").locator(".kept")).to_have_text("dropped · not routed")
    expect(store_row(step, "far").locator(".method")).to_have_text("–")
    expect(store_row(step, "broke").locator(".kept")).to_have_text(
        "dropped · error: budget used up"
    )
    expect(step.locator("ul.counts")).to_have_count(0)
    expect(step.locator("select, input, button")).to_have_count(0)  # read only
    expect(rerun_button(step)).to_have_count(0)
    # The chat's panel, outside it, re-runs the message with the settings in force.
    expect(rerun_button(settings_panel(page))).to_be_enabled()


def test_settings_panel_shows_only_under_the_latest_message_with_inherited_values(page, seeded):
    conn = seeded.app.state.conn
    search_defaults.save(conn, {"evidence_size": 6})
    try:
        chat_id = open_chat(page, seeded, ("c",))
        expect(page.locator("#log details.panel")).to_have_count(0)  # no message yet
        chat_ask(page, "enzyme kinetics rate experiment")
        expect(page.locator("#msg-1 details.panel")).to_have_count(1)
        chat_ask(page, "and the volcano?")
        expect(page.locator("#msg-2 .reply")).to_have_count(1)
        expect(page.locator("#log details.panel")).to_have_count(1)
        expect(page.locator("#msg-2 details.panel")).to_have_count(1)
        expect(page.locator("details.retrieval details.panel")).to_have_count(0)
        panel = settings_panel(page)
        size = panel.get_by_label("Evidence size")
        expect(size).to_have_value("")
        expect(size).to_have_attribute("placeholder", "6 · Settings")
        rerank = panel.get_by_label("Rerank", exact=True)
        expect(rerank.locator("option")).to_have_text(["Inherit (on · recommended)", "On", "Off"])
        expect(rerank).to_have_value("")
        rewrite = panel.get_by_label("Rewrite follow-ups")
        expect(rewrite.locator("option")).to_have_text(["Inherit (off · Settings)", "On", "Off"])
        c = edit_box(panel, "c")
        expect(c.locator("legend")).to_have_text("c")
        method = c.get_by_label("Method")
        expect(method.locator("option:checked")).to_have_text("Inherit (Hybrid · recommended)")
        expect(c.get_by_label("Candidates")).to_have_value("")
        expect(c.get_by_label("Candidates")).to_have_attribute("placeholder", "20 · recommended")
        expect(c.get_by_role("button", name="Set as default")).to_be_disabled()  # nothing to save

        # A value the chat holds is shown; the placeholder still says what blank inherits.
        r = httpx.patch(f"{seeded.url}/chats/{chat_id}", json={"evidence_size": 3, "rerank": False})
        assert r.status_code == 200, r.text
        page.reload()
        panel = settings_panel(page)
        size = panel.get_by_label("Evidence size")
        expect(size).to_have_value("3")
        expect(size).to_have_attribute("placeholder", "6 · Settings")
        rerank = panel.get_by_label("Rerank", exact=True)
        expect(rerank).to_have_value("off")
        expect(rerank.locator("option").first).to_have_text("Inherit (on · recommended)")

        # With Search defaults unread, what a held value inherits is unknown: its source is
        # said, never a 0.
        page.expected = ["500"]
        page.route(
            "**/search-defaults",
            lambda r: r.fulfill(status=500, json={"detail": "search defaults broke"}),
        )
        page.reload()
        panel = settings_panel(page)
        expect(panel.get_by_label("Evidence size")).to_have_attribute("placeholder", "Settings")
        rerank = panel.get_by_label("Rerank", exact=True)
        expect(rerank.locator("option").first).to_have_text("Inherit (Settings)")
        rewrite = panel.get_by_label("Rewrite follow-ups")  # the chat holds none: in force says
        expect(rewrite.locator("option").first).to_have_text("Inherit (off · Settings)")
    finally:
        search_defaults.save(conn, {"evidence_size": None})


def test_one_rerun_sends_every_change_and_the_panel_survives_a_reload(page, seeded):
    sent = attempt_posts(page)
    open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "enzyme kinetics rate experiment")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    panel = settings_panel(page)
    panel.get_by_label("Evidence size").fill("3")
    edit_box(panel, "c").get_by_label("Candidates").fill("5")
    with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts")) as got:
        rerun_button(panel).click()
    assert sent == [{"settings": {"evidence_size": 3}, "stores": {"c": {"candidates": 5}}}]
    ran = got.value.json()
    assert ran["pipeline"]["evidence_size"] == 3
    assert next(d for d in ran["decisions"] if d["store"] == "c")["candidates"] == 5
    expect(turn.get_by_role("tab", name="Attempt 2")).to_have_attribute("aria-selected", "true")
    page.reload()
    panel = settings_panel(page)
    expect(panel.get_by_label("Evidence size")).to_have_value("3")
    expect(edit_box(panel, "c").get_by_label("Candidates")).to_have_value("5")
    expect(edit_box(panel, "vo").get_by_label("Candidates")).to_have_value("")


def test_chat_methods_have_plain_labels_and_no_hybrid_rerank(page, seeded):
    chat_methods = httpx.get(f"{seeded.url}/catalog").json()["chat_methods"]
    open_chat(page, seeded, ("c",))  # c has no vocabulary
    chat_ask(page, "enzyme kinetics rate experiment")
    expect(page.locator("#msg-1 .reply")).to_have_count(1)
    method = edit_box(settings_panel(page), "c").get_by_label("Method")
    options = method.locator("option")
    expect(options.filter(has_text=re.compile("^Hybrid$"))).to_have_count(1)
    values = options.evaluate_all("os => os.map(o => o.value)")
    assert values == ["", *chat_methods]
    assert "hybrid_rerank" not in values and "iterative" not in values
    expect(method.locator('option[value="kw_individual"]')).to_have_text(
        "Keywords per list (may cost AI)"
    )
    keyword_rag = method.locator('option[value="keyword_rag"]')
    expect(keyword_rag).to_have_attribute("disabled", "")
    expect(keyword_rag).to_contain_text("needs a vocabulary")
    expect(method.locator('option[value="hybrid"]')).not_to_have_attribute("disabled", "")


def test_set_as_default_confirms_that_it_re_runs_then_saves(page, seeded):
    conn = seeded.app.state.conn
    before = db.get_chat_method(conn, "c"), db.get_chat_k(conn, "c")
    defaults = []
    page.on(
        "request",
        lambda r: defaults.append(r.url) if r.url.endswith("/stores/c/default") else None,
    )
    sent = attempt_posts(page)
    try:
        open_chat(page, seeded, ("c",))
        chat_ask(page, "enzyme kinetics rate experiment")
        turn = page.locator("#msg-1")
        expect(turn.locator(".reply")).to_have_count(1)
        expect(turn.get_by_role("tab")).to_have_count(0)  # one attempt
        c = edit_box(settings_panel(page), "c")
        default = c.get_by_role("button", name="Set as default")
        expect(default).to_be_disabled()  # no chat value and no edit
        c.get_by_label("Method").select_option("vector")
        c.get_by_label("Candidates").fill("5")
        expect(default).to_be_enabled()

        default.click()
        question = (
            "Save Vector search and 5 candidates as c's chat settings? "
            "This also re-runs this message."
        )
        expect(c.locator(".confirm .ask")).to_have_text(question)
        c.get_by_role("button", name="Cancel").click()
        expect(c.locator(".confirm")).to_have_count(0)
        assert defaults == [] and sent == []
        expect(c.get_by_label("Candidates")).to_have_value("5")  # the edit is kept

        default.click()
        with page.expect_response(lambda r: r.url.endswith("/stores/c/default")):
            c.get_by_role("button", name="Yes, save").click()
        expect(c.locator(".edit-note")).to_have_text("Saved as c's chat settings.")
        assert sent == [{"settings": {}, "stores": {"c": {"method": "vector", "candidates": 5}}}]
        expect(turn.get_by_role("tab")).to_have_count(2)  # the re-run came first
        expect(turn.get_by_role("tab", name="Attempt 2")).to_have_attribute("aria-selected", "true")
        chat_settings = httpx.get(f"{seeded.url}/collections/c").json()["chat_settings"]
        assert chat_settings["method"] == {"value": "vector", "from": "store"}
        assert chat_settings["candidates"] == {"value": 5, "from": "store"}
        # The chat holds them no more: they are the store's.
        method = c.get_by_label("Method")
        expect(method.locator("option:checked")).to_have_text("Inherit (Vector search · store)")
        expect(c.get_by_label("Candidates")).to_have_value("")
        expect(c.get_by_label("Candidates")).to_have_attribute("placeholder", "5 · store")
        expect(default).to_be_disabled()

        # A value the chat holds, with nothing unsent, is saved without a re-run.
        c.get_by_label("Candidates").fill("4")
        with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts")):
            rerun_button(settings_panel(page)).click()
        expect(turn.get_by_role("tab")).to_have_count(3)
        default.click()
        expect(c.locator(".confirm .ask")).to_have_text(
            "Save the inherited method and 4 candidates as c's chat settings?"
        )
        with page.expect_response(lambda r: r.url.endswith("/stores/c/default")):
            c.get_by_role("button", name="Yes, save").click()
        expect(c.locator(".edit-note")).to_have_text("Saved as c's chat settings.")
        assert len(sent) == 2
        expect(turn.get_by_role("tab")).to_have_count(3)
        assert db.get_chat_k(conn, "c") == 4
    finally:
        db.set_chat_method(conn, "c", before[0])
        db.set_chat_k(conn, "c", before[1])


def test_a_panel_edit_that_changes_nothing_is_dropped_and_set_as_default_clears(page, seeded):
    conn = seeded.app.state.conn
    before = db.get_chat_method(conn, "c"), db.get_chat_k(conn, "c")
    sent = attempt_posts(page)
    try:
        open_chat(page, seeded, ("c",))
        chat_ask(page, "enzyme kinetics rate experiment")
        turn = page.locator("#msg-1")
        expect(turn.locator(".reply")).to_have_count(1)
        panel = settings_panel(page)
        method = edit_box(panel, "c").get_by_label("Method")
        method.select_option("vector")
        with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts")):
            rerun_button(panel).click()  # the chat now holds vector for c
        expect(turn.get_by_role("tab")).to_have_count(2)

        # Away and back: no edit, so Set as default saves with no re-run.
        c = edit_box(settings_panel(page), "c")
        c.get_by_label("Method").select_option("hybrid")
        c.get_by_label("Method").select_option("vector")
        default = c.get_by_role("button", name="Set as default")
        default.click()
        ask = "Save Vector search and the inherited candidates as c's chat settings?"
        expect(c.locator(".confirm .ask")).to_have_text(ask)
        c.get_by_label("Candidates").fill("7")  # an edit while it asks: the question is gone
        expect(c.locator(".confirm")).to_have_count(0)
        c.get_by_label("Candidates").fill("")
        default.click()
        expect(c.locator(".confirm .ask")).to_have_text(ask)
        with page.expect_response(lambda r: r.url.endswith("/stores/c/default")):
            c.get_by_role("button", name="Yes, save").click()
        expect(c.locator(".edit-note")).to_have_text("Saved as c's chat settings.")
        assert len(sent) == 1
        expect(c.get_by_label("Method").locator("option:checked")).to_have_text(
            "Inherit (Vector search · store)"
        )
        expect(default).to_be_disabled()
        with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts")):
            rerun_button(settings_panel(page)).click()
        assert sent[-1] == {"settings": {}, "stores": {}}
    finally:
        db.set_chat_method(conn, "c", before[0])
        db.set_chat_k(conn, "c", before[1])


def test_rewrite_follow_ups_is_hidden_without_a_rewrite_model(page, seeded):
    chat_id = make_chat(seeded, ["c"], "a chat with no rewrite model")
    r = httpx.post(f"{seeded.url}/chats/{chat_id}/messages", json={"content": "enzyme kinetics"})
    assert r.status_code == 200, r.text
    page.goto(f"{seeded.url}/chat#{chat_id}")
    panel = settings_panel(page)
    expect(panel.get_by_label("Rewrite follow-ups")).to_be_visible()

    def no_rewrite_model(route):
        got = route.fetch()
        chat = got.json()
        chat["in_force"]["rewrite_available"] = False
        route.fulfill(response=got, json=chat)

    page.route(f"**/chats/{chat_id}", no_rewrite_model)
    page.reload()
    panel = settings_panel(page)
    expect(panel.get_by_label("Rerank", exact=True)).to_be_visible()
    expect(panel.get_by_label("Rewrite follow-ups")).to_be_hidden()


def test_a_rerun_that_rewrites_again_asks_first(page, seeded):
    sent = attempt_posts(page)
    open_chat(page, seeded, ("c",))
    chat_ask(page, "enzyme kinetics rate experiment")
    expect(page.locator("#msg-1 .reply")).to_have_count(1)
    chat_ask(page, "and the rate?")
    expect(page.locator("#msg-2 .reply")).to_have_count(1)
    expect(retrieval_step(page, 2).locator(".from")).to_have_text("with the previous message")
    panel = settings_panel(page)
    panel.get_by_label("Rewrite follow-ups").select_option("on")
    rerun_button(panel).click()
    question = "Re-run rewrites this follow-up with AI: one AI call."
    expect(panel.locator(".confirm .ask")).to_have_text(question)
    panel.get_by_role("button", name="Cancel").click()
    expect(panel.locator(".confirm")).to_have_count(0)
    assert sent == []
    expect(panel.get_by_label("Rewrite follow-ups")).to_have_value("on")  # the edit is kept

    rerun_button(panel).click()
    with page.expect_response(lambda r: r.url.endswith("/messages/2/attempts")):
        panel.get_by_role("button", name="Yes, re-run").click()
    assert sent == [{"settings": {"rewrite": True}, "stores": {}}]
    expect(retrieval_step(page, 2).locator(".from")).to_have_text("rewritten by AI")
    # Rewritten already, a re-run keeps that text: nothing to ask.
    with page.expect_response(lambda r: r.url.endswith("/messages/2/attempts")):
        rerun_button(settings_panel(page)).click()
    assert len(sent) == 2
    expect(page.locator("#msg-2").get_by_role("tab")).to_have_count(3)


def test_rerun_with_an_override_adds_an_attempt_tab_and_answer_from_it(page, seeded):
    sent = attempt_posts(page)
    chat_id = open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "enzyme kinetics rate experiment, re-run")  # a prompt no other test caches
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    expect(turn.get_by_role("tab")).to_have_count(0)  # one attempt: no tabs
    step = retrieval_step(page)
    expect(count_line(step, "vo").locator(".count")).to_contain_text("vo: 3 candidates")
    panel = settings_panel(page)
    vo = edit_box(panel, "vo")
    expect(vo.get_by_label("Method")).to_have_value("")
    expect(vo.get_by_label("Candidates")).to_have_value("")
    vo.get_by_label("Candidates").fill("2")
    with page.expect_response(lambda r: r.url.endswith(f"/chats/{chat_id}/messages/1/attempts")):
        rerun_button(panel).click()
    assert sent == [{"settings": {}, "stores": {"vo": {"candidates": 2}}}]  # only the change

    second = turn.get_by_role("tab", name="Attempt 2")
    expect(second).to_have_attribute("aria-selected", "true")
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "false")
    expect(step).to_have_attribute("open", "")
    expect(count_line(step, "vo").locator(".count")).to_contain_text("vo: 2 candidates")
    expect(turn.locator(".reply")).to_have_count(0)  # a re-run is not answered
    expect(edit_box(panel, "vo").get_by_label("Candidates")).to_have_value("2")  # now the chat's

    answer = turn.get_by_role("button", name="Answer from this attempt")
    with page.expect_response(lambda r: r.url.endswith("/messages/1/attempts/2/answer")):
        answer.click()
    expect(turn.locator(".reply .md")).to_have_text("Answer: A")
    expect(answer).to_have_count(0)
    saved = httpx.get(f"{seeded.url}/chats/{chat_id}").json()["messages"][0]["attempts"]
    assert saved[0]["answer"] and saved[1]["answer"] == "Answer: A"

    # The first attempt is still there, as it was; the panel stays under the message.
    turn.get_by_role("tab", name="Attempt 1").click()
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "true")
    expect(count_line(step, "vo").locator(".count")).to_contain_text("vo: 3 candidates")
    expect(turn.locator("details.panel")).to_have_count(1)

    # The chosen tab and an edit not yet re-run survive the log's redraw on the next reply,
    # and the panel moves under the newest message.
    chat_ask(page, "and the volcano?")
    expect(page.locator("#msg-2 .reply")).to_have_count(1)
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "true")
    expect(step).to_have_attribute("open", "")
    edit_box(settings_panel(page), "c").get_by_label("Candidates").fill("5")
    chat_ask(page, "and the market?")
    expect(page.locator("#msg-3 .reply")).to_have_count(1)
    expect(page.locator("#msg-3 details.panel")).to_have_count(1)
    expect(edit_box(settings_panel(page), "c").get_by_label("Candidates")).to_have_value("5")
    expect(turn.get_by_role("tab", name="Attempt 1")).to_have_attribute("aria-selected", "true")


def test_rerun_refused_shows_the_reason_and_keeps_the_edit(page, seeded):
    page.expected = ["422", "409"]  # refusals are the point
    sent = attempt_posts(page)
    open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "enzyme kinetics rate experiment")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply")).to_have_count(1)
    panel = settings_panel(page)
    c = edit_box(panel, "c")
    note = panel.locator(".panel-note")

    # A value out of bounds is said in the panel, and nothing is sent.
    c.get_by_label("Candidates").fill("61")
    rerun_button(panel).click()
    expect(note).to_have_text("Candidates is a whole number from 1 to 60.")
    panel.get_by_label("Evidence size").fill("21")
    c.get_by_label("Candidates").fill("4")
    rerun_button(panel).click()
    expect(note).to_have_text("Evidence size is a whole number from 1 to 20.")
    assert sent == []
    panel.get_by_label("Evidence size").fill("")

    # The server's refusals, 422 and 409, are said in the panel, and the edits stay.
    refusals = [
        (422, "method 'x' is not a chat method"),
        (409, "vector can't run on 'c' just now"),
    ]

    def refuse(route):
        status, why = refusals.pop(0)
        route.fulfill(status=status, json={"detail": why})

    page.route("**/messages/1/attempts", refuse)
    c.get_by_label("Method").select_option("vector")
    rerun_button(panel).click()
    expect(note).to_have_text("method 'x' is not a chat method")
    expect(note).to_have_class(re.compile("err"))
    rerun_button(panel).click()
    expect(note).to_have_text("vector can't run on 'c' just now")
    expect(c.get_by_label("Method")).to_have_value("vector")
    expect(c.get_by_label("Candidates")).to_have_value("4")
    assert sent == [{"settings": {}, "stores": {"c": {"method": "vector", "candidates": 4}}}] * 2
    expect(turn.get_by_role("tab")).to_have_count(0)  # no attempt was added


DECISION = {
    "store": "c",
    "method": "hybrid",
    "k": 8,
    "search": {"weights": {"meaning": 1, "words": 1, "keywords": 1}, "query_tags": "similarity"},
    "tags": None,
    "usd": 0.0,
    "error": None,
    "note": None,
}
GONE = {**DECISION, "store": "gone", "method": None, "k": None, "search": None, "note": "deleted"}


def test_stores_table_shows_errors_deleted_stores_and_the_no_passages_note(page, seeded):
    # An attempt saved before the pipeline, its decisions with notes but no counts.
    old_gone = {k: v for k, v in GONE.items() if k != "note"}  # saved before decisions had notes
    old_gone |= {"store": "old-gone", "kept": False, "reason": "deleted"}
    decisions = [
        DECISION,
        {**DECISION, "store": "vo"},
        {**DECISION, "store": "old-kw", "method": "keyword_rag"},  # saved before tags were kept
        {**DECISION, "store": "broke", "error": "budget used up"},
        GONE,
        old_gone,
    ]
    answer = {"answer": "No store passages were used.", "citations": [], "evidence": []}
    answer |= {"decisions": decisions, "pipeline": None}
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, **answer)))
    open_chat(page, seeded, ("c", "vo"))
    chat_ask(page, "something no store holds")
    turn = page.locator("#msg-1")
    expect(turn.locator(".reply .md")).to_have_text("No store passages were used.")
    # Said under the answer, with the step still closed.
    note = turn.locator(".reply .no-passages")
    expect(note).to_have_text("The stores found no passages, so this answer used none.")
    expect(note).to_be_visible()
    step = retrieval_step(page)
    expect(step.locator("table.decisions thead th")).to_have_text(["Store", "Method", "Result"])
    expect(store_row(step, "c").locator(".kept")).to_have_text("kept")
    expect(store_row(step, "old-kw").locator(".method")).to_have_text(
        "keyword_rag · query tags: similarity"
    )
    expect(store_row(step, "broke").locator(".kept")).to_have_text(
        "dropped · error: budget used up"
    )
    gone = store_row(step, "gone")
    expect(gone.locator(".kept")).to_have_text("deleted")
    expect(gone.locator(".method")).to_have_text("–")
    expect(gone).to_have_class(re.compile("dropped"))
    expect(store_row(step, "old-gone").locator(".kept")).to_have_text("deleted")
    expect(step.locator("ol.evidence li")).to_have_count(0)
    # Only the chat's own stores can be set.
    panel = settings_panel(page)
    expect(panel.locator("fieldset[data-edit]")).to_have_count(2)
    expect(edit_box(panel, "c")).to_be_visible()
    expect(edit_box(panel, "vo")).to_be_visible()


def test_no_passages_says_the_answer_used_none(page, seeded):
    answer = {"answer": "Nothing matched.", "citations": [], "evidence": []}
    answer["decisions"] = [{**CHAT_DECISION, "found": 0, "used": 0}]
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, **answer)))
    open_chat(page, seeded, ("c",))
    chat_ask(page, "something the store has no passage for")
    expect(page.locator("#msg-1 .reply .no-passages")).to_have_text(
        "The stores found no passages, so this answer used none."
    )
    step = retrieval_step(page)
    expect(count_line(step, "c").locator(".count")).to_have_text("c: 0 candidates, 0 kept")
    expect(step).to_contain_text("No passages.")


# -- The embedding model: Settings, the store page, the home form and the worker -------------

REBUILD_MODEL = "BAAI/bge-small-en-v1.5"


def test_settings_page_picks_the_embedding_model_for_new_stores(page, seeded):
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        field = page.locator('#search label:has([name="dense_model"])')
        expect(field).to_contain_text("Embedding model for new stores")
        assert page.locator("#search label").first.inner_text().startswith("Embedding model")
        select = page.locator('#search [name="dense_model"]')
        small = select.locator('option[value^="BAAI/bge-small-en-v1.5"]')
        expect(small).to_contain_text("BAAI/bge-small-en-v1.5 · ")
        expect(small).to_contain_text("0.067 GB")
        expect(small).to_contain_text("fast")
        expect(select.locator('option[value="fake/dense"]')).to_contain_text("(recommended)")
        expect(field.locator("[data-recommended]")).to_have_text("Recommended: fake/dense")
        expect(field.locator("[data-embedder-note]")).to_have_text(
            "Existing stores keep theirs; rebuild one from its page."
        )
        select.select_option("BAAI/bge-base-en-v1.5")
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["dense_model"] == "BAAI/bge-base-en-v1.5"


def test_store_page_names_its_embedding_model_and_rebuilds_with_another(page, seeded):
    seeded.build("rb")
    page.goto(seeded.url + "/c/rb")
    expect(page.locator("#embedding")).to_contain_text("Built with fake/dense · 256 dimensions")
    select = page.locator("#embedding select")
    expect(select.locator('option[value="fake/dense"]')).to_have_count(0)
    select.select_option(REBUILD_MODEL)
    page.get_by_role("button", name="Rebuild", exact=True).click()
    expect(page.locator("#embedding-confirm")).to_contain_text(
        f"Rebuild rb with {REBUILD_MODEL}? Every item is embedded again on this machine: free"
    )
    page.get_by_role("button", name="Yes, rebuild").click()
    job = httpx.get(seeded.url + "/collections/rb").json()["rebuild"]["job_id"]
    for _ in range(600):  # the rebuild runs on fakes: moments
        if httpx.get(f"{seeded.url}/jobs/{job}").json()["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert httpx.get(f"{seeded.url}/jobs/{job}").json()["status"] == "done"
    page.goto(seeded.url + "/c/rb")
    expect(page.locator("#embedding")).to_contain_text(f"Built with {REBUILD_MODEL}")


def test_store_page_shows_rebuild_progress_cancel_and_a_stopped_switch(page, seeded):
    seeded.build("rbp")
    rebuild = {
        "job_id": "job-1",
        "status": "running",
        "model": REBUILD_MODEL,
        "from": "fake/dense",
        "progress": 0.4,
        "switching": False,
    }
    current = {"rebuild": rebuild, "state": "rebuilding"}

    def store(route):
        body = route.fetch().json()
        route.fulfill(json={**body, **current})

    posts = []

    def job(route):
        posts.append((route.request.url.rsplit("/", 2)[-2:], route.request.post_data))
        route.fulfill(status=202, json={})

    page.route("**/collections/rbp", store)
    page.route("**/jobs/job-1/*", job)
    page.goto(seeded.url + "/c/rbp")
    box = page.locator("#embedding")
    expect(box).to_contain_text(f"Rebuilding with {REBUILD_MODEL}: 40%")
    box.get_by_role("button", name="Cancel").click()
    expect(box.locator("select")).to_be_disabled()
    expect(box).to_contain_text("Wait for the rebuilding job to finish.")
    assert posts and posts[0][0] == ["job-1", "cancel"]
    posts.clear()
    current["rebuild"] = {**rebuild, "status": "interrupted", "switching": True}
    current["state"] = "ready"
    page.goto(seeded.url + "/c/rbp")
    expect(box).to_contain_text(
        f"The switch to {REBUILD_MODEL} stopped part-way; search is refused until it finishes."
    )
    with page.expect_request("**/jobs/job-1/resume"):
        box.get_by_role("button", name="Resume").click()
    assert posts[0][0] == ["job-1", "resume"] and json.loads(posts[0][1]) == {}
    current["rebuild"] = {**rebuild, "status": "cancelled"}
    page.goto(seeded.url + "/c/rbp")
    expect(box).to_contain_text(
        f"The rebuild with {REBUILD_MODEL} stopped; the store still uses fake/dense."
    )
    current["rebuild"] = {**rebuild, "switching": True}
    page.goto(seeded.url + "/c/rbp")
    expect(box).to_contain_text(f"Switching to {REBUILD_MODEL}…")
    expect(box.get_by_role("button", name="Cancel")).to_have_count(0)


def test_new_store_form_says_which_model_it_embeds_with(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    expect(page.locator("#form")).to_contain_text(
        "Embedded with fake/dense (Settings → Search defaults)"
    )


def test_worker_page_offers_resume_for_a_stopped_rebuild(page, seeded):
    job = {
        "id": "abcdef12-0000",
        "kind": "reembed",
        "collection": "rb",
        "status": "cancelled",
        "error": None,
        "progress": 0.2,
        "created_at": "2026-11-08T10:00:00+00:00",
        "updated_at": "2026-11-08T10:05:00+00:00",
        "setup": {"model": REBUILD_MODEL},
    }
    page.route("**/jobs?limit=200", lambda route: route.fulfill(json=[job]))
    page.goto(seeded.url + "/worker")
    row = page.locator("[data-finished-job]")
    expect(row).to_contain_text("Re-embedding rb")
    expect(row.get_by_role("button", name="Resume")).to_be_visible()


# -- Plan 1 page leftovers ---------------------------------------------------------------------


def test_chat_shows_connect_ai_after_a_failed_ai_read(page, seeded):
    failed = []

    def ai(route):
        if not failed:
            failed.append(1)
            route.fulfill(status=500, json={"detail": "down"})
        else:
            route.continue_()

    page.expected = ["500"]  # the failed read, on purpose
    page.route("**/ai", ai)
    open_chat(page, seeded)
    note = page.locator("#compose #note")
    expect(note).to_have_text("Connect AI in Settings to get written answers")
    expect(page.locator("#model-pick")).to_be_hidden()
    page.evaluate("window.dispatchEvent(new Event('focus'))")  # no reload
    expect(note).to_be_hidden()
    expect(page.locator("#model option")).not_to_have_count(0)
    expect(page.locator("#model-pick")).to_be_visible()
    expect(page.locator("#send")).to_be_enabled()


def test_chat_reads_ai_again_before_a_send_while_the_last_read_failed(page, seeded):
    reads = []

    def ai(route):
        reads.append(1)
        if len(reads) == 1:
            route.fulfill(status=500, json={"detail": "down"})
        else:
            route.continue_()

    page.expected = ["500"]  # the failed read, on purpose
    page.route("**/ai", ai)
    open_chat(page, seeded)
    expect(page.locator("#compose #note")).to_be_visible()
    chat_ask(page, "enzyme kinetics rate experiment")
    expect(page.locator("#msg-1 .reply")).to_have_count(1)
    assert len(reads) == 2
    expect(page.locator("#compose #note")).to_be_hidden()


def test_set_as_default_yes_is_disabled_while_a_message_is_pending(page, seeded):
    conn = seeded.app.state.conn
    before = db.get_chat_method(conn, "c"), db.get_chat_k(conn, "c")
    held = []
    try:
        open_chat(page, seeded, ("c",))
        chat_ask(page, "enzyme kinetics rate experiment")
        expect(page.locator("#msg-1 .reply")).to_have_count(1)
        c = edit_box(settings_panel(page), "c")
        c.get_by_label("Method").select_option("vector")
        c.get_by_role("button", name="Set as default").click()
        yes = c.get_by_role("button", name="Yes, save")
        expect(yes).to_be_enabled()
        page.route("**/chats/*/messages", lambda route: held.append(route))
        chat_ask(page, "a second question")
        expect(page.locator("#send")).to_be_disabled()  # the message is on its way
        expect(yes).to_be_disabled()
        held[0].continue_()
        expect(page.locator("#msg-2 .reply")).to_have_count(1)
    finally:
        db.set_chat_method(conn, "c", before[0])
        db.set_chat_k(conn, "c", before[1])


def test_settings_sends_null_when_the_reranker_pick_is_the_recommended_one(page, seeded):
    listed = httpx.get(seeded.url + "/search-defaults").json()
    own = listed["recommended"]["rerank_model"]
    turbo = "jinaai/jina-reranker-v1-turbo-en"
    puts: list[dict] = []

    def record(route):
        if route.request.method == "PUT":
            puts.append(route.request.post_data_json)
        route.continue_()

    page.route("**/search-defaults", record)
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        select = page.locator('#search [name="rerank_model"]')
        select.select_option(turbo)
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        select.select_option(own)
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        assert puts == [{"rerank_model": turbo}, {"rerank_model": None}]


def test_chat_escapes_markup_in_store_names_notes_and_errors(page, seeded):
    chat_id = make_chat(seeded, ("c",))
    markup = "<b>x</b>"
    decision = {**CHAT_DECISION, "store": markup, "note": markup, "found": 0, "used": 0}
    broke = {**CHAT_DECISION, "store": "broke", "error": markup}
    attempt = {**CHAT_ATTEMPT, "decisions": [decision, broke], "error": markup, "evidence": []}

    def chat_get(route):
        got = route.fetch().json()
        got["messages"] = [{"n": 1, "content": "asked", "attempts": [attempt]}]
        route.fulfill(json=got)

    page.route(f"**/chats/{chat_id}", chat_get)
    page.goto(f"{seeded.url}/chat#{chat_id}")
    step = retrieval_step(page)
    counts = step.locator("ul.counts")
    expect(counts).to_contain_text(f"{markup}: 0 candidates, 0 kept")
    expect(counts).to_contain_text(f"error: {markup}")
    expect(counts).to_contain_text(f"· {markup}")
    expect(page.locator("#msg-1 .meta.warn:not(.no-passages)")).to_have_text(markup)
    expect(page.locator("#log b")).to_have_count(0)  # all of it is text


# -- using a chat from your own app ---------------------------------------------------------------

BRIEF_KEYS = {"n", "answer", "citations", "passages", "passages_only", "error", "usd"}


def your_app(page):
    """The chat's Use from your app section, opened: its two <pre> lines."""
    page.locator("#chat-head summary", has_text="Use from your app").click()
    return page.locator("#chat-head details pre")


def test_chat_shows_the_curl_lines_for_this_chat(page, seeded):
    a = make_chat(seeded, ("c",), title="first")
    b = make_chat(seeded, ("c",), title="second")
    page.goto(f"{seeded.url}/chat#{a}")
    expect(page.locator("#chips .chip")).to_have_count(1)
    details = page.locator("#chat-head details")
    expect(details).not_to_have_attribute("open", "")
    pres = your_app(page)
    expect(details).to_contain_text("Send the next message:")
    expect(details).to_contain_text("Start a new conversation set up like this one:")
    expect(details).to_contain_text('Add "answer":false to get the passages without an answer')
    expect(pres.nth(0)).to_have_text(
        f"curl -s '{seeded.url}/chats/{a}/messages?brief=true' -H 'content-type: application/json'"
        """ -d '{"content":"…"}'"""
    )
    expect(pres.nth(1)).to_have_text(
        f"curl -s {seeded.url}/chats -H 'content-type: application/json'"
        f""" -d '{{"from":"{a}"}}'"""
    )
    chat_link(page, b).click()
    expect(pres.nth(0)).to_contain_text(f"'{seeded.url}/chats/{b}/messages?brief=true'")
    expect(pres.nth(1)).to_contain_text(f'"from":"{b}"')


def test_the_copied_message_curl_runs(page, seeded):
    chat_id = open_chat(page, seeded, ("c",))
    line = your_app(page).nth(0).inner_text().replace("…", "what speeds up an enzyme?")
    words = shlex.split(line)
    assert words[:2] == ["curl", "-s"] and words[3] == "-H" and words[5] == "-d"
    url, header, body = words[2], words[4], words[6]
    key, _, value = header.partition(": ")
    r = page.request.post(url, headers={key: value}, data=body)
    assert r.status == 200, r.text()
    got = r.json()
    assert set(got) == BRIEF_KEYS
    assert got["n"] == 1 and got["error"] is None
    saved = httpx.get(f"{seeded.url}/chats/{chat_id}").json()["messages"]
    assert saved[-1]["content"] == "what speeds up an enzyme?"


def test_eval_set_page_save_keeps_judged_ids(page, seeded):
    seeded.build("judged1")
    got = httpx.get(f"{seeded.url}/eval-sets/judged1").json()
    questions = got["questions"]
    questions[0]["judged_ids"] = ["x"]
    r = httpx.put(
        f"{seeded.url}/eval-sets/judged1/questions",
        json={"questions": questions, "base_version": got["version"]},
    )
    assert r.status_code == 200, r.text
    first = questions[0]["id"]
    page.goto(seeded.url + "/e/judged1")
    rows = page.locator("#questions tbody tr")
    other = rows.filter(has=page.locator("td", has_text=questions[1]["id"])).first
    other.get_by_role("button", name="Edit").click()
    other.locator("[data-field=text]").fill("reworded for the judged test")
    other.get_by_role("button", name="Done").click()
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    after = {q["id"]: q for q in httpx.get(f"{seeded.url}/eval-sets/judged1").json()["questions"]}
    assert after[first]["judged_ids"] == ["x"]


# -- labelling a Documents eval set ------------------------------------------------------------

POOL_LOADING = "The ranking model is still loading, so passages are in search order, not best first"


def labelling_set(seeded, name, store=True):
    """A Documents store `name` holding two passages of x.md, and an eval set `<name>q` of
    Documents linked to it (with `store`) or to none."""
    documents_store(seeded, name)
    body = {"name": f"{name}q", "qa_type": "documents"}
    if store:
        body["store"] = name
    r = httpx.post(f"{seeded.url}/eval-sets", json=body)
    assert r.status_code == 201, r.text


def mark(passage, which):
    passage.get_by_role("button", name=which, exact=True).click()


def pressed(passage, which):
    return passage.get_by_role("button", name=which, exact=True)


def test_documents_eval_set_labels_a_question_and_saves_it(page, seeded):
    labelling_set(seeded, "lab1")
    page.goto(seeded.url + "/e/lab1q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("how long does alpha last?")
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    expect(passages.first).to_contain_text("x.md, part 1 of 2")
    shown = panel.inner_text()
    for word in ("vector", "keyword", "hybrid"):
        assert word not in shown.lower(), word
    assert "Vector search" not in shown
    first, second = (passages.nth(i).get_attribute("data-passage") for i in range(2))
    mark(passages.nth(0), "y")
    mark(passages.nth(1), "n")
    expect(page.locator("#lab-count")).to_have_text("1 right, 1 wrong, 0 not judged")
    panel.get_by_label("Reference answer (optional; short is better)").fill("Ten days")
    panel.get_by_role("button", name="Save and write the next").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(panel.get_by_label("Question", exact=True)).to_have_value("")
    got = httpx.get(f"{seeded.url}/eval-sets/lab1q").json()["questions"]
    assert len(got) == 1
    q = got[0]
    assert q["gold_ids"] == [first]
    assert q["judged_ids"] == [first, second]
    assert q["gold_answers"] == ["Ten days"]
    assert q["meta"]["source"] == "hand"
    rows = page.locator("#lab-questions tbody tr")
    expect(rows).to_have_count(1)
    expect(rows.first.locator("td").nth(1)).to_have_text("1")
    expect(rows.first.locator("td").nth(2)).to_have_text("2")
    rows.first.get_by_role("button", name="Label").click()
    expect(passages).to_have_count(2)
    expect(page.locator("#lab-count")).to_have_text("1 right, 1 wrong, 0 not judged")
    expect(panel.get_by_label("Reference answer (optional; short is better)")).to_have_value(
        "Ten days"
    )


def test_labelling_save_question_twice_saves_one_question(page, seeded):
    labelling_set(seeded, "lab2")
    page.goto(seeded.url + "/e/lab2q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("how long does alpha last?")
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    first = passages.first.get_attribute("data-passage")
    mark(passages.first, "y")
    panel.get_by_role("button", name="Save question").click()
    msg = page.locator("#lab-msg")
    expect(msg).to_contain_text("Saved as version")
    said = msg.inner_text()
    mark(passages.first, "n")
    panel.get_by_role("button", name="Save question").click()
    expect(msg).not_to_have_text(said)
    expect(msg).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/lab2q").json()["questions"]
    assert len(got) == 1
    assert got[0]["gold_ids"] == []
    assert got[0]["judged_ids"] == [first]


def test_labelling_a_mark_clicked_twice_is_unjudged(page, seeded):
    labelling_set(seeded, "lab3")
    page.goto(seeded.url + "/e/lab3q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("how long does alpha last?")
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    mark(passages.first, "y")
    expect(pressed(passages.first, "y")).to_have_attribute("aria-pressed", "true")
    mark(passages.first, "y")
    expect(pressed(passages.first, "y")).to_have_attribute("aria-pressed", "false")
    expect(page.locator("#lab-count")).to_have_text("0 right, 0 wrong, 2 not judged")


def test_labelling_page_searches_the_store_for_a_missing_passage(page, seeded):
    labelling_set(seeded, "lab4")
    r = httpx.post(f"{seeded.url}/eval-sets/lab4q/pool", json={"question": "alpha"})
    assert r.status_code == 200, r.text
    pool = r.json()
    first, second = pool["passages"]
    assert first["text"].lower().count("alpha") > second["text"].lower().count("alpha")
    page.route(
        "**/eval-sets/lab4q/pool", lambda route: route.fulfill(json={**pool, "passages": [first]})
    )
    page.goto(seeded.url + "/e/lab4q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("alpha")
    panel.get_by_role("button", name="Find passages").click()
    expect(panel.locator("li[data-passage]")).to_have_count(1)
    panel.get_by_label("None of these? Search the store").fill("brine")
    with page.expect_request("**/query") as sent:
        panel.get_by_role("button", name="Search", exact=True).click()
    body = sent.value.post_data_json
    assert body["methods"] == ["hybrid"]
    assert body["k"] == 10
    assert body["collection"] == "lab4"
    found = page.locator("#lab-found")
    expect(found).to_contain_text("From your search")
    expect(found.locator("li[data-passage]")).to_have_count(1)
    hit = found.locator(f'li[data-passage="{second["id"]}"]')
    expect(hit).to_contain_text("x.md, part 2 of 2")
    mark(hit, "y")
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/lab4q").json()["questions"]
    assert [q["gold_ids"] for q in got] == [[second["id"]]]


def test_labelling_keeps_marks_when_another_labeller_saved_first(page, seeded):
    labelling_set(seeded, "lab5")
    r = httpx.post(
        f"{seeded.url}/eval-sets/lab5q/questions",
        json={"question": {"text": "how long does alpha last?"}},
    )
    assert r.status_code == 201, r.text
    qid = r.json()["id"]
    page.goto(seeded.url + "/e/lab5q")
    page.locator("#lab-questions tbody tr").first.get_by_role("button", name="Label").click()
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    mark(passages.nth(0), "y")
    mark(passages.nth(1), "n")
    stored = question_map(seeded, "lab5q")[qid]
    changed = {**stored, "text": "reworded by someone else"}
    r = httpx.put(
        f"{seeded.url}/eval-sets/lab5q/questions/{qid}", json={"question": changed, "base": stored}
    )
    assert r.status_code == 200, r.text
    page.expected = ["409"]
    panel.get_by_role("button", name="Save question").click()
    msg = page.locator("#lab-msg")
    expect(msg).to_contain_text("This question was changed by someone else since you opened it")
    expect(msg.get_by_role("button", name="Reload it (discard my marks)")).to_be_visible()
    expect(pressed(passages.nth(0), "y")).to_have_attribute("aria-pressed", "true")
    expect(pressed(passages.nth(1), "n")).to_have_attribute("aria-pressed", "true")
    assert question_map(seeded, "lab5q")[qid]["gold_ids"] == []
    msg.get_by_role("button", name="Reload it (discard my marks)").click()
    expect(panel.get_by_label("Question", exact=True)).to_have_value("reworded by someone else")
    expect(passages).to_have_count(2)
    expect(page.locator("#lab-count")).to_have_text("0 right, 0 wrong, 2 not judged")
    expect(msg).to_have_text("")


def test_labelling_page_shows_the_pool_notes(page, seeded):
    labelling_set(seeded, "lab6")
    passage = {
        "id": "x.md#aaaaaaaaaaaa",
        "text": "alpha",
        "file": "x.md",
        "part": 1,
        "parts": 2,
        "relevance": None,
    }
    reply = {
        "store": "lab6",
        "methods": [],
        "passages": [passage],
        "merged_by": "rank",
        "notes": [POOL_LOADING],
        "labelled": [],
        "missing": [],
    }
    page.route("**/eval-sets/lab6q/pool", lambda route: route.fulfill(json=reply))
    page.goto(seeded.url + "/e/lab6q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("alpha")
    panel.get_by_role("button", name="Find passages").click()
    expect(panel.locator("li[data-passage]")).to_have_count(1)
    expect(panel.get_by_text(POOL_LOADING, exact=True)).to_be_visible()


def test_documents_eval_set_page_flags_missing_and_unlabelled_questions(page, seeded):
    labelling_set(seeded, "lab7")
    lost = "x.md#000000000000"
    for question in (
        {"text": "where did alpha go?", "gold_ids": [lost]},
        {"text": "nothing marked on brine"},
    ):
        r = httpx.post(f"{seeded.url}/eval-sets/lab7q/questions", json={"question": question})
        assert r.status_code == 201, r.text
    page.goto(seeded.url + "/e/lab7q")
    main = page.locator("main")
    expect(main.get_by_text("1 labelled passage is no longer in lab7", exact=True)).to_be_visible()
    expect(
        main.get_by_text("1 question has no right passage; it is not scored", exact=True)
    ).to_be_visible()
    rows = page.locator("#lab-questions tbody tr")
    expect(rows).to_have_count(2)
    q1 = rows.filter(has_text="where did alpha go?")
    q2 = rows.filter(has_text="nothing marked on brine")
    expect(q1).to_contain_text("(1 no longer in lab7)")
    expect(q2).not_to_contain_text("no longer in")
    expect(page.get_by_role("heading", name="Leaked questions")).to_have_count(0)
    expect(page.get_by_role("columnheader", name="Choices")).to_have_count(0)
    expect(page.locator("th", has_text="Choices")).to_have_count(0)
    q1.get_by_role("button", name="Label").click()
    gone = page.locator(f'#lab-panel li[data-passage="{lost}"]')
    expect(gone).to_contain_text(lost)
    expect(pressed(gone, "y")).to_have_attribute("aria-pressed", "true")
    q2.get_by_role("button", name="Delete").click()
    expect(main.get_by_text("Delete this question and its marks?")).to_be_visible()
    assert len(httpx.get(f"{seeded.url}/eval-sets/lab7q").json()["questions"]) == 2
    page.get_by_role("button", name="Yes, delete it").click()
    expect(rows).to_have_count(1)
    got = httpx.get(f"{seeded.url}/eval-sets/lab7q").json()["questions"]
    assert [q["text"] for q in got] == ["where did alpha go?"]


def test_a_documents_set_without_a_store_asks_for_one(page, seeded):
    labelling_set(seeded, "lab8", store=False)
    page.goto(seeded.url + "/e/lab8q")
    expect(page.get_by_text("Pick a store to label against:")).to_be_visible()
    expect(page.get_by_role("button", name="Find passages")).to_be_hidden()
    page.get_by_label("Pick a store to label against:").select_option("lab8")
    page.get_by_role("button", name="Link", exact=True).click()
    expect(page.locator("#lab-store")).to_have_text("Labelled against lab8.")
    expect(page.locator('#lab-store a[href="/c/lab8"]')).to_have_text("lab8")
    expect(page.get_by_role("button", name="Find passages")).to_be_visible()
    assert httpx.get(f"{seeded.url}/eval-sets/lab8q").json()["store"] == "lab8"


def labelled_questions(seeded, name, texts):
    """Adds questions of `texts` to the set `<name>q`; returns their ids in order."""
    ids = []
    for text in texts:
        r = httpx.post(
            f"{seeded.url}/eval-sets/{name}q/questions", json={"question": {"text": text}}
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    return ids


def test_labelling_a_save_in_flight_keeps_each_question_its_own(page, seeded):
    labelling_set(seeded, "lab9")
    q1, q2 = labelled_questions(seeded, "lab9", ["alpha first?", "brine second?"])
    held = []
    page.route(f"**/eval-sets/lab9q/questions/{q1}", lambda route: held.append(route))
    page.goto(seeded.url + "/e/lab9q")
    rows = page.locator("#lab-questions tbody tr")
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    rows.filter(has_text="alpha first?").get_by_role("button", name="Label").click()
    expect(passages).to_have_count(2)
    first = passages.first.get_attribute("data-passage")
    mark(passages.first, "y")
    panel.get_by_role("button", name="Save question").click()
    deadline = time.time() + 5
    while not held and time.time() < deadline:
        page.wait_for_timeout(20)
    assert held, "the PUT never left"
    expect(
        rows.filter(has_text="brine second?").get_by_role("button", name="Label")
    ).to_be_disabled()
    expect(
        rows.filter(has_text="brine second?").get_by_role("button", name="Delete")
    ).to_be_disabled()
    expect(panel.get_by_role("button", name="Save question")).to_be_disabled()
    expect(panel.get_by_role("button", name="Find passages")).to_be_disabled()
    held[0].continue_()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    page.unroute(f"**/eval-sets/lab9q/questions/{q1}")
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    expect(panel.get_by_label("Question", exact=True)).to_have_value("brine second?")
    expect(passages).to_have_count(2)
    mark(passages.nth(1), "y")
    second = passages.nth(1).get_attribute("data-passage")
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    got = question_map(seeded, "lab9q")
    assert (got[q1]["text"], got[q1]["gold_ids"]) == ("alpha first?", [first])
    assert (got[q2]["text"], got[q2]["gold_ids"]) == ("brine second?", [second])


def test_labelling_a_double_click_on_save_makes_one_question(page, seeded):
    labelling_set(seeded, "lab10")
    page.goto(seeded.url + "/e/lab10q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("  how long does alpha last?  ")
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    mark(passages.first, "y")
    panel.get_by_role("button", name="Save question").dblclick()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(page.locator("#lab-questions tbody tr")).to_have_count(1)
    got = httpx.get(f"{seeded.url}/eval-sets/lab10q").json()["questions"]
    assert [q["text"] for q in got] == ["how long does alpha last?"]


def test_labelling_a_deleted_question_saves_its_marks_as_a_new_one(page, seeded):
    labelling_set(seeded, "lab11")
    (qid,) = labelled_questions(seeded, "lab11", ["alpha soon deleted?"])
    page.goto(seeded.url + "/e/lab11q")
    page.locator("#lab-questions tbody tr").first.get_by_role("button", name="Label").click()
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    first, second = (passages.nth(i).get_attribute("data-passage") for i in range(2))
    mark(passages.nth(0), "y")
    mark(passages.nth(1), "n")
    r = httpx.delete(f"{seeded.url}/eval-sets/lab11q/questions/{qid}")
    assert r.status_code == 200, r.text
    page.expected = ["404"]
    panel.get_by_role("button", name="Save question").click()
    msg = page.locator("#lab-msg")
    expect(msg).to_contain_text("This question was deleted since you opened it")
    expect(page.locator("#lab-questions tbody tr")).to_have_count(0)
    expect(pressed(passages.nth(0), "y")).to_have_attribute("aria-pressed", "true")
    expect(pressed(passages.nth(1), "n")).to_have_attribute("aria-pressed", "true")
    panel.get_by_role("button", name="Save question").click()
    expect(msg).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/lab11q").json()["questions"]
    assert len(got) == 1
    assert got[0]["text"] == "alpha soon deleted?"
    assert got[0]["gold_ids"] == [first]
    assert got[0]["judged_ids"] == [first, second]


UNSAVED = "This question has unsaved marks."


def test_labelling_label_on_another_row_asks_before_dropping_unsaved_marks(page, seeded):
    labelling_set(seeded, "lab12")
    labelled_questions(seeded, "lab12", ["alpha first?", "brine second?"])
    page.goto(seeded.url + "/e/lab12q")
    rows = page.locator("#lab-questions tbody tr")
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    question = panel.get_by_label("Question", exact=True)
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    expect(passages).to_have_count(2)
    rows.filter(has_text="alpha first?").get_by_role("button", name="Label").click()
    expect(question).to_have_value("alpha first?")  # nothing unsaved: it just opens
    expect(passages).to_have_count(2)
    mark(passages.first, "y")
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    notice = panel.get_by_text(UNSAVED, exact=True)
    expect(notice).to_be_visible()
    expect(question).to_have_value("alpha first?")
    panel.get_by_role("button", name="Keep editing").click()
    expect(notice).to_have_count(0)
    expect(question).to_have_value("alpha first?")
    expect(pressed(passages.first, "y")).to_have_attribute("aria-pressed", "true")
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    panel.get_by_role("button", name="Discard them").click()
    expect(question).to_have_value("brine second?")
    expect(notice).to_have_count(0)
    expect(passages).to_have_count(2)
    expect(page.locator("#lab-count")).to_have_text("0 right, 0 wrong, 2 not judged")


def test_labelling_delete_asks_before_dropping_unsaved_marks(page, seeded):
    labelling_set(seeded, "lab13")
    labelled_questions(seeded, "lab13", ["alpha first?", "brine second?"])
    page.goto(seeded.url + "/e/lab13q")
    rows = page.locator("#lab-questions tbody tr")
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    main = page.locator("main")
    confirm = main.get_by_text("Delete this question and its marks?")
    rows.filter(has_text="alpha first?").get_by_role("button", name="Label").click()
    expect(passages).to_have_count(2)
    mark(passages.first, "y")
    rows.filter(has_text="brine second?").get_by_role("button", name="Delete").click()
    notice = panel.get_by_text(UNSAVED, exact=True)
    expect(notice).to_be_visible()
    expect(confirm).to_have_count(0)
    panel.get_by_role("button", name="Keep editing").click()
    expect(notice).to_have_count(0)
    expect(confirm).to_have_count(0)
    expect(pressed(passages.first, "y")).to_have_attribute("aria-pressed", "true")
    assert len(httpx.get(f"{seeded.url}/eval-sets/lab13q").json()["questions"]) == 2
    rows.filter(has_text="brine second?").get_by_role("button", name="Delete").click()
    panel.get_by_role("button", name="Discard them").click()
    expect(confirm).to_be_visible()
    page.get_by_role("button", name="Yes, delete it").click()
    expect(rows).to_have_count(1)
    got = httpx.get(f"{seeded.url}/eval-sets/lab13q").json()["questions"]
    assert [q["text"] for q in got] == ["alpha first?"]


# -- the AI's answer on the labelling panel -----------------------------------------------------

ASK_AI = "Get the AI's answer"
ADDED = "Added as the reference answer. Save question to keep it."
REFERENCE = "Reference answer (optional; short is better)"


@contextmanager
def replies(seeded, reply):
    """The fake AI answers every prompt with reply(prompt) until the block ends."""
    fake_replies(seeded.app.state.ai, reply)
    try:
        yield
    finally:
        fake_replies(seeded.app.state.ai, lambda p: "Answer: A")


@contextmanager
def ai_calls(seeded):
    """The purpose of each complete_many the server makes until the block ends, in order."""
    ai = seeded.app.state.ai
    original = ai.complete_many
    purposes = []

    def counting(*args, **kw):
        purposes.append(kw["purpose"])
        return original(*args, **kw)

    ai.complete_many = counting
    try:
        yield purposes
    finally:
        ai.complete_many = original


def ai_panel(page, seeded, name, text):
    """Opens the set `<name>q`, types `text` and finds its two passages. Returns the panel,
    its passages and the body of every /answer request the page sends, as it sends them."""
    sent = []

    def record(request):
        if request.url.endswith(f"/eval-sets/{name}q/answer"):
            sent.append(request.post_data_json)

    page.on("request", record)
    page.goto(f"{seeded.url}/e/{name}q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill(text)
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    return panel, passages, sent


def wait_for(page, held):
    """Waits until a held route has caught its request."""
    deadline = time.time() + 5
    while not held and time.time() < deadline:
        page.wait_for_timeout(20)
    assert held, "the request never left"


def test_labelling_ai_answer_asks_before_it_spends(page, seeded):
    labelling_set(seeded, "ai1")
    panel, passages, sent = ai_panel(page, seeded, "ai1", "How long is leave? (ai1)")
    first = passages.first.get_attribute("data-passage")
    mark(passages.first, "y")
    panel.get_by_role("button", name=ASK_AI).click()
    start = page.locator("#lab-ai-confirm").get_by_role("button", name="Start")
    expect(start).to_be_visible()
    answer = page.locator("#lab-ai-answer")
    expect(answer).to_be_empty()
    question = "How long is leave? (ai1)"
    assert sent == [{"question": question, "passages": [first], "confirm": False}]
    with ai_calls(seeded) as purposes:
        start.click()
        expect(answer.get_by_text("Is this the right answer?", exact=True)).to_be_visible()
        expect(answer.locator(".full")).to_have_text("Answer: A")
    assert purposes == ["reference_answer"]
    assert sent[1:] == [{"question": question, "passages": [first], "confirm": True}]
    expect(page.locator("#lab-ai-confirm")).to_be_empty()
    answer.get_by_role("button", name="y", exact=True).click()
    expect(panel.get_by_label(REFERENCE)).to_have_value("Answer: A")
    expect(page.locator("#lab-ai-msg")).to_have_text(ADDED)
    expect(answer).to_be_empty()
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(page.locator("#lab-ai-msg")).to_have_text("")  # it is kept now
    got = httpx.get(f"{seeded.url}/eval-sets/ai1q").json()["questions"]
    assert [q["gold_answers"] for q in got] == [["Answer: A"]]


def test_labelling_ai_answer_n_keeps_nothing_and_survives_a_mark(page, seeded):
    labelling_set(seeded, "ai2")
    panel, passages, _ = ai_panel(page, seeded, "ai2", "How long is leave? (ai2)")
    mark(passages.first, "y")
    panel.get_by_role("button", name=ASK_AI).click()
    page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
    answer = page.locator("#lab-ai-answer")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    mark(passages.nth(1), "n")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    answer.get_by_role("button", name="n", exact=True).click()
    expect(answer).to_be_empty()
    expect(panel.get_by_label(REFERENCE)).to_have_value("")
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/ai2q").json()["questions"]
    assert [q["gold_answers"] for q in got] == [[]]


def test_labelling_ai_answer_waits_for_a_y_mark(page, seeded):
    labelling_set(seeded, "ai3")
    panel, passages, sent = ai_panel(page, seeded, "ai3", "How long is leave? (ai3)")
    ask = panel.get_by_role("button", name=ASK_AI)
    expect(ask).to_be_visible()
    expect(
        page.get_by_text("The AI answers from the passages marked y.", exact=True)
    ).to_be_visible()
    expect(ask).to_be_disabled()
    mark(passages.first, "n")
    expect(ask).to_be_disabled()
    mark(passages.nth(1), "y")
    expect(ask).to_be_enabled()
    panel.get_by_label("Question", exact=True).fill("")
    expect(ask).to_be_disabled()
    assert sent == []


def test_labelling_ai_answer_price_is_dropped_when_the_panel_changes(page, seeded):
    labelling_set(seeded, "ai4")
    labelled_questions(seeded, "ai4", ["What is leave for? (ai4)"])
    held = []
    page.route("**/eval-sets/ai4q/answer", lambda route: held.append(route))
    panel, passages, sent = ai_panel(page, seeded, "ai4", "How long is leave? (ai4)")
    ask = panel.get_by_role("button", name=ASK_AI)
    start = page.locator("#lab-ai-confirm").get_by_role("button", name="Start")
    with ai_calls(seeded) as purposes:
        mark(passages.first, "y")
        ask.click()
        wait_for(page, held)
        mark(passages.nth(1), "n")  # while its price is out
        with page.expect_response("**/eval-sets/ai4q/answer") as reply:
            held[0].continue_()
        reply.value.finished()
        page.wait_for_timeout(300)  # the reply is read; a stale Start would be drawn by now
        expect(start).to_have_count(0)
        page.unroute("**/eval-sets/ai4q/answer")
        ask.click()
        expect(start).to_be_visible()
        panel.get_by_label("Question", exact=True).fill("How long is leave, really? (ai4)")
        expect(start).to_have_count(0)
        ask.click()
        expect(start).to_be_visible()
        rows = page.locator("#lab-questions tbody tr")
        rows.filter(has_text="What is leave for? (ai4)").get_by_role("button", name="Label").click()
        panel.get_by_role("button", name="Discard them").click()
        expect(panel.get_by_label("Question", exact=True)).to_have_value("What is leave for? (ai4)")
        expect(start).to_have_count(0)
        assert purposes == []
    assert len(sent) == 3
    assert not any(body["confirm"] for body in sent)


def test_labelling_ai_answer_waits_for_a_save(page, seeded):
    labelling_set(seeded, "ai5")
    panel, passages, _ = ai_panel(page, seeded, "ai5", "How long is leave? (ai5)")
    mark(passages.first, "y")
    panel.get_by_role("button", name=ASK_AI).click()
    page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
    answer = page.locator("#lab-ai-answer")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    held = []
    page.route("**/eval-sets/ai5q/questions", lambda route: held.append(route))
    panel.get_by_role("button", name="Save question").click()
    wait_for(page, held)
    y = answer.get_by_role("button", name="y", exact=True)
    n = answer.get_by_role("button", name="n", exact=True)
    expect(y).to_be_disabled()
    expect(n).to_be_disabled()
    y.dispatch_event("click")
    expect(panel.get_by_label(REFERENCE)).to_have_value("")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    held[0].continue_()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(y).to_be_enabled()
    expect(n).to_be_enabled()
    expect(panel.get_by_label(REFERENCE)).to_have_value("")
    expect(page.locator("#lab-ai-msg")).to_have_text("")


def test_labelling_ai_answer_goes_with_its_question(page, seeded):
    labelling_set(seeded, "ai8")
    labelled_questions(seeded, "ai8", ["What is leave for? (ai8)"])
    panel, passages, _ = ai_panel(page, seeded, "ai8", "How long is leave? (ai8)")
    ask = panel.get_by_role("button", name=ASK_AI)
    start = page.locator("#lab-ai-confirm").get_by_role("button", name="Start")
    answer = page.locator("#lab-ai-answer")
    msg = page.locator("#lab-ai-msg")
    label = (
        page.locator("#lab-questions tbody tr")
        .filter(has_text="What is leave for? (ai8)")
        .get_by_role("button", name="Label")
    )
    mark(passages.first, "y")
    ask.click()
    start.click()
    answer.get_by_role("button", name="y", exact=True).click()
    expect(msg).to_have_text(ADDED)
    panel.get_by_label(REFERENCE).press_sequentially(" or so")
    expect(msg).to_have_text("")  # edited by hand, it is no longer the AI's answer
    label.click()
    panel.get_by_role("button", name="Discard them").click()
    expect(panel.get_by_label("Question", exact=True)).to_have_value("What is leave for? (ai8)")
    expect(msg).to_have_text("")
    expect(passages).to_have_count(2)
    mark(passages.first, "y")
    ask.click()
    start.click()
    expect(answer.locator(".full")).to_have_text("Answer: A")
    label.click()
    panel.get_by_role("button", name="Discard them").click()
    expect(answer).to_be_empty()
    # A paid answer that lands after another question opened is not shown.
    held = []

    def hold_paid(route):
        if route.request.post_data_json["confirm"]:
            held.append(route)
        else:
            route.continue_()

    page.route("**/eval-sets/ai8q/answer", hold_paid)
    expect(passages).to_have_count(2)
    mark(passages.nth(1), "y")
    ask.click()
    start.click()
    wait_for(page, held)
    label.click()
    panel.get_by_role("button", name="Discard them").click()
    with page.expect_response("**/eval-sets/ai8q/answer") as reply:
        held[0].continue_()
    reply.value.finished()
    page.wait_for_timeout(300)
    expect(answer).to_be_empty()
    expect(msg).to_have_text("")


def test_labelling_without_an_answer_model_hides_the_ai_answer(page, seeded):
    labelling_set(seeded, "ai6")

    def no_model(route):
        data = route.fetch().json()
        data["answer_available"] = False
        route.fulfill(json=data)

    page.route("**/eval-sets/ai6q", no_model)
    panel, passages, sent = ai_panel(page, seeded, "ai6", "How long is leave? (ai6)")
    mark(passages.first, "y")
    expect(page.locator("#lab-ai")).to_have_count(1)  # drawn, but hidden
    expect(panel.get_by_role("button", name=ASK_AI)).to_be_hidden()
    expect(page.locator("#lab-ai-hint")).to_be_hidden()
    panel.get_by_label(REFERENCE).fill("Ten days")
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/ai6q").json()["questions"]
    assert [q["gold_answers"] for q in got] == [["Ten days"]]
    assert sent == []


def test_labelling_ai_answer_strips_citations(page, seeded):
    labelling_set(seeded, "ai7")
    panel, passages, _ = ai_panel(page, seeded, "ai7", "How long is leave? (ai7)")
    mark(passages.first, "y")
    with replies(seeded, lambda p: "Ten days [1]."):
        panel.get_by_role("button", name=ASK_AI).click()
        page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
        expect(page.locator("#lab-ai-answer .full")).to_have_text("Ten days.")


# -- versions, Create one, Run against its store, the run form's warning -------------------------


def test_eval_set_page_head_counts_its_questions(page, seeded):
    page.goto(seeded.url + "/e/c")
    expect(page.locator("#head")).to_contain_text("40 questions")
    expect(page.locator("#questions tbody tr")).to_have_count(40)
    assert "NaN" not in page.locator("main").inner_text()


def put_questions(seeded, name, questions, base=None):
    r = httpx.put(
        f"{seeded.url}/eval-sets/{name}/questions",
        json={"questions": questions, "base_version": base},
    )
    assert r.status_code == 200, r.text
    return r.json()["version"]


def test_eval_set_page_views_and_restores_an_old_version(page, seeded):
    seeded.build("ver1")
    old = httpx.get(f"{seeded.url}/eval-sets/ver1").json()
    was = next(q for q in old["questions"] if q["id"] == "q0")["text"]
    changed = [{**q, "text": "q0 reworded"} if q["id"] == "q0" else q for q in old["questions"]]
    put_questions(seeded, "ver1", changed, old["version"])
    page.goto(seeded.url + "/e/ver1")
    first = old["version"][:8]
    expect(page.locator("#versions li")).to_have_count(2)
    page.locator("#versions li", has_text=first).get_by_role("button", name="View").click()
    view = page.locator("#ver-view")
    expect(view).to_contain_text(f"Version {first}: 40 questions")
    row = view.locator("tbody tr").filter(has=page.locator("td", has_text=re.compile(r"^q0$")))
    expect(row).to_contain_text(was)
    expect(view).not_to_contain_text("q0 reworded")
    view.get_by_role("button", name="Restore this version").click()
    expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first}.")
    expect(view).to_be_empty()
    assert httpx.get(f"{seeded.url}/eval-sets/ver1").json()["version"] == old["version"]
    expect(page.locator("#head")).to_contain_text(f"version {first}")
    expect(page.locator("#questions")).to_contain_text(was)


def test_eval_set_page_shows_the_newest_20_versions_first(page, seeded):
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": "ver2", "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    for i in range(22):
        one = {"id": "s0", "text": f"version two text {i}", "choices": ["x", "y"]}
        put_questions(seeded, "ver2", [{**one, "gold_answers": ["A"]}])
    page.goto(seeded.url + "/e/ver2")
    items = page.locator("#versions li")
    expect(items).to_have_count(20)
    page.get_by_role("button", name="Show all versions").click()
    expect(items).to_have_count(22)
    expect(page.get_by_role("button", name="Show all versions")).to_have_count(0)


def test_eval_set_page_back_to_the_current_version_closes_the_view(page, seeded):
    mcq_set(seeded, "ver5", 3)
    before = httpx.get(f"{seeded.url}/eval-sets/ver5").json()
    put_questions(seeded, "ver5", before["questions"][:2], before["version"])
    page.goto(seeded.url + "/e/ver5")
    old = page.locator("#versions li", has_text=before["version"][:8])
    old.get_by_role("button", name="View").click()
    view = page.locator("#ver-view")
    expect(view).to_contain_text("3 questions")
    view.get_by_role("button", name="Back to the current version").click()
    expect(view).to_be_empty()
    assert len(httpx.get(f"{seeded.url}/eval-sets/ver5").json()["questions"]) == 2


def test_eval_set_page_restore_waits_for_unsaved_edits(page, seeded):
    mcq_set(seeded, "ver4", 3)
    before = httpx.get(f"{seeded.url}/eval-sets/ver4").json()
    put_questions(seeded, "ver4", before["questions"][:2], before["version"])
    page.goto(seeded.url + "/e/ver4")
    page.locator("#versions li", has_text=before["version"][:8]).get_by_role(
        "button", name="View"
    ).click()
    restore = page.get_by_role("button", name="Restore this version")
    expect(restore).to_be_enabled()
    row = page.locator("#questions tbody tr").first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("an edit not saved")
    expect(restore).to_be_disabled()
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    expect(restore).to_be_enabled()


def test_eval_set_page_restore_waits_for_unsaved_marks(page, seeded):
    labelling_set(seeded, "ver3")
    labelled_questions(seeded, "ver3", ["alpha first?", "brine second?"])
    older = httpx.get(f"{seeded.url}/eval-sets/ver3q").json()["versions"][1]["version"]
    page.goto(seeded.url + "/e/ver3q")
    page.locator("#versions li", has_text=older[:8]).get_by_role("button", name="View").click()
    view = page.locator("#ver-view")
    expect(view).to_contain_text(f"Version {older[:8]}: 1 question")
    expect(view).to_contain_text("alpha first?")
    restore = view.get_by_role("button", name="Restore this version")
    expect(restore).to_be_enabled()
    rows = page.locator("#lab-questions tbody tr")
    rows.filter(has_text="alpha first?").get_by_role("button", name="Label").click()
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    expect(restore).to_be_enabled()  # opening a question is not a change
    mark(passages.first, "y")
    expect(restore).to_be_disabled()
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(restore).to_be_enabled()
    panel.get_by_label("Reference answer (optional; short is better)").fill("soon")
    expect(restore).to_be_disabled()
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    expect(restore).to_be_disabled()  # Label asks first: the typing is still there
    panel.get_by_role("button", name="Discard them").click()
    expect(restore).to_be_enabled()  # opening another question drops the typing
    panel.get_by_label("Question", exact=True).fill("brine second, reworded?")
    expect(restore).to_be_disabled()


def restorable(seeded, name):
    """The set `<name>q` of one question, first right on the store's first passage and judged
    on both, then saved with no right passage and only the other judged. Returns the
    question's id, the first version and the two passage ids, the right one first."""
    labelling_set(seeded, name)
    pool = httpx.post(f"{seeded.url}/eval-sets/{name}q/pool", json={"question": "alpha"})
    assert pool.status_code == 200, pool.text
    right, other = (p["id"] for p in pool.json()["passages"])
    question = {"text": "alpha restored?", "gold_ids": [right], "judged_ids": [right, other]}
    r = httpx.post(f"{seeded.url}/eval-sets/{name}q/questions", json={"question": question})
    assert r.status_code == 201, r.text
    qid, first = r.json()["id"], r.json()["version"]
    stored = question_map(seeded, f"{name}q")[qid]
    changed = {**stored, "gold_ids": [], "judged_ids": [other]}
    r = httpx.put(
        f"{seeded.url}/eval-sets/{name}q/questions/{qid}",
        json={"question": changed, "base": stored},
    )
    assert r.status_code == 200, r.text
    return qid, first, right, other


def hold_restore(page, seeded, name, first):
    """Opens the question in the panel, views version `first` and presses Restore with its
    PUT held; returns the held routes."""
    page.goto(f"{seeded.url}/e/{name}q")
    page.locator("#lab-questions tbody tr").first.get_by_role("button", name="Label").click()
    expect(page.locator("#lab-panel li[data-passage]")).to_have_count(2)
    page.locator("#versions li", has_text=first[:8]).get_by_role("button", name="View").click()
    held = []
    page.route(f"**/eval-sets/{name}q/questions", lambda route: held.append(route))
    page.get_by_role("button", name="Restore this version").click()
    deadline = time.time() + 5
    while not held and time.time() < deadline:
        page.wait_for_timeout(20)
    assert held, "the Restore PUT never left"
    return held


def test_eval_set_page_restore_holds_the_panel_still_and_brings_back_judged_ids(page, seeded):
    qid, first, right, other = restorable(seeded, "rs1")
    held = hold_restore(page, seeded, "rs1", first)
    panel = page.locator("#lab-panel")
    row = page.locator("#lab-questions tbody tr").first
    for button in ("Save question", "Save and write the next", "Find passages"):
        expect(panel.get_by_role("button", name=button)).to_be_disabled()
    expect(row.get_by_role("button", name="Label")).to_be_disabled()
    expect(row.get_by_role("button", name="Delete")).to_be_disabled()
    for passage in panel.locator("li[data-passage]").all():
        expect(pressed(passage, "y")).to_be_disabled()
        expect(pressed(passage, "n")).to_be_disabled()
    held[0].continue_()
    expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first[:8]}.")
    got = question_map(seeded, "rs1q")[qid]
    assert (got["gold_ids"], got["judged_ids"]) == ([right], [right, other])
    expect(panel.get_by_label("Question", exact=True)).to_have_value("")  # nothing unsaved
    expect(row.get_by_role("button", name="Label")).to_be_enabled()
    expect(row.get_by_role("button", name="Delete")).to_be_enabled()
    expect(row.locator("td").nth(1)).to_have_text("1")
    expect(row.locator("td").nth(2)).to_have_text("2")


def test_eval_set_page_restore_keeps_what_was_typed_while_it_ran(page, seeded):
    _, first, _, other = restorable(seeded, "rs2")
    held = hold_restore(page, seeded, "rs2", first)
    panel = page.locator("#lab-panel")
    answer = panel.get_by_label("Reference answer (optional; short is better)")
    answer.fill("typed meanwhile")
    held[0].continue_()
    expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first[:8]}.")
    expect(panel.get_by_label("Question", exact=True)).to_have_value("alpha restored?")
    expect(answer).to_have_value("typed meanwhile")
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    kept = page.locator(f'#lab-panel li[data-passage="{other}"]')
    expect(pressed(kept, "n")).to_have_attribute("aria-pressed", "true")
    expect(panel.get_by_role("button", name="Save question")).to_be_enabled()


def test_eval_set_page_lists_the_current_version_after_restoring_an_old_one(page, seeded):
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": "ver6", "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    made = []
    for i in range(22):
        one = {"id": "s0", "text": f"version six text {i}", "choices": ["x", "y"]}
        made.append(put_questions(seeded, "ver6", [{**one, "gold_answers": ["A"]}]))
    old = httpx.get(f"{seeded.url}/eval-sets/ver6", params={"version": made[0]}).json()
    assert put_questions(seeded, "ver6", old["questions"], made[-1]) == made[0]
    page.goto(seeded.url + "/e/ver6")
    items = page.locator("#versions li")
    current = items.filter(has_text="current")
    expect(current).to_have_count(1)
    expect(current).to_contain_text(made[0][:8])
    expect(items).to_have_count(21)  # the current one, then the newest 20
    page.get_by_role("button", name="Show all versions").click()
    expect(items).to_have_count(22)
    expect(current).to_have_count(1)


def test_linking_an_older_documents_set_picks_its_own_store_first(page, seeded):
    for store in ("pk1a", "pk1b", "pk1c"):
        documents_store(seeded, store, passages=False)
    for name in ("pk1b", "pk1x", "pk1z"):
        r = httpx.post(f"{seeded.url}/eval-sets", json={"name": name, "qa_type": "documents"})
        assert r.status_code == 201, r.text
    r = httpx.patch(f"{seeded.url}/collections/pk1c", json={"eval_set": "pk1x"})
    assert r.status_code == 200, r.text
    pick = page.get_by_label("Pick a store to label against:")
    page.goto(seeded.url + "/e/pk1x")  # pk1c's default set
    expect(pick).to_have_value("pk1c")
    page.goto(seeded.url + "/e/pk1b")  # named after pk1b
    expect(pick).to_have_value("pk1b")
    page.goto(seeded.url + "/e/pk1z")  # no store of its own: the first
    expect(pick.locator("option")).not_to_have_count(0)
    expect(pick).to_have_value(pick.locator("option").first.get_attribute("value") or "")


def test_create_one_on_a_store_page_makes_a_linked_set(page, seeded):
    documents_store(seeded, "mk1", passages=False)
    page.goto(seeded.url + "/c/mk1")
    page.locator("#asked").get_by_role("link", name="Create one").click()
    expect(page).to_have_url(seeded.url + "/eval-sets?store=mk1")
    expect(page.get_by_label("Name")).to_have_value("mk1")
    expect(page.locator("#f-type")).to_have_value("documents")
    expect(page.locator("#f-type")).to_be_disabled()
    expect(page.get_by_text("Linked store: mk1", exact=True)).to_be_visible()
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/e/mk1")
    assert httpx.get(f"{seeded.url}/eval-sets/mk1").json()["store"] == "mk1"
    assert httpx.get(f"{seeded.url}/collections/mk1").json()["eval_set"] == "mk1"


def test_new_eval_set_form_links_a_documents_store(page, seeded):
    documents_store(seeded, "link1", passages=False)
    page.goto(seeded.url + "/eval-sets")
    page.get_by_role("button", name="+ New eval set").click()
    expect(page.get_by_label("Store to label against")).to_be_hidden()
    page.get_by_label("Name").fill("link1q")
    page.locator("#f-type").select_option("documents")
    pick = page.get_by_label("Store to label against")
    expect(pick.locator("option").first).to_have_text("(none)")
    values = pick.locator("option").evaluate_all("o => o.map(x => x.value)")
    assert "link1" in values and "c" not in values  # Documents stores only
    pick.select_option("link1")
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/e/link1q")
    assert httpx.get(f"{seeded.url}/eval-sets/link1q").json()["store"] == "link1"


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


def test_an_eval_set_without_a_store_offers_no_run_link(page, seeded):
    mcq_set(seeded, "run2", 2)
    page.goto(seeded.url + "/e/run2")
    expect(page.locator("#questions tbody tr")).to_have_count(2)
    expect(page.get_by_role("link", name="Run against its store")).to_have_count(0)


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


# -- a Documents run's page ---------------------------------------------------------------------

RUN_LEAD = "How well does this store find the right passages?"
NOT_TRACKED = (
    "This eval set doesn't record which passages were judged, so score reliability can't be checked"
)


def documents_run(
    seeded, name, methods, judged_all=True, answer=False, judge=False, references=0, held=True
):
    """A Documents store `name` and its linked set `<name>q` of two questions, each right on
    the store's first passage and judged on both passages (with `judged_all`) or on that one
    alone; then a finished run of `methods` on the set. The first `references` questions carry a
    reference answer; with `held=False` the right passage is not in the store. Returns the run id
    and the ids of the two passages, the right one first."""
    labelling_set(seeded, name)
    pool = httpx.post(f"{seeded.url}/eval-sets/{name}q/pool", json={"question": "alpha"})
    assert pool.status_code == 200, pool.text
    ids = [p["id"] for p in pool.json()["passages"]]
    assert len(ids) == 2, ids
    judged = ids if judged_all else ids[:1]
    gold = ids[:1] if held else [f"{name}-gone"]
    for n, text in enumerate(("how long does alpha last?", "what does alpha hold?")):
        question = {"text": f"{text} ({name})", "gold_ids": gold, "judged_ids": judged}
        if n < references:
            question["gold_answers"] = ["Ten days"]
        r = httpx.post(f"{seeded.url}/eval-sets/{name}q/questions", json={"question": question})
        assert r.status_code == 201, r.text
    body = {
        "collection": name,
        "eval_set": f"{name}q",
        "split": "full",
        "methods": methods,
        "answer": answer,
        "judge": judge,
        "confirm": True,
    }
    r = httpx.post(f"{seeded.url}/eval", json=body)
    assert r.status_code == 202, r.text
    run_id = r.json()["job_id"]
    assert seeded.wait(run_id)["status"] == "done"
    return run_id, ids


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
    expect(page.locator("#gaps table")).to_have_count(0)
    expect(page.locator("[data-gap-row]")).to_have_count(0)
    expect(page.locator("#note")).to_have_text("")


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
    expect(page.get_by_role("button", name="Cancel")).to_be_visible()
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
    expect(page.locator("#runs tbody tr")).to_have_count(1)
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


# -- a Documents run's answers line -------------------------------------------------------------

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


def test_lab_names_stores_not_collections(page, seeded):
    page.goto(seeded.url + "/lab")
    expect(page.locator("label", has=page.locator("#collection"))).to_contain_text("Store")
    expect(page.locator("label", has=page.locator("#collection"))).not_to_contain_text("Collection")
    page.route("**/collections", lambda route: route.fulfill(json=[]))
    page.goto(seeded.url + "/lab")
    expect(page.locator("#status")).to_have_text("No stores yet. Create one from the Stores page.")


def test_chat_delete_prompt_has_one_question_mark(page, seeded):
    page.goto(seeded.url + "/chat")
    page.locator("#new").click()
    page.locator('#picking input[value="c"]').check()
    page.get_by_role("button", name="Start chat").click()
    chat_ask(page, "What is alpha?")
    expect(page.locator("#log .reply")).to_have_count(1)
    chat_id = page.url.split("#")[1]
    chat_link(page, chat_id).locator("xpath=..").get_by_role("button", name="Delete").click()
    expect(page.locator("#deleting p")).to_have_text("Delete chat What is alpha?")


def test_chat_store_picker_shows_type_labels(page, seeded):
    seeded.build("docsty", qa_type="documents")
    page.goto(seeded.url + "/chat")
    page.locator("#new").click()
    row = page.locator("#picking label", has=page.locator('input[value="docsty"]'))
    expect(row.locator(".muted")).to_have_text("Documents")
    page.locator('#picking input[value="c"]').check()
    page.get_by_role("button", name="Start chat").click()
    expect(page.locator("#add-store option[value=docsty]")).to_have_text("docsty · Documents")


@pytest.mark.parametrize("late", ["collections", "catalog"])
def test_chat_new_chat_waits_for_the_stores_and_the_catalog(page, seeded, late):
    label = httpx.get(seeded.url + "/catalog").json()["qa_types"]["mcq"]["label"]
    assert label != "mcq"
    held = []
    page.route(
        f"**/{late}",
        lambda route: held.append(route) if route.request.method == "GET" else route.fallback(),
    )
    page.goto(seeded.url + "/chat")
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, f"/{late} was never read"
    page.wait_for_timeout(300)  # the rest of the boot, had it not waited
    expect(page.locator("#new")).to_be_disabled()
    for route in held:  # the side list reads the stores too
        route.continue_()
    expect(page.locator("#new")).to_be_enabled()
    page.locator("#new").click()
    row = page.locator("#picking label", has=page.locator('input[value="c"]'))
    expect(row.locator(".muted")).to_have_text(label)


def test_chat_store_picker_falls_back_to_the_raw_type_without_a_catalog(page, seeded):
    seeded.build("docsraw", qa_type="documents")
    page.expected = ["500"]
    page.route("**/catalog", lambda route: route.fulfill(status=500, json={"detail": "down"}))
    page.goto(seeded.url + "/chat")
    page.locator("#new").click()
    row = page.locator("#picking label", has=page.locator('input[value="docsraw"]'))
    expect(row.locator(".muted")).to_have_text("documents")


def test_chat_phone_layout_puts_the_conversation_first(page, seeded):
    page.set_viewport_size({"width": 400, "height": 800})
    page.goto(seeded.url + "/chat")
    page.locator("#new").click()
    page.locator('#picking input[value="c"]').check()
    page.get_by_role("button", name="Start chat").click()
    expect(page.locator("#chat-head")).to_be_visible()
    assert page.locator("#convo").bounding_box()["y"] < page.locator("#chats").bounding_box()["y"]


def test_store_types_read_as_whole_sentences(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    options = page.locator("#f-type option")
    expect(options).not_to_have_count(0)
    texts = [t.strip() for t in options.all_text_contents()]
    for want in (
        "Science problems: retrieves worked examples whose reasoning carries over, "
        "with or without choices",
        "Exam multiple choice: retrieves worked examples whose reasoning carries over",
        "Tables: retrieves the table and text that hold the figures for the answer",
    ):
        assert want in texts


def test_stopped_load_shows_on_the_stores_row_and_the_store_page(page, seeded):
    seeded.build("lf")
    worker = seeded.app.state.worker
    worker.stop()
    try:
        conn = seeded.app.state.conn
        job = db.create_job(conn, "load", "lf", {})
        db.update_job(conn, job, status="failed", error="HF said no")
    finally:
        worker.start()
    page.goto(seeded.url + "/")
    row = page.locator("a.row-link[href='/c/lf']")
    expect(row).to_contain_text("Last load stopped: HF said no")
    page.goto(seeded.url + "/c/lf")
    expect(page.locator("#benchmark")).to_contain_text("Last load stopped: HF said no")


# -- Plan 4, Task 4: Create store opens the store; the store page starts a chat ----------------

EMPTY_WHY = "Add items first: an empty store has nothing to search."


def test_create_store_opens_its_page(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("opened")
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/c/opened")
    expect(page.locator('#side a[href="/c/opened"]')).to_be_visible()


def test_store_page_starts_a_chat_with_the_store(page, seeded):
    seeded.build("chatstore")
    page.goto(seeded.url + "/c/chatstore")
    button = page.locator("#chat-store")
    expect(button).to_be_enabled()
    button.click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    chat_id = page.url.split("#", 1)[1]
    assert httpx.get(f"{seeded.url}/chats/{chat_id}").json()["stores"] == ["chatstore"]
    expect(page.locator("#chips .chip")).to_have_text([re.compile("^chatstore")])


def test_store_page_chat_button_makes_one_chat_after_refreshes(page, seeded):
    seeded.build("chatonce")
    before = len(httpx.get(f"{seeded.url}/chats").json())
    page.clock.install()
    page.goto(seeded.url + "/c/chatonce")
    expect(page.locator("#chat-store")).to_be_enabled()
    seen = []
    page.on(
        "response", lambda r: seen.append(1) if r.url.endswith("/collections/chatonce") else None
    )
    for want in (1, 2):  # two idle refreshes, 15 s apart (the load's own fetch came first)
        for _ in range(50):  # refresh() schedules its next timer only after it has painted: retry
            if len(seen) >= want:
                break
            page.clock.fast_forward(16000)
            page.wait_for_timeout(100)
        assert len(seen) >= want
    page.locator("#chat-store").click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    assert len(httpx.get(f"{seeded.url}/chats").json()) == before + 1


def test_store_page_chat_button_works_again_after_back(page, seeded):
    seeded.build("chatback")
    held = []  # the chat POST never answers: the page stays as it was when it left
    page.route("**/chats", lambda r: held.append(r) if r.request.method == "POST" else r.fallback())
    page.goto(seeded.url + "/c/chatback")
    button = page.locator("#chat-store")
    expect(button).to_be_enabled()
    button.click()
    expect(button).to_be_disabled()
    assert held
    # Playwright's Chromium runs without the back/forward cache, so a restore is dispatched
    page.evaluate("window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }))")
    expect(button).to_be_enabled()


def test_chat_with_an_empty_store_is_disabled(page, seeded):
    r = httpx.post(f"{seeded.url}/collections", json={"name": "nochat", "qa_type": "documents"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/c/nochat")
    expect(page.locator("#chat-store-why")).to_have_text(EMPTY_WHY)
    expect(page.locator("#chat-store")).to_be_disabled()
    expect(page.locator("#chat-store")).to_have_attribute("title", EMPTY_WHY)


def test_create_store_refused_stays_on_the_form(page, seeded):
    page.expected = ["409"]
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("c")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("#f-msg")).to_contain_text("already exists")
    expect(page).to_have_url(seeded.url + "/")
    expect(page.locator("#form")).to_be_visible()
