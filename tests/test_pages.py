"""The console pages, driven in a real browser against the app on fakes."""

import json
import re
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
    page.get_by_role("button", name="+ New collection").click()
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
        "What does it get asked?",
        "How are items described?",
    ]
    assert titles[-1] == "Delete this collection"


def test_collection_page_shows_verdict_gap_bar_and_runs(page, seeded):
    page.goto(seeded.url + "/c/c")
    expect(page.locator("main svg[role=img]").first).to_be_visible()
    expect(page.locator("#runs tbody tr")).to_have_count(1)
    expect(page.get_by_text("run 1 on this collection", exact=False)).to_be_visible()


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
    claude = seeded.app.state.claude
    fake_replies(claude, lambda p: '{"topic": ["enzymes"], "method": ["assay"]}')
    try:
        page.goto(seeded.url + "/c/prop")
        expect(page.locator("#kw-topic")).to_have_value("enzyme\nmarket\nvolcano\noptics")
        page.get_by_role("button", name="Propose with Claude").click()
        expect(page.get_by_text("Claude call", exact=False)).to_be_visible()
        assert page.locator("#kw-topic").input_value().startswith("enzyme\n")  # not proposed yet
        page.get_by_role("button", name="Start").click()
        expect(page.locator("#kw-topic")).to_have_value("enzymes")
    finally:
        fake_replies(claude, lambda p: "Answer: A")


def test_new_run_estimates_then_starts(page, seeded):
    seeded.build("runs", run=True)
    page.goto(seeded.url + "/c/runs")
    expect(page.locator("#runs tbody tr")).to_have_count(1)
    page.get_by_role("button", name="New run").click()
    expect(page.get_by_label("kw_individual")).to_be_checked()
    expect(page.get_by_label("random_example")).to_be_checked()
    page.get_by_role("button", name="Estimate").click()
    expect(page.get_by_text("Claude calls", exact=False)).to_be_visible()
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
    expect(page.get_by_text("Claude calls", exact=False)).to_be_visible()
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
    expect(page.get_by_text("Claude call", exact=False)).to_be_visible()
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
    expect(page.get_by_text("Claude call", exact=False)).to_be_visible()
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
    page.get_by_role("button", name="+ New collection").click()
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
    expect(page.get_by_text("51 Claude calls")).to_be_visible()
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
    page.get_by_role("button", name="Set up the four benchmark collections").click()
    expect(page.get_by_text("5 Claude calls")).to_be_visible()
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
    page.get_by_role("button", name="+ New collection").click()
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


def credit_label(page, name):
    return page.locator(f"label.check:has(input[value='{name}'])").locator(".credit")


def test_paid_methods_carry_a_uses_credit_mark(page, seeded):
    seeded.build("credit")
    page.goto(seeded.url + "/c/credit")
    page.get_by_role("button", name="New run").click()
    expect(credit_label(page, "iterative")).to_have_text("uses credit")
    expect(credit_label(page, "hybrid")).to_have_count(0)
    # a keyword method spends only with summaries, and this collection has none
    expect(credit_label(page, "kw_individual")).to_have_count(0)
    expect(
        page.locator("label.check", has=page.locator("#rf-answer")).locator(".credit")
    ).to_have_text("uses credit")


def test_keyword_methods_say_uses_credit_on_a_summarising_collection(page, seeded):
    seeded.build("sumcol", summarize=True)
    page.goto(seeded.url + "/c/sumcol")
    page.get_by_role("button", name="New run").click()
    expect(credit_label(page, "kw_individual")).to_have_text("uses credit")
    expect(credit_label(page, "iterative")).to_have_text("uses credit")


def test_lab_marks_paid_methods_and_the_answer_box(page, seeded):
    page.goto(seeded.url + "/lab")
    expect(page.locator("#methods label", has_text="iterative").locator(".credit")).to_have_text(
        "uses credit"
    )
    expect(
        page.locator("#methods label", has_text="hybrid").first.locator(".credit")
    ).to_have_count(0)
    expect(
        page.locator("#methods label", has_text="kw_individual").locator(".credit")
    ).to_have_count(0)  # `c` has no summaries
    expect(page.locator("label", has=page.locator("#answer")).locator(".credit")).to_have_text(
        "uses credit"
    )


def test_lab_marks_keyword_methods_on_a_summarising_collection(page, seeded):
    seeded.build("labsum", summarize=True)
    page.goto(seeded.url + "/lab")
    page.locator("#collection").select_option("labsum")
    expect(
        page.locator("#methods label", has_text="kw_individual").locator(".credit")
    ).to_have_text("uses credit")


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
    expect(box).to_contain_text("Claude call")
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
    expect(page.locator("#imp-items-source")).to_have_value("notes")
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("3 rows")
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
        "Delete source upload:notes: its 3 items and the questions imported with it?"
    )
    page.get_by_role("button", name="Yes, delete upload:notes").click()
    expect(page.locator("#sources tbody tr", has_text="upload:notes")).to_have_count(0)


def test_import_questions_shows_leaked_and_include(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "qimp", "qa_type": "mcq"})
    item = {
        "id": "tide",
        "text": "the tides of the ocean follow the moon every single day",
        "answer": "A",
        "explanation": "",
        "meta": {},
    }
    r = httpx.post(
        f"{seeded.url}/collections/qimp/documents", json={"items": [item], "confirm": True}
    )
    assert r.status_code == 200, r.text
    rows = [
        {
            "id": "u1",
            "text": "tides of the ocean follow the moon",
            "choices": ["x", "y"],
            "answer": "A",
        },
        {"id": "u2", "text": "an unrelated one", "choices": ["x", "y"], "answer": "B"},
    ]
    qs = tmp_path / "qs.jsonl"
    qs.write_text("\n".join(json.dumps(x) for x in rows))
    page.goto(seeded.url + "/c/qimp")
    upload_and_preview(page, "questions", qs)
    expect(page.locator("#imp-questions-preview")).to_contain_text("2 rows")
    page.locator("#imp-questions-import").click()
    expect(page.get_by_text(re.compile(r"Queued \(job \w{8}\)"))).to_be_visible()
    jobs_done(seeded, "qimp")
    page.reload()
    asked = page.locator("#asked")
    expect(asked).to_contain_text("1 question")
    expect(asked).to_contain_text(
        "1 question appears among the items (leaked) and is left out of runs"
    )
    expect(asked).to_contain_text("Worked solutions: 0 of 1")
    expect(asked).to_contain_text("Evidence ids: —")
    page.get_by_label("Include them anyway").check()
    expect(asked).to_contain_text("2 questions")
    assert httpx.get(f"{seeded.url}/collections/qimp").json()["include_leaked"] is True


def test_own_science_collection_offers_import_and_benchmark(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "own", "qa_type": "science"})
    page.goto(seeded.url + "/c/own")
    expect(page.locator("#imp-items-file")).to_be_visible()
    expect(page.locator("#imp-questions-file")).to_be_visible()
    expect(page.get_by_text("Or load the built-in benchmark, scibench")).to_be_visible()
    expect(page.get_by_role("button", name="Load scibench")).to_be_visible()
    notes = tmp_path / "own notes!.csv"
    notes.write_text("text\nfirst thing about reefs\nsecond thing about tides\n")
    page.locator("#imp-items-file").set_input_files(str(notes))
    expect(page.locator("#imp-items-source")).to_have_value("own notes-")
    page.locator("#imp-items-go").click()
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
