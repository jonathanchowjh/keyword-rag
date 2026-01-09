"""The console pages, driven in a real browser against the app on fakes."""

from contextlib import contextmanager

import httpx
import pytest
from playwright.sync_api import expect

from ragqa_core import db
from ragqa_core.fakes import fake_replies

pytestmark = pytest.mark.pages

ORIGINAL_TOPICS = "enzyme\nmarket\nvolcano\noptics"  # the topic list `seeded.build` saves


def test_collections_page_shows_the_answer(page, seeded):
    page.goto(seeded.url + "/")
    row = page.locator('main a.row-link[href="/c/c"]')
    expect(row).to_be_visible()
    expect(row).to_contain_text("Keywords per list")
    expect(row).to_contain_text("run " + seeded.run_id[:4])
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
    expect(page.locator("main table tbody tr")).to_have_count(1)
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
    expect(page.get_by_text("run " + run_id[:4]).first).to_be_visible(timeout=15000)
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
    expect(page.locator("main table tbody tr")).to_have_count(1)
    page.get_by_role("button", name="New run").click()
    expect(page.get_by_label("kw_individual")).to_be_checked()
    expect(page.get_by_label("random_example")).to_be_checked()
    page.get_by_role("button", name="Estimate").click()
    expect(page.get_by_text("Claude calls", exact=False)).to_be_visible()
    page.get_by_role("button", name="Start").click()
    expect(page.locator("main table tbody tr")).to_have_count(2)


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
    expect(page.locator("#head")).to_contain_text("Run " + run_id[:4])
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
        expect(page.get_by_text("The worker thread has stopped")).to_be_visible()
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
    expect(page.get_by_text("Queued (job ab12)")).to_be_visible()
    assert [b.get("confirm") for b in seen] == [False, True]


def test_collection_load_that_is_free_queues_at_once(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "freeload", "qa_type": "mcq"})
    seen = route_load(page, "freeload", lambda b: (202, {"job_id": "ee99ff"}))
    page.goto(seeded.url + "/c/freeload")
    page.get_by_role("button", name="Load scienceqa").click()
    expect(page.get_by_text("Queued (job ee99)")).to_be_visible()
    assert len(seen) == 1


def test_setup_benchmarks_sums_one_estimate_and_confirms_the_paying_loads(page, seeded):
    for n in ("sqa", "mq", "tq"):
        assert n not in [c["name"] for c in httpx.get(f"{seeded.url}/collections").json()]
    paying = {"sqa": 0.10, "tq": 0.25}
    seen = {n: route_load(page, n, _reply(paying, n)) for n in ("sqa", "mq", "tq")}
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="Set up the three benchmark collections").click()
    expect(page.get_by_text("5 Claude calls")).to_be_visible()
    expect(page.get_by_text("$0.35")).to_be_visible()
    expect(page.get_by_role("button", name="Start")).to_have_count(1)
    assert [b.get("confirm") for b in seen["sqa"]] == [None] and len(seen["tq"]) == 1
    page.get_by_role("button", name="Start").click()
    expect(page.get_by_text("sqa: loading scienceqa (queued, job sqa0)")).to_be_visible()
    expect(page.get_by_text("tq: loading tatqa (queued, job tq01)")).to_be_visible()
    assert [b.get("confirm") for b in seen["sqa"]] == [None, True]
    assert [b.get("confirm") for b in seen["tq"]] == [None, True]
    assert len(seen["mq"]) == 1  # already queued by its first answer, never posted twice


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
    expect(page.locator(".verdict")).to_contain_text(CAVEAT)
    page.goto(f"{seeded.url}/c/c/runs/{seeded.run_id}")
    expect(page.locator(".verdict")).not_to_contain_text(CAVEAT)


def test_collection_offers_finish_loading_after_an_interrupted_load(page, seeded):
    seeded.build("fl")
    job = [{"id": "dead0000", "kind": "load", "collection": "fl", "status": "interrupted"}]
    page.route("**/jobs?*kind=load*", lambda route: route.fulfill(json=job))
    page.goto(seeded.url + "/c/fl")
    expect(page.get_by_role("button", name="Finish loading")).to_be_visible()


def test_collection_without_questions_offers_finish_loading(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "noq", "qa_type": "mcq"})
    r = httpx.post(
        f"{seeded.url}/collections/noq/documents", json={"items": ITEMS_ONE, "confirm": True}
    )
    assert r.status_code == 200, r.text
    page.goto(seeded.url + "/c/noq")
    expect(page.get_by_role("button", name="Finish loading")).to_be_visible()
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


def test_paid_methods_carry_a_uses_credit_mark(page, seeded):
    seeded.build("credit")
    page.goto(seeded.url + "/c/credit")
    page.get_by_role("button", name="New run").click()
    paid = page.locator("label.check", has=page.get_by_label("iterative"))
    free = page.locator("label.check", has=page.get_by_label("hybrid", exact=True))
    expect(paid.locator(".credit")).to_have_text("uses credit")
    expect(free.locator(".credit")).to_have_count(0)
    expect(
        page.locator("label.check", has=page.locator("#rf-answer")).locator(".credit")
    ).to_have_text("uses credit")


def test_lab_marks_paid_methods_and_the_answer_box(page, seeded):
    page.goto(seeded.url + "/lab")
    expect(page.locator("#methods label", has_text="iterative").locator(".credit")).to_have_text(
        "uses credit"
    )
    expect(
        page.locator("#methods label", has_text="hybrid").first.locator(".credit")
    ).to_have_count(0)
    expect(page.locator("label", has=page.locator("#answer")).locator(".credit")).to_have_text(
        "uses credit"
    )


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
