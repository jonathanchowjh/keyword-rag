"""The Lab and the Worker page."""

import httpx
import pytest
from page_helpers import (
    REBUILD_MODEL,
    documents_store,
    interrupt,
    worker_stopped,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.pages


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
        page.locator("[data-queued-job] button", has_text="Stop job").click()
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


def test_lab_names_stores_not_collections(page, seeded):
    page.goto(seeded.url + "/lab")
    expect(page.locator("label", has=page.locator("#collection"))).to_contain_text("Store")
    expect(page.locator("label", has=page.locator("#collection"))).not_to_contain_text("Collection")
    page.route("**/collections", lambda route: route.fulfill(json=[]))
    page.goto(seeded.url + "/lab")
    expect(page.locator("#status")).to_have_text("No stores yet. Create one from the Stores page.")


def test_lab_hides_multiple_choice_parts_for_a_documents_store(page, seeded):
    documents_store(seeded, "labdocs")
    page.goto(seeded.url + "/lab")
    choices = page.get_by_text("Choices, one per line", exact=False)
    page.locator("#collection").select_option("labdocs")
    expect(page.locator("#methods input[value=hybrid]")).to_be_visible()
    expect(choices).to_be_hidden()
    expect(page.locator("#head .sub")).not_to_contain_text("Highlighted chips")
    expect(page.locator("#head .sub")).to_contain_text(
        "One question, several methods side by side."
    )
    r = httpx.post(f"{seeded.url}/collections", json={"name": "labsci", "qa_type": "science"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/lab")
    page.locator("#collection").select_option("labsci")
    expect(page.locator("#methods")).not_to_be_empty()
    expect(choices).to_be_visible()
    page.locator("#collection").select_option("c")
    expect(page.locator("#methods input[value=kw_individual]")).to_be_visible()
    expect(choices).to_be_visible()
    expect(page.locator("#head .sub")).to_contain_text("Highlighted chips")
