"""The Stores page: the Start here strip, New store, and the built-in benchmarks."""

import httpx
import pytest
from page_helpers import (
    route_load,
)
from playwright.sync_api import expect

from ragqa_core import db

pytestmark = pytest.mark.pages


def test_collections_page_shows_the_answer(page, seeded):
    page.goto(seeded.url + "/")
    row = page.locator('main a.row-link[href="/c/c"]')
    expect(row).to_be_visible()
    expect(row).to_contain_text("Do keywords help? ")
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


def test_one_path_end_to_end(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("e2e")
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/c/e2e")
    expect(page.get_by_text("No score yet: run Hybrid + reranker to get one").first).to_be_visible()
    page.goto(seeded.url + "/worker")
    expect(page.locator("#main")).to_be_visible()


def test_setup_benchmarks_sums_one_estimate_and_confirms_the_paying_loads(page, seeded):
    for n in ("sqa", "mq", "tq", "sb"):
        assert n not in [c["name"] for c in httpx.get(f"{seeded.url}/collections").json()]
    paying = {"sqa": 0.10, "tq": 0.25}
    seen = {n: route_load(page, n, _reply(paying, n)) for n in ("sqa", "mq", "tq", "sb")}
    page.goto(seeded.url + "/")
    page.get_by_text("Try a built-in benchmark").click()
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
    # Each load says the price it showed, not the sum.
    assert (seen["sqa"][1]["calls"], seen["sqa"][1]["est_usd"]) == (2, 0.10)
    assert (seen["tq"][1]["calls"], seen["tq"][1]["est_usd"]) == (3, 0.25)
    assert len(seen["mq"]) == 1  # already queued by its first answer, never posted twice


def test_setup_benchmarks_with_an_unpriced_load_shows_no_total(page, seeded):
    names = ("sqa", "mq", "tq", "sb")
    for n in names:  # the benchmark stores, as the test above may have left them
        httpx.delete(f"{seeded.url}/collections/{n}")
    paying = {"sqa": 0.10, "tq": None}  # tq: no model to price it
    for n in names:
        route_load(page, n, _reply(paying, n))
    page.goto(seeded.url + "/")
    page.get_by_text("Try a built-in benchmark").click()
    page.get_by_role("button", name="Set up the four benchmark stores").click()
    expect(page.get_by_text("5 AI calls")).to_be_visible()
    cost = page.locator("#steps .confirm")
    expect(cost).to_contain_text("cost unknown: no AI model is set up")
    expect(cost).not_to_contain_text("$")


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


def test_stores_page_leads_with_new_store(page, seeded):
    page.goto(seeded.url + "/")
    assert page.title().startswith("keyword-rag")
    expect(page.locator("#side")).to_contain_text("keyword-rag")
    expect(page.locator("#benchmarks")).not_to_have_attribute("open", "")
    expect(page.get_by_role("button", name="Add scienceqa")).to_be_hidden()
    expect(page.locator("#defaults button")).not_to_have_count(0)  # drawn, only folded away
    order = page.evaluate(
        """() => {
          const all = [...document.querySelectorAll("main button")]
          .map((b) => b.textContent.trim());
          return [all.indexOf("+ New store"), all.findIndex((t) => t.startsWith("Add scienceqa"))];
        }"""
    )
    assert 0 <= order[0] < order[1]
    page.get_by_text("Try a built-in benchmark").click()
    expect(page.get_by_role("button", name="Add scienceqa")).to_be_visible()


def test_add_a_default_store_lists_every_dataset_and_loads_one(page, seeded):
    datasets = httpx.get(f"{seeded.url}/catalog").json()["datasets"]
    seen = route_load(page, "gpqa", lambda b: (202, {"job_id": "dd11ee22"}))
    page.goto(seeded.url + "/")
    expect(page.get_by_role("heading", name="Make a store from your documents")).to_be_visible()
    page.get_by_text("Try a built-in benchmark").click()
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
    page.get_by_text("Try a built-in benchmark").click()
    page.get_by_role("button", name="Add tatqa").click()
    expect(page.get_by_text("7 AI calls")).to_be_visible()
    assert [b.get("confirm") for b in seen] == [False]
    page.locator("#defaults-confirm").get_by_role("button", name="Start").click()
    expect(page.get_by_text("Queued (job ff00aa11)")).to_be_visible()
    assert [b.get("confirm") for b in seen] == [False, True]
    assert (seen[1]["calls"], seen[1]["est_usd"]) == (7, 0.2)  # what was shown


def test_add_a_default_store_shows_a_hub_build_as_free(page, seeded):
    estimate = {"est_usd": 0, "items": 5, "calls": 0, "hub": {"entry": "x/y"}}
    route_load(page, "musique", lambda b: (200, {"estimate": estimate}))
    page.goto(seeded.url + "/")
    page.get_by_text("Try a built-in benchmark").click()
    page.get_by_role("button", name="Add musique").click()
    expect(page.locator("#defaults-confirm")).to_contain_text("Free")


def test_a_default_store_that_exists_is_opened_not_added_again(page, seeded):
    r = httpx.post(f"{seeded.url}/collections", json={"name": "scienceqa", "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/")
    page.get_by_text("Try a built-in benchmark").click()
    expect(page.locator("#defaults").get_by_role("link", name="Open scienceqa")).to_have_attribute(
        "href", "/c/scienceqa"
    )
    expect(page.get_by_role("button", name="Add scienceqa")).to_have_count(0)
    expect(page.get_by_role("button", name="Add scibench")).to_be_visible()


def test_new_store_form_says_which_model_it_embeds_with(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    expect(page.locator("#form")).to_contain_text(
        "Embedded with fake/dense (Settings → Search defaults)"
    )


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


def test_create_store_opens_its_page(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("opened")
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/c/opened")
    expect(page.locator('#side a[href="/c/opened"]')).to_be_visible()


def test_create_store_refused_stays_on_the_form(page, seeded):
    page.expected = ["409"]
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("c")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("#f-msg")).to_contain_text("already exists")
    expect(page).to_have_url(seeded.url + "/")
    expect(page.locator("#form")).to_be_visible()


def test_create_store_is_disabled_while_it_saves(page, seeded):
    held = []

    def hold(route):
        if route.request.method != "POST":
            return route.fallback()
        held.append(route)

    def next_held():
        for _ in range(100):
            if held:
                return held.pop()
            page.wait_for_timeout(20)
        raise AssertionError("the POST never reached the route")

    page.route("**/collections", hold)
    page.expected = ["409"]
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    page.get_by_label("Name").fill("slowmade")
    create = page.locator("#f-create")
    create.click()
    expect(create).to_be_disabled()
    next_held().fulfill(status=409, json={"detail": "a store named 'slowmade' already exists"})
    expect(page.locator("#f-msg")).to_have_text("a store named 'slowmade' already exists")
    expect(create).to_be_enabled()
    create.click()
    expect(create).to_be_disabled()
    next_held().continue_()
    expect(page).to_have_url(seeded.url + "/c/slowmade")


def test_create_works_again_after_back(page, seeded):
    page.expected = ["409"]  # each held POST is refused at the end, so the next page is clean
    held = []  # the POST never answers: the page stays as it was when it left
    page.route(
        "**/collections", lambda r: held.append(r) if r.request.method == "POST" else r.fallback()
    )
    page.route(
        "**/eval-sets", lambda r: held.append(r) if r.request.method == "POST" else r.fallback()
    )
    for path, button, name in (
        ("/", "+ New store", "backmade"),
        ("/eval-sets", "+ New eval set", "backset"),
    ):
        page.goto(seeded.url + path)
        page.get_by_role("button", name=button).click()
        page.get_by_label("Name").fill(name)
        create = page.locator("#f-create")
        create.click()
        expect(create).to_be_disabled()
        # Playwright's Chromium runs without the back/forward cache, so a restore is dispatched
        page.evaluate(
            "window.dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }))"
        )
        expect(create).to_be_enabled()
        for _ in range(100):
            if held:
                break
            page.wait_for_timeout(20)
        held.pop().fulfill(status=409, json={"detail": "held"})  # settled before the next page
        expect(page.locator("#f-msg")).to_have_text("held")
