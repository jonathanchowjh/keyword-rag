"""A store's page: its imports, sources, rebuild and keyword tags."""

import json
import re

import httpx
import pytest
from page_helpers import (
    REBUILD_MODEL,
    documents_run,
    documents_store,
    jobs_done,
    mcq_set,
    open_chat,
    route_load,
    search_defaults_restored,
    sources,
    tag_jobs,
    two_sections,
    upload_and_preview,
)
from playwright.sync_api import expect

from ragqa_core import db
from ragqa_core.fakes import fake_replies, until

pytestmark = pytest.mark.pages

ORIGINAL_TOPICS = "enzyme\nmarket\nvolcano\noptics"  # the topic list `seeded.build` saves


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
    expect(page.get_by_text("Saving re-keys 3 examples, free")).to_be_visible()
    page.locator("#kw-topic").fill("alpha\nbeta")
    page.get_by_role("button", name="Save lists").click()
    expect(page.get_by_text("re-keys 3 examples", exact=False).first).to_be_visible()
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


def test_delete_asks_inline_first(page, seeded):
    page.goto(seeded.url + "/c/c")
    page.get_by_role("button", name="Delete store…").click()
    expect(page.get_by_role("button", name="Yes, delete c")).to_be_visible()
    page.get_by_role("button", name="Cancel").click()
    expect(page.get_by_role("button", name="Yes, delete c")).to_have_count(0)


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
    assert (seen[1]["calls"], seen[1]["est_usd"]) == (51, 0.12)  # what was shown


def test_collection_load_that_is_free_queues_at_once(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "freeload", "qa_type": "mcq"})
    seen = route_load(page, "freeload", lambda b: (202, {"job_id": "ee99ff"}))
    page.goto(seeded.url + "/c/freeload")
    page.get_by_role("button", name="Load scienceqa").click()
    expect(page.get_by_text("Queued (job ee99ff)")).to_be_visible()
    assert len(seen) == 1


def test_science_collection_offers_scibench_and_runs(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "sci0", "qa_type": "science"})
    page.goto(seeded.url + "/c/sci0")
    expect(page.get_by_role("button", name="Load scibench")).to_be_visible()
    run_id = seeded.build("sci", run=True, qa_type="science")
    page.goto(f"{seeded.url}/c/sci/runs/{run_id}")
    page.get_by_text("Each method's own score").click()
    expect(page.get_by_text("Example tests the same skill").first).to_be_visible()


def test_finish_loading_uses_the_dataset_the_load_started_with(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "gq", "qa_type": "mcq"})
    job = [
        {
            "id": "dead0001",
            "kind": "load",
            "collection": "gq",
            "status": "cancelled",
            "setup": {"dataset": "gpqa"},
        }
    ]
    page.route("**/jobs?*kind=load*", lambda route: route.fulfill(json=job))
    seen = route_load(page, "gq", lambda b: (202, {"job_id": "ff00ee"}))
    page.goto(seeded.url + "/c/gq")
    page.get_by_role("button", name="Finish loading gpqa").click()
    expect(page.get_by_text("Queued (job ff00ee)")).to_be_visible()
    assert seen[0]["dataset"] == "gpqa"
    expect(page.locator("#ld-dataset")).to_be_disabled()  # an unfinished load keeps its dataset


def test_a_new_store_picks_among_its_types_datasets(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "pick", "qa_type": "science"})
    seen = route_load(page, "pick", lambda b: (202, {"job_id": "aa11bb"}))
    page.goto(seeded.url + "/c/pick")
    options = page.locator("#ld-dataset option")
    expect(options).to_have_text(["scibench", "bright_aops", "bright_theoremqa"])
    page.locator("#ld-dataset").select_option("bright_aops")
    page.get_by_role("button", name="Load bright_aops").click()
    expect(page.get_by_text("Queued (job aa11bb)")).to_be_visible()
    assert seen[0]["dataset"] == "bright_aops"


def test_a_store_named_after_a_dataset_offers_that_dataset(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "gpqa", "qa_type": "mcq"})
    try:
        page.goto(seeded.url + "/c/gpqa")
        expect(page.get_by_role("button", name="Load gpqa")).to_be_visible()
    finally:  # a dataset-named store would turn the Stores page's "Add gpqa" into "Open gpqa"
        httpx.delete(f"{seeded.url}/collections/gpqa")


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


def test_runs_table_says_what_each_run_said(page, seeded):
    page.goto(seeded.url + "/c/c")
    said = page.locator("[data-result]").first
    expect(said).to_have_text(re.compile(r"^(Yes|No|Can't tell yet) [+−]\d+ pts?"))
    expect(said.locator("strong.word")).to_have_count(1)
    page.goto(seeded.url + "/c/vo")
    expect(page.locator("[data-result]").first).to_have_text("–")


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
    expect(preview).to_have_text(re.compile(r"^Added 3 worked examples\. Next: "))
    jobs_done(seeded, "imp")
    page.reload()
    row = page.locator("#sources tbody tr", has_text="notes (uploaded)")
    expect(row).to_contain_text("3")
    box = row.get_by_role("checkbox", name="Use notes (uploaded) in retrieval")
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
    page.locator("#sources tbody tr", has_text="notes (uploaded)").get_by_role("button").click()
    expect(page.locator("#sources-confirm")).to_contain_text(
        "Delete source notes (uploaded): its 3 worked examples?"
    )
    page.get_by_role("button", name="Yes, delete notes (uploaded)").click()
    expect(page.locator("#sources tbody tr", has_text="notes (uploaded)")).to_have_count(0)


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
    expect(preview).to_have_text(re.compile(r"^Added 2 passages\. Next: "))
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
    expect(preview).to_have_text(re.compile(r"^Added 3 worked examples\. Next: "))
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


def test_store_import_redraws_a_price_that_changed(page, seeded, tmp_path):
    seeded.build("repriced")  # keyword lists made: the import is free, 0 AI calls
    notes = tmp_path / "notes.jsonl"
    notes.write_text(json.dumps({"id": "p1", "text": "a note about reefs"}))
    page.goto(f"{seeded.url}/c/repriced")
    upload_and_preview(page, "items", notes)
    button = page.locator("#imp-items-import")
    expect(button).to_be_enabled()
    confirms, previews = [], []
    changed = "the price changed since you saw it: check it again"

    def route(r):
        sent = json.loads(r.request.post_data)
        if not sent["confirm"]:
            return previews.append(r)  # held, to see Import wait for the fresh price
        confirms.append(sent)
        if len(confirms) == 1:
            return r.fulfill(status=409, json={"detail": changed})
        r.fallback()

    page.route("**/collections/repriced/imports", route)
    page.expected = ["409"]
    button.click()
    expect(page.locator("#imp-items-msg")).to_have_text(changed)
    assert (confirms[0]["calls"], confirms[0]["est_usd"]) == (0, 0)  # what was drawn
    for _ in range(100):
        if previews:
            break
        page.wait_for_timeout(20)
    assert len(previews) == 1, "the price was never asked for again"
    expect(button).to_be_disabled()  # until the fresh estimate is drawn
    previews[0].continue_()
    expect(button).to_be_enabled()
    expect(page.locator("#imp-items-msg")).to_have_text(changed)
    button.click()
    expect(page.locator("#imp-items-preview")).to_contain_text("Added 1 ")
    assert len(confirms) == 2 and len(import_jobs(seeded, "repriced")) == 1


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
    expect(preview).to_have_text(re.compile(r"^Added 2 passages\. Next: "))


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
        expect(preview).to_have_text(re.compile("^" + re.escape(said) + " Next: "))


def two_files(tmp_path):
    """a.md and b.md, two passages each."""
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_text(two_sections())
    b.write_text("# Coral\n\n" + "coral " * 160 + "\n\n# Delta\n\n" + "delta " * 160)
    return a, b


def test_import_several_files(page, seeded, tmp_path):
    documents_store(seeded, "severalmd", passages=False)
    a, b = two_files(tmp_path)
    queued = import_posts(page, "severalmd")
    page.goto(seeded.url + "/c/severalmd")
    file = page.locator("#imp-items-file")
    expect(file).to_have_attribute("multiple", "")
    file.set_input_files([str(a), str(b)])
    expect(page.locator("#imp-items-source")).to_be_hidden()  # each file names its own source
    page.get_by_role("button", name="Preview").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text(re.compile(r"2 files: 4 passages to add · at most (\$|<)"))
    expect(preview).to_contain_text("a.md: 2 passages")
    expect(preview).to_contain_text("b.md: 2 passages")
    button = page.locator("#imp-items-import")
    expect(button).to_have_count(1)
    expect(button).to_have_text(re.compile(r"^Import \(at most (\$|<).*\)$"))
    button.click()
    expect(preview).to_have_text(re.compile(r"^Added 4 passages\. Next: "))
    first, second = (json.loads(r.post_data) for r in queued)
    assert (first["source_name"], second["source_name"]) == ("a", "b")
    assert "calls" in first and "est_usd" in first
    assert "calls" not in second and "est_usd" in second  # a price that rose is still refused
    assert {"upload:a", "upload:b"} <= {s["source"] for s in sources(seeded, "severalmd")}


def test_import_several_files_one_refused(page, seeded, tmp_path):
    documents_store(seeded, "manyref", passages=False)
    a, _ = two_files(tmp_path)
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(b"%PDF-1.4 not text")
    queued = import_posts(page, "manyref")
    page.goto(seeded.url + "/c/manyref")
    page.expected = ["422"]  # the .pdf is refused, which is the point
    page.locator("#imp-items-file").set_input_files([str(a), str(pdf)])
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("c.pdf: A Documents store takes .md and .txt files")
    expect(preview).to_contain_text("a.md: 2 passages")
    expect(preview).to_contain_text(re.compile(r"1 file: 2 passages to add · (\$|<)"))
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text(re.compile(r"^Added 2 passages\. Next: "))
    assert [json.loads(r.post_data)["source_name"] for r in queued] == ["a"]


def test_import_several_files_price_rise_stops_there(page, seeded, tmp_path):
    documents_store(seeded, "rise", passages=False)
    a, b = two_files(tmp_path)
    confirms = []
    changed = "the price changed since you saw it: check it again"

    def route(r):
        sent = json.loads(r.request.post_data)
        if sent["confirm"]:
            confirms.append(sent)
            if sent["source_name"] == "b" and len(confirms) == 2:
                return r.fulfill(status=409, json={"detail": changed})
        r.fallback()

    page.route("**/collections/rise/imports", route)
    page.goto(seeded.url + "/c/rise")
    page.expected = ["409"]
    page.locator("#imp-items-file").set_input_files([str(a), str(b)])
    page.locator("#imp-items-go").click()
    button = page.locator("#imp-items-import")
    expect(button).to_be_enabled()
    button.click()
    expect(page.locator("#imp-items-msg")).to_have_text("The price of b.md changed: check it again")
    preview = page.locator("#imp-items-preview")
    # a.md is in; b.md is priced again and left for the next click
    expect(preview).to_contain_text(re.compile(r"1 file: 2 passages to add · (\$|<)"))
    expect(preview).not_to_contain_text("a.md")
    expect(button).to_be_enabled()
    assert len(import_jobs(seeded, "rise")) == 1
    # the store page shows a.md without a reload
    expect(page.locator("#sources tbody tr", has_text="a (uploaded)")).to_be_visible()
    button.click()
    expect(preview).to_have_text(re.compile(r"^Added 2 passages\. Next: "))
    assert [c["source_name"] for c in confirms] == ["a", "b", "b"]
    assert "calls" in confirms[2]  # now the first file of its own import
    assert len(import_jobs(seeded, "rise")) == 2


def test_import_several_files_one_unpriced(page, seeded, tmp_path):
    documents_store(seeded, "manyunp", passages=False)
    a, b = two_files(tmp_path)

    def unpriced(route):
        body = route.fetch().json()
        if body["filename"] == "b.md":
            body["estimate"] = {**body["estimate"], "est_usd": None, "calls": 3}
        route.fulfill(json=body)

    page.route("**/collections/manyunp/uploads?*", unpriced)
    queued = import_posts(page, "manyunp")
    page.goto(seeded.url + "/c/manyunp")
    page.locator("#imp-items-file").set_input_files([str(a), str(b)])
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text("Can't price b.md: set up a model in Settings")
    expect(page.locator("#imp-items-import")).to_be_disabled()
    assert queued == []


def test_several_files_without_ai_disable_import(page, seeded, tmp_path):
    documents_store(seeded, "manynoai", passages=False)
    a, b = two_files(tmp_path)
    pdf = tmp_path / "c.pdf"
    pdf.write_bytes(b"%PDF-1.4 not text")

    def missing(route):
        response = route.fetch()
        body = response.json()
        if response.ok:
            body["estimate"] = {**body["estimate"], "missing": ["summary"]}
        route.fulfill(response=response, json=body)

    page.route("**/collections/manynoai/uploads?*", missing)
    page.goto(seeded.url + "/c/manynoai")
    page.expected = ["422"]  # the .pdf is refused
    page.locator("#imp-items-file").set_input_files([str(a), str(b), str(pdf)])
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview.get_by_role("link", name="Connect AI first")).to_have_attribute(
        "href", "/settings"
    )
    expect(page.locator("#imp-items-import")).to_be_disabled()
    expect(preview.locator("li")).to_have_count(3)


def test_mcq_import_takes_one_file(page, seeded):
    httpx.post(f"{seeded.url}/collections", json={"name": "onefile", "qa_type": "mcq"})
    page.goto(seeded.url + "/c/onefile")
    file = page.locator("#imp-items-file")
    expect(file).to_be_visible()
    expect(file).not_to_have_attribute("multiple", "")
    expect(page.locator("#imp-items-source")).to_be_visible()
    expect(page.locator("#imp-items-go")).to_have_text("Preview")


def test_own_science_collection_offers_import_and_benchmark(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "own", "qa_type": "science"})
    page.goto(seeded.url + "/c/own")
    expect(page.locator("#imp-items-file")).to_be_visible()
    expect(page.locator("#imp-questions-file")).to_have_count(0)
    expect(page.get_by_text("Or load a built-in benchmark.")).to_be_visible()
    expect(page.get_by_role("button", name="Load scibench")).to_be_visible()
    notes = tmp_path / "own notes!.csv"
    notes.write_text("text\nfirst thing about reefs\nsecond thing about tides\n")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    expect(page.locator("#imp-items-source")).to_have_value("own notes")
    # no keyword lists yet, so the import pays to make them: its price sits beside Import,
    # and the one click on Import queues it
    page.locator("#imp-items-import").click()
    expect(page.locator("#imp-items-preview")).to_have_text(
        re.compile(r"^Added 2 worked examples\. Next: ")
    )
    jobs_done(seeded, "own")
    page.reload()
    expect(page.locator("#imp-items-file")).to_be_visible()
    expect(page.get_by_text("built-in benchmark")).to_have_count(0)
    expect(page.get_by_role("button", name="Load scibench")).to_have_count(0)


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


def test_an_empty_documents_store_offers_no_benchmark(page, seeded):
    r = httpx.post(f"{seeded.url}/collections", json={"name": "docs0", "qa_type": "documents"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/c/docs0")
    expect(page.get_by_text("0 passages.", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name=re.compile(r"^(Load|Finish loading)"))).to_have_count(0)
    expect(page.get_by_text("built-in benchmark")).to_have_count(0)
    assert "undefined" not in page.locator("main").inner_text()


NOT_BUILT = (
    "Not built. Keyword tags let Keyword-RAG search this store: an AI proposes a vocabulary "
    "from your passages, then tags each passage by similarity."
)


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
    expect(preview).to_have_text(re.compile(r"^Added 2 passages\. Next: "))
    jobs_done(seeded, "dmd")
    assert httpx.get(f"{seeded.url}/collections/dmd").json()["count"] == 2
    httpx.post(
        f"{seeded.url}/eval-sets", json={"name": "dmd", "qa_type": "documents"}
    )  # New run waits for a set
    page.reload()
    expect(page.get_by_text("2 passages.", exact=True)).to_be_visible()
    expect(page.locator("#head .pill")).to_have_text("Ready")
    expect(page.get_by_text("Or load a built-in benchmark.")).to_have_count(0)
    expect(page.get_by_text("How are items described?")).to_be_hidden()
    expect(page.locator("#kw-topic")).to_be_hidden()
    expect(page.get_by_role("button", name="Propose with AI")).to_be_hidden()
    page.get_by_role("button", name="New run").click()
    expect(page.locator("input[name=method][value=hybrid]")).to_be_checked()
    expect(page.locator("#rf-answer")).not_to_be_checked()
    expect(page.locator("input[name=method][value=kw_individual]")).to_have_count(0)


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
        until(lambda: tag_jobs(seeded, "dtag"), 10, 0.05, "the tag job being queued")
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
    assert resumes == [{"confirm": False}, {"confirm": True, "calls": 3, "est_usd": 0.02}]


def test_keyword_tags_message_goes_when_the_state_changes(page, seeded):
    documents_store(seeded, "dmsg")
    shown = {"tags": {"state": "none"}}
    route_keyword_tags(page, "dmsg", shown)
    refusal = {"detail": "connect a model for keywords first"}
    page.route("**/collections/dmsg/keyword-tags", lambda r: r.fulfill(status=503, json=refusal))
    page.expected = ["503"]  # the refusal is the point
    page.clock.install()
    page.goto(seeded.url + "/c/dmsg")
    page.get_by_role("button", name="Add keyword tags").click()
    msg = page.locator("#tags-msg")
    expect(msg).to_have_text(refusal["detail"])
    shown["tags"] = {"state": "building", "job_id": "tagjob03", "progress": 0}
    page.clock.run_for(16000)  # the next poll repaints the section into the new state
    expect(page.locator("#tags")).to_contain_text("Adding keyword tags")
    expect(msg).to_have_text("")


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
    with page.expect_response("**/collections/rb/reembed") as r:
        page.get_by_role("button", name="Yes, rebuild").click()
    job = r.value.json()["job_id"]
    until(  # the rebuild runs on fakes: moments
        lambda: httpx.get(f"{seeded.url}/jobs/{job}").json()["status"] not in ("queued", "running"),
        30,
        0.05,
        "the rebuild finishing",
    )
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
    current = {"rebuild": rebuild, "state": "rebuilding", "busy": True}

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
    box.get_by_role("button", name="Stop job").click()
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
    expect(box.get_by_role("button", name="Stop job")).to_have_count(0)


def test_store_page_polls_slowly_on_a_stopped_switch(page, seeded):
    seeded.build("rbs")
    job = {"id": "job-1", "kind": "reembed", "status": "interrupted", "progress": 0.4}
    stopped = {
        "rebuild": {
            "job_id": "job-1",
            "status": "interrupted",
            "model": REBUILD_MODEL,
            "from": "fake/dense",
            "progress": 0.4,
            "switching": True,
        },
        "state": "rebuilding",
        "busy": True,
        "job": job,
    }
    gets = []

    def store(route):
        gets.append(1)
        route.fulfill(json={**route.fetch().json(), **stopped})

    page.route("**/collections/rbs", store)
    page.clock.install()
    page.goto(seeded.url + "/c/rbs")
    expect(page.locator("#embedding")).to_contain_text("stopped part-way")
    first = len(gets)
    page.clock.run_for(4_000)
    page.wait_for_timeout(500)
    assert len(gets) == first  # not the 3 s poll of a job in flight
    page.clock.run_for(12_000)
    page.wait_for_timeout(500)
    assert len(gets) == first + 1  # the 15 s poll
    expect(page.get_by_role("button", name="Rebuild", exact=True)).to_be_disabled()
    expect(page.locator("#sources input[data-source]").first).to_be_disabled()


def test_documents_store_shows_its_chat_search_score(page, seeded):
    documents_run(seeded, "dd", ["hybrid_rerank", "vector"])
    page.goto(seeded.url + "/")
    expect(page.locator(".th")).to_contain_text("Result")
    expect(page.locator('a.row-link[href="/c/dd"]')).to_contain_text(
        "for Hybrid + reranker, the search your chats use"
    )
    page.goto(seeded.url + "/c/dd")
    expect(page.locator("#answer-title")).to_have_text(
        "How well does chat find the right passages?"
    )
    expect(page.get_by_role("heading", name="What's the answer?")).to_have_count(0)
    expect(page.locator("#answer")).to_contain_text(
        "Chats can choose another method; this scores the store's setting."
    )
    expect(page.locator("#answer [data-gap-bar], #answer svg")).to_have_count(0)
    expect(page.locator("body")).not_to_contain_text("Keywords per list")


def test_documents_score_copy_shows_headline_range_and_n(page, seeded):
    run_id, _ = documents_run(seeded, "dd9", ["hybrid_rerank", "vector"])
    conn = seeded.app.state.conn
    row = db.run_items(conn, run_id, "hybrid_rerank")[0]  # one of two questions scores 50
    fields = {k: v for k, v in row.items() if k not in ("method", "question_id", "complete")}
    fields["retrieval"] = {**row["retrieval"], "ndcg@10": 0.5}
    db.upsert_run_item(conn, run_id, "hybrid_rerank", row["question_id"], fields, True)
    line = (
        "Ranking quality nDCG@10: 75% (50–100) for Hybrid + reranker, "
        "the search your chats use, on 2 questions"
    )
    page.goto(seeded.url + "/c/dd9")
    expect(page.locator("#answer")).to_contain_text(line)
    expect(page.locator("#answer code.muted")).to_have_text("nDCG@10")
    page.goto(f"{seeded.url}/c/dd9/runs/{run_id}")
    expect(page.locator("#answer")).to_contain_text(line)


def test_documents_unjudged_says_label_more(page, seeded):
    run_id, ids = documents_run(seeded, "dd3", ["hybrid_rerank", "vector"])
    conn = seeded.app.state.conn
    for r in db.run_items(conn, run_id, "hybrid_rerank"):
        row = {k: v for k, v in r.items() if k not in ("method", "question_id", "complete")}
        row["hit_ids"] = ids + [f"dd3-never-{i}" for i in range(7)]
        db.upsert_run_item(conn, run_id, "hybrid_rerank", r["question_id"], row, True)
    page.goto(seeded.url + "/c/dd3")
    expect(page.locator("#answer")).to_contain_text(
        "No score yet: most of the top 10 from Hybrid + reranker were never judged. "
        "Label more of its passages."
    )
    expect(page.locator("#answer")).not_to_contain_text("nDCG")


def test_documents_score_few_questions_says_label_more(page, seeded):
    documents_run(seeded, "dd4", ["hybrid_rerank", "vector"])
    page.goto(seeded.url + "/c/dd4")
    expect(page.locator("#answer")).to_contain_text(
        "only 2 questions: label more for a firmer score"
    )


def test_documents_no_score_names_the_method(page, seeded):
    documents_run(seeded, "dd6", ["hybrid"])
    r = httpx.put(f"{seeded.url}/search-defaults", json={"method": "vector"})
    assert r.status_code == 200, r.text
    try:
        page.goto(seeded.url + "/c/dd6")
        expect(page.locator("#answer")).to_contain_text(
            "No score yet: run Vector search to get one"
        )
        page.goto(seeded.url + "/")
        expect(page.locator('a.row-link[href="/c/dd6"]')).to_contain_text(
            "No score yet: run Vector search to get one"
        )
    finally:
        httpx.put(f"{seeded.url}/search-defaults", json={"method": "hybrid"})


def test_store_import_says_which_old_passages_it_will_remove(page, seeded, tmp_path):
    documents_store(seeded, "oldpass", passages=False)
    handbook = tmp_path / "x.md"
    handbook.write_text(two_sections())
    page.goto(seeded.url + "/c/oldpass")
    preview = page.locator("#imp-items-preview")
    upload_and_preview(page, "items", handbook)
    expect(page.locator("#imp-items-import")).to_be_enabled()
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text(re.compile(r"^Added 2 passages\. Next: "))
    handbook.write_text(two_sections().replace("brine " * 160, "brine " * 159 + "cocoa"))
    upload_and_preview(page, "items", handbook)
    expect(page.locator("[data-removed]")).to_have_text(
        "1 old passage will be removed, since the text changed."
    )
    page.locator("#imp-items-import").click()
    expect(preview).to_have_text(
        re.compile(r"^Added 1 passage\. 1 already in the store\. 1 old passage removed\. Next: ")
    )
    upload_and_preview(page, "items", handbook)
    expect(page.locator("#imp-items-import")).to_be_enabled()
    expect(page.locator("[data-removed]")).to_have_count(0)


def test_documents_nouns_are_passages(page, seeded):
    documents_store(seeded, "pass")
    page.goto(seeded.url + "/c/pass")
    expect(page.locator("#sources thead")).to_contain_text("Passages")
    expect(page.locator("#sources thead")).not_to_contain_text("Items")
    open_chat(page, seeded, ("pass",))
    expect(page.locator("#chips .chip")).to_contain_text("· 2 passages")


def test_sources_show_upload_label(page, seeded):
    documents_store(seeded, "lbl")
    page.goto(seeded.url + "/c/lbl")
    expect(page.locator("#sources tbody tr").first).to_contain_text("x (uploaded)")
    got = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
          return [m.sourceLabel('upload:leave'), m.sourceLabel('hand'), m.sourceLabel('mmlu')]; }"""
    )
    assert got == ["leave (uploaded)", "added by hand", "mmlu"]


def top_of(page, selector):
    return page.locator(selector).bounding_box()["y"]


def test_empty_store_shows_import_first(page, seeded):
    documents_store(seeded, "emptyfirst", passages=False)
    page.goto(seeded.url + "/c/emptyfirst")
    expect(page.locator("#imp-items-file")).to_be_visible()
    assert top_of(page, "#imp-title") < top_of(page, "#answer")
    assert top_of(page, "#imp-title") < top_of(page, "#new-run")
    page.goto(seeded.url + "/c/c")  # a store with items keeps the order
    expect(page.locator("#new-run")).to_be_visible()
    assert top_of(page, "#answer") < top_of(page, "#imp-title")


def test_import_result_links_next_steps(page, seeded, tmp_path):
    documents_store(seeded, "nextsteps", passages=False)
    handbook = tmp_path / "handbook.md"
    handbook.write_text(two_sections())
    page.goto(seeded.url + "/c/nextsteps")
    upload_and_preview(page, "items", handbook)
    page.locator("#imp-items-import").click()
    preview = page.locator("#imp-items-preview")
    expect(preview).to_contain_text(
        "Added 2 passages. Next: Label questions · Chat with this store"
    )
    expect(preview.get_by_role("link", name="Label questions")).to_have_attribute(
        "href", "/eval-sets?store=nextsteps"
    )
    httpx.post(
        f"{seeded.url}/eval-sets",
        json={"name": "nextsteps", "qa_type": "documents", "store": "nextsteps"},
    )
    a, b = two_files(tmp_path)
    page.goto(seeded.url + "/c/nextsteps")
    page.locator("#imp-items-file").set_input_files([str(a), str(b)])
    page.get_by_role("button", name="Preview").click()
    page.locator("#imp-items-import").click()
    expect(preview).to_contain_text("Next: Label questions · Chat with this store")
    expect(preview.get_by_role("link", name="Label questions")).to_have_attribute(
        "href", "/e/nextsteps"
    )
    preview.get_by_role("link", name="Chat with this store").click()
    expect(page).to_have_url(re.compile(r"/chat#"))


def test_import_many_clears_the_message_and_stops_on_a_new_choice(page, seeded, tmp_path):
    documents_store(seeded, "manymsg", passages=False)
    a, b = two_files(tmp_path)
    confirms, held = [], []
    changed = "the price changed since you saw it: check it again"

    def route(r):
        sent = json.loads(r.request.post_data)
        if sent["confirm"]:
            confirms.append(sent)
            if len(confirms) == 2:
                return r.fulfill(status=409, json={"detail": changed})
            if len(confirms) == 3:
                return held.append(r)
        r.fallback()

    page.route("**/collections/manymsg/imports", route)
    page.goto(seeded.url + "/c/manymsg")
    page.expected = ["409"]
    page.locator("#imp-items-file").set_input_files([str(a), str(b)])
    page.locator("#imp-items-go").click()
    button = page.locator("#imp-items-import")
    expect(button).to_be_enabled()
    button.click()
    msg = page.locator("#imp-items-msg")
    expect(msg).to_have_text("The price of b.md changed: check it again")
    expect(button).to_be_enabled()
    button.click()  # b alone now; its message from the last round is gone as it is sent
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, "the second Import was never sent"
    expect(msg).to_have_text("")
    held[0].fallback()
    expect(page.locator("#imp-items-preview")).to_have_text(
        re.compile(r"^Added 2 passages\. Next: ")
    )

    # a new file choice mid-import: the files not yet sent are never sent
    documents_store(seeded, "manystop", passages=False)
    confirms.clear()
    held.clear()

    def hold_first(r):
        if json.loads(r.request.post_data)["confirm"]:
            confirms.append(r)
            if len(confirms) == 1:
                return held.append(r)
        r.fallback()

    page.route("**/collections/manystop/imports", hold_first)
    page.goto(seeded.url + "/c/manystop")
    page.locator("#imp-items-file").set_input_files([str(a), str(b)])
    page.locator("#imp-items-go").click()
    expect(page.locator("#imp-items-import")).to_be_enabled()
    page.locator("#imp-items-import").click()
    for _ in range(100):
        if held:
            break
        page.wait_for_timeout(20)
    assert held, "the first file was never sent"
    page.locator("#imp-items-file").set_input_files([str(b)])
    held[0].fallback()
    # the first file is in and the store page shows it: the loop reached its stop, with b unsent
    expect(page.locator("#sources tbody tr", has_text="a (uploaded)")).to_be_visible()
    assert len(confirms) == 1


def test_run_form_drops_its_start_after_a_price_change(page, seeded):
    seeded.build("stalerf", run=True)

    def refuse(route):
        detail = "the price changed since you saw it: check it again"
        if route.request.method == "POST" and route.request.post_data_json.get("confirm"):
            return route.fulfill(status=409, json={"detail": detail})
        route.fallback()

    page.expected = ["409"]
    page.route("**/eval", refuse)
    page.goto(seeded.url + "/c/stalerf")
    page.get_by_role("button", name="New run").click()
    page.get_by_role("button", name="Estimate").click()
    page.locator("#rf-confirm button").click()
    expect(page.locator("#rf-confirm button")).to_have_count(0)
    expect(page.locator("#rf-msg")).to_contain_text("check it again")
    page.unroute("**/eval")
    page.get_by_role("button", name="Estimate").click()
    expect(page.locator("#rf-confirm button")).to_be_visible()
    expect(page.locator("#rf-msg")).to_have_text("")


def test_an_empty_documents_store_points_at_import_not_a_run(page, seeded):
    documents_store(seeded, "emptyd", passages=False)
    page.goto(seeded.url + "/c/emptyd")
    expect(page.locator("#answer")).to_contain_text("No score yet: add your documents to start")
    tags = page.get_by_role("button", name="Add keyword tags")
    expect(tags).to_be_disabled()
    expect(
        page.get_by_text("Add passages first: an empty store has nothing to tag.")
    ).to_be_visible()


def test_a_documents_store_without_an_eval_set_points_at_labelling(page, seeded):
    documents_store(seeded, "nolab")
    page.goto(seeded.url + "/c/nolab")
    expect(page.locator("#answer")).to_contain_text(
        "No score yet: label a few questions to get one"
    )


def test_add_documents_with_a_missing_model_offers_connect_ai(page, seeded, tmp_path):
    seeded.build("nomodel")

    def missing(route):
        body = route.fetch().json()
        body["estimate"] = {**body["estimate"], "missing": ["summary"]}
        route.fulfill(json=body)

    page.route("**/collections/nomodel/uploads?*", missing)
    notes = tmp_path / "notes.jsonl"
    notes.write_text(json.dumps({"id": "u1", "text": "a note about reefs"}))
    page.goto(seeded.url + "/c/nomodel")
    page.locator("#imp-items-file").set_input_files(str(notes))
    page.locator("#imp-items-go").click()
    preview = page.locator("#imp-items-preview")
    expect(preview.get_by_role("link", name="Connect AI first")).to_have_attribute(
        "href", "/settings"
    )
    expect(page.locator("#imp-items-import")).to_be_disabled()
