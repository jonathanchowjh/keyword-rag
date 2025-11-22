"""The console pages, driven in a real browser against the app on fakes."""

import httpx
import pytest
from playwright.sync_api import expect

from ragqa_core.fakes import fake_replies

pytestmark = pytest.mark.pages


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
