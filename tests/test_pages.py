"""The console pages, driven in a real browser against the app on fakes."""

import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.pages


def test_collections_page_shows_the_answer(page, seeded):
    page.goto(seeded.url + "/")
    row = page.locator("main a.row-link", has_text="Keywords per list")
    expect(row).to_be_visible()
    expect(row).to_have_attribute("href", "/c/c")
    expect(row).to_contain_text("run " + seeded.run_id[:4])
    expect(row).to_contain_text("Ready")
    expect(page.get_by_text("Keywords per list").first).to_be_visible()


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


def test_lists_save_shows_rekey_count(page, seeded):
    page.goto(seeded.url + "/c/c")
    page.get_by_role("button", name="Save lists").click()
    expect(page.get_by_text("re-keys 3 items", exact=False)).to_be_visible()


def test_new_run_estimates_then_starts(page, seeded):
    page.goto(seeded.url + "/c/c")
    expect(page.get_by_text("Ready").first).to_be_visible(timeout=15000)
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
