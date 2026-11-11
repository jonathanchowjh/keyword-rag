"""What every page shares: sidebar, theme, polling, messages, the Name field, the curl Copy."""

import re

import httpx
import pytest
from page_helpers import (
    chat_ask,
    chat_message,
    chat_route,
    labelling_set,
    mcq_set,
    open_chat,
    worker_stopped,
    your_app,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.pages


def test_gap_bar_is_accessible_and_clamped(page, seeded):
    page.goto(seeded.url + "/")
    svg = page.evaluate("import('/static/ui.js').then(m => m.gapBar(-30, 50, 80, 10))")
    assert 'role="img"' in svg and 'aria-label="Gap +50 pts (−30 to 80)"' in svg
    cx = float(svg.split('<circle cx="')[1].split('"')[0])
    assert cx <= 220  # clamped to the axis, not drawn at 50/10 of the way out


def test_pts_and_clock_format(page, seeded):
    page.goto(seeded.url + "/")
    out = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
      return [m.pts(6.2, 1.8, 10.4), m.pts(-7, -12, -2),
              m.clock('2026-10-21T06:02:00Z', new Date('2026-10-23T00:00:00Z'))] }"""
    )
    assert out[0] == "+6 pts (2–10)" and out[1] == "−7 pts (−12 to −2)" and "21 Oct" in out[2]


def test_confirm_spend_prints_sub_cent_and_zero(page, seeded):
    page.goto(seeded.url + "/")
    out = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
      return [0.004, 0, 0.41].map((usd) => { const el = document.createElement('div');
        m.confirmSpend(el, { est_usd: usd, calls: 2 }, () => {}); return el.textContent; }) }"""
    )
    assert "< $0.01" in out[0] and "≈ $0.00" in out[1] and "< $0.01" not in out[1]
    assert "$0.41" in out[2]


def test_keyword_rag_costs_money_only_when_the_ai_picks_its_tags(page, seeded):
    page.goto(seeded.url + "/")
    marks = page.evaluate(
        """async () => { const m = await import('/static/ui.js');
        const cat = await (await fetch('/catalog')).json();
        return ['model', 'classifier'].map(t =>
          m.costMark('keyword_rag', {summarize: false, search: {query_tags: t}}, cat)); }"""
    )
    assert "costs money" in marks[0] and marks[1] == ""


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


def test_buttons_use_one_wording(page, seeded):
    old = ["Keep it", "Keep editing", "Delete…", "New run", "New chat"]

    def no_old_wording():
        names = [b.inner_text().strip() for b in page.get_by_role("button").all()]
        for word in old:
            assert word not in names, f"{page.url}: a button still reads {word!r}"

    seeded.build("bw")
    page.goto(seeded.url + "/c/bw")
    expect(page.get_by_role("button", name="+ New run")).to_be_visible()
    expect(page.get_by_role("button", name="Delete store…")).to_be_visible()
    sources = page.locator("#sources tbody tr button")
    expect(sources.first).to_be_visible()
    assert {t.strip() for t in sources.all_inner_texts()} == {"Delete source…"}
    no_old_wording()
    page.goto(seeded.url + "/e/bw")  # the draft rows of a set are removed unsaved: no confirm
    no_old_wording()
    labelling_set(seeded, "bwd")
    q = {"text": "how long does alpha last?", "gold_ids": []}
    made = httpx.post(f"{seeded.url}/eval-sets/bwdq/questions", json={"question": q})
    assert made.status_code == 201, made.text
    page.goto(seeded.url + "/e/bwdq")
    saved = page.locator("main button[data-del]")
    expect(saved.first).to_be_visible()
    assert {t.strip() for t in saved.all_inner_texts()} == {"Delete question…"}
    no_old_wording()
    page.goto(seeded.url + "/")
    expect(page.get_by_role("button", name="+ New store")).to_be_visible()
    no_old_wording()
    page.goto(seeded.url + "/eval-sets")
    expect(page.get_by_role("button", name="+ New eval set")).to_be_visible()
    no_old_wording()
    page.goto(seeded.url + "/chat")
    expect(page.get_by_role("button", name="+ New chat")).to_be_visible()
    no_old_wording()
    done = seeded.build("bwr", run=True)
    page.goto(f"{seeded.url}/c/bwr/runs/{done}")
    expect(page.get_by_role("button", name="Delete run…")).to_be_visible()
    no_old_wording()
    page.goto(seeded.url + "/chat")
    with worker_stopped(seeded, "bwq") as run_id:
        page.goto(f"{seeded.url}/c/bwq/runs/{run_id}")
        expect(page.get_by_role("button", name="Stop run")).to_be_visible()
        expect(page.get_by_role("button", name="Cancel")).to_have_count(0)
        no_old_wording()
        page.goto(seeded.url + "/worker")
        expect(page.locator("[data-queued-job] button")).to_have_text("Stop job")
        expect(page.locator("[data-queued-job] button", has_text="Cancel")).to_have_count(0)


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


def test_nav_lists_eval_sets(page, seeded):
    mcq_set(seeded, "navset", 1)
    page.goto(seeded.url + "/e/navset")
    expect(page.locator('#side a.sub[href="/e/navset"]')).to_have_text("navset")
    expect(page.locator('#side a.sub[href="/e/c"]')).to_have_text("c")  # seeded; the store c too
    expect(page.locator('#side a.sub[href="/e/navset"]')).to_have_class("sub active")
    expect(page.locator('#side a.sub[href="/e/c"]')).not_to_have_class(re.compile(r"\bactive\b"))
    expect(page.locator('#side a.sub[href="/c/c"]')).to_have_class("sub")
    expect(page.locator('#side a[href="/eval-sets"]')).not_to_have_class(re.compile(r"\bactive\b"))
    hrefs = [a.get_attribute("href") for a in page.locator("#side a").all()]
    # c is both a store and an eval set: each link sits under its own heading
    under_stores = hrefs[hrefs.index("/") + 1 : hrefs.index("/chat")]
    under_eval_sets = hrefs[hrefs.index("/eval-sets") + 1 : hrefs.index("/lab")]
    assert "/c/c" in under_stores and "/e/c" not in under_stores
    assert "/e/c" in under_eval_sets and "/c/c" not in under_eval_sets
    assert "/e/navset" in under_eval_sets and "/e/navset" not in under_stores
    page.goto(seeded.url + "/eval-sets")
    expect(page.locator('#side a[href="/eval-sets"]')).to_have_class(re.compile(r"\bactive\b"))


LOGIN_LINE = "With a login set, add -u user:password: the RAGQA_USERNAME and RAGQA_PASSWORD"


def test_curl_box_copy_and_login_line(page, seeded):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_chat(page, seeded, ("c",))
    pres = your_app(page)
    details = page.locator("#chat-head details")
    expect(details).to_contain_text(LOGIN_LINE + " in the server's .env.")
    expect(details).not_to_contain_text("add -u user:password. ")
    buttons = details.locator("button[data-copy]")  # its name changes while it says Copied
    expect(buttons).to_have_count(2)
    expect(buttons).to_have_text(["Copy", "Copy"])
    buttons.nth(1).click()
    expect(buttons.nth(1)).to_have_text("Copied")
    assert page.evaluate("navigator.clipboard.readText()") == pres.nth(1).inner_text()
    expect(buttons.nth(1)).to_have_text("Copy", timeout=4000)


def test_curl_box_copy_falls_back_to_selecting(page, seeded):
    page.add_init_script("Object.defineProperty(navigator, 'clipboard', {value: undefined})")
    open_chat(page, seeded, ("c",))
    pres = your_app(page)
    button = page.locator("#chat-head details button[data-copy]").first
    button.click()
    expect(button).to_have_text("Press Ctrl+C to copy")
    assert page.evaluate("getSelection().toString()") == pres.first.inner_text()


def test_name_hint_suggests_slug(page, seeded):
    page.goto(seeded.url + "/")
    page.get_by_role("button", name="+ New store").click()
    expect(page.get_by_text("Letters, digits, - or _; no spaces")).to_be_visible()
    expect(page.get_by_role("button", name="Use team-handbook")).to_have_count(0)
    page.get_by_label("Name").fill("Team Handbook")
    page.get_by_role("button", name="Use team-handbook").click()
    expect(page.locator("#f-name")).to_have_value("team-handbook")
    expect(page.get_by_role("button", name=re.compile("^Use "))).to_have_count(0)
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/c/team-handbook")


def test_eval_set_name_hint_suggests_slug(page, seeded):
    page.goto(seeded.url + "/eval-sets")
    page.get_by_role("button", name="+ New eval set").click()
    page.get_by_label("Name").fill("My Set")
    page.get_by_role("button", name="Use my-set").click()
    expect(page.locator("#f-name")).to_have_value("my-set")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("main a.row-link", has_text="my-set")).to_be_visible()


def test_page_slug_matches_the_server_slug(page, seeded):
    from ragqa_core.types import slug

    raws = ["Team Handbook", "Café notes", "***", "  a__b  ", "x" * 80, "Ünï cødé"]
    page.goto(seeded.url + "/")
    got = page.evaluate(
        "async (raws) => { const m = await import('/static/ui.js'); return raws.map(m.slug); }",
        raws,
    )
    assert got == [slug(r) for r in raws]


def test_a_poll_keeps_focus_inside_the_import_box(page, seeded):
    page.clock.install()
    page.goto(seeded.url + "/c/c")  # a store with items: Import stays after them
    file = page.locator("#imp-items-file")
    expect(file).to_be_visible()
    file.focus()
    page.evaluate(
        """() => {
            window.moved = 0; // times the import section was taken out of the page
            new MutationObserver((list) => {
                for (const m of list) {
                    for (const n of m.removedNodes) if (n.id === "sec-import") window.moved++;
                }
            }).observe(document.getElementById("main"), { childList: true });
        }"""
    )
    with page.expect_response(lambda r: r.url.endswith("/search-defaults")):
        page.clock.run_for(16000)  # the next poll repaints the page
    page.evaluate("() => new Promise((done) => setTimeout(done, 0))")  # let the repaint finish
    assert page.evaluate("window.moved") == 0
    expect(file).to_be_focused()


def test_theme_button_cycles_and_remembers(page, seeded):
    page.goto(seeded.url + "/settings")
    html = page.locator("html")
    button = page.locator("#theme")
    try:
        expect(button).to_have_text("Theme: System")
        expect(html).not_to_have_attribute("data-theme", re.compile(".*"))
        for label, value in (("Light", "light"), ("Dark", "dark")):
            button.click()
            expect(button).to_have_text(f"Theme: {label}")
            expect(html).to_have_attribute("data-theme", value)
            expect(button).to_be_focused()  # the redraw keeps the keyboard on the button
        page.keyboard.press("Enter")  # so a keyboard user can keep cycling
        expect(button).to_have_text("Theme: System")
        button.click()
        button.click()  # back to Dark for the next page
        page.goto(seeded.url + "/chat")
        expect(html).to_have_attribute("data-theme", "dark")
        expect(page.locator("#theme")).to_have_text("Theme: Dark")
        page.locator("#theme").click()
        expect(page.locator("#theme")).to_have_text("Theme: System")
        expect(html).not_to_have_attribute("data-theme", re.compile(".*"))
    finally:
        page.evaluate("localStorage.removeItem('theme')")


def test_dark_choice_beats_a_light_system(page, seeded):
    bg = "getComputedStyle(document.documentElement).getPropertyValue('--bg').trim()"
    try:
        for system, choice, expected in (("light", "dark", "#141416"), ("dark", "light", "#fff")):
            page.emulate_media(color_scheme=system)
            page.goto(seeded.url + "/settings")
            page.evaluate(f"localStorage.setItem('theme', '{choice}')")
            page.reload()
            assert page.evaluate(bg) == expected
    finally:
        page.evaluate("localStorage.removeItem('theme')")


def test_poll_pauses_while_hidden_and_refreshes_on_return(page, seeded):
    gets = []

    def store_get(r):
        if r.method == "GET" and r.url.endswith("/collections/ph"):
            gets.append(r.url)

    def show(hidden):
        state = "hidden" if hidden else "visible"
        page.evaluate(
            """([hidden, state]) => {
              const own = (key, value) =>
                Object.defineProperty(document, key, { configurable: true, get: () => value });
              own("hidden", hidden);
              own("visibilityState", state);
              document.dispatchEvent(new Event("visibilitychange"));
            }""",
            [hidden, state],
        )

    with worker_stopped(seeded, "ph"):
        assert httpx.get(f"{seeded.url}/collections/ph").json()["state"] == "queued"
        page.on("request", store_get)
        page.goto(seeded.url + "/c/ph")
        expect(page.locator("#head")).to_contain_text("ph")
        page.wait_for_timeout(3500)
        assert len(gets) >= 2, "a busy store's page polls every 3 s"
        show(True)
        page.wait_for_timeout(500)  # a tick that began just before the hide sends its GET now
        gets.clear()
        page.wait_for_timeout(7000)
        assert gets == []
        with page.expect_request(
            lambda r: r.method == "GET" and r.url.endswith("/collections/ph"), timeout=1000
        ):
            show(False)


LAYOUT_PROPS = [
    "display", "gap", "white-space", "margin", "padding", "font-size", "border-radius",
    "width", "min-height", "max-width", "background-color",
]  # fmt: skip


def has_layout(page, selector, **want):
    """The first element matching `selector` computes to `want` (CSS names, `_` for `-`)."""
    got = page.locator(selector).first.evaluate(
        """(el, props) => Object.fromEntries(
            props.map((p) => [p, getComputedStyle(el).getPropertyValue(p)]))""",
        LAYOUT_PROPS,
    )
    got = {k.replace("-", "_"): v for k, v in got.items()}
    assert {k: got[k] for k in want} == want, selector


def test_settings_form_layout_is_unchanged(page, seeded):
    # The values were read at b8e36a0, before the page styles moved into ui.css.
    page.goto(seeded.url + "/settings")
    expect(page.locator(".conn").first).to_be_visible()
    has_layout(page, "#add-url", width="352px", display="block", font_size="12px")
    has_layout(page, "#cap", width="144px", padding="6px 8px", border_radius="6px")
    has_layout(page, "#add .row", display="flex", gap="12px")
    has_layout(page, "#roles", display="grid", gap="10px", max_width="900px")
    has_layout(page, ".conn", display="grid", gap="8px", margin="10px 0px", padding="12px 14px")
    has_layout(page, ".conn", border_radius="8px", background_color="rgb(246, 246, 248)")
    has_layout(page, ".conn .top", display="flex", gap="6px 12px", font_size="14px")
    has_layout(page, ".conn .keyline input", width="256px", font_size="12px")
    has_layout(page, ".result", font_size="12px", display="block")

    hit = {
        "id": "i1", "text": "A ball falls.", "score": 0.5, "keywords": {}, "matched": ["drag"],
        "answer": None, "explanation": None, "meta": {}, "tags": ["drag", "energy"],
        "channels": ["meaning"],
    }  # fmt: skip
    empty = {"hits": [], "answer": None, "usd": 0.0, "error": None, "query_tags": []}
    results = {"keyword_rag": {**empty, "hits": [hit], "query_tags": ["drag"]}}
    page.route("**/query", lambda route: route.fulfill(json={"results": results}))
    page.goto(seeded.url + "/lab")
    page.locator("#question").fill("how fast does it fall")
    page.locator("#run").click()
    expect(page.locator(".hit .chip")).to_have_count(2)
    has_layout(page, "#form", display="grid", gap="10px", margin="0px 0px 16px", max_width="900px")
    has_layout(page, "#question", min_height="70px", white_space="pre-wrap", width="900px")
    has_layout(page, "#choices", min_height="50px")
    has_layout(page, "#k", width="70px")
    has_layout(page, "#rrfk", width="80px")
    has_layout(page, "label:has(#answer)", display="flex", gap="6px", font_size="12px")
    # The Lab's own chip beats ui.css's `.chip` (inline-flex, 13px, 12px radius).
    plain = ".hit .chip:not(.on)"
    has_layout(page, plain, display="inline-block", gap="2px", margin="2px 2px 0px 0px")
    has_layout(page, plain, padding="1px 6px", font_size="11px", border_radius="10px")
    has_layout(page, plain, background_color="rgba(0, 0, 0, 0)")
    has_layout(page, ".hit .chip.on", background_color="rgb(255, 224, 138)", padding="1px 6px")
    has_layout(page, ".hit", margin="6px 0px", padding="8px", border_radius="6px")
    page.unroute_all()

    chat_route(
        page, lambda route: route.fulfill(json=chat_message(route, answer=None, error="boom"))
    )
    open_chat(page, seeded)
    chat_ask(page, "how fast are enzymes?")
    expect(page.locator("#log .msg.err")).to_be_visible()
    has_layout(page, "#log .user", white_space="pre-wrap", max_width="85%", padding="10px 12px")
    has_layout(page, "#log .user", background_color="rgb(246, 246, 248)", border_radius="8px")
    has_layout(page, "#log .msg.err", white_space="pre-wrap", padding="10px 12px")

    page.goto(seeded.url + "/c/c")  # another page's .err keeps its normal wrapping
    expect(page.locator("#chat-store-msg")).to_have_count(1)
    has_layout(page, "#chat-store-msg", white_space="normal", font_size="14px")


def test_sidebar_asks_for_spend_once_on_load(page, seeded):
    budgets = []
    page.on("request", lambda r: budgets.append(r.url) if r.url.endswith("/budget") else None)
    page.goto(seeded.url + "/worker")
    expect(page.locator("#spend")).to_contain_text("this month")
    expect(page.locator('#side a[href="/c/c"]')).to_be_visible()  # only the redraw adds it
    page.wait_for_timeout(3000)  # past the redraws that /collections and /eval-sets cause
    assert len(budgets) == 1, budgets
    expect(page.locator("#spend")).to_contain_text("this month")  # a redraw keeps what was fetched


def test_copy_is_announced(page, seeded):
    page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    open_chat(page, seeded, ("c",))
    your_app(page)
    status = page.locator("#chat-head details .sr-only[role=status]")
    expect(status).to_have_text("")
    page.locator("#chat-head details button[data-copy]").first.click()
    expect(status).to_have_text("Copied")


def test_copy_is_announced_when_the_clipboard_fails(page, seeded):
    page.add_init_script("Object.defineProperty(navigator, 'clipboard', {value: undefined})")
    open_chat(page, seeded, ("c",))
    your_app(page)
    page.locator("#chat-head details button[data-copy]").first.click()
    expect(page.locator("#chat-head details .sr-only[role=status]")).to_have_text(
        "Press Ctrl+C to copy"
    )


@pytest.mark.parametrize(
    "path, text, link",
    [
        ("/c/nope", "No store named nope.", "Stores"),
        ("/e/nope", "No eval set named nope.", "Eval sets"),
        ("/c/c/runs/abc", "No run abc on store c.", "Back to c"),
        ("/c/nope/runs/abc", "No store named nope.", "Stores"),
    ],
)
def test_a_missing_page_says_so_and_offers_a_way_back(page, seeded, path, text, link):
    page.expected = ["404"]
    page.clock.install()
    page.goto(seeded.url + path)
    expect(page.locator("[data-not-found]")).to_have_text(text)
    expect(page.get_by_role("link", name=link, exact=True).last).to_be_visible()
    expect(page.locator("main button")).to_have_count(0)  # no Save, Delete or Propose left
    page.clock.run_for(20_000)  # past the store page's slow poll: nothing redraws, nothing throws
    expect(page.locator("[data-not-found]")).to_have_text(text)


def test_a_missing_chat_says_so_at_the_top(page, seeded):
    page.expected = ["404"]
    page.goto(seeded.url + "/chat#doesnotexist")
    expect(page.locator("#empty")).to_have_text(
        "No chat doesnotexist: it may have been deleted. "
        "Start a new chat, or open one from the list."
    )


def test_an_unknown_page_is_an_html_page(page, seeded):
    page.expected = ["404"]
    page.goto(seeded.url + "/nothing")
    expect(page.locator("h1")).to_have_text("Page not found")
    expect(page.get_by_role("link", name="Stores", exact=True).last).to_be_visible()
