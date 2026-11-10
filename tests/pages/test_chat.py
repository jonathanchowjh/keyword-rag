"""Chat: the conversation, its settings panel, the retrieval step, and chat from your own app."""

import json
import re
import shlex
import threading
from contextlib import contextmanager

import httpx
import pytest
from page_helpers import (
    CHAT_ATTEMPT,
    chat_ask,
    chat_message,
    chat_route,
    make_chat,
    open_chat,
    sources,
    your_app,
)
from playwright.sync_api import expect

from ragqa_core import db, search_defaults
from ragqa_core.fakes import fake_replies, until

pytestmark = pytest.mark.pages

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
    row.get_by_role("button", name="Delete chat").click()
    expect(page.get_by_text("Delete chat the second of two saved chats?")).to_be_visible()
    page.get_by_role("button", name="Cancel").click()
    expect(page.get_by_text("Delete chat the second of two saved chats?")).to_have_count(0)
    row.get_by_role("button", name="Delete chat").click()
    page.get_by_role("button", name="Yes, delete").click()
    expect(chat_link(page, second)).to_have_count(0)
    assert httpx.get(f"{seeded.url}/chats/{second}").status_code == 404
    expect(page.locator("#log .user")).to_have_text("the first of two saved chats")  # still open
    # Deleting the open chat closes it.
    open_row = page.locator("#chat-list li", has=page.locator(f'a[href="#{first}"]'))
    open_row.get_by_role("button", name="Delete chat").click()
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
    until(
        lambda: len(httpx.get(f"{seeded.url}/chats/{chat_id}").json()["messages"]) == count,
        4,
        0.02,
        f"the server saving {count} messages",
    )


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
    expect(first.locator(".what")).to_have_text("[1] enzyme-item · added by hand · c")
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


def test_chat_sources_name_the_file_and_part_of_a_split_file(page, seeded):
    part = {"n": 1, "id": "x.md#2", "text": "x.md\n\nbrine", "source": "upload:x", "store": "c"}
    citations = [{**part, "file": "x.md", "part": 2, "parts": 3}, CHAT_ATTEMPT["citations"][1]]
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, citations=citations)))
    open_chat(page, seeded)
    chat_ask(page, "parts?")
    sources_ = page.locator("#log .reply .sources li .what")
    expect(sources_.first).to_have_text("[1] x.md · part 2 of 3 · x (uploaded) · c")
    expect(sources_.first.locator('[title="x.md#2"]')).to_have_text("x.md · part 2 of 3")
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
    row.get_by_role("button", name="Delete chat").click()
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
        expect(chip).to_contain_text("0 worked examples")
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
        expect(chip).to_contain_text("1 worked example")
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
    expect(passage.locator(".what")).to_have_text(
        "[1] enzyme-item · c · added by hand · rank 1 in c"
    )
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
        "[1] enzyme-item · c · added by hand · rank 1 in c · rerank 0.77"
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


BRIEF_KEYS = {"n", "answer", "citations", "passages", "passages_only", "error", "usd"}


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


def test_chat_delete_prompt_has_one_question_mark(page, seeded):
    page.goto(seeded.url + "/chat")
    page.locator("#new").click()
    page.locator('#picking input[value="c"]').check()
    page.get_by_role("button", name="Start chat").click()
    chat_ask(page, "What is alpha?")
    expect(page.locator("#log .reply")).to_have_count(1)
    chat_id = page.url.split("#")[1]
    chat_link(page, chat_id).locator("xpath=..").get_by_role("button", name="Delete chat").click()
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


EMPTY_WHY = "Add passages first: an empty store has nothing to search."  # a Documents store


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


def click_chat_store(page, name):
    page.locator("#chat-store").click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    chat_id = page.url.split("#", 1)[1]
    page.goto(f"{page.url.split('/chat')[0]}/c/{name}")
    expect(page.locator("#chat-store")).to_be_enabled()
    return chat_id


def test_chat_with_store_reuses_its_chat(page, seeded):
    seeded.build("reuse1")
    page.goto(seeded.url + "/c/reuse1")
    first = click_chat_store(page, "reuse1")
    page.locator("#chat-store").click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    assert page.url.split("#", 1)[1] == first


def test_chat_with_store_ignores_api_made_chats(page, seeded):
    seeded.build("reuse2")
    made = make_chat(seeded, ("reuse2",))
    page.goto(seeded.url + "/c/reuse2")
    page.locator("#chat-store").click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    assert page.url.split("#", 1)[1] != made


def test_chat_with_store_new_after_edit(page, seeded):
    seeded.build("reuse3")
    page.goto(seeded.url + "/c/reuse3")
    first = click_chat_store(page, "reuse3")
    r = httpx.patch(f"{seeded.url}/chats/{first}", json={"stores": ["reuse3", "c"]})
    assert r.status_code == 200, r.text
    page.locator("#chat-store").click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    new = page.url.split("#", 1)[1]
    assert new != first
    assert httpx.get(f"{seeded.url}/chats/{new}").json()["stores"] == ["reuse3"]
    assert httpx.get(f"{seeded.url}/chats/{first}").json()["stores"] == ["reuse3", "c"]  # kept


def test_chat_with_store_new_after_its_chat_is_deleted(page, seeded):
    seeded.build("reuse4")
    page.goto(seeded.url + "/c/reuse4")
    first = click_chat_store(page, "reuse4")
    assert httpx.delete(f"{seeded.url}/chats/{first}").status_code == 204
    page.expected = ["404"]
    page.locator("#chat-store").click()
    page.wait_for_url(re.compile(r"/chat#.+"))
    assert page.url.split("#", 1)[1] != first


def test_chat_with_an_empty_store_is_disabled(page, seeded):
    r = httpx.post(f"{seeded.url}/collections", json={"name": "nochat", "qa_type": "documents"})
    assert r.status_code == 201, r.text
    page.goto(seeded.url + "/c/nochat")
    expect(page.locator("#chat-store-why")).to_have_text(EMPTY_WHY)
    expect(page.locator("#chat-store")).to_be_disabled()
    expect(page.locator("#chat-store")).to_have_attribute("title", EMPTY_WHY)


def test_chat_passage_stays_open_on_redraw(page, seeded):
    chat_route(page, lambda route: route.fulfill(json=chat_message(route)))
    open_chat(page, seeded)
    chat_ask(page, "kinetics?")
    first = page.locator("#log .reply .sources li").first
    first.locator("summary").click()
    expect(first.locator("details")).to_have_attribute("open", "")
    page.locator("#log .reply a.cite").nth(1).click()  # opens the retrieval step: a redraw
    expect(page.locator("#msg-1 details.retrieval")).to_have_attribute("open", "")
    expect(page.locator("#log .reply .sources li").first.locator("details")).to_have_attribute(
        "open", ""
    )


def test_evidence_shows_file_and_part(page, seeded):
    evidence = [
        {**CHAT_ATTEMPT["evidence"][0], "id": "x.md#2", "file": "x.md", "part": 2, "parts": 3},
        {**CHAT_ATTEMPT["evidence"][1], "id": "lone", "file": None, "part": None, "parts": None},
    ]
    chat_route(page, lambda route: route.fulfill(json=chat_message(route, evidence=evidence)))
    open_chat(page, seeded)
    chat_ask(page, "where?")
    page.locator("#msg-1 details.retrieval > summary").first.click()
    items = page.locator("#msg-1 ol.evidence > li")
    expect(items.first.locator(".what")).to_contain_text("[1] x.md · part 2 of 3")
    expect(items.first.locator(".what span[title]")).to_have_attribute("title", "x.md#2")
    expect(items.nth(1).locator(".what")).to_contain_text("[2] lone")
