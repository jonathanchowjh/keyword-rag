"""Labelling a Documents eval set: marking passages, Find passages and the AI's answer."""

import time
from contextlib import contextmanager

import httpx
import pytest
from page_helpers import (
    labelled_questions,
    labelling_set,
    mark,
    pressed,
    question_map,
    replies,
)
from playwright.sync_api import expect

from ragqa_core.fakes import until

pytestmark = pytest.mark.pages

POOL_LOADING = "The ranking model is still loading, so passages are in search order, not best first"


def test_documents_eval_set_labels_a_question_and_saves_it(page, seeded):
    labelling_set(seeded, "lab1")
    page.goto(seeded.url + "/e/lab1q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("how long does alpha last?")
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    expect(passages.first).to_contain_text("x.md · part 1 of 2")
    shown = panel.inner_text()
    for word in ("vector", "keyword", "hybrid"):
        assert word not in shown.lower(), word
    assert "Vector search" not in shown
    first, second = (passages.nth(i).get_attribute("data-passage") for i in range(2))
    mark(passages.nth(0), "right")
    mark(passages.nth(1), "wrong")
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
    mark(passages.first, "right")
    panel.get_by_role("button", name="Save question").click()
    msg = page.locator("#lab-msg")
    expect(msg).to_contain_text("Saved as version")
    said = msg.inner_text()
    mark(passages.first, "wrong")
    panel.get_by_role("button", name="Save question").click()
    page.get_by_role("button", name="Save anyway").click()
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
    mark(passages.first, "right")
    expect(pressed(passages.first, "right")).to_have_attribute("aria-pressed", "true")
    mark(passages.first, "right")
    expect(pressed(passages.first, "right")).to_have_attribute("aria-pressed", "false")
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
    expect(hit).to_contain_text("x.md · part 2 of 2")
    mark(hit, "right")
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
    mark(passages.nth(0), "right")
    mark(passages.nth(1), "wrong")
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
    expect(pressed(passages.nth(0), "right")).to_have_attribute("aria-pressed", "true")
    expect(pressed(passages.nth(1), "wrong")).to_have_attribute("aria-pressed", "true")
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
    expect(pressed(gone, "right")).to_have_attribute("aria-pressed", "true")
    q2.get_by_role("button", name="Delete question…").click()
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
    mark(passages.first, "right")
    panel.get_by_role("button", name="Save question").click()
    deadline = time.time() + 5
    while not held and time.time() < deadline:
        page.wait_for_timeout(20)
    assert held, "the PUT never left"
    expect(
        rows.filter(has_text="brine second?").get_by_role("button", name="Label")
    ).to_be_disabled()
    expect(
        rows.filter(has_text="brine second?").get_by_role("button", name="Delete question…")
    ).to_be_disabled()
    expect(panel.get_by_role("button", name="Save question")).to_be_disabled()
    expect(panel.get_by_role("button", name="Find passages")).to_be_disabled()
    held[0].continue_()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    page.unroute(f"**/eval-sets/lab9q/questions/{q1}")
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    expect(panel.get_by_label("Question", exact=True)).to_have_value("brine second?")
    expect(passages).to_have_count(2)
    mark(passages.nth(1), "right")
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
    mark(passages.first, "right")
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
    mark(passages.nth(0), "right")
    mark(passages.nth(1), "wrong")
    r = httpx.delete(f"{seeded.url}/eval-sets/lab11q/questions/{qid}")
    assert r.status_code == 200, r.text
    page.expected = ["404"]
    panel.get_by_role("button", name="Save question").click()
    msg = page.locator("#lab-msg")
    expect(msg).to_contain_text("This question was deleted since you opened it")
    expect(page.locator("#lab-questions tbody tr")).to_have_count(0)
    expect(pressed(passages.nth(0), "right")).to_have_attribute("aria-pressed", "true")
    expect(pressed(passages.nth(1), "wrong")).to_have_attribute("aria-pressed", "true")
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
    mark(passages.first, "right")
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    notice = panel.get_by_text(UNSAVED, exact=True)
    expect(notice).to_be_visible()
    expect(question).to_have_value("alpha first?")
    panel.get_by_role("button", name="Cancel").click()
    expect(notice).to_have_count(0)
    expect(question).to_have_value("alpha first?")
    expect(pressed(passages.first, "right")).to_have_attribute("aria-pressed", "true")
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
    mark(passages.first, "right")
    rows.filter(has_text="brine second?").get_by_role("button", name="Delete question…").click()
    notice = panel.get_by_text(UNSAVED, exact=True)
    expect(notice).to_be_visible()
    expect(confirm).to_have_count(0)
    panel.get_by_role("button", name="Cancel").click()
    expect(notice).to_have_count(0)
    expect(confirm).to_have_count(0)
    expect(pressed(passages.first, "right")).to_have_attribute("aria-pressed", "true")
    assert len(httpx.get(f"{seeded.url}/eval-sets/lab13q").json()["questions"]) == 2
    rows.filter(has_text="brine second?").get_by_role("button", name="Delete question…").click()
    panel.get_by_role("button", name="Discard them").click()
    expect(confirm).to_be_visible()
    page.get_by_role("button", name="Yes, delete it").click()
    expect(rows).to_have_count(1)
    got = httpx.get(f"{seeded.url}/eval-sets/lab13q").json()["questions"]
    assert [q["text"] for q in got] == ["alpha first?"]


def test_labelling_a_mark_made_after_delete_was_asked_withdraws_the_delete(page, seeded):
    labelling_set(seeded, "lab14")
    labelled_questions(seeded, "lab14", ["alpha to delete?"])
    page.goto(seeded.url + "/e/lab14q")
    row = page.locator("#lab-questions tbody tr").first
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    row.get_by_role("button", name="Label").click()
    expect(passages).to_have_count(2)
    row.get_by_role("button", name="Delete question…").click()
    expect(page.locator("main").get_by_text("Delete this question and its marks?")).to_be_visible()
    mark(passages.first, "right")
    expect(page.locator("#lab-del")).to_be_empty()
    row.get_by_role("button", name="Delete question…").click()
    expect(panel.get_by_text(UNSAVED, exact=True)).to_be_visible()
    assert len(httpx.get(f"{seeded.url}/eval-sets/lab14q").json()["questions"]) == 1


ASK_AI = "Get the AI's answer"

ADDED = "Added as the reference answer. Save question to keep it."

REFERENCE = "Reference answer (optional; short is better)"


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
    mark(passages.first, "right")
    panel.get_by_role("button", name=ASK_AI).click()
    start = page.locator("#lab-ai-confirm").get_by_role("button", name="Start")
    expect(start).to_be_visible()
    answer = page.locator("#lab-ai-answer")
    expect(answer).to_be_empty()
    question = "How long is leave? (ai1)"
    assert sent == [{"question": question, "passages": [first], "confirm": False}]
    shown = httpx.post(f"{seeded.url}/eval-sets/ai1q/answer", json=sent[0]).json()["estimate"]
    with ai_calls(seeded) as purposes:
        start.click()
        expect(answer.get_by_text("Is this the right answer?", exact=True)).to_be_visible()
        expect(answer.locator(".full")).to_have_text("Answer: A")
    assert purposes == ["reference_answer"]
    assert sent[1:] == [{"question": question, "passages": [first], "confirm": True, **shown}]
    expect(page.locator("#lab-ai-confirm")).to_be_empty()
    answer.get_by_role("button", name="Right", exact=True).click()
    expect(panel.get_by_label(REFERENCE)).to_have_value("Answer: A")
    expect(page.locator("#lab-ai-msg")).to_have_text(ADDED)
    expect(answer).to_be_empty()
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(page.locator("#lab-ai-msg")).to_have_text("")  # it is kept now
    got = httpx.get(f"{seeded.url}/eval-sets/ai1q").json()["questions"]
    assert [q["gold_answers"] for q in got] == [["Answer: A"]]


def test_reference_answer_sends_the_price_it_showed(page, seeded):
    labelling_set(seeded, "ai10")
    panel, passages, sent = ai_panel(page, seeded, "ai10", "How long is leave? (ai10)")
    mark(passages.first, "right")
    with page.expect_response("**/eval-sets/ai10q/answer") as priced:
        panel.get_by_role("button", name=ASK_AI).click()
    estimate = priced.value.json()["estimate"]
    assert estimate["calls"] == 1 and estimate["est_usd"] > 0
    page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
    expect(page.locator("#lab-ai-answer .full")).to_have_text("Answer: A")
    confirming = sent[1]
    assert confirming["confirm"] is True
    assert (confirming["calls"], confirming["est_usd"]) == (estimate["calls"], estimate["est_usd"])


def test_labelling_ai_answer_n_keeps_nothing_and_survives_a_mark(page, seeded):
    labelling_set(seeded, "ai2")
    panel, passages, _ = ai_panel(page, seeded, "ai2", "How long is leave? (ai2)")
    mark(passages.first, "right")
    panel.get_by_role("button", name=ASK_AI).click()
    page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
    answer = page.locator("#lab-ai-answer")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    mark(passages.nth(1), "wrong")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    answer.get_by_role("button", name="Wrong", exact=True).click()
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
        page.get_by_text("The AI answers from the passages marked Right.", exact=True)
    ).to_be_visible()
    expect(ask).to_be_disabled()
    mark(passages.first, "wrong")
    expect(ask).to_be_disabled()
    mark(passages.nth(1), "right")
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
        mark(passages.first, "right")
        ask.click()
        wait_for(page, held)
        mark(passages.nth(1), "wrong")  # while its price is out
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
    mark(passages.first, "right")
    panel.get_by_role("button", name=ASK_AI).click()
    page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
    answer = page.locator("#lab-ai-answer")
    expect(answer.locator(".full")).to_have_text("Answer: A")
    held = []
    page.route("**/eval-sets/ai5q/questions", lambda route: held.append(route))
    panel.get_by_role("button", name="Save question").click()
    wait_for(page, held)
    y = answer.get_by_role("button", name="Right", exact=True)
    n = answer.get_by_role("button", name="Wrong", exact=True)
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
    mark(passages.first, "right")
    ask.click()
    start.click()
    answer.get_by_role("button", name="Right", exact=True).click()
    expect(msg).to_have_text(ADDED)
    panel.get_by_label(REFERENCE).press_sequentially(" or so")
    expect(msg).to_have_text("")  # edited by hand, it is no longer the AI's answer
    label.click()
    panel.get_by_role("button", name="Discard them").click()
    expect(panel.get_by_label("Question", exact=True)).to_have_value("What is leave for? (ai8)")
    expect(msg).to_have_text("")
    expect(passages).to_have_count(2)
    mark(passages.first, "right")
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
    mark(passages.nth(1), "right")
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
    mark(passages.first, "right")
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
    mark(passages.first, "right")
    with replies(seeded, lambda p: "Ten days [1]."):
        panel.get_by_role("button", name=ASK_AI).click()
        page.locator("#lab-ai-confirm").get_by_role("button", name="Start").click()
        expect(page.locator("#lab-ai-answer .full")).to_have_text("Ten days.")


def test_labelling_answer_start_cannot_be_bought_twice(page, seeded):
    labelling_set(seeded, "ai9")
    held = []

    def hold_paid(route):
        if route.request.post_data_json["confirm"] and not held:
            held.append(route)
        else:
            route.continue_()  # a second paid call goes through, so the test sees it

    page.route("**/eval-sets/ai9q/answer", hold_paid)
    panel, passages, sent = ai_panel(page, seeded, "ai9", "How long is leave? (ai9)")
    mark(passages.first, "right")
    start = page.locator("#lab-ai-confirm").get_by_role("button", name="Start")
    with ai_calls(seeded) as purposes:
        panel.get_by_role("button", name=ASK_AI).click()
        start.click()
        wait_for(page, held)
        panel.get_by_role("button", name="Save question").click()
        expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
        expect(start).to_be_disabled()
        start.dispatch_event("click")
        page.wait_for_timeout(300)  # a second paid call would have left by now
        with page.expect_response("**/eval-sets/ai9q/answer") as reply:
            held[0].continue_()
        reply.value.finished()
        expect(page.locator("#lab-ai-answer .full")).to_have_text("Answer: A")
    page.unroute("**/eval-sets/ai9q/answer")
    assert purposes == ["reference_answer"]
    assert [body["confirm"] for body in sent] == [False, True]


def test_labelling_answer_start_waits_out_a_restore(page, seeded):
    # restorable() leaves no right passage, so the AI could not be asked: this one is the
    # other way round, with the current version right on the first passage.
    labelling_set(seeded, "rs3")
    pool = httpx.post(f"{seeded.url}/eval-sets/rs3q/pool", json={"question": "alpha"})
    assert pool.status_code == 200, pool.text
    right, other = (p["id"] for p in pool.json()["passages"])
    question = {"text": "alpha restored?", "judged_ids": [other]}
    r = httpx.post(f"{seeded.url}/eval-sets/rs3q/questions", json={"question": question})
    assert r.status_code == 201, r.text
    qid, first = r.json()["id"], r.json()["version"]
    stored = question_map(seeded, "rs3q")[qid]
    current = {**stored, "gold_ids": [right], "judged_ids": [right, other]}
    r = httpx.put(
        f"{seeded.url}/eval-sets/rs3q/questions/{qid}",
        json={"question": current, "base": stored},
    )
    assert r.status_code == 200, r.text
    sent = []

    def record(request):
        if request.url.endswith("/eval-sets/rs3q/answer"):
            sent.append(request.post_data_json)

    page.on("request", record)
    page.goto(f"{seeded.url}/e/rs3q")
    page.locator("#lab-questions tbody tr").first.get_by_role("button", name="Label").click()
    panel = page.locator("#lab-panel")
    expect(panel.locator("li[data-passage]")).to_have_count(2)
    kept = page.locator(f'#lab-panel li[data-passage="{right}"]')
    expect(pressed(kept, "right")).to_have_attribute("aria-pressed", "true")
    start = page.locator("#lab-ai-confirm").get_by_role("button", name="Start")
    with ai_calls(seeded) as purposes:
        panel.get_by_role("button", name=ASK_AI).click()
        expect(start).to_be_visible()
        page.locator("#versions li", has_text=first[:8]).get_by_role("button", name="View").click()
        held = []
        page.route("**/eval-sets/rs3q/questions", lambda route: held.append(route))
        page.get_by_role("button", name="Restore this version").click()
        wait_for(page, held)
        expect(start).to_be_disabled()
        start.dispatch_event("click")
        page.wait_for_timeout(300)  # a paid call would have left by now
        held[0].continue_()
        expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first[:8]}.")
        page.unroute("**/eval-sets/rs3q/questions")
        expect(page.locator("#lab-ai-confirm")).to_be_empty()
    assert "reference_answer" not in purposes
    assert not any(body["confirm"] for body in sent)


def test_labelling_buttons_read_right_wrong(page, seeded):
    labelling_set(seeded, "rw1")
    page.goto(seeded.url + "/e/rw1q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("alpha")
    panel.get_by_role("button", name="Find passages").click()
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    expect(passages.first.get_by_role("button", name="Right", exact=True)).to_be_visible()
    expect(passages.first.get_by_role("button", name="Wrong", exact=True)).to_be_visible()
    expect(passages.first.get_by_role("button", name="y", exact=True)).to_have_count(0)
    expect(page.locator("#lab-ai-hint")).to_have_text(
        "The AI answers from the passages marked Right."
    )


def test_long_passage_collapses(page, seeded):
    labelling_set(seeded, "rw2")
    r = httpx.post(f"{seeded.url}/eval-sets/rw2q/pool", json={"question": "alpha"})
    pool = r.json()
    long = {**pool["passages"][0], "text": "long " * 120}
    page.route(
        "**/eval-sets/rw2q/pool", lambda route: route.fulfill(json={**pool, "passages": [long]})
    )
    page.goto(seeded.url + "/e/rw2q")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("alpha")
    panel.get_by_role("button", name="Find passages").click()
    item = panel.locator("li[data-passage]")
    expect(item.locator("details")).to_have_count(1)
    expect(item.locator(".full")).to_be_hidden()
    expect(item.locator("summary")).to_be_visible()
    item.locator("summary").click()
    expect(item.locator(".full")).to_be_visible()
    assert len(item.locator(".full").inner_text()) > 590


def test_search_store_says_nothing_new(page, seeded):
    labelling_set(seeded, "rw3")
    page.goto(seeded.url + "/e/rw3q")
    panel = page.locator("#lab-panel")
    box = panel.get_by_label("None of these? Search the store")
    box.fill("alpha")
    panel.get_by_role("button", name="Search", exact=True).click()
    expect(page.locator("#lab-found li[data-passage]")).not_to_have_count(0)
    expect(page.locator("#lab-search-msg")).to_have_text("")
    panel.get_by_role("button", name="Search", exact=True).click()
    expect(page.locator("#lab-search-msg")).to_have_text(
        "Nothing new: every passage the search found is already listed."
    )


def test_new_set_header_has_no_version(page, seeded):
    labelling_set(seeded, "rw4")
    page.goto(seeded.url + "/e/rw4q")
    expect(page.locator("h1")).to_have_text("rw4q")
    expect(page.locator(".head .aside")).to_have_count(0)
    assert "version" not in page.locator(".head").inner_text().lower()


def test_labelling_error_shows_in_its_message(page, seeded):
    labelling_set(seeded, "lerr")
    page.expected = ["500"]
    held = []
    page.route("**/eval-sets/lerrq/pool", lambda route: held.append(route))
    page.route("**/query", lambda route: held.append(route))

    def fail_unstyled(msg, detail):
        """Answers the held request with a 500 once the message has lost its class, so only the
        error's own say can make it "err" again."""
        for _ in range(100):
            if held:
                break
            page.wait_for_timeout(20)
        page.evaluate(f"document.getElementById('{msg}').className = ''")
        held.pop().fulfill(status=500, json={"detail": detail})

    page.goto(seeded.url + "/e/lerrq")
    panel = page.locator("#lab-panel")
    panel.get_by_label("Question", exact=True).fill("alpha")
    panel.get_by_role("button", name="Find passages").click()
    fail_unstyled("lab-find-msg", "the pool broke")
    expect(page.locator("#lab-find-msg")).to_have_text("the pool broke")
    expect(page.locator("#lab-find-msg")).to_have_class("err")
    panel.get_by_label("None of these? Search the store").fill("brine")
    panel.get_by_role("button", name="Search", exact=True).click()
    fail_unstyled("lab-search-msg", "the search broke")
    expect(page.locator("#lab-search-msg")).to_have_text("the search broke")
    expect(page.locator("#lab-search-msg")).to_have_class("err")


def labelled_count(seeded, name):
    return len(question_map(seeded, name))


def test_save_waits_for_find_and_asks_when_nothing_is_right(page, seeded):
    labelling_set(seeded, "savewait")
    page.goto(seeded.url + "/e/savewaitq")
    page.locator("#lab-text").fill("Where is the office?")
    expect(page.locator("#lab-save")).to_be_disabled()
    expect(page.locator("#lab-save-why")).to_have_text(
        "Find passages first, so you can mark the right ones."
    )
    page.get_by_role("button", name="Find passages").click()
    expect(page.locator("#lab-save")).to_be_enabled()
    page.locator("#lab-save").click()
    expect(
        page.get_by_text("No passage marked Right: this question won't be scored.")
    ).to_be_visible()
    page.get_by_role("button", name="Cancel").click()
    assert labelled_count(seeded, "savewaitq") == 0  # nothing was saved
    page.locator("#lab-save").click()
    page.get_by_role("button", name="Save anyway").click()
    until(lambda: labelled_count(seeded, "savewaitq") == 1, 5, 0.05, "the question saved")
    page.locator("#lab-save").click()  # the same question again: the ask comes back
    expect(
        page.get_by_text("No passage marked Right: this question won't be scored.")
    ).to_be_visible()


def test_save_turns_on_when_find_returns_nothing(page, seeded):
    labelling_set(seeded, "findnone")
    page.route(
        "**/eval-sets/findnoneq/pool",
        lambda route: route.fulfill(
            json={"passages": [], "labelled": [], "missing": [], "notes": []}
        ),
    )
    page.goto(seeded.url + "/e/findnoneq")
    page.locator("#lab-text").fill("Anything at all?")
    page.get_by_role("button", name="Find passages").click()
    expect(page.locator("#lab-save")).to_be_enabled()


def test_a_question_opened_with_label_saves_at_once(page, seeded):
    # A guard: this passes before the change too; it keeps the U4 rule off opened questions.
    labelling_set(seeded, "reopen")
    labelled_questions(seeded, "reopen", ["What is an enzyme?"])
    page.goto(seeded.url + "/e/reopenq")
    page.get_by_role("button", name="Label").first.click()
    expect(page.locator("#lab-save")).to_be_enabled()
