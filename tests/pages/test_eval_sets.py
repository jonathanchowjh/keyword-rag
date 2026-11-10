"""The eval-set list and an eval set's page: questions, saving and versions."""

import json
import re
import time

import httpx
import pytest
from page_helpers import (
    documents_store,
    labelled_questions,
    labelling_set,
    mark,
    mcq_set,
    pressed,
    put_questions,
    question_map,
    upload_and_preview,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.pages


def test_import_questions_on_the_eval_set_page_then_include_leaked(page, seeded, tmp_path):
    httpx.post(f"{seeded.url}/collections", json={"name": "qimp", "qa_type": "mcq"})
    httpx.post(f"{seeded.url}/eval-sets", json={"name": "qimp", "qa_type": "mcq"})
    rows = [
        {"id": "u1", "text": "tides follow the moon", "choices": ["x", "y"], "answer": "A"},
        {"id": "u2", "text": "an unrelated one", "choices": ["x", "y"], "answer": "B"},
    ]
    qs = tmp_path / "qs.jsonl"
    qs.write_text("\n".join(json.dumps(x) for x in rows))
    page.goto(seeded.url + "/e/qimp")
    upload_and_preview(page, "questions", qs)
    expect(page.locator("#imp-questions-preview")).to_contain_text("2 rows")
    page.locator("#imp-questions-import").click()
    expect(page.locator("#imp-questions-preview")).to_contain_text("2 new")
    expect(page.locator("#questions tbody tr")).to_have_count(2)
    expect(page.locator("#fields")).to_contain_text("Worked solutions: 0 of 2")
    expect(page.locator("#fields")).to_contain_text("Evidence ids: –")
    page.get_by_label("Include leaked questions").check()
    expect(page.locator("#leaked-msg")).to_have_text("")
    expect(page.get_by_label("Include leaked questions")).to_be_checked()
    assert httpx.get(f"{seeded.url}/eval-sets/qimp").json()["include_leaked"] is True


def test_eval_sets_page_lists_and_creates(page, seeded):
    page.goto(seeded.url + "/eval-sets")
    expect(page.locator("main h1")).to_have_text("Eval sets")
    expect(page.locator('main a.row-link[href="/e/c"]')).to_contain_text("40 questions")
    page.get_by_role("button", name="+ New eval set").click()
    page.get_by_label("Name").fill("made-here")
    page.locator("#f-type").select_option("science")
    page.get_by_role("button", name="Create").click()
    expect(page.locator("main a.row-link", has_text="made-here")).to_be_visible()
    got = httpx.get(f"{seeded.url}/eval-sets/made-here").json()
    assert got["qa_type"] == "science"


def test_eval_set_page_edits_and_saves_a_version_keeping_meta(page, seeded):
    seeded.build("edit")
    before = httpx.get(f"{seeded.url}/eval-sets/edit").json()
    page.goto(seeded.url + "/e/edit")
    rows = page.locator("#questions tbody tr")
    expect(rows).to_have_count(40)
    row = rows.filter(has=page.locator("td", has_text="q0")).first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("enzyme question zero, reworded")
    row.get_by_role("button", name="Done").click()
    q1 = rows.filter(has=page.locator("td", has_text="q1")).first
    q1.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("button", name="Add question").click()
    fresh = rows.last
    fresh.locator("[data-field=text]").fill("a brand new question")
    fresh.locator("[data-field=choices]").fill("one\ntwo")
    fresh.locator("[data-field=gold_answers]").fill("B")
    fresh.get_by_role("button", name="Done").click()
    assert httpx.get(f"{seeded.url}/eval-sets/edit").json() == before  # nothing written yet
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    got = httpx.get(f"{seeded.url}/eval-sets/edit").json()
    by_id = {q["id"]: q for q in got["questions"]}
    assert len(got["questions"]) == 40
    assert "q1" not in by_id
    assert by_id["q0"]["text"] == "enzyme question zero, reworded"
    assert by_id["q0"]["meta"] == next(q for q in before["questions"] if q["id"] == "q0")["meta"]
    added = [q for q in got["questions"] if q["text"] == "a brand new question"]
    assert [q["gold_answers"] for q in added] == [["B"]]
    assert [q["meta"] for q in added] == [{"source": "hand"}]  # checked for leaks
    assert len(got["versions"]) == len(before["versions"]) + 1
    expect(page.locator("#versions li").first).to_contain_text(got["version"][:8])


def test_eval_set_page_keeps_edits_on_a_stale_save(page, seeded):
    seeded.build("stalee")
    page.goto(seeded.url + "/e/stalee")
    expect(page.locator("#questions tbody tr")).to_have_count(40)
    now = httpx.get(f"{seeded.url}/eval-sets/stalee").json()
    qs = [q for q in now["questions"] if q["id"] != "q5"]
    r = httpx.put(
        f"{seeded.url}/eval-sets/stalee/questions",
        json={"questions": qs, "base_version": now["version"]},
    )
    assert r.status_code == 200, r.text
    row = page.locator("#questions tbody tr").first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("my edit survives")
    page.expected = ["409"]
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("changed")
    expect(page.locator("[data-field=text]").first).to_have_value("my edit survives")
    assert question_map(seeded, "stalee")["q0"]["text"] != "my edit survives"
    page.get_by_role("button", name="Reload (discard my edits)").click()
    expect(page.locator("#questions tbody tr")).to_have_count(39)
    expect(page.locator("[data-field=text]")).to_have_count(0)
    expect(page.locator("#save-msg")).to_have_text("")
    expect(page.locator("#dirty")).to_have_text("")


def test_eval_set_page_pages_fifty_at_a_time(page, seeded):
    mcq_set(seeded, "many", 60)
    page.goto(seeded.url + "/e/many")
    expect(page.locator("#questions tbody tr")).to_have_count(50)
    page.get_by_role("button", name="Next").click()
    expect(page.locator("#questions tbody tr")).to_have_count(10)


def test_eval_set_page_deletes_after_saying_how_many_runs(page, seeded):
    run_id = seeded.build("gonee", run=True)
    assert run_id
    page.goto(seeded.url + "/e/gonee")
    page.get_by_role("button", name="Delete eval set…").click()
    expect(page.locator("#del-confirm")).to_contain_text("Delete eval set gonee and its 1 run?")
    page.get_by_role("button", name="Yes, delete gonee").click()
    expect(page).to_have_url(seeded.url + "/eval-sets")
    assert httpx.get(f"{seeded.url}/eval-sets/gonee").status_code == 404


def test_eval_set_page_will_not_import_over_unsaved_edits(page, seeded, tmp_path):
    seeded.build("dirt")
    before = httpx.get(f"{seeded.url}/eval-sets/dirt").json()
    qs = tmp_path / "more.jsonl"
    qs.write_text(
        json.dumps({"id": "u1", "text": "one more", "choices": ["x", "y"], "answer": "A"})
    )
    page.goto(seeded.url + "/e/dirt")
    expect(page.locator("#questions tbody tr")).to_have_count(40)
    expect(page.locator("#imp-questions-go")).to_be_enabled()
    row = page.locator("#questions tbody tr").first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("my unsaved edit")
    expect(page.locator("#imp-questions-go")).to_be_disabled()  # typing is an edit already
    row.get_by_role("button", name="Done").click()
    expect(page.locator("#imp-note")).to_have_text("Save or discard your changes before importing")
    expect(page.locator("#imp-questions-file")).to_be_disabled()
    expect(page.locator("#imp-questions-go")).to_be_disabled()
    expect(page.locator("#questions tbody tr").first).to_contain_text("my unsaved edit")
    assert httpx.get(f"{seeded.url}/eval-sets/dirt").json() == before  # nothing imported
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    expect(page.locator("#imp-note")).to_have_text("")
    upload_and_preview(page, "questions", qs)
    page.locator("#imp-questions-import").click()
    expect(page.locator("#imp-questions-preview")).to_contain_text("1 new")
    expect(page.locator("#questions tbody tr")).to_have_count(41)
    assert question_map(seeded, "dirt")["q0"]["text"] == "my unsaved edit"


def test_eval_set_versions_mark_the_current_one_after_a_revert(page, seeded):
    mcq_set(seeded, "rev", 2)
    first = httpx.get(f"{seeded.url}/eval-sets/rev").json()
    r = httpx.put(
        f"{seeded.url}/eval-sets/rev/questions", json={"questions": first["questions"][:1]}
    )
    second = r.json()["version"]
    r = httpx.put(f"{seeded.url}/eval-sets/rev/questions", json={"questions": first["questions"]})
    assert r.json()["version"] == first["version"]  # the same content: the same version
    page.goto(seeded.url + "/e/rev")
    items = page.locator("#versions li")
    expect(items).to_have_count(2)
    expect(items.filter(has_text=first["version"][:8])).to_contain_text("current")
    expect(items.filter(has_text=second[:8])).not_to_contain_text("current")


def test_eval_set_page_save_keeps_judged_ids(page, seeded):
    seeded.build("judged1")
    got = httpx.get(f"{seeded.url}/eval-sets/judged1").json()
    questions = got["questions"]
    questions[0]["judged_ids"] = ["x"]
    r = httpx.put(
        f"{seeded.url}/eval-sets/judged1/questions",
        json={"questions": questions, "base_version": got["version"]},
    )
    assert r.status_code == 200, r.text
    first = questions[0]["id"]
    page.goto(seeded.url + "/e/judged1")
    rows = page.locator("#questions tbody tr")
    other = rows.filter(has=page.locator("td", has_text=questions[1]["id"])).first
    other.get_by_role("button", name="Edit").click()
    other.locator("[data-field=text]").fill("reworded for the judged test")
    other.get_by_role("button", name="Done").click()
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    after = {q["id"]: q for q in httpx.get(f"{seeded.url}/eval-sets/judged1").json()["questions"]}
    assert after[first]["judged_ids"] == ["x"]


def test_eval_set_page_head_counts_its_questions(page, seeded):
    page.goto(seeded.url + "/e/c")
    expect(page.locator("#head")).to_contain_text("40 questions")
    expect(page.locator("#questions tbody tr")).to_have_count(40)
    assert "NaN" not in page.locator("main").inner_text()


def test_eval_set_page_views_and_restores_an_old_version(page, seeded):
    seeded.build("ver1")
    old = httpx.get(f"{seeded.url}/eval-sets/ver1").json()
    was = next(q for q in old["questions"] if q["id"] == "q0")["text"]
    changed = [{**q, "text": "q0 reworded"} if q["id"] == "q0" else q for q in old["questions"]]
    put_questions(seeded, "ver1", changed, old["version"])
    page.goto(seeded.url + "/e/ver1")
    first = old["version"][:8]
    expect(page.locator("#versions li")).to_have_count(2)
    page.locator("#versions li", has_text=first).get_by_role("button", name="View").click()
    view = page.locator("#ver-view")
    expect(view).to_contain_text(f"Version {first}: 40 questions")
    row = view.locator("tbody tr").filter(has=page.locator("td", has_text=re.compile(r"^q0$")))
    expect(row).to_contain_text(was)
    expect(view).not_to_contain_text("q0 reworded")
    view.get_by_role("button", name="Restore this version").click()
    expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first}.")
    expect(view).to_be_empty()
    assert httpx.get(f"{seeded.url}/eval-sets/ver1").json()["version"] == old["version"]
    expect(page.locator("#head")).to_contain_text(f"version {first}")
    expect(page.locator("#questions")).to_contain_text(was)


def test_eval_set_page_shows_the_newest_20_versions_first(page, seeded):
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": "ver2", "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    for i in range(22):
        one = {"id": "s0", "text": f"version two text {i}", "choices": ["x", "y"]}
        put_questions(seeded, "ver2", [{**one, "gold_answers": ["A"]}])
    page.goto(seeded.url + "/e/ver2")
    items = page.locator("#versions li")
    expect(items).to_have_count(20)
    page.get_by_role("button", name="Show all versions").click()
    expect(items).to_have_count(22)
    expect(page.get_by_role("button", name="Show all versions")).to_have_count(0)


def test_eval_set_page_back_to_the_current_version_closes_the_view(page, seeded):
    mcq_set(seeded, "ver5", 3)
    before = httpx.get(f"{seeded.url}/eval-sets/ver5").json()
    put_questions(seeded, "ver5", before["questions"][:2], before["version"])
    page.goto(seeded.url + "/e/ver5")
    old = page.locator("#versions li", has_text=before["version"][:8])
    old.get_by_role("button", name="View").click()
    view = page.locator("#ver-view")
    expect(view).to_contain_text("3 questions")
    view.get_by_role("button", name="Back to the current version").click()
    expect(view).to_be_empty()
    assert len(httpx.get(f"{seeded.url}/eval-sets/ver5").json()["questions"]) == 2


def test_eval_set_page_restore_waits_for_unsaved_edits(page, seeded):
    mcq_set(seeded, "ver4", 3)
    before = httpx.get(f"{seeded.url}/eval-sets/ver4").json()
    put_questions(seeded, "ver4", before["questions"][:2], before["version"])
    page.goto(seeded.url + "/e/ver4")
    page.locator("#versions li", has_text=before["version"][:8]).get_by_role(
        "button", name="View"
    ).click()
    restore = page.get_by_role("button", name="Restore this version")
    expect(restore).to_be_enabled()
    row = page.locator("#questions tbody tr").first
    row.get_by_role("button", name="Edit").click()
    row.locator("[data-field=text]").fill("an edit not saved")
    expect(restore).to_be_disabled()
    page.get_by_role("button", name="Save changes").click()
    expect(page.locator("#save-msg")).to_contain_text("Saved as version")
    expect(restore).to_be_enabled()


def test_eval_set_page_restore_waits_for_unsaved_marks(page, seeded):
    labelling_set(seeded, "ver3")
    labelled_questions(seeded, "ver3", ["alpha first?", "brine second?"])
    older = httpx.get(f"{seeded.url}/eval-sets/ver3q").json()["versions"][1]["version"]
    page.goto(seeded.url + "/e/ver3q")
    page.locator("#versions li", has_text=older[:8]).get_by_role("button", name="View").click()
    view = page.locator("#ver-view")
    expect(view).to_contain_text(f"Version {older[:8]}: 1 question")
    expect(view).to_contain_text("alpha first?")
    restore = view.get_by_role("button", name="Restore this version")
    expect(restore).to_be_enabled()
    rows = page.locator("#lab-questions tbody tr")
    rows.filter(has_text="alpha first?").get_by_role("button", name="Label").click()
    panel = page.locator("#lab-panel")
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    expect(restore).to_be_enabled()  # opening a question is not a change
    mark(passages.first, "right")
    expect(restore).to_be_disabled()
    panel.get_by_role("button", name="Save question").click()
    expect(page.locator("#lab-msg")).to_contain_text("Saved as version")
    expect(restore).to_be_enabled()
    panel.get_by_label("Reference answer (optional; short is better)").fill("soon")
    expect(restore).to_be_disabled()
    rows.filter(has_text="brine second?").get_by_role("button", name="Label").click()
    expect(restore).to_be_disabled()  # Label asks first: the typing is still there
    panel.get_by_role("button", name="Discard them").click()
    expect(restore).to_be_enabled()  # opening another question drops the typing
    panel.get_by_label("Question", exact=True).fill("brine second, reworded?")
    expect(restore).to_be_disabled()


def restorable(seeded, name):
    """The set `<name>q` of one question, first right on the store's first passage and judged
    on both, then saved with no right passage and only the other judged. Returns the
    question's id, the first version and the two passage ids, the right one first."""
    labelling_set(seeded, name)
    pool = httpx.post(f"{seeded.url}/eval-sets/{name}q/pool", json={"question": "alpha"})
    assert pool.status_code == 200, pool.text
    right, other = (p["id"] for p in pool.json()["passages"])
    question = {"text": "alpha restored?", "gold_ids": [right], "judged_ids": [right, other]}
    r = httpx.post(f"{seeded.url}/eval-sets/{name}q/questions", json={"question": question})
    assert r.status_code == 201, r.text
    qid, first = r.json()["id"], r.json()["version"]
    stored = question_map(seeded, f"{name}q")[qid]
    changed = {**stored, "gold_ids": [], "judged_ids": [other]}
    r = httpx.put(
        f"{seeded.url}/eval-sets/{name}q/questions/{qid}",
        json={"question": changed, "base": stored},
    )
    assert r.status_code == 200, r.text
    return qid, first, right, other


def hold_restore(page, seeded, name, first):
    """Opens the question in the panel, views version `first` and presses Restore with its
    PUT held; returns the held routes."""
    page.goto(f"{seeded.url}/e/{name}q")
    page.locator("#lab-questions tbody tr").first.get_by_role("button", name="Label").click()
    expect(page.locator("#lab-panel li[data-passage]")).to_have_count(2)
    page.locator("#versions li", has_text=first[:8]).get_by_role("button", name="View").click()
    held = []
    page.route(f"**/eval-sets/{name}q/questions", lambda route: held.append(route))
    page.get_by_role("button", name="Restore this version").click()
    deadline = time.time() + 5
    while not held and time.time() < deadline:
        page.wait_for_timeout(20)
    assert held, "the Restore PUT never left"
    return held


def test_eval_set_page_restore_holds_the_panel_still_and_brings_back_judged_ids(page, seeded):
    qid, first, right, other = restorable(seeded, "rs1")
    held = hold_restore(page, seeded, "rs1", first)
    panel = page.locator("#lab-panel")
    row = page.locator("#lab-questions tbody tr").first
    for button in ("Save question", "Save and write the next", "Find passages"):
        expect(panel.get_by_role("button", name=button)).to_be_disabled()
    expect(row.get_by_role("button", name="Label")).to_be_disabled()
    expect(row.get_by_role("button", name="Delete question…")).to_be_disabled()
    for passage in panel.locator("li[data-passage]").all():
        expect(pressed(passage, "right")).to_be_disabled()
        expect(pressed(passage, "wrong")).to_be_disabled()
    held[0].continue_()
    expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first[:8]}.")
    got = question_map(seeded, "rs1q")[qid]
    assert (got["gold_ids"], got["judged_ids"]) == ([right], [right, other])
    expect(panel.get_by_label("Question", exact=True)).to_have_value("")  # nothing unsaved
    expect(row.get_by_role("button", name="Label")).to_be_enabled()
    expect(row.get_by_role("button", name="Delete question…")).to_be_enabled()
    expect(row.locator("td").nth(1)).to_have_text("1")
    expect(row.locator("td").nth(2)).to_have_text("2")


def test_eval_set_page_restore_keeps_what_was_typed_while_it_ran(page, seeded):
    _, first, _, other = restorable(seeded, "rs2")
    held = hold_restore(page, seeded, "rs2", first)
    panel = page.locator("#lab-panel")
    answer = panel.get_by_label("Reference answer (optional; short is better)")
    answer.fill("typed meanwhile")
    held[0].continue_()
    expect(page.locator("#ver-msg")).to_have_text(f"Restored version {first[:8]}.")
    expect(panel.get_by_label("Question", exact=True)).to_have_value("alpha restored?")
    expect(answer).to_have_value("typed meanwhile")
    passages = panel.locator("li[data-passage]")
    expect(passages).to_have_count(2)
    kept = page.locator(f'#lab-panel li[data-passage="{other}"]')
    expect(pressed(kept, "wrong")).to_have_attribute("aria-pressed", "true")
    expect(panel.get_by_role("button", name="Save question")).to_be_enabled()


def test_eval_set_page_lists_the_current_version_after_restoring_an_old_one(page, seeded):
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": "ver6", "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    made = []
    for i in range(22):
        one = {"id": "s0", "text": f"version six text {i}", "choices": ["x", "y"]}
        made.append(put_questions(seeded, "ver6", [{**one, "gold_answers": ["A"]}]))
    old = httpx.get(f"{seeded.url}/eval-sets/ver6", params={"version": made[0]}).json()
    assert put_questions(seeded, "ver6", old["questions"], made[-1]) == made[0]
    page.goto(seeded.url + "/e/ver6")
    items = page.locator("#versions li")
    current = items.filter(has_text="current")
    expect(current).to_have_count(1)
    expect(current).to_contain_text(made[0][:8])
    expect(items).to_have_count(21)  # the current one, then the newest 20
    page.get_by_role("button", name="Show all versions").click()
    expect(items).to_have_count(22)
    expect(current).to_have_count(1)


def test_linking_an_older_documents_set_picks_its_own_store_first(page, seeded):
    for store in ("pk1a", "pk1b", "pk1c"):
        documents_store(seeded, store, passages=False)
    for name in ("pk1b", "pk1x", "pk1z"):
        r = httpx.post(f"{seeded.url}/eval-sets", json={"name": name, "qa_type": "documents"})
        assert r.status_code == 201, r.text
    r = httpx.patch(f"{seeded.url}/collections/pk1c", json={"eval_set": "pk1x"})
    assert r.status_code == 200, r.text
    pick = page.get_by_label("Pick a store to label against:")
    page.goto(seeded.url + "/e/pk1x")  # pk1c's default set
    expect(pick).to_have_value("pk1c")
    page.goto(seeded.url + "/e/pk1b")  # named after pk1b
    expect(pick).to_have_value("pk1b")
    page.goto(seeded.url + "/e/pk1z")  # no store of its own: the first
    expect(pick.locator("option")).not_to_have_count(0)
    expect(pick).to_have_value(pick.locator("option").first.get_attribute("value") or "")


def test_create_one_on_a_store_page_makes_a_linked_set(page, seeded):
    documents_store(seeded, "mk1", passages=False)
    page.goto(seeded.url + "/c/mk1")
    page.locator("#asked").get_by_role("link", name="Create one").click()
    expect(page).to_have_url(seeded.url + "/eval-sets?store=mk1")
    expect(page.get_by_label("Name")).to_have_value("mk1")
    expect(page.locator("#f-type")).to_have_value("documents")
    expect(page.locator("#f-type")).to_be_disabled()
    expect(page.get_by_text("Linked store: mk1", exact=True)).to_be_visible()
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/e/mk1")
    assert httpx.get(f"{seeded.url}/eval-sets/mk1").json()["store"] == "mk1"
    assert httpx.get(f"{seeded.url}/collections/mk1").json()["eval_set"] == "mk1"


def test_new_eval_set_form_links_a_documents_store(page, seeded):
    documents_store(seeded, "link1", passages=False)
    page.goto(seeded.url + "/eval-sets")
    page.get_by_role("button", name="+ New eval set").click()
    page.locator("#f-type").select_option("mcq")
    expect(page.get_by_label("Store to label against")).to_be_hidden()
    page.get_by_label("Name").fill("link1q")
    page.locator("#f-type").select_option("documents")
    pick = page.get_by_label("Store to label against")
    expect(pick.locator("option").first).to_have_text("(none)")
    values = pick.locator("option").evaluate_all("o => o.map(x => x.value)")
    assert "link1" in values and "c" not in values  # Documents stores only
    pick.select_option("link1")
    page.get_by_role("button", name="Create").click()
    expect(page).to_have_url(seeded.url + "/e/link1q")
    assert httpx.get(f"{seeded.url}/eval-sets/link1q").json()["store"] == "link1"


def test_an_eval_set_without_a_store_offers_no_run_link(page, seeded):
    mcq_set(seeded, "run2", 2)
    page.goto(seeded.url + "/e/run2")
    expect(page.locator("#questions tbody tr")).to_have_count(2)
    expect(page.get_by_role("link", name="Run against its store")).to_have_count(0)


def test_create_eval_set_prefills_store_name(page, seeded):
    documents_store(seeded, "prefillme", passages=False)
    page.goto(seeded.url + "/eval-sets?store=prefillme")
    expect(page.locator("#f-name")).to_have_value("prefillme")
