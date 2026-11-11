"""The Settings page: search defaults, connections, models and jobs."""

import httpx
import pytest
from page_helpers import (
    ai_state,
    search_defaults_restored,
)
from playwright.sync_api import expect

pytestmark = pytest.mark.pages


def test_settings_page_search_defaults_save_and_reset_to_recommended(page, seeded):
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        box = page.locator('#search [name="candidates"]')
        box.fill("30")
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["candidates"] == 30
        page.get_by_role("button", name="Reset to recommended").click()
        expect(page.locator("#search-msg")).to_have_text("Back to recommended")
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["candidates"] == 20
        expect(box).to_have_value("20")
        expect(box.locator("..").locator("[data-recommended]")).to_have_text("Recommended: 20")


def test_settings_page_says_what_each_job_does_and_what_works_without_ai(page, seeded):
    page.goto(seeded.url + "/settings")
    expect(page.locator("#head .sub")).to_have_text(
        "AI writes chat answers, rewrites follow-ups, and builds keyword tags. "
        "Search and eval runs on retrieval work without it."
    )
    expect(page.locator('#roles label:has([data-role="rewrite"]) .muted')).to_have_text(
        "Rewrites a chat follow-up as one standalone question"
    )
    expect(
        page.get_by_text("Used by chat and POST /chat, and by POST /query when no method is given.")
    ).to_be_visible()


def test_settings_jobs_use_labels(page, seeded):
    page.goto(seeded.url + "/settings")
    labels = {
        "default": "Default model",
        "answer": "Answers",
        "summary": "Summaries",
        "keywords": "Keyword lists and tags",
        "query": "Query keywords",
        "rewrite": "Follow-up rewriting",
    }
    for role, label in labels.items():
        row = page.locator(f'#roles label:has([data-role="{role}"])')
        expect(row).to_contain_text(label)
        expect(row.locator("select option").first).to_have_text(
            "none chosen" if role == "default" else "same as default"
        )
    expect(page.get_by_text("Each change is saved at once.")).to_be_visible()


def test_rewrite_note_follows_default(page, seeded):
    def resolved(rewrite):
        def handler(route):
            body = route.fetch().json()
            route.fulfill(json={**body, "resolved": {**body["resolved"], "rewrite": rewrite}})

        return handler

    note = page.locator(
        '#search label:has([name="rewrite"]) .muted', has_text="needs a rewrite model"
    )
    page.route("**/ai", resolved(False))
    page.goto(seeded.url + "/settings")
    expect(page.locator('#search [name="rewrite"]')).to_be_attached()
    expect(note).to_be_visible()
    page.unroute("**/ai")
    page.route("**/ai", resolved(True))
    page.reload()
    expect(page.locator('#search [name="rewrite"]')).to_be_attached()
    expect(note).to_be_hidden()


def test_settings_save_buttons_name_their_part(page, seeded):
    page.goto(seeded.url + "/settings")
    expect(page.get_by_role("button", name="Save budget", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Save search defaults", exact=True)).to_be_visible()
    expect(page.get_by_role("button", name="Save prices", exact=True).first).to_be_visible()
    expect(page.get_by_role("button", name="Save", exact=True)).to_have_count(0)
    expect(page.get_by_role("heading", name="Add a connection")).to_be_visible()
    expect(page.get_by_label("Connection id (a short name you choose)")).to_be_visible()


def test_settings_says_why_rerank_is_off_by_default(page, seeded):
    def with_note(route):
        body = route.fetch().json()
        route.fulfill(
            json={**body, "notes": {"rerank": "Reranking is off until the model is ready."}}
        )

    page.route("**/search-defaults", with_note)
    page.goto(seeded.url + "/settings")
    expect(page.locator("#search [data-rerank-note]")).to_have_text(
        "Reranking is off until the model is ready."
    )


def test_settings_page_picks_a_reranker_and_a_pool(page, seeded):
    listed = httpx.get(seeded.url + "/search-defaults").json()
    names = [r["name"] for r in listed["rerankers"]]
    own = listed["recommended"]["rerank_model"]  # the fixture's own, not fastembed's
    turbo = "jinaai/jina-reranker-v1-turbo-en"
    tiny_name = "jinaai/jina-reranker-v1-tiny-en"
    assert own not in names and turbo in names and tiny_name in names
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        select = page.locator('#search [name="rerank_model"]')
        expect(select.locator("option")).to_have_count(len(names) + 1)
        for name in [*names, own]:
            expect(select.locator(f'option[value="{name}"]')).to_have_count(1)
        expect(select.locator(f'option[value="{own}"]')).to_contain_text("(recommended)")
        tiny = select.locator(f'option[value="{tiny_name}"]')
        expect(tiny).to_contain_text("about 1.5 s for 20 passages on an M4 Pro")
        bge = select.locator('option[value^="BAAI/bge-reranker-"]').first
        expect(bge).to_contain_text("slower; for a faster machine")
        select.select_option(turbo)
        page.locator('#search [name="rerank_pool"]').fill("30")
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        expect(page.locator("#search [data-downloading]")).to_have_count(0)  # the fake's is ready
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["rerank_model"] == turbo
        assert got["in_force"]["rerank_pool"] == 30


def test_settings_shows_the_reranker_warm_up_the_service_reports(page, seeded):
    state = {"rerank_status": "loading", "rerank_error": None}
    puts: list[dict] = []

    def reported(route):
        if route.request.method == "PUT":
            puts.append(route.request.post_data_json)
        route.fulfill(json={**route.fetch().json(), **state})

    page.route("**/search-defaults", reported)
    page.goto(seeded.url + "/settings")
    hint, error = (
        page.locator("#search [data-downloading]"),
        page.locator("#search [data-rerank-error]"),
    )
    expect(hint).to_have_text(
        "Downloading in the background; chat merges by rank until it is ready."
    )
    expect(error).to_have_count(0)

    state.update(rerank_status="failed", rerank_error="hub down")
    page.reload()
    expect(error).to_have_text(
        "Could not load it: hub down. Chat merges by rank; Save tries again."
    )
    expect(hint).to_have_count(0)
    with search_defaults_restored(seeded):
        page.locator("#search-go").click()  # nothing edited: Save sends the reranker again
        expect(page.locator("#search-msg")).to_have_text("Saved")
        assert puts == [{"rerank_model": None}]  # as saved: the service's own

    state.update(rerank_status="ready", rerank_error=None)
    page.reload()
    expect(page.locator('#search [name="rerank_model"]')).to_be_visible()
    expect(hint).to_have_count(0)
    expect(error).to_have_count(0)


def add_connection(page, seeded, preset, id, key=""):
    page.goto(seeded.url + "/settings")
    page.locator("#add-preset").select_option(label=preset)
    page.locator("#add-id").fill(id)
    if key:
        page.locator("#add-key").fill(key)
    page.get_by_role("button", name="Add connection").click()
    row = page.locator(f'#connections [data-connection="{id}"]')
    expect(row).to_be_visible()
    return row


def test_settings_page_adds_a_connection_whose_models_need_a_price(page, seeded, monkeypatch):
    ai = seeded.app.state.ai
    monkeypatch.setattr(ai, "_list_models", lambda connection: ["llama3"])
    monkeypatch.setattr(
        ai, "_send", lambda connection, model_id, system, prompt, max_tokens: ("OK", 3, 1)
    )
    row = add_connection(page, seeded, "OpenRouter", "router", key="sk-or-TEST123456")
    expect(row).to_contain_text("OpenRouter")
    row.get_by_role("button", name="Fetch models").click()
    expect(page.locator('#models [data-model="router/llama3"]')).to_be_visible()
    # Fetched at $0, so the Test call can go at once.
    row.get_by_role("button", name="Test").click()
    expect(row.locator(".result")).to_contain_text("Worked with llama3")


def test_settings_page_shows_a_connection_answered_by_hand_without_key_or_test(page, seeded):
    row = add_connection(page, seeded, "Answered by hand", "hand")
    expect(row).to_contain_text("ragqa_core.replies")
    for name in ("Test", "Fetch models", "Save key"):
        expect(row.get_by_role("button", name=name)).to_have_count(0)


def test_settings_page_never_shows_key(page, seeded):
    row = add_connection(page, seeded, "OpenAI", "keyed", key="sk-proj-SECRET123456")
    expect(row).to_contain_text("…3456")
    assert "SECRET" not in page.content()
    assert page.locator("#add-key").input_value() == ""
    page.reload()
    expect(page.locator('#connections [data-connection="keyed"]')).to_be_visible()
    assert "SECRET" not in page.content()


def test_a_fetched_model_starts_at_zero_and_takes_a_listed_typed_or_no_price(
    page, seeded, monkeypatch
):
    monkeypatch.setattr(seeded.app.state.ai, "_list_models", lambda connection: ["gpt-new"])
    row = add_connection(page, seeded, "OpenAI", "remote", key="sk-remote-key-0000")
    row.get_by_role("button", name="Fetch models").click()
    model = page.locator('#models [data-model="remote/gpt-new"]')
    price = model.locator("select[name=price]")
    expect(price).to_have_value("0/0")  # $0 by default, and usable at once
    expect(model.locator("[data-free]")).to_contain_text("the budget does not limit it")
    expect(model.locator("input[name=price_in]")).to_be_hidden()
    roles = page.locator("#roles select").first
    expect(roles.locator("option", has_text="gpt-new")).to_have_count(1)

    def save():
        with page.expect_response(
            lambda r: r.request.method == "PUT" and "remote/gpt-new" in r.url
        ):
            model.get_by_role("button", name="Save prices").click()
        return next(m for m in ai_state(seeded)["models"] if m["ref"] == "remote/gpt-new")

    price.select_option(label="Haiku 5.5 rates: $0.1 in, $0.5 out")
    stored = save()
    assert (stored["price_in"], stored["price_out"]) == (0.1, 0.5)
    expect(model.locator("[data-free]")).to_have_count(0)
    price.select_option(label="Enter by hand…")
    expect(model.locator("input[name=price_in]")).to_be_visible()
    model.locator("input[name=price_in]").fill("1")
    model.locator("input[name=price_out]").fill("2")
    stored = save()
    assert (stored["price_in"], stored["price_out"]) == (1, 2)
    expect(price).to_have_value("hand")  # a typed price that is not listed stays typed
    price.select_option(label="No price (can't be used)")
    stored = save()
    assert stored["price_in"] is None
    expect(model).to_contain_text("needs a price before use")
    expect(roles.locator("option", has_text="gpt-new")).to_have_count(0)


def test_half_a_price_is_not_sent_and_typing_is_kept(page, seeded, monkeypatch):
    monkeypatch.setattr(seeded.app.state.ai, "_list_models", lambda connection: ["gpt-half"])
    row = add_connection(page, seeded, "OpenAI", "half", key="sk-half-key-00000")
    row.get_by_role("button", name="Fetch models").click()
    model = page.locator('#models [data-model="half/gpt-half"]')
    expect(model).to_be_visible()
    page.locator('#models [data-connection="anthropic"] input[name=new-id]').fill("typed-id")
    model.locator("select[name=price]").select_option(label="Enter by hand…")
    model.locator("input[name=price_in]").fill("1")
    model.locator("input[name=price_out]").fill("")  # the boxes start at the price in force
    model.get_by_role("button", name="Save prices").click()
    expect(page.locator("#msg")).to_contain_text("Give both prices")
    assert model.locator("input[name=price_in]").input_value() == "1"
    assert (
        page.locator('#models [data-connection="anthropic"] input[name=new-id]').input_value()
        == "typed-id"
    )
    stored = next(m for m in ai_state(seeded)["models"] if m["ref"] == "half/gpt-half")
    assert stored["price_in"] == 0  # still the default


def test_env_connection_has_no_remove(page, seeded):
    page.goto(seeded.url + "/settings")
    env = page.locator('#connections [data-connection="anthropic"]')
    expect(env).to_be_visible()
    expect(env.get_by_role("button", name="Delete connection…")).to_have_count(0)
    add_connection(page, seeded, "OpenRouter", "removable", key="sk-or-TEST123456")
    expect(
        page.locator('#connections [data-connection="removable"]').get_by_role(
            "button", name="Delete connection…"
        )
    ).to_have_count(1)


def test_delete_connection_asks_first(page, seeded):
    row = add_connection(page, seeded, "OpenRouter", "askdel", key="sk-or-TEST123456")
    row.get_by_role("button", name="Delete connection…").click()
    assert any(c["id"] == "askdel" for c in ai_state(seeded)["connections"])
    row.get_by_role("button", name="Cancel").click()
    expect(row.locator(".ask")).to_be_empty()
    row.get_by_role("button", name="Delete connection…").click()
    row.get_by_role("button", name="Yes, delete").click()
    expect(page.locator('#connections [data-connection="askdel"]')).to_have_count(0)
    assert not any(c["id"] == "askdel" for c in ai_state(seeded)["connections"])


def test_delete_connection_in_use_says_why(page, seeded):
    page.expected = ["409"]
    before = ai_state(seeded)["roles"].get("answer")
    add_connection(page, seeded, "OpenRouter", "inuse", key="sk-or-TEST123456")
    page.request.post(seeded.url + "/ai/connections/inuse/models", data={"model_id": "m"})
    page.request.put(seeded.url + "/ai/roles/answer", data={"model": "inuse/m"})
    try:
        page.reload()
        row = page.locator('#connections [data-connection="inuse"]')
        row.get_by_role("button", name="Delete connection…").click()
        row.get_by_role("button", name="Yes, delete").click()
        expect(row.locator(".ask .err")).to_contain_text("is used by answer")
        assert any(c["id"] == "inuse" for c in ai_state(seeded)["connections"])
    finally:  # seeded is module-scoped: leave it as found
        page.request.put(seeded.url + "/ai/roles/answer", data={"model": before})
        page.request.delete(seeded.url + "/ai/connections/inuse")


def test_empty_save_key_leaves_the_key_alone(page, seeded):
    row = add_connection(page, seeded, "OpenAI", "keep", key="sk-proj-KEEPME123456")
    expect(row.get_by_role("button", name="Save key")).to_be_disabled()
    before = next(c for c in ai_state(seeded)["connections"] if c["id"] == "keep")["key"]
    assert before
    row.locator("input[name=key]").fill("x")
    row.locator("input[name=key]").fill("")
    expect(row.get_by_role("button", name="Save key")).to_be_disabled()
    after = next(c for c in ai_state(seeded)["connections"] if c["id"] == "keep")["key"]
    assert after == before


def test_a_job_on_a_model_it_can_no_longer_use_shows_it_and_can_be_cleared(page, seeded):
    ai = seeded.app.state.ai
    ai.add_model("anthropic", "pinned-off", label="Pinned off", price_in=1, price_out=1)
    ai.add_model(
        "anthropic", "pinned-unpriced", label="Pinned unpriced", price_in=None, price_out=None
    )
    ai.set_role("answer", "anthropic/pinned-off")
    ai.set_role("summary", "anthropic/pinned-unpriced")
    ai.update_model("anthropic", "pinned-off", enabled=False)
    try:
        page.goto(seeded.url + "/settings")
        answer = page.locator('#roles select[data-role="answer"]')
        expect(answer.locator("option:checked")).to_have_text("Pinned off (disabled)")
        summary = page.locator('#roles select[data-role="summary"]')
        expect(summary.locator("option:checked")).to_have_text("Pinned unpriced (needs a price)")
        with page.expect_response(
            lambda r: r.request.method == "PUT" and r.url.endswith("/ai/roles/answer")
        ):
            answer.select_option(label="same as default")
        assert "answer" not in ai_state(seeded)["roles"]
        expect(answer.locator("option:checked")).to_have_text("same as default")
        expect(answer.locator("option", has_text="Pinned off")).to_have_count(0)
    finally:
        ai.set_role("answer", None)
        ai.set_role("summary", None)


def test_a_job_on_a_model_whose_connection_has_no_key_says_no_key(page, seeded):
    ai = seeded.app.state.ai
    ai.add_connection("nokey", "anthropic", "No key")
    ai.add_model("nokey", "m", label="Keyless", price_in=1, price_out=1)
    ai.set_role("answer", "nokey/m")
    try:
        page.goto(seeded.url + "/settings")
        answer = page.locator('#roles select[data-role="answer"]')
        expect(answer.locator("option:checked")).to_have_text("Keyless (no key)")
        expect(answer.locator("option", has_text="Keyless")).to_have_count(1)
    finally:  # seeded is session-scoped: leave it as found
        ai.set_role("answer", None)
        ai.remove_connection("nokey")


def test_default_job_set_by_the_environment_is_read_only(page, seeded):
    def env_default(route):
        response = route.fetch()
        route.fulfill(response=response, json={**response.json(), "default_from_env": True})

    page.route("**/ai", env_default)
    page.goto(seeded.url + "/settings")
    expect(page.locator('#roles select[data-role="default"]')).to_be_disabled()
    expect(page.locator("#roles")).to_contain_text("set by environment")
    expect(page.locator('#roles select[data-role="answer"]')).to_be_enabled()


def test_add_form_id_follows_the_server_rule(page, seeded):
    page.goto(seeded.url + "/settings")
    box = page.locator("#add-id")

    def valid(value):
        box.fill(value)
        return box.evaluate("e => e.checkValidity()")

    assert valid("gateway-2") and valid("a" * 32)
    assert not valid("a_b") and not valid("-a") and not valid("a" * 33)


def test_settings_page_picks_the_embedding_model_for_new_stores(page, seeded):
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        field = page.locator('#search label:has([name="dense_model"])')
        expect(field).to_contain_text("Embedding model for new stores")
        assert page.locator("#search label").first.inner_text().startswith("Embedding model")
        select = page.locator('#search [name="dense_model"]')
        small = select.locator('option[value^="BAAI/bge-small-en-v1.5"]')
        expect(small).to_contain_text("BAAI/bge-small-en-v1.5 · ")
        expect(small).to_contain_text("0.067 GB")
        expect(small).to_contain_text("fast")
        expect(select.locator('option[value="fake/dense"]')).to_contain_text("(recommended)")
        expect(field.locator("[data-recommended]")).to_have_text("Recommended: fake/dense")
        expect(field.locator("[data-embedder-note]")).to_have_text(
            "Existing stores keep theirs; rebuild one from its page."
        )
        select.select_option("BAAI/bge-base-en-v1.5")
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        got = httpx.get(seeded.url + "/search-defaults").json()
        assert got["in_force"]["dense_model"] == "BAAI/bge-base-en-v1.5"


def test_settings_sends_null_when_the_reranker_pick_is_the_recommended_one(page, seeded):
    listed = httpx.get(seeded.url + "/search-defaults").json()
    own = listed["recommended"]["rerank_model"]
    turbo = "jinaai/jina-reranker-v1-turbo-en"
    puts: list[dict] = []

    def record(route):
        if route.request.method == "PUT":
            puts.append(route.request.post_data_json)
        route.continue_()

    page.route("**/search-defaults", record)
    with search_defaults_restored(seeded):
        page.goto(seeded.url + "/settings")
        select = page.locator('#search [name="rerank_model"]')
        select.select_option(turbo)
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        select.select_option(own)
        page.locator("#search-go").click()
        expect(page.locator("#search-msg")).to_have_text("Saved")
        assert puts == [{"rerank_model": turbo}, {"rerank_model": None}]


def test_settings_draws_connections_when_catalog_fails(page, seeded):
    page.expected = ["500"]

    def with_connection(route):
        reply = route.fetch().json()
        mine = {"id": "kept", "label": "Kept", "kind": "manual", "base_url": None, "from_env": []}
        route.fulfill(json={**reply, "connections": [mine]})

    page.route("**/ai", with_connection)
    page.route(
        "**/catalog", lambda route: route.fulfill(status=500, json={"detail": "catalog down"})
    )
    page.goto(seeded.url + "/settings")
    expect(page.locator('#connections [data-connection="kept"]')).to_be_visible()
    expect(page.locator("#msg")).to_have_text("catalog down")


def test_a_double_click_on_test_sends_one_request(page, seeded):
    held = []
    page.route("**/ai/connections/*/test", lambda route: held.append(route))
    page.goto(seeded.url + "/settings")
    row = page.locator('#connections [data-connection="anthropic"]')
    test = row.get_by_role("button", name="Test")
    test.click()
    expect(test).to_be_disabled()  # the mechanism: the second click lands on a disabled button
    test.click(force=True)
    held[0].fulfill(json={"ok": True, "model": "m", "usd": 0})
    expect(row.locator(".result")).to_contain_text("Worked with m")
    assert len(held) == 1


def test_a_double_click_on_save_prices_sends_one_request(page, seeded):
    held = []
    page.route("**/ai/models/**", lambda route: held.append(route))
    page.goto(seeded.url + "/settings")
    model = page.locator('#models [data-connection="anthropic"] .model').first
    save = model.get_by_role("button", name="Save prices")
    save.click()
    expect(save).to_be_disabled()  # the mechanism: the second click lands on a disabled button
    save.click(force=True)
    held[0].continue_()  # the row saves as it is, so the shared app is unchanged
    expect(save).to_be_enabled()
    assert len(held) == 1
