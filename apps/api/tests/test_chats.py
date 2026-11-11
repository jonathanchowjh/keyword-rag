"""The chat routes, and the one-shot /query and /chat."""

import pytest
from api_helpers import (
    add_collection,
    add_documents,
    add_text,
    chat_body,
    documents_with_passages,
    ids_of,
    long_text,
    make_client,
    new_chat,
    record_queries,
    reembed,
    reference_with_vocabulary,
    small_model,
    source_ids,
    store_import,
    wait,
)

from ragqa_core import db, search_defaults
from ragqa_core.fakes import ITEMS, fake_replies
from ragqa_core.prompts import CHAT_TOKENS
from ragqa_core.types import Item, Question


def test_create_add_query_flow(client):
    add_collection(client)
    body = {"collection": "c", "question": "enzyme rate", "methods": ["hybrid", "kw_individual"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["hybrid"]["hits"][0]["id"] == "enzyme-item"
    assert r["results"]["kw_individual"]["hits"][0]["matched"]
    assert r["results"]["hybrid"]["answer"] is None and r["results"]["hybrid"]["error"] is None


def test_query_with_answer(client):
    add_collection(client)
    body = {
        "collection": "c",
        "question": "enzyme rate?",
        "choices": ["x", "y"],
        "methods": ["hybrid"],
        "answer": True,
    }
    result = client.post("/query", json=body).json()["results"]["hybrid"]
    assert result["answer"] == "A" and result["usd"] > 0


def test_query_on_a_science_collection_without_choices_returns_free_text(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Answer: 3.1 J") as client:
        client.post("/collections", json={"name": "sci", "qa_type": "science", "summarize": False})
        client.post("/collections/sci/documents", json={"items": ITEMS, "confirm": True})
        body = {"collection": "sci", "question": "enzyme rate?", "methods": ["vector"]}
        result = client.post("/query", json={**body, "answer": True}).json()["results"]["vector"]
    assert result["answer"] == "3.1 J"


def test_query_answers_from_the_program_and_the_example_setting(tmp_path):
    sent = []

    def reply(prompt):
        sent.append(prompt)
        return "plan\n```python\nprint(6*7)\n```\nAnswer: 41"

    with make_client(tmp_path, reply=reply) as client:
        client.post("/collections", json={"name": "sci", "qa_type": "science", "summarize": False})
        client.post("/collections/sci/documents", json={"items": ITEMS, "confirm": True})
        body = {"collection": "sci", "question": "enzyme rate?", "methods": ["vector"]}
        ask = {**body, "answer": True, "search": {"program": True, "example": "off"}}
        result = client.post("/query", json=ask).json()["results"]["vector"]
        assert result["answer"] == "42" and result["program"]["ran"] is True
        assert "```python" in sent[-1] and "Example" not in sent[-1]
        chosen = {**ask, "choices": ["x", "y"], "search": {"program": True}}
        result = client.post("/query", json=chosen).json()["results"]["vector"]
        assert result["answer"] == "" and result["program"] is None
        assert "```python" not in sent[-1] and "Example problem" in sent[-1]


def test_query_on_other_types_sends_the_prompt_it_always_did(tmp_path):
    """Byte for byte what /query built before it shared the run's answer step: answer_prompt
    over answer_hits, with answer_tokens."""
    from ragqa_core.evaluator import answer_hits, answer_tokens
    from ragqa_core.prompts import answer_prompt
    from ragqa_core.types import Hit

    reply = "plan\n```python\nprint(2)\n```\nAnswer: A"
    with make_client(tmp_path, reply=lambda p: reply) as client:
        add_collection(client)
        client.post("/collections", json={"name": "mh", "qa_type": "multihop", "summarize": False})
        r = client.post("/collections/mh/documents", json={"items": ITEMS, "confirm": True})
        assert r.status_code == 200, r.text
        calls = record_queries(client)
        asks = [
            ("c", "mcq", ["x", "y"], {}),
            ("mh", "multihop", None, {"program": True}),
        ]
        for name, qa_type, choices, search in asks:
            body = {"collection": name, "question": "enzyme rate?", "choices": choices}
            body |= {"methods": ["vector"], "answer": True, "search": search}
            result = client.post("/query", json=body).json()["results"]["vector"]
            assert result["error"] is None and result["answer"]
            hits = [Hit(**h) for h in result["hits"]]
            spec = client.app.state.retriever.spec(name)
            example = search.get("example", spec.search.example)
            program = search.get("program", spec.search.program) and not choices
            system, user = answer_prompt(
                qa_type,
                "enzyme rate?",
                choices,
                answer_hits(qa_type, hits, example),
                example,
                search.get("program", spec.search.program),
            )
            assert calls[-1] == (system, user, answer_tokens(qa_type, program))
        assert calls[-1][2] > answer_tokens("multihop", False)  # the program's budget


def test_query_documents_answers_like_a_run(client):
    """A Documents store's /query answers as its run does: chat's prompt over the in-force
    evidence size, CHAT_SYSTEM (with no passages for question_only), CHAT_TOKENS, and the
    reply shown without its [n] markers."""
    from ragqa_core.evaluator import run_prompt
    from ragqa_core.types import Hit, SearchSettings

    add_documents(client)
    passages = [Item(f"p{n}", f"alpha beta passage {n}") for n in range(1, 7)]
    client.app.state.indexer.add(
        db.get_collection(client.app.state.indexer.conn, "c"), passages, "upload:docs"
    )
    search_defaults.save(client.app.state.conn, {"evidence_size": 3})
    fake_replies(client.app.state.ai, lambda p: "Ten days [1].")
    calls = record_queries(client)
    body = {"collection": "c", "question": "alpha beta", "methods": ["hybrid", "question_only"]}
    results = client.post("/query", json={**body, "k": 6, "answer": True}).json()["results"]
    question = Question("lab", "alpha beta", [], [], [])
    search = client.app.state.retriever.spec("c").search or SearchSettings()
    for method, sent in zip(body["methods"], calls, strict=True):
        result = results[method]
        assert result["error"] is None
        hits = [Hit(**h) for h in result["hits"]]
        assert len(hits) == (6 if method == "hybrid" else 0)
        retrieves = method != "question_only"
        system, user = run_prompt("documents", question, hits[:3], search, retrieves)
        assert sent == (system, user, CHAT_TOKENS)
        assert result["answer"] == "Ten days." and result["program"] is None
    assert len(calls) == 2


def test_query_documents_answer_reads_the_evidence_size(client):
    add_documents(client)
    store_import(client, "x.md", long_text([f"w{i}" for i in range(10)]))
    assert client.app.state.store.count("c") == 10
    fake_replies(client.app.state.ai, lambda p: "Ten days [1].")
    q = {"collection": "c", "question": "w1", "methods": ["hybrid"], "k": 5}
    hits = client.post("/query", json={**q, "answer": True}).json()["results"]["hybrid"]["hits"]
    assert len(hits) == 8  # Settings' evidence size, the recommended 8
    hits = client.post("/query", json=q).json()["results"]["hybrid"]["hits"]
    assert len(hits) == 5


def test_query_usd_counts_every_claude_call_of_the_method(client):
    add_collection(client, summarize=True)
    body = {"collection": "c", "question": "enzyme rate", "methods": ["kw_individual", "hybrid"]}
    results = client.post("/query", json=body).json()["results"]
    assert results["kw_individual"]["usd"] > 0  # the question's summary
    assert results["hybrid"]["usd"] == 0


def test_a_query_answered_by_hand_says_its_prompt_waits(client):
    add_collection(client)
    ai = client.app.state.ai
    client.post("/ai/connections", json={"preset": "manual", "id": "hand"})
    client.post(
        "/ai/connections/hand/models", json={"model_id": "m", "price_in": 0, "price_out": 0}
    )
    ai.set_role("answer", "hand/m")
    body = {"collection": "c", "question": "enzyme?", "choices": ["x", "y"], "answer": True}
    result = client.post("/query", json={**body, "methods": ["hybrid"]}).json()["results"]
    assert "1 prompt wait for replies by hand" in result["hybrid"]["error"]
    [waiting] = ai.waiting()
    ai.answer({waiting["key"]: "Answer: B"})
    result = client.post("/query", json={**body, "methods": ["hybrid"]}).json()["results"]
    assert result["hybrid"]["answer"] == "B" and result["hybrid"]["usd"] == 0


def test_query_reports_unavailable_method_per_method(client):
    add_collection(client, lists=False)
    body = {"collection": "c", "question": "x", "methods": ["hybrid", "kw_all"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["hybrid"]["error"] is None
    assert "keyword lists not set" in r["results"]["kw_all"]["error"]


def test_query_without_key_keeps_free_methods(client):
    add_collection(client, summarize=True)
    client.app.state.ai.update_connection("anthropic", api_key=None)
    body = {"collection": "c", "question": "enzyme", "methods": ["keyword", "kw_individual"]}
    r = client.post("/query", json=body).json()
    assert r["results"]["keyword"]["hits"]
    assert "'query' job" in r["results"]["kw_individual"]["error"]


def test_query_unknown_method_422(client):
    add_collection(client)
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["nope"]})
    assert r.status_code == 422


def test_query_returns_channels_and_tags(client):
    reference_with_vocabulary(client)
    assert "keyword_rag" in client.get("/collections/r").json()["methods_available"]
    body = {"collection": "r", "question": "enzyme rate", "methods": ["keyword_rag", "hybrid"]}
    results = client.post("/query", json=body).json()["results"]
    found = results["keyword_rag"]
    assert found["error"] is None and found["query_tags"][0] == "enzyme"
    top = found["hits"][0]
    assert top["id"] == "enzyme-item" and top["tags"][0] == "enzyme"
    assert top["channels"] == ["meaning", "words", "keywords"] and "enzyme" in top["matched"]
    assert "query_tags" not in results["hybrid"]
    weighted = {**body, "search": {"weights": {"meaning": 0, "words": 0}}}
    found = client.post("/query", json=weighted).json()["results"]["keyword_rag"]
    assert {tuple(h["channels"]) for h in found["hits"]} == {("keywords",)}


def test_query_model_tags_are_the_ais_pick_and_counted_in_its_usd(client):
    reference_with_vocabulary(client)
    fake_replies(client.app.state.ai, lambda p: '{"terms": ["volcano"]}')
    body = {"collection": "r", "question": "enzyme rate", "methods": ["keyword_rag"]}
    found = client.post("/query", json={**body, "search": {"query_tags": "model"}}).json()
    found = found["results"]["keyword_rag"]
    assert found["error"] is None and found["query_tags"] == ["volcano"] and found["usd"] > 0


def test_query_oracle_is_422(client):
    reference_with_vocabulary(client)
    body = {"collection": "r", "question": "enzyme", "methods": ["keyword_rag"]}
    r = client.post("/query", json={**body, "search": {"query_tags": "oracle"}})
    assert r.status_code == 422 and "oracle tags need a question's gold" in r.json()["detail"]


def test_query_keyword_rag_without_vocabulary_is_a_method_error(client):
    add_collection(client)
    body = {"collection": "c", "question": "enzyme", "methods": ["keyword_rag", "hybrid"]}
    results = client.post("/query", json=body).json()["results"]
    assert "no vocabulary for store 'c'" in results["keyword_rag"]["error"]
    assert results["hybrid"]["hits"]


def test_a_store_keeps_its_model_when_the_services_default_changes(client):
    add_collection(client)
    client.app.state.models.dense_model = "other/model"
    body = {"collection": "c", "question": "enzyme", "methods": ["hybrid"]}
    result = client.post("/query", json=body).json()["results"]["hybrid"]
    assert result["error"] is None and result["hits"]


def test_a_recreated_collection_does_not_reuse_the_controls_id_cache(client):
    add_collection(client, lists=False, questions=False)
    query = {"collection": "c", "question": "enzyme", "methods": ["random_example"], "k": 3}

    def hit_ids() -> set[str]:
        hits = client.post("/query", json=query).json()["results"]["random_example"]["hits"]
        return {h["id"] for h in hits}

    assert hit_ids() == {i["id"] for i in ITEMS}
    assert client.delete("/collections/c").status_code == 204
    client.post("/collections", json={"name": "c", "qa_type": "mcq"})
    fresh = [{**i, "id": "new-" + i["id"]} for i in ITEMS]
    client.post("/collections/c/documents", json={"items": fresh, "confirm": True})
    assert hit_ids() == {i["id"] for i in fresh}


def test_a_long_text_upload_adds_one_item_per_chunk_and_chat_cites_the_part(tmp_path):
    reply = lambda p: " ".join(f"[{i}]" for i in range(1, 21))  # noqa: E731
    with make_client(tmp_path, reply=reply) as client:
        add_collection(client, questions=False, lists=False)
        body = long_text(["alpha", "brine", "cocoa"])
        assert len(body) > 2900
        ids = ids_of("x.md", body)
        job = add_text(client, "x.md", body)
        assert job["result"]["new"] == 3 and job["result"]["removed"] == 0
        assert source_ids(client, "upload:docs") == sorted(ids)
        again = add_text(client, "x.md", body)  # the same file: the same ids, nothing new
        assert again["result"]["unchanged"] == 3 and again["result"]["new"] == 0
        assert again["result"]["removed"] == 0
        assert source_ids(client, "upload:docs") == sorted(ids)
        chat = {**chat_body("brine"), "method": "vector", "k": 20}
        got = client.post("/chat", json=chat).json()
        part = next(c for c in got["citations"] if c["id"] == ids[1])
        assert (part["part"], part["parts"], part["source"]) == (2, 3, "upload:docs")
        assert part["text"].startswith("x.md\n\nbrine brine")
        whole = next(c for c in got["citations"] if c["id"] == "enzyme-item")
        assert whole["part"] is None and whole["parts"] is None


def test_an_unreachable_provider_is_one_methods_error_in_a_query(client):
    # Each method reports its own error, so what the others spent is still shown.
    add_collection(client)

    def down(*args, **kwargs):
        raise ConnectionError("connection refused")

    client.app.state.ai._send = down
    r = client.post(
        "/query",
        json={"collection": "c", "question": "enzyme rate", "methods": ["hybrid"], "answer": True},
    )
    assert r.status_code == 200, r.text
    assert "connection refused" in r.json()["results"]["hybrid"]["error"]


def test_chat_answers_with_citations(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Rates rise [1]. Answer: x") as client:
        add_collection(client)
        r = client.post("/chat", json=chat_body())
        assert r.status_code == 200, r.text
        got = r.json()
        assert got["answer"].startswith("Rates rise") and got["error"] is None
        assert got["citations"][0]["id"] == got["hit_ids"][0] == "enzyme-item"
        assert got["citations"][0]["n"] == 1 and "enzyme" in got["citations"][0]["text"]
        assert got["usd"] > 0 and got["method"] == "hybrid" and got["query_tags"] is None


@pytest.mark.parametrize(
    ("answer", "cited"),
    [
        ("Rates rise [1, 2].", [1, 2]),
        ("Rates rise [2-3].", [2, 3]),
        ("Rates rise [3–2].", []),  # a range read backwards cites nothing
        ("Rates rise [1]. Again [1].", [1]),
        ("Rates rise [9] and [3].", [3]),  # 3 passages: 9 is not one of them
        ("Rates rise [2] [1, 3].", [2, 1, 3]),
        ("Rates rise [\u0661].", []),  # an Arabic-Indic one: not a digit to the page either
        # too long a run to be a mark, and no raise
        pytest.param("Rates rise [" + "1" * 5000 + "].", [], id="5000-digits"),
    ],
)
def test_chat_cites_lists_and_ranges(tmp_path, answer, cited):
    with make_client(tmp_path, reply=lambda p: answer) as client:
        add_collection(client)
        ai = client.app.state.ai
        complete, calls = ai.complete, []

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return complete(*args, **kwargs)

        ai.complete = spy
        got = client.post("/chat", json=chat_body()).json()
    hits = got["hit_ids"]
    assert len(hits) == 3
    shown = hits[0::2] + hits[1::2][::-1]  # the passages are numbered best first and last
    assert [c["n"] for c in got["citations"]] == cited
    assert [c["id"] for c in got["citations"]] == [shown[n - 1] for n in cited]
    assert [(c["purpose"], c["max_tokens"]) for c in calls] == [("chat", CHAT_TOKENS)]


def test_chat_cites_what_the_model_saw(tmp_path):
    with make_client(tmp_path, reply=lambda p: "Long [1]") as client:
        add_collection(client, name="long", questions=False, lists=False)
        long_text = "enzyme " + "x" * 4993
        r = client.post(
            "/collections/long/documents",
            json={"items": [{"id": "big", "text": long_text}], "confirm": True},
        )
        assert r.status_code == 200, r.text
        body = {**chat_body("enzyme x"), "collection": "long", "method": "vector"}
        got = client.post("/chat", json=body).json()
        big = [c for c in got["citations"] if c["id"] == "big"]
        assert big and len(big[0]["text"]) == 1500


def test_chat_method_defaults_and_refusals(client):
    add_collection(client)
    assert client.post("/chat", json=chat_body()).json()["method"] == "hybrid"
    allowed = client.post("/chat", json=chat_body(method="kw_individual"))
    assert allowed.status_code == 200
    add_collection(client, name="bare", lists=False)
    r = client.post("/chat", json={**chat_body(method="kw_individual"), "collection": "bare"})
    assert r.status_code == 409 and "needs keyword lists" in r.json()["detail"]
    assert client.post("/chat", json=chat_body(method="nope")).status_code == 422
    assert client.post("/chat", json=chat_body(method="hybrid_rerank")).status_code == 422
    reference_with_vocabulary(client)
    body = {**chat_body(), "collection": "r"}
    got = client.post("/chat", json=body).json()
    assert got["method"] == "hybrid" and got["query_tags"] is None  # the store's chain
    body = {**body, "method": "keyword_rag"}
    got = client.post("/chat", json=body).json()
    assert got["method"] == "keyword_rag" and got["query_tags"][0] == "enzyme"
    put = client.put("/collections/r/search", json={"query_tags": "oracle"})
    assert put.status_code == 200
    r = client.post("/chat", json=body)
    assert r.status_code == 200 and r.json()["error"] is None and r.json()["query_tags"]


def test_post_chat_refuses_a_paid_method_with_no_ai_and_reports_an_unpriced_model(client):
    add_collection(client, summarize=True)  # kw_individual summarises the query: paid
    ai = client.app.state.ai
    ai.add_model("anthropic", "unpriced-model", price_in=None, price_out=None)
    got = client.post("/chat", json=chat_body(model="anthropic/unpriced-model"))
    assert got.status_code == 200, got.text
    assert got.json()["answer"] is None and "unpriced-model" in got.json()["error"]
    ai.set_role("default", None)
    r = client.post("/chat", json=chat_body(method="kw_individual"))
    assert r.status_code == 503 and "needs AI" in r.json()["detail"]


def test_chat_needs_ai_is_503(client):
    add_collection(client, summarize=True)  # kw_individual summarises the query: paid
    add_collection(client, name="bare", lists=False)
    chat = new_chat(client)
    assert client.post(f"/chats/{chat['id']}/messages", json={"content": "rate"}).status_code == 200
    client.app.state.ai.set_role("default", None)  # no AI resolves
    r = client.post("/chat", json=chat_body(method="kw_individual"))
    assert (r.status_code, r.json()["detail"]) == (503, "kw_individual needs AI on 'c'")
    # A method the store can't run is still 409: the store, not the model, is missing it.
    r = client.post("/chat", json={**chat_body(method="kw_individual"), "collection": "bare"})
    assert r.status_code == 409 and "needs AI" not in r.json()["detail"]
    # A chat's own method is refused the same way when a re-run sets it.
    url = f"/chats/{chat['id']}/messages/1/attempts"
    r = client.post(url, json={"stores": {"c": {"method": "kw_individual"}}})
    assert (r.status_code, r.json()["detail"]) == (503, "kw_individual needs AI on 'c'")


def chat_spend(client) -> float:
    rows = client.app.state.ai.sql("select sum(usd) as usd from ai_ledger where purpose = 'chat'")
    return rows[0]["usd"] or 0.0


def test_post_chat_uses_the_settings_and_the_pipeline(client):
    add_collection(client)
    conn = client.app.state.conn
    search_defaults.save(conn, {"method": "vector"})
    before = chat_spend(client)
    got = client.post("/chat", json=chat_body()).json()
    assert got["method"] == "vector" and got["error"] is None
    assert isinstance(got["notes"], list) and got["notes"] == []
    assert got["usd"] > 0 and got["usd"] == pytest.approx(chat_spend(client) - before)
    assert got["hit_ids"][0] == "enzyme-item"  # reranked against the question
    search_defaults.save(conn, {"evidence_size": 2})
    got = client.post("/chat", json=chat_body()).json()
    assert 0 < len(got["hit_ids"]) <= 2
    assert len(client.post("/chat", json=chat_body(k=3)).json()["hit_ids"]) == 3
    r = client.post("/chat", json=chat_body(method="hybrid_rerank"))
    assert r.status_code == 422 and "is not a chat method" in r.json()["detail"]
    assert client.post("/chat", json=chat_body(method="question_only")).status_code == 422
    client.app.state.models.rerank_status = lambda model=None: "loading"  # type: ignore[attr-defined]
    got = client.post("/chat", json=chat_body()).json()
    assert got["notes"] == ["reranker still loading: merged by rank"] and got["answer"]


def test_query_without_methods_uses_the_settings_method(client):
    add_collection(client)
    search_defaults.save(client.app.state.conn, {"method": "keyword", "rrf_k": 5})
    retriever, seen = client.app.state.retriever, []
    real = retriever.retrieve

    def retrieve(collection, question, method, k, rrf_k, *args, **kw):
        seen.append((method, rrf_k))
        return real(collection, question, method, k, rrf_k, *args, **kw)

    retriever.retrieve = retrieve
    r = client.post("/query", json={"collection": "c", "question": "enzyme rate"})
    assert r.status_code == 200, r.text
    assert r.json()["results"].keys() == {"keyword"} and seen == [("keyword", 5)]
    body = {"collection": "c", "question": "enzyme rate", "methods": ["vector"], "rrf_k": 9}
    assert client.post("/query", json=body).json()["results"].keys() == {"vector"}
    assert seen[-1] == ("vector", 9)


def test_chat_over_budget_is_that_replys_error(tmp_path):
    with make_client(tmp_path, cap_usd=0) as client:
        add_collection(client)
        r = client.post("/chat", json=chat_body())
        assert r.status_code == 200
        got = r.json()
        assert got["answer"] is None and "cap" in got["error"]


def test_chat_sends_only_the_last_six_turns(tmp_path):
    seen = []

    def reply(prompt):
        seen.append(prompt)
        return "ok"

    with make_client(tmp_path, reply=reply) as client:
        add_collection(client)
        turns = [
            {"role": "user" if i % 2 else "assistant", "content": f"turn {i}"} for i in range(1, 9)
        ]
        last = {"role": "user", "content": "enzyme rate"}
        r = client.post("/chat", json={"collection": "c", "messages": [*turns, last]})
        assert r.status_code == 200, r.text
    prompt = seen[-1]
    assert "turn 2" not in prompt and "turn 3" in prompt and "turn 8" in prompt


def test_chat_needs_the_user_to_speak_last(client):
    add_collection(client)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert client.post("/chat", json={"collection": "c", "messages": messages}).status_code == 422
    assert client.post("/chat", json={"collection": "c", "messages": []}).status_code == 422


def test_chats_crud_and_errors(client):
    add_collection(client)
    assert client.get("/chats").json() == []
    chat = new_chat(client, title="Mine")
    assert chat["title"] == "Mine" and chat["stores"] == ["c"]
    assert client.get("/chats").json()[0]["id"] == chat["id"]
    got = client.get(f"/chats/{chat['id']}").json()
    assert got["messages"] == []
    assert got["settings"] == {"evidence_size": None, "rewrite": None, "rerank": None, "stores": {}}
    r = client.patch(
        f"/chats/{chat['id']}", json={"title": "New", "evidence_size": 3, "rewrite": True}
    )
    assert r.status_code == 200 and r.json()["title"] == "New"
    assert r.json()["settings"]["evidence_size"] == 3 and r.json()["settings"]["rewrite"] is True
    for body in ({"evidence_size": 0}, {"evidence_size": 21}, {"stores": []}, {"title": " "}):
        assert client.patch(f"/chats/{chat['id']}", json=body).status_code == 422
    assert client.patch(f"/chats/{chat['id']}", json={"stores": ["nope"]}).status_code == 404
    # A null title or stores is refused, not ignored: leave a field out to keep it.
    for field in ("title", "stores"):
        r = client.patch(f"/chats/{chat['id']}", json={field: None})
        assert r.status_code == 422, field
        assert field in r.json()["detail"][0]["msg"]  # r.text echoes the input: no test
    assert client.post("/chats", json={"stores": []}).status_code == 422
    assert client.post("/chats", json={"stores": ["nope"]}).status_code == 404
    assert client.get("/chats/nope").status_code == 404
    assert client.patch("/chats/nope", json={"title": "x"}).status_code == 404
    assert client.delete("/chats/nope").status_code == 404
    assert client.post(f"/chats/{chat['id']}/messages", json={"content": ""}).status_code == 422
    for typo in ({"answers": False}, {"Answer": False}):  # refused, not answered and paid for
        r = client.post(f"/chats/{chat['id']}/messages", json={"content": "hi", **typo})
        assert r.status_code == 422, typo
    assert client.get(f"/chats/{chat['id']}").json()["messages"] == []
    assert client.post("/chats/nope/messages", json={"content": "hi"}).status_code == 404
    assert client.post(f"/chats/{chat['id']}/messages/9/attempts", json={}).status_code == 404
    assert (
        client.post(f"/chats/{chat['id']}/messages/9/attempts/1/answer", json={}).status_code == 404
    )
    assert client.post(f"/chats/{chat['id']}/stores/nope/default").status_code == 404
    assert client.delete(f"/chats/{chat['id']}").status_code == 204
    assert client.get(f"/chats/{chat['id']}").status_code == 404


def test_chat_message_returns_decisions_evidence_and_cost(client):
    add_collection(client)
    chat = new_chat(client)
    r = client.post(f"/chats/{chat['id']}/messages", json={"content": "enzyme rate"})
    assert r.status_code == 200, r.text
    message = r.json()
    assert message["n"] == 1 and message["content"] == "enzyme rate"
    (attempt,) = message["attempts"]
    assert attempt["attempt"] == 1 and attempt["answer"] and attempt["usd"] > 0
    assert attempt["decisions"][0]["store"] == "c" and attempt["decisions"][0]["error"] is None
    assert attempt["evidence"] and attempt["evidence"][0]["store"] == "c"
    assert client.get(f"/chats/{chat['id']}").json()["messages"][0]["attempts"] == [attempt]
    assert client.get(f"/chats/{chat['id']}").json()["in_force"]["stores"]["c"]["candidates"] == {
        "value": 20,
        "from": "recommended",
    }
    r = client.post(f"/chats/{chat['id']}/messages/1/attempts", json={})
    assert r.status_code == 200 and r.json()["attempt"] == 2 and r.json()["answer"] is None
    url = f"/chats/{chat['id']}/messages/1/attempts"
    assert client.post(f"{url}/2/answer", json={}).json()["answer"]
    assert client.post(f"{url}/2/answer", json={}).status_code == 409  # already answered
    assert client.post(f"{url}/1/answer", json={}).status_code == 409
    assert client.post(f"{url}/3/answer", json={}).status_code == 404
    bad = client.post(f"/chats/{chat['id']}/messages", json={"content": "x", "model": "no/such"})
    assert bad.status_code == 422
    assert len(client.get(f"/chats/{chat['id']}").json()["messages"]) == 1


def test_chat_attempt_override_rejects_a_method_the_store_cannot_run(client):
    add_collection(client, name="bare", lists=False)
    chat = new_chat(client, stores=["bare"])
    client.post(f"/chats/{chat['id']}/messages", json={"content": "enzyme"})
    url = f"/chats/{chat['id']}/messages/1/attempts"
    r = client.post(url, json={"stores": {"bare": {"method": "kw_individual"}}})
    assert r.status_code == 409 and "needs keyword lists" in r.json()["detail"]
    for bad in (
        {"candidates": 0},
        {"candidates": 61},
        {"method": "nope"},
        {"method": "hybrid_rerank"},
        {"k": 2},
        {"search": {"example": "off"}},
    ):
        r = client.post(url, json={"stores": {"bare": bad}})
        assert r.status_code == 422, bad
    for body in (
        {"overrides": {}},
        {"settings": {"colour": 1}},
        {"settings": {"evidence_size": 21}},
    ):
        assert client.post(url, json=body).status_code == 422, body
    r = client.post(url, json={"stores": {"other": {"candidates": 2}}})
    assert r.status_code == 404
    r = client.post(url, json={"stores": {"bare": {"candidates": 2, "method": None}}})
    assert r.status_code == 200 and r.json()["decisions"][0]["candidates"] == 2


def test_set_as_default_from_a_chat_changes_the_store_method_and_candidates(client):
    add_collection(client)
    chat = new_chat(client)
    client.post(f"/chats/{chat['id']}/messages", json={"content": "enzyme rate"})
    search = client.get("/collections/c").json()["search"]
    url = f"/chats/{chat['id']}/messages/1/attempts"
    r = client.post(url, json={"stores": {"c": {"method": "vector", "candidates": 12}}})
    assert r.status_code == 200, r.text
    assert client.get("/collections/c").json()["chat_k"] is None  # not saved yet
    r = client.post(f"/chats/{chat['id']}/stores/c/default")
    assert r.status_code == 200, r.text
    from_store = {
        "method": {"value": "vector", "from": "store"},
        "candidates": {"value": 12, "from": "store"},
    }
    assert r.json() == {"chat_method": "vector", "chat_k": 12, "in_force": from_store}
    detail = client.get("/collections/c").json()
    assert (detail["chat_method"], detail["chat_k"]) == ("vector", 12)
    assert detail["chat_settings"] == from_store
    assert detail["search"] == search  # a chat never edits a store's search
    got = client.get(f"/chats/{chat['id']}").json()
    assert got["in_force"]["stores"]["c"] == from_store
    assert got["settings"]["stores"]["c"] == {"method": None, "candidates": None}


def test_set_as_default_refuses_a_method_the_store_cannot_run(client):
    add_collection(client)
    chat = new_chat(client)
    conn = client.app.state.conn
    settings = {"stores": {"c": {"method": "keyword_rag", "candidates": None}}}
    db.update_chat(conn, chat["id"], settings=settings)
    r = client.post(f"/chats/{chat['id']}/stores/c/default")
    assert r.status_code == 409, r.text
    assert "keyword_rag needs a vocabulary on 'c'" in r.json()["detail"]
    assert client.get("/collections/c").json()["chat_method"] is None
    assert db.get_chat(conn, chat["id"])["settings"] == settings  # nothing cleared


def test_chat_patch_takes_null_as_inherit_and_refuses_a_null_title(client):
    add_collection(client)
    chat = new_chat(client)
    url = f"/chats/{chat['id']}"
    r = client.patch(url, json={"evidence_size": 5, "rewrite": True, "rerank": False})
    assert r.status_code == 200, r.text
    assert r.json()["settings"] == {
        "evidence_size": 5,
        "rewrite": True,
        "rerank": False,
        "stores": {},
    }
    force = client.get(url).json()["in_force"]
    assert force["evidence_size"] == {"value": 5, "from": "chat"}
    assert force["rerank"] == {"value": False, "from": "chat"}
    r = client.patch(url, json={"evidence_size": None, "rewrite": None, "rerank": None})
    assert r.status_code == 200, r.text
    force = client.get(url).json()["in_force"]
    assert force["evidence_size"] == {"value": 8, "from": "recommended"}
    assert force["rewrite"] == {"value": False, "from": "settings"}  # make_client turns it off
    assert force["rerank"] == {"value": True, "from": "recommended"}
    for field in ("title", "stores"):
        r = client.patch(url, json={field: None})
        assert r.status_code == 422, field
        assert field in r.json()["detail"][0]["msg"]
    assert client.get(url).json()["title"] == "New chat"


def test_chat_attempt_saves_its_settings_with_the_chat(client):
    add_collection(client)
    chat = new_chat(client)
    client.post(f"/chats/{chat['id']}/messages", json={"content": "enzyme rate"})
    url = f"/chats/{chat['id']}/messages/1/attempts"
    body = {
        "settings": {"evidence_size": 1},
        "stores": {"c": {"method": "vector", "candidates": 5}},
    }
    r = client.post(url, json=body)
    assert r.status_code == 200, r.text
    ran = r.json()["decisions"][0]
    assert (ran["method"], ran["candidates"]) == ("vector", 5) and len(r.json()["evidence"]) == 1
    got = client.get(f"/chats/{chat['id']}").json()
    assert got["settings"] == {
        "evidence_size": 1,
        "rewrite": None,
        "rerank": None,
        "stores": {"c": {"method": "vector", "candidates": 5}},
    }
    assert got["in_force"]["stores"]["c"]["candidates"] == {"value": 5, "from": "chat"}
    # null inherits; an absent field keeps the chat's value
    r = client.post(url, json={"stores": {"c": {"candidates": None}}})
    assert r.status_code == 200, r.text
    ran = r.json()["decisions"][0]
    assert (ran["method"], ran["candidates"]) == ("vector", 20)


def test_old_post_chat_still_answers(client):
    add_collection(client)
    r = client.post("/chat", json=chat_body())
    assert r.status_code == 200 and r.json()["answer"]


def test_the_store_reports_a_stopped_switch_and_refuses_search(client, monkeypatch):
    small_model(client, monkeypatch)
    add_collection(client)
    app = client.app.state
    store, copy = app.store, app.store.copy

    def dies(source, target):
        raise RuntimeError("the process stopped")

    store.copy = dies  # the switch empties the store, then stops before refilling it
    job_id = reembed(client).json()["job_id"]
    assert wait(client, job_id)["status"] == "failed"
    db.update_job(app.conn, job_id, status="interrupted")  # as a restart leaves it
    body = client.get("/collections/c").json()
    assert body["rebuild"]["switching"] is True and body["rebuild"]["status"] == "interrupted"
    ask = {"collection": "c", "question": "enzyme rate", "methods": ["hybrid"]}
    r = client.post("/query", json=ask)
    assert r.status_code == 409 and "switching to fake/small" in r.json()["detail"]
    r = reembed(client)
    assert r.status_code == 409
    assert r.json()["detail"] == "the switch to fake/small stopped part-way; resume it first"
    store.copy = copy
    r = client.post(f"/jobs/{job_id}/resume")
    assert r.status_code == 202 and r.json() == {"job_id": job_id}
    assert wait(client, job_id)["status"] == "done"
    body = client.get("/collections/c").json()
    assert (body["dense_model"], body["dense_dim"], body["rebuild"]) == ("fake/small", 64, None)
    assert body["count"] == len(ITEMS) and db.switching_model(app.conn, "c") is None
    hits = client.post("/query", json=ask).json()["results"]["hybrid"]["hits"]
    assert hits[0]["id"] == "enzyme-item"


BRIEF_KEYS = {"n", "answer", "citations", "passages", "passages_only", "error", "usd"}


PASSAGE_KEYS = {"n", "store", "id", "text", "file", "source", "part", "parts", "relevance"}


def brief(client, chat_id, content="enzyme rate", **body):
    r = client.post(f"/chats/{chat_id}/messages?brief=true", json={"content": content, **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_new_chat_takes_stores_or_from(client):
    add_collection(client)
    first = new_chat(client)
    r = client.post("/chats", json={"from": first["id"]})
    assert r.status_code == 201 and r.json()["stores"] == ["c"]
    r = client.post("/chats", json={"from": first["id"], "title": "Bot"})
    assert r.status_code == 201 and r.json()["title"] == "Bot"
    r = client.post("/chats", json={})
    assert r.status_code == 422
    assert r.json()["detail"] == "a chat needs stores, or from: a chat to copy"
    r = client.post("/chats", json={"stores": ["c"], "from": first["id"]})
    assert r.status_code == 422 and r.json()["detail"] == "send stores or from, not both"
    assert client.post("/chats", json={"from": "nope"}).status_code == 404
    assert client.post("/chats", json={"stores": ["c"], "routing": "all"}).status_code == 422


def test_brief_reply_is_the_answer_citations_and_passages(client):
    add_collection(client)
    chat = new_chat(client)
    one = brief(client, chat["id"])
    assert set(one) == BRIEF_KEYS and one["n"] == 1
    assert one["answer"] and one["passages_only"] is False and one["error"] is None
    assert one["passages"] and all(set(p) == PASSAGE_KEYS for p in one["passages"])
    saved = client.get(f"/chats/{chat['id']}").json()["messages"][-1]["attempts"][0]
    assert one["citations"] == saved["citations"] and one["usd"] == saved["usd"]
    for p, e in zip(one["passages"], saved["evidence"], strict=True):
        assert (p["id"], p["relevance"]) == (e["id"], e["relevance"])


def test_brief_reply_without_an_answer_model_is_passages_only(client):
    add_collection(client)
    chat = new_chat(client)
    client.app.state.ai.set_role("default", None)
    one = brief(client, chat["id"])
    assert one["answer"] is None and one["passages_only"] is True and one["error"] is None
    assert one["citations"] == [] and one["passages"] and one["usd"] == 0


def test_brief_reply_on_a_chat_with_a_deleted_store_answers_from_the_rest(client):
    add_collection(client)
    add_collection(client, name="d")
    first = new_chat(client, stores=("c", "d"))
    copy = client.post("/chats", json={"from": first["id"]}).json()
    assert client.delete("/collections/d").status_code == 204
    one = brief(client, copy["id"])
    assert one["passages"] and {p["store"] for p in one["passages"]} == {"c"}
    attempt = client.get(f"/chats/{copy['id']}").json()["messages"][0]["attempts"][0]
    assert next(d for d in attempt["decisions"] if d["store"] == "d")["note"] == "deleted"


def test_brief_reply_on_an_empty_documents_store_has_no_passages(client):
    add_documents(client)
    chat = new_chat(client)
    one = brief(client, chat["id"])
    assert one["passages"] == [] and one["error"] is None


def test_brief_reply_with_answer_false_is_passages_only(client):
    add_collection(client)
    chat = new_chat(client)
    one = brief(client, chat["id"], answer=False)
    assert one["answer"] is None and one["passages_only"] is True and one["error"] is None
    assert one["passages"] and one["usd"] == 0


def test_brief_passages_name_their_file(client):
    documents_with_passages(client)  # store c: x.md's three passages
    chat = new_chat(client)
    one = brief(client, chat["id"], content="alpha")
    assert one["passages"] and all(p["file"] == "x.md" for p in one["passages"])
    saved = client.get(f"/chats/{chat['id']}").json()["messages"][-1]["attempts"][0]
    assert [p["file"] for p in one["passages"]] == [e["file"] for e in saved["evidence"]]


def test_messages_without_brief_keep_the_full_reply(client):
    add_collection(client)
    chat = new_chat(client)
    for query in ("", "?brief=false"):
        r = client.post(f"/chats/{chat['id']}/messages{query}", json={"content": "enzyme rate"})
        (attempt,) = r.json()["attempts"]
        assert {"decisions", "evidence", "pipeline"} <= set(attempt)
