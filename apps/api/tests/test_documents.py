"""The Documents routes: adding items, and the uploads and imports that bring files in."""

import zipfile
from dataclasses import asdict
from pathlib import Path

import pytest
from api_helpers import (
    LISTS,
    NOTES,
    PROPOSING,
    WORKED,
    add_collection,
    add_documents,
    add_tags,
    add_text,
    bare_client,
    do_import,
    documents_with_passages,
    handbook,
    ids_of,
    import_items,
    jsonl,
    ledger_purposes,
    long_text,
    make_client,
    passages,
    source_ids,
    store_import,
    terms_reply,
    three_items,
    upload,
    wait,
)

from ragqa_core import db, imports
from ragqa_core.fakes import ITEMS
from ragqa_core.types import Item


def test_documents_need_confirm_when_they_cost(client):
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/documents", json={"items": ITEMS})
    assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0
    assert client.get("/collections/s").json()["count"] == 0


def test_a_confirm_that_saw_no_price_is_refused_once_priced(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        client.app.state.worker.stop()
        ai = client.app.state.ai
        model = ai.roles()["default"]
        ai.set_role("default", None)  # nothing can price the summaries
        two = {"items": [{"id": f"t{i}", "text": f"tide note {i}"} for i in range(2)]}
        seen = client.post("/collections/s/documents", json=two).json()["estimate"]
        assert seen["calls"] == 2 and seen["est_usd"] is None
        shown = {**two, "confirm": True, "calls": 2, "est_usd": None}
        r = client.post("/collections/s/documents", json=shown)  # still unpriced: no model
        assert r.status_code == 503 and "'summary'" in r.json()["detail"]
        ai.set_role("default", model)  # now priced, above the nothing that was shown
        r = client.post("/collections/s/documents", json=shown)
        assert r.status_code == 409
        assert r.json()["detail"] == "the price changed since you saw it: check it again"
        assert client.get("/collections/s").json()["count"] == 0 and ledger_purposes(client) == []
        r = client.post("/collections/s/documents", json={**two, "confirm": True})
        assert r.status_code == 200 and r.json() == {"added": 2}  # no calls, no est_usd: no check


def test_budget_exceeded_is_402(tmp_path):
    client = make_client(tmp_path, cap_usd=0.0)
    client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
    r = client.post("/collections/s/documents", json={"items": ITEMS, "confirm": True})
    assert r.status_code == 402 and "monthly cap reached" in r.json()["detail"]


def uploads_left(tmp_path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "uploads").glob("*"))


def test_summarising_previews_work_with_no_ai_and_leave_no_upload(tmp_path):
    with bare_client(tmp_path) as client:
        client.post("/collections", json={"name": "c", "qa_type": "mcq", "summarize": True})
        up = upload(client)
        assert up.status_code == 200, up.text
        assert up.json()["estimate"]["est_usd"] is None and up.json()["estimate"]["calls"] == 3
        r = do_import(client, up.json(), confirm=True)
        assert r.status_code == 503 and "'summary' job" in r.json()["detail"]
        assert uploads_left(tmp_path) == [up.json()["upload_id"]]  # kept for a retry
        few = {"items": ITEMS}
        r = client.post("/collections/c/documents", json=few)
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] is None
        r = client.post("/collections/c/documents", json={**few, "confirm": True})
        assert r.status_code == 503 and "'summary' job" in r.json()["detail"]
        many = {"items": [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]}
        r = client.post("/collections/c/documents", json=many)
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] is None
        r = client.post("/collections/c/documents", json={**many, "confirm": True})
        assert r.status_code == 503 and "'summary' job" in r.json()["detail"]
        assert uploads_left(tmp_path) == [up.json()["upload_id"]]
        assert client.get("/collections/c").json()["count"] == 0
        assert db.list_jobs(client.app.state.conn, collection="c", limit=db.ALL) == []


def test_a_failed_upload_estimate_leaves_no_upload(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/collections", json={"name": "c", "qa_type": "mcq", "summarize": True})

        def broken(*args, **kwargs):
            raise RuntimeError("estimate failed")

        client.app.state.loader.estimate_import = broken
        with pytest.raises(RuntimeError):
            upload(client)
        many = {"items": [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]}
        with pytest.raises(RuntimeError):
            client.post("/collections/c/documents", json=many)
        assert uploads_left(tmp_path) == []


def test_documents_are_refused_while_a_load_rekey_or_eval_is_queued_or_running(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    body = {"items": ITEMS[:1], "confirm": True}
    for kind in ("load", "rekey", "eval"):
        for status in ("queued", "running"):
            job = db.create_job(conn, kind, "c", {})
            db.update_job(conn, job, status=status)
            r = client.post("/collections/c/documents", json=body)
            assert r.status_code == 409 and f"a {kind} job" in r.json()["detail"], (kind, status)
            db.update_job(conn, job, status="done")
    assert client.post("/collections/c/documents", json=body).status_code == 200


def test_upload_jsonl_previews_with_a_guessed_mapping(client):
    add_collection(client)
    r = upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "notes.jsonl" and len(body["upload_id"]) == 32
    assert body["preview"]["ready"] and body["preview"]["mapping"]["text"] == "text"
    assert body["estimate"]["new"] == 3 and body["estimate"]["est_usd"] == 0


def test_upload_not_ready_has_no_estimate(client):
    add_collection(client)
    body = upload(client, jsonl({"foo": "bar"})).json()
    assert body["preview"]["ready"] is False and body["estimate"] is None


def test_unmapped_import_preview_still_estimates_nothing(client):
    add_collection(client)
    up = upload(client).json()
    r = do_import(client, up, mapping={"text": None})
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"preview", "estimate"}
    assert r.json()["preview"]["ready"] is False and r.json()["estimate"] is None


def test_upload_over_limit_is_413(client, monkeypatch):
    from ragqa_core import imports

    add_collection(client)
    monkeypatch.setattr(imports, "UPLOAD_LIMIT", 10)
    assert upload(client).status_code == 413
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))


def test_upload_unsupported_type_is_422(client):
    add_collection(client)
    r = upload(client, b"x", filename="notes.exe")
    assert r.status_code == 422
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))


def test_store_upload_with_accept_text_refuses_other_files(client):
    add_collection(client)
    r = client.post(
        "/collections/c/uploads",
        params={"filename": "notes.pdf", "kind": "items", "accept": "text"},
        content=b"x",
    )
    assert r.status_code == 422 and "only .txt and .md" in r.json()["detail"]
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))
    ok = client.post(
        "/collections/c/uploads",
        params={"filename": "Q3 (final).md", "kind": "items", "accept": "text"},
        content=b"some text",
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["source_name"] == "Q3 -final"
    assert upload(client, filename="notes.jsonl").status_code == 200  # no accept: as before


def test_re_adding_a_shorter_text_file_leaves_only_its_new_chunks(client):
    add_collection(client, questions=False, lists=False)
    add_text(client, "x.md", long_text(["alpha", "brine", "cocoa"]))
    add_text(client, "y.md", long_text(["delta", "egret", "finch"]))  # same source
    add_text(client, "x.md", long_text(["gamma"]), source_name="other")  # another source
    job = add_text(client, "x.md", long_text(["alpha", "brine"]))
    assert job["result"]["removed"] == 1 and job["result"]["unchanged"] == 2
    x, y = (
        ids_of("x.md", long_text(["alpha", "brine"])),
        ids_of("y.md", long_text(["delta", "egret", "finch"])),
    )
    assert source_ids(client, "upload:docs") == sorted(x + y)
    assert source_ids(client, "upload:other") == ids_of("x.md", long_text(["gamma"]))
    held = client.app.state.store.get("c", [x[1], y[2]])
    assert [h.meta["parts"] for h in held] == [2, 3]


def test_a_text_file_shrunk_to_one_chunk_is_one_item_part_1_of_1(client):
    add_collection(client, questions=False, lists=False)
    add_text(client, "x.md", long_text(["alpha", "brine", "cocoa"]))
    job = add_text(client, "x.md", b"just a line now")
    assert job["result"]["removed"] == 3 and job["result"]["new"] == 1
    [one] = ids_of("x.md", b"just a line now")
    assert source_ids(client, "upload:docs") == [one]
    [held] = client.app.state.store.get("c", [one])
    assert held.text == "just a line now"
    assert (held.meta["file"], held.meta["part"], held.meta["parts"]) == ("x.md", 1, 1)
    grown = add_text(client, "x.md", long_text(["alpha", "brine"]))
    assert grown["result"]["removed"] == 1
    assert source_ids(client, "upload:docs") == sorted(
        ids_of("x.md", long_text(["alpha", "brine"]))
    )


def test_a_file_named_like_a_chunk_of_another_is_not_its_chunk(client, tmp_path):
    add_collection(client, questions=False, lists=False)
    store_import(client, "a.md", long_text(["alpha", "brine", "cocoa"]))
    store_import(client, "a.md#b.md", b"one chunk of its own")
    zipped = tmp_path / "x.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("a.md#old/x.md", long_text(["delta", "egret"]).decode())
        z.writestr("a.md#7.txt", b"also its own")
    store_import(client, "x.zip", zipped.read_bytes())
    before = source_ids(client, "upload:docs")
    others = [
        *ids_of("a.md#b.md", b"one chunk of its own"),
        *ids_of("a.md#old/x.md", long_text(["delta", "egret"])),
        *ids_of("a.md#7.txt", b"also its own"),
    ]
    assert len(others) == 4 and set(others) <= set(before)
    job = store_import(client, "a.md", long_text(["alpha", "brine"]))
    assert job["result"]["removed"] == 1
    cocoa = ids_of("a.md", long_text(["alpha", "brine", "cocoa"]))[2]
    assert source_ids(client, "upload:docs") == sorted(set(before) - {cocoa})


def test_appending_to_a_split_file_re_indexes_only_the_changed_chunks(client):
    add_collection(client, questions=False, lists=False)
    add_text(client, "x.md", long_text(["alpha", "brine", "cocoa"]))
    job = add_text(client, "x.md", long_text(["alpha", "brine", "cocoa", "delta"]))
    assert job["result"]["new"] == 1 and job["result"]["unchanged"] == 3
    assert job["result"]["updated"] == 0 and job["result"]["removed"] == 0
    held = client.app.state.store.get(
        "c", ids_of("x.md", long_text(["alpha", "brine", "cocoa", "delta"]))
    )
    assert [(h.meta["part"], h.meta["parts"]) for h in held] == [(i, 4) for i in (1, 2, 3, 4)]
    assert all(h.meta["source"] == "upload:docs" and h.meta["file"] == "x.md" for h in held)
    again = add_text(client, "x.md", long_text(["alpha", "brine", "cocoa", "delta"]))
    assert again["result"]["unchanged"] == 4


def test_a_failed_text_import_removes_nothing(client):
    add_collection(client, questions=False, lists=False)
    add_text(client, "x.md", long_text(["alpha", "brine", "cocoa"]))
    loader = client.app.state.loader

    def broken(*args, **kwargs):
        raise RuntimeError("the add broke")

    loader._add = broken
    up = client.post(
        "/collections/c/uploads",
        params={"filename": "x.md", "kind": "items", "accept": "text"},
        content=b"short now",
    ).json()
    r = do_import(client, up, "docs", confirm=True)
    assert wait(client, r.json()["job_id"])["status"] == "failed"
    assert source_ids(client, "upload:docs") == sorted(
        ids_of("x.md", long_text(["alpha", "brine", "cocoa"]))
    )


def test_an_edit_above_a_passage_keeps_its_id_and_skips_its_embedding(client):
    add_collection(client, questions=False, lists=False)
    old = handbook("")
    assert len(old) > 4000
    add_text(client, "handbook.md", old)
    before = source_ids(client, "upload:docs")
    assert len(before) == 4
    models = client.app.state.models
    embedded: list[str] = []
    embed = models.embed

    def counting(texts, model=None):
        embedded.extend(texts)
        return embed(texts, model)

    models.embed = counting
    edited = handbook(long_text(["eagle"]).decode())  # a new paragraph opens section 1
    job = add_text(client, "handbook.md", edited)
    after = ids_of("handbook.md", edited)
    assert len(after) == 5  # section 1 is now two chunks
    held = {h.id: h for h in client.app.state.store.get("c", after)}
    unchanged = [
        i
        for i in after
        if "brine" in held[i].text or "cocoa" in held[i].text or "delta" in held[i].text
    ]
    assert len(unchanged) == 3 and set(unchanged) <= set(before)  # sections 2-4 keep their ids
    counts = {k: job["result"][k] for k in ("new", "unchanged", "updated", "removed")}
    assert counts == {"new": 2, "unchanged": 3, "updated": 0, "removed": 1}
    assert source_ids(client, "upload:docs") == sorted(after)
    # Item vectors for the two new chunks only, and no piece of sections 2-4 embedded again.
    fresh = [held[i].text for i in after if i not in before]
    stored = {h.text for h in held.values()}
    assert sorted(t for t in embedded if t in stored) == sorted(fresh)
    assert not any("brine" in t or "cocoa" in t or "delta" in t for t in embedded)
    # their new place is written as a payload: parts 3-5 of 5
    assert [(held[i].meta["part"], held[i].meta["parts"]) for i in unchanged] == [
        (3, 5),
        (4, 5),
        (5, 5),
    ]
    assert all(held[i].meta["source"] == "upload:docs" for i in after)


def test_adding_a_file_again_drops_its_positional_ids(client):
    add_collection(client, questions=False, lists=False)
    state = client.app.state
    source = {"source": "upload:docs"}
    old = [  # as a load before content ids wrote them
        Item("x.md#1", "x.md\n\nalpha", meta={**source, "file": "x.md", "part": 1, "parts": 2}),
        Item("x.md#2", "x.md\n\nbrine", meta={**source, "file": "x.md", "part": 2, "parts": 2}),
        Item("y.md", "a short file", meta=source),
    ]
    texts = [i.text for i in old]
    state.store.upsert_items(
        "c", old, state.models.embed(texts), state.models.sparse(texts), [None] * 3
    )
    ids = [i.id for i in old]
    state.store.upsert_chunks("c", ids, texts, state.models.embed(texts), ["upload:docs"] * 3)
    assert source_ids(client, "upload:docs") == ["x.md#1", "x.md#2", "y.md"]
    x_body, y_body = long_text(["alpha", "brine"]), b"a short file"
    assert add_text(client, "x.md", x_body)["result"]["removed"] == 2
    assert add_text(client, "y.md", y_body)["result"]["removed"] == 1
    assert source_ids(client, "upload:docs") == sorted(
        ids_of("x.md", x_body) + ids_of("y.md", y_body)
    )
    assert not state.store.chunk_points("c", ids)  # their chunks went with them


def test_a_text_file_that_is_not_utf8_is_refused_by_name(client, tmp_path):
    add_collection(client, questions=False, lists=False)
    message = "notes.txt is not UTF-8 text; save it as UTF-8 and upload it again"
    windows = "café au lait".encode("cp1252")
    r = upload(client, windows, filename="notes.txt")
    assert r.status_code == 422 and r.json()["detail"] == message
    zipped = tmp_path / "x.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("ok.md", "fine")
        z.writestr("notes.txt", windows)
    r = upload(client, zipped.read_bytes(), filename="x.zip")
    assert r.status_code == 422 and r.json()["detail"] == message
    chat_way = client.post(
        "/collections/c/uploads",
        params={"filename": "notes.txt", "kind": "items", "accept": "text"},
        content=windows,
    )
    assert chat_way.status_code == 422 and chat_way.json()["detail"] == message
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))
    assert source_ids(client, "upload:notes") == []
    # JSON keeps replacing a bad byte, as the benchmark loaders need
    loose = upload(client, b'[{"id": "j1", "text": "caf\xe9 au lait"}]', filename="notes.json")
    assert loose.status_code == 200 and loose.json()["preview"]["ready"]
    r = do_import(client, loose.json(), "loose", confirm=True)
    assert wait(client, r.json()["job_id"])["status"] == "done"
    assert client.app.state.store.get("c", ["j1"])[0].text == "caf\ufffd au lait"


def test_a_text_import_drops_the_store_leak_scans(client):
    add_collection(client, questions=False, lists=False)
    forgot = []
    evaluator = client.app.state.evaluator
    forget = evaluator.forget
    evaluator.forget = lambda name: (forgot.append(name), forget(name))
    add_text(client, "x.md", long_text(["alpha", "brine", "cocoa"]))
    assert forgot == ["c"]


def test_upload_to_an_unknown_collection_is_404(client):
    assert upload(client, name="nope").status_code == 404


def test_import_unknown_upload_is_404(client):
    add_collection(client)
    body = {"upload_id": "../x", "kind": "items", "mapping": None, "source_name": "notes"}
    assert client.post("/collections/c/imports", json=body).status_code == 404
    body["upload_id"] = "0" * 32
    assert client.post("/collections/c/imports", json=body).status_code == 404


def test_import_rejects_a_bad_source_name_and_mapping(client):
    add_collection(client)
    up = upload(client).json()
    assert do_import(client, up, "../evil").status_code == 422
    assert do_import(client, up, mapping={"bogus": "text"}).status_code == 422
    assert do_import(client, up, mapping={"text": 3}).status_code == 422


def test_import_estimate_then_confirm_then_reimport_unchanged(client):
    add_collection(client)
    up = upload(client).json()
    r = do_import(client, up)
    assert r.status_code == 200 and r.json()["estimate"]["new"] == 3
    assert r.json()["preview"]["ready"]
    assert client.get("/collections/c").json()["count"] == 3  # nothing yet
    job = import_items(client)
    assert job["status"] == "done" and job["setup"]["source"] == "upload:notes"
    assert job["result"]["new"] == 3
    detail = client.get("/collections/c").json()
    assert detail["count"] == 6
    again = import_items(client)
    assert again["result"]["new"] == 0 and again["result"]["updated"] == 0
    assert again["result"]["unchanged"] == 3
    sources = {s["source"]: s for s in client.get("/collections/c/sources").json()}
    assert sources["upload:notes"] == {"source": "upload:notes", "items": 3, "enabled": True}


def test_import_not_ready_is_422(client):
    add_collection(client)
    up = upload(client, jsonl({"foo": "bar"})).json()
    r = do_import(client, up, confirm=True)
    assert r.status_code == 422 and "text" in str(r.json()["detail"])


def test_import_that_costs_needs_a_key(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "c", "qa_type": "mcq", "summarize": True})
        up = upload(client).json()
        assert up["estimate"]["est_usd"] > 0
        assert do_import(client, up, confirm=True).status_code == 503


def test_an_import_priced_without_a_keywords_model_records_no_proposal(tmp_path):
    with make_client(tmp_path, reply=PROPOSING) as client:
        add_collection(client, lists=False)
        client.app.state.worker.stop()
        ai = client.app.state.ai
        ai.update_connection("anthropic", api_key=None)
        r = do_import(client, upload(client).json(), confirm=True)
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]
        assert client.get(f"/jobs/{job_id}").json()["setup"]["propose"] is False
        ai.update_connection("anthropic", api_key="test")  # set up while the job waits
        client.app.state.worker.start()
        job = wait(client, job_id)
        assert job["status"] == "done" and job["result"]["lists_source"] is None
        assert "propose" not in ledger_purposes(client)


def test_an_import_whose_price_changed_since_it_was_shown_is_refused(tmp_path):
    with make_client(tmp_path, reply=PROPOSING) as client:
        add_collection(client, lists=False)
        client.app.state.worker.stop()
        ai = client.app.state.ai
        ai.update_connection("anthropic", api_key=None)
        up = upload(client).json()
        assert up["estimate"]["calls"] == 0
        ai.update_connection("anthropic", api_key="test")  # now it would propose
        r = do_import(client, up, confirm=True, calls=0)
        assert r.status_code == 409
        assert r.json()["detail"] == "the price changed since you saw it: check it again"
        assert db.list_jobs(client.app.state.conn, collection="c", limit=db.ALL) == []
        r = do_import(client, up, confirm=True)  # without calls: as before
        assert r.status_code == 202 and set(r.json()) == {"job_id"}


def summarised_import_repriced(client, times: float) -> tuple[dict, dict]:
    """A summarised store's upload previewed, then the default model's price scaled by
    `times`: the calls stay the same, the price moves."""
    add_collection(client, summarize=True, lists=False)
    client.app.state.worker.stop()
    up = upload(client).json()
    shown = up["estimate"]
    assert shown["calls"] > 0 and shown["est_usd"] > 0
    ai = client.get("/ai").json()
    ref = ai["roles"]["default"]
    model = next(m for m in ai["models"] if m["ref"] == ref)
    price = {"price_in": model["price_in"] * times, "price_out": model["price_out"] * times}
    assert client.put(f"/ai/models/{ref}", json=price).status_code == 200
    again = do_import(client, up).json()["estimate"]
    assert again["calls"] == shown["calls"] and again["est_usd"] != shown["est_usd"]
    return up, shown


def test_an_import_whose_price_rose_with_the_same_calls_is_refused(tmp_path):
    with make_client(tmp_path) as client:
        up, shown = summarised_import_repriced(client, 3)
        r = do_import(client, up, confirm=True, calls=shown["calls"], est_usd=shown["est_usd"])
        assert r.status_code == 409
        assert r.json()["detail"] == "the price changed since you saw it: check it again"
        assert db.list_jobs(client.app.state.conn, collection="c", limit=db.ALL) == []


def test_an_import_whose_price_fell_is_queued(tmp_path):
    with make_client(tmp_path) as client:
        up, shown = summarised_import_repriced(client, 0.5)
        r = do_import(client, up, confirm=True, calls=shown["calls"], est_usd=shown["est_usd"])
        assert r.status_code == 202 and set(r.json()) == {"job_id"}


def test_store_question_routes_are_gone(client):
    add_collection(client)
    assert client.get("/collections/c/questions").status_code in (404, 405)
    r = client.patch("/collections/c/questions/leaked", json={"include": True})
    assert r.status_code in (404, 405)
    r = client.post(
        "/collections/c/uploads",
        params={"filename": "q.jsonl", "kind": "questions"},
        content=b"{}\n",
    )
    assert r.status_code == 422


def test_documents_over_20_items_queue_an_import(client):
    add_collection(client, questions=False)
    many = [
        {"id": f"m{i}", "text": f"note number {i} about reefs", "answer": "a"} for i in range(21)
    ]
    r = client.post("/collections/c/documents", json={"items": many})
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["kind"] == "import" and job["status"] == "done"
    assert job["setup"]["source"] == "hand" and job["result"]["new"] == 21
    assert client.get("/collections/c").json()["count"] == 24


def test_documents_over_20_items_estimate_first(tmp_path):
    with make_client(tmp_path) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        r = client.post("/collections/s/documents", json={"items": many})
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0
        assert client.get("/collections/s").json()["count"] == 0


def test_queued_documents_need_a_key_when_they_cost(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        r = client.post("/collections/s/documents", json={"items": many, "confirm": True})
        assert r.status_code == 503
        assert db.list_jobs(client.app.state.conn, collection="s", limit=db.ALL) == []


def test_preview_warns_when_ids_would_move_from_another_source(client):
    add_collection(client)
    import_items(client)
    up = upload(client, filename="again.jsonl").json()
    warning = "3 items already belong to upload:notes and would move to this source"
    assert warning in up["preview"]["problems"]
    same = do_import(client, up, "notes").json()["preview"]["problems"]
    assert not [p for p in same if "already belong" in p]
    other = do_import(client, up, "other").json()["preview"]["problems"]
    assert warning in other


def test_documents_are_refused_while_an_import_or_backfill_is_queued_or_running(client):
    add_collection(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    body = {"items": ITEMS[:1], "confirm": True}
    for kind in ("import", "backfill"):
        for status in ("queued", "running"):
            job = db.create_job(conn, kind, "c", {})
            db.update_job(conn, job, status=status)
            r = client.post("/collections/c/documents", json=body)
            assert r.status_code == 409 and f"a {kind} job" in r.json()["detail"], (kind, status)
            db.update_job(conn, job, status="done")


def test_add_now_checks_summary_model(tmp_path):
    with make_client(tmp_path, api_key=None) as client:
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        two = [{"id": f"t{i}", "text": f"tide note {i}"} for i in range(2)]
        r = client.post("/collections/s/documents", json={"items": two})
        assert r.status_code == 200 and r.json()["estimate"]["calls"] == 2
        r = client.post("/collections/s/documents", json={"items": two, "confirm": True})
        assert r.status_code == 503 and "'summary'" in r.json()["detail"]
        assert client.get("/collections/s").json()["count"] == 0
        assert ledger_purposes(client) == []


def test_load_and_documents_refuse_changed_price(tmp_path, monkeypatch):
    three_items(monkeypatch)
    changed = "the price changed since you saw it: check it again"
    with make_client(tmp_path, reply=PROPOSING) as client:
        client.post("/collections", json={"name": "m", "qa_type": "multihop"})
        client.post("/collections", json={"name": "s", "qa_type": "mcq", "summarize": True})
        client.app.state.worker.stop()
        conn = client.app.state.conn
        seen = client.post("/collections/m/load", json={"dataset": "musique"}).json()["estimate"]
        assert seen["calls"] == 1 and seen["est_usd"] > 0  # the proposal
        load = {"dataset": "musique", "confirm": True}
        for stale in ({"calls": 0}, {"calls": 0, "est_usd": seen["est_usd"]}):
            r = client.post("/collections/m/load", json={**load, **stale})
            assert r.status_code == 409 and r.json()["detail"] == changed
        r = client.post("/collections/m/load", json={**load, "est_usd": seen["est_usd"] / 2})
        assert r.status_code == 409 and r.json()["detail"] == changed
        two = [{"id": f"t{i}", "text": f"tide note {i}"} for i in range(2)]
        many = [{"id": f"m{i}", "text": f"note {i}"} for i in range(21)]
        # A summary each; queued, the proposal too (a sync add makes no lists: decision H).
        for items, calls in ((two, 2), (many, 22)):
            shown = client.post("/collections/s/documents", json={"items": items}).json()
            assert shown["estimate"]["calls"] == calls
            r = client.post(
                "/collections/s/documents", json={"items": items, "confirm": True, "calls": 1}
            )
            assert r.status_code == 409 and r.json()["detail"] == changed
        assert db.list_jobs(conn, limit=db.ALL) == [] and ledger_purposes(client) == []
        assert client.get("/collections/s").json()["count"] == 0
        assert list((client.app.state.settings.home / "uploads").glob("*")) == []
        # What was shown still holds: accepted. Without calls or est_usd: as before.
        held = {"calls": 1, "est_usd": seen["est_usd"]}
        r = client.post("/collections/m/load", json={**load, **held})
        assert r.status_code == 202, r.text
        priced = client.post("/collections/s/documents", json={"items": two}).json()["estimate"]
        r = client.post(
            "/collections/s/documents",
            json={"items": two, "confirm": True, "calls": 2, "est_usd": priced["est_usd"]},
        )
        assert r.status_code == 200 and r.json() == {"added": 2}
        r = client.post("/collections/s/documents", json={"items": many, "confirm": True})
        assert r.status_code == 202 and set(r.json()) == {"job_id"}


def test_worked_documents_whose_build_failed_are_built_by_sending_them_again(tmp_path):
    def reply(prompt: str) -> str:
        if '"principle"' in prompt:
            return '{"principle": "P.", "method": "M.", "answer": "A"}'
        return '{"terms": ["rates"]}'

    with make_client(tmp_path, reply=reply) as client:
        client.post("/collections", json=WORKED)
        builder = client.app.state.builder

        def boom(*args, **kwargs):
            raise RuntimeError("provider down")

        builder.abstract = boom
        with pytest.raises(RuntimeError):
            client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True})
        del builder.abstract
        assert client.get("/collections/w").json()["count"] == 3  # added and tagged only
        r = client.post("/collections/w/documents", json={"items": ITEMS})
        assert r.status_code == 200 and r.json()["estimate"]["est_usd"] > 0  # the abstracts
        r = client.post("/collections/w/documents", json={"items": ITEMS, "confirm": True})
        assert r.status_code == 200
        hits = client.app.state.store.get("w", [i["id"] for i in ITEMS])
        assert all(h.tags == ["rates"] and h.abstracted for h in hits)
        r = client.post("/collections/w/documents", json={"items": ITEMS})
        assert r.status_code == 200 and r.json() == {"added": 3}  # built: free, nothing to confirm


def test_a_sync_add_holds_off_jobs_on_its_collection_until_it_ends(client):
    """A sync add runs outside the worker, so the jobs table can't show it: a retag claimed
    during it would leave the new items tagged under the old vocabulary."""
    add_collection(client)
    client.post("/collections", json=WORKED)
    state = client.app.state
    state.worker.stop()
    seen = []
    build = state.builder.build
    state.builder.build = lambda spec, todo: (seen.append(set(state.adding)), build(spec, todo))
    r = client.post("/collections/c/documents", json={"items": ITEMS[:1], "confirm": True})
    assert r.status_code == 200 and seen == [{"c"}] and state.adding == set()

    state.adding.update({"c", "w"})  # as while each one's add runs

    def versions():
        lists = state.conn.execute("select count(*) as n from keyword_lists")[0]["n"]
        return lists, db.list_vocabularies(state.conn, "w")

    before = versions()
    refused = {
        "rekey": client.put("/collections/c/keyword-lists", json=asdict(LISTS)),
        "retag": client.put("/collections/w/vocabulary", json={"terms": ["x"], "confirm": True}),
        "load": client.post("/collections/c/load", json={"dataset": "scienceqa", "confirm": True}),
        "copy": client.post(
            "/collections/c/copy", json={"name": "c2", "recipe": "paper_2024", "confirm": True}
        ),
        "eval": client.post(
            "/eval", json={"collection": "c", "methods": ["vector"], "confirm": True}
        ),
        "add": client.post("/collections/c/documents", json={"items": ITEMS[:1], "confirm": True}),
    }
    for kind, r in refused.items():
        assert r.status_code == 409 and "being added" in r.json()["detail"], (kind, r.text)
    assert db.list_jobs(state.conn, status="queued", limit=db.ALL) == []
    assert versions() == before  # a refused rekey or retag saves no version
    state.adding.clear()
    assert (
        client.put("/collections/w/vocabulary", json={"terms": ["x"], "confirm": True}).status_code
        == 202
    )


DOCUMENTS_ONLY = "A Documents store takes .md and .txt files"


def test_documents_upload_takes_md_and_txt_only(client, tmp_path):
    add_documents(client)
    zipped = tmp_path / "x.zip"
    with zipfile.ZipFile(zipped, "w") as z:
        z.writestr("a.md", "some text")
    refused = (("notes.csv", b"text\nsome text\n"), ("notes.jsonl", NOTES), ("x.zip", zipped))
    for filename, body in refused:
        content = body.read_bytes() if isinstance(body, Path) else body
        r = upload(client, content, filename=filename)
        assert r.status_code == 422 and r.json()["detail"] == DOCUMENTS_ONLY
    assert not list((client.app.state.settings.home / "uploads").glob("*/file"))
    for filename in ("a.md", "b.txt", "C.MD"):
        r = upload(client, b"some text", filename=filename)
        assert r.status_code == 200, r.text
        assert r.json()["preview"]["ready"] and r.json()["preview"]["usable"] == 1
    questions = jsonl({"text": "where is some text", "gold_ids": "x"})
    r = upload(client, questions, filename="set.jsonl", kind="questions")  # items only
    assert r.status_code == 422, r.text
    add_collection(client, name="m", lists=False, questions=False)
    assert upload(client, filename="notes.jsonl", name="m").status_code == 200  # other types


def test_documents_upload_keeps_passages_whole(client):
    add_documents(client)
    body = long_text(["alpha", "brine", "cocoa"])
    rows = imports.text_rows("x.md", body.decode())
    assert len(rows) == 3
    job = store_import(client, "x.md", body)
    assert job["result"]["new"] == 3
    held = client.app.state.store.get("c", [r["file"] for r in rows])
    assert {h.id: h.text for h in held} == {r["file"]: r["text"] for r in rows}
    assert client.app.state.store.count("c") == 3


def test_an_empty_documents_store_reads_and_a_blank_file_is_refused(client):
    add_documents(client)
    r = client.get("/collections/c")
    assert r.status_code == 200, r.text
    assert set(r.json()["measurable"]) == {"ndcg@10", "recall@10", "mrr@10"}
    assert r.json()["methods_unavailable"]["keyword_rag"] == (
        "needs keyword tags: Add keyword tags on the store page"
    )
    up = upload(client, b"  \n\n\t \n", filename="blank.md")
    assert up.status_code == 200, up.text
    shown = up.json()["preview"]
    assert shown["usable"] == 0 and shown["ready"] is False and up.json()["estimate"] is None
    r = do_import(client, up.json(), "blank", confirm=True)
    assert r.status_code == 422 and r.json()["detail"] == "no usable rows"
    assert client.app.state.store.count("c") == 0
    query = {"collection": "c", "question": "anything", "methods": ["hybrid"]}
    r = client.post("/query", json=query)
    assert r.status_code == 200, r.text
    assert r.json()["results"]["hybrid"]["hits"] == []
    r = client.post("/query", json={"collection": "c", "question": "x", "methods": ["keyword_rag"]})
    assert r.json()["results"]["keyword_rag"]["error"] == (
        "needs keyword tags: Add keyword tags on the store page"
    )


def test_add_documents_waits_for_a_tag_build(client):
    documents_with_passages(client)
    client.app.state.worker.stop()
    assert add_tags(client, True).status_code == 202
    body = {"items": [{"id": "n", "text": "a new passage"}], "confirm": True}
    r = client.post("/collections/c/documents", json=body)
    assert r.status_code == 409 and "a tag_build job" in r.json()["detail"], r.text
    assert client.app.state.store.count("c") == 3


def test_an_import_waits_for_a_tag_build(client):
    documents_with_passages(client)
    conn = client.app.state.conn
    client.app.state.worker.stop()
    imports_before = len(db.list_jobs(conn, kind="import"))
    up = upload(client, b"a fourth passage on dunes", filename="y.md").json()
    assert up["estimate"]["est_usd"] == 0  # priced while the store is still untagged
    assert add_tags(client, True).status_code == 202
    r = do_import(client, up, "docs", confirm=True)
    assert r.status_code == 409, r.text
    assert r.json()["detail"] == "a tag_build job is queued or running on 'c'; wait"
    assert len(db.list_jobs(conn, kind="import")) == imports_before


def test_adding_passages_to_a_tagged_documents_store_estimates_the_build(tmp_path):
    with make_client(tmp_path, reply=terms_reply) as client:
        documents_with_passages(client)
        add_documents(client, "u")  # stays without keyword tags
        assert wait(client, add_tags(client, True).json()["job_id"])["status"] == "done"
        up = upload(client, b"a fourth passage on dunes", filename="y.md").json()
        # The build is priced: with the vocabulary in force, tagging by similarity is free.
        assert up["estimate"]["est_usd"] == 0 and up["estimate"]["calls"] == 0
        r = do_import(client, up, "docs", confirm=True)
        assert wait(client, r.json()["job_id"])["status"] == "done"
        assert len(passages(client)) == 4 and all(h.tags for h in passages(client))
        assert "keyword_rag" in client.get("/collections/c").json()["methods_available"]
        client.app.state.ai.set_role("default", None)  # no model for keywords now
        # With its vocabulary in force, the tagged store's build is free: no model is needed.
        tagged = upload(client, b"a fifth passage", filename="z.md").json()
        untagged = upload(client, b"a fifth passage", filename="z.md", name="u").json()
        assert tagged["estimate"]["est_usd"] == 0 and tagged["estimate"]["calls"] == 0
        assert untagged["estimate"]["est_usd"] == 0  # the untagged one builds nothing
        r = do_import(client, tagged, "more", confirm=True)
        assert r.status_code == 202, r.text
        assert wait(client, r.json()["job_id"])["status"] == "done"
        body = {"items": [{"id": "n", "text": "a sixth passage on dunes"}], "confirm": True}
        r = client.post("/collections/c/documents", json=body)
        assert r.status_code == 200 and r.json() == {"added": 1}, r.text
        assert len(passages(client)) == 6 and all(h.tags for h in passages(client))
        assert ledger_purposes(client).count("induce") == 1  # induced once, by Add keyword tags
