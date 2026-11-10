"""What the API tests share: the app on fakes, and the stores, items and imports they set up."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from fastapi.testclient import TestClient

from ragqa_api.app import create_app
from ragqa_core import db, fakes, imports, search_defaults
from ragqa_core.fakes import ITEMS, fake_ai, fake_embeddings, fake_replies, until
from ragqa_core.settings import load_settings
from ragqa_core.types import Item, KeywordLists, Question

LISTS = KeywordLists(**fakes.LISTS)
OTHER_LISTS = KeywordLists(topic=["enzyme", "market"], method=["experiment"], common=["rate"])


QUESTION = Question(
    "q1", "enzyme rate", ["x", "y"], ["A"], [], {"skill": "enzymes", "solution": "Rates rise."}
)


EVAL = {"collection": "c", "methods": ["hybrid"], "answer": False}


PAIR = ["kw_individual", "hybrid"]


TURBO = "jinaai/jina-reranker-v1-turbo-en"


SEARCH_FIELDS = list(search_defaults.RECOMMENDED)


def settings(tmp_path: Path, **overrides):
    base = load_settings({})
    fake = {
        "dense_model": "fake/dense",
        "judge_model": "fake/judge",
        "rerank_model": "fake/rerank",
        "build_hub": "",  # tests never read the real Hub
    }
    return replace(base, home=tmp_path, **{**fake, **overrides})


def make_client(
    tmp_path,
    reply=lambda p: "Answer: A",
    api_key: str | None = "test",
    cap_usd: float = 20.0,
    **overrides,
) -> TestClient:
    """Not started: enter it (`with`) for the lifespan, which runs the worker. The key is
    stored, not from the environment, so a test can clear it; with none, models are still
    set up and priced, so estimates work and spending is refused."""
    app = create_app(settings(tmp_path, **overrides))
    fake_embeddings(app.state.models)
    fake_ai(app.state.conn, reply, api_key, cap_usd)  # stored in the app's file
    fake_replies(app.state.ai, reply)
    search_defaults.save(app.state.conn, {"rewrite": False})  # follow-ups search the joined text
    app.state.worker.idle_seconds = 0.01
    return TestClient(app)


def wait(client, job_id, seconds=10):
    def finished():
        job = client.get(f"/jobs/{job_id}").json()
        return job if job["status"] not in ("queued", "running") else None

    return until(finished, seconds, 0.02, f"job {job_id} finishing")


def add_collection(client, summarize=False, lists=True, name="c", questions=True):
    r = client.post("/collections", json={"name": name, "qa_type": "mcq", "summarize": summarize})
    assert r.status_code == 201, r.text
    r = client.post(f"/collections/{name}/documents", json={"items": ITEMS, "confirm": True})
    assert r.status_code == 200, r.text
    if questions:
        db.save_questions(client.app.state.conn, name, [QUESTION])
    if lists:
        r = client.put(f"/collections/{name}/keyword-lists", json=asdict(LISTS))
        assert r.status_code == 202, r.text
        assert wait(client, r.json()["job_id"])["status"] == "done"


def record_queries(client) -> list[tuple[str, str, int]]:
    """Every /query answer call from here on as (system, user, max_tokens); they go through."""
    calls, complete_many = [], client.app.state.ai.complete_many

    def recording(prompts, **kw):
        if kw["purpose"] == "query":
            calls.extend((kw.get("system", ""), p, kw["max_tokens"]) for p in prompts)
        return complete_many(prompts, **kw)

    client.app.state.ai.complete_many = recording
    return calls


def ledger_purposes(client):
    rows = client.app.state.conn.execute("select purpose from ai_ledger")
    return [r["purpose"] for r in rows]


def three_items(monkeypatch):
    from ragqa_core import loader
    from ragqa_core.types import Item

    items = [Item(d["id"], d["text"] + " shared words", d["answer"]) for d in ITEMS]
    monkeypatch.setattr(loader, "load_dataset", lambda name, cache: (items, []))
    monkeypatch.setattr("ragqa_core.loader.load_dataset", lambda name, cache: (items, []))
    return items


WORKED = {"name": "w", "qa_type": "science", "recipe": "worked_examples"}


def stored_worked(client) -> None:
    """Collection w (worked examples, AI tagging) with ITEMS stored."""
    client.post("/collections", json=WORKED)
    app = client.app.state
    spec = app.indexer.spec("w")
    app.indexer.add(spec, [Item(**i) for i in ITEMS], "hand")


def reference_with_vocabulary(client) -> None:
    """Collection r (reference recipe) with ITEMS stored and tagged from a vocabulary."""
    client.post("/collections", json={"name": "r", "qa_type": "multihop", "recipe": "reference"})
    app = client.app.state
    app.indexer.add(app.indexer.spec("r"), [Item(**i) for i in ITEMS], "hand")
    terms = {"terms": ["enzyme", "market", "volcano"], "confirm": True}
    r = client.put("/collections/r/vocabulary", json=terms)
    assert r.status_code == 202, r.text
    assert wait(client, r.json()["job_id"])["status"] == "done"


def set_running(conn, job_id, progress=0.0):
    conn.execute("update jobs set status='running', progress=? where id=?", (progress, job_id))


def stopped_eval_on(client, name: str) -> str:
    """A stopped answered eval on the store, made as test_resume_flow_estimates_then_requeues
    makes one."""
    add_collection(client, name=name)
    client.app.state.worker.stop()
    body = {**EVAL, "collection": name, "methods": ["kw_individual"], "answer": True}
    job = client.post("/eval", json={**body, "confirm": True}).json()["job_id"]
    db.update_job(client.app.state.conn, job, status="interrupted")
    return job


def stale(seen: dict) -> dict:
    """A confirm that saw another number of calls."""
    return {"confirm": True, "calls": seen["calls"] + 1, "est_usd": seen["est_usd"]}


def run_pair(client) -> str:
    body = {"collection": "c", "methods": PAIR + ["random_example"], "answer": False}
    job_id = client.post("/eval", json={**body, "confirm": True}).json()["job_id"]
    assert wait(client, job_id)["status"] == "done"
    return job_id


def row(hit, right=None, retrieval=None):
    out = {"hit_ids": [hit], "hit_skills": ["s"], "hit_matched": [["rate"]], "retrieval": {}}
    if retrieval is not None:
        out["retrieval"] = retrieval
    if right is not None:
        out["pred"] = "A"
        out["answer"] = {"accuracy": float(right)}
    return out


def jsonl(*rows) -> bytes:
    return "\n".join(json.dumps(r) for r in rows).encode()


NOTES = jsonl(
    {"id": "n1", "text": "tides follow the moon", "answer": "x"},
    {"id": "n2", "text": "glaciers carve valleys", "answer": "y"},
    {"id": "n3", "text": "reefs need warm shallow water", "answer": "z"},
)


def upload(client, body=NOTES, filename="notes.jsonl", kind="items", name="c"):
    return client.post(
        f"/collections/{name}/uploads", params={"filename": filename, "kind": kind}, content=body
    )


def do_import(client, up, source_name="notes", kind="items", name="c", **extra):
    body = {"upload_id": up["upload_id"], "kind": kind, "source_name": source_name, **extra}
    body["mapping"] = body.get("mapping", up["preview"]["mapping"])
    return client.post(f"/collections/{name}/imports", json=body)


def import_items(client, source_name="notes", body=NOTES, name="c"):
    up = upload(client, body, name=name).json()
    r = do_import(client, up, source_name, name=name, confirm=True)
    assert r.status_code == 202, r.text
    return wait(client, r.json()["job_id"])


def long_text(words: list[str]) -> bytes:
    """One paragraph of 1,000 characters per word: 'w w w …', so each is one chunk."""
    return "\n\n".join(" ".join([w] * (1000 // (len(w) + 1))) for w in words).encode()


def ids_of(filename: str, body: bytes) -> list[str]:
    """The content ids a text file's chunks take, in file order (test_imports pins them)."""
    return [r["file"] for r in imports.text_rows(filename, body.decode())]


def store_import(client, filename, body, source_name="docs"):
    """A file imported the store page's way: no accept=text, the guessed mapping."""
    up = upload(client, body, filename=filename).json()
    r = do_import(client, up, source_name, confirm=True)
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    return job


def handbook(first: str) -> bytes:
    """Four sections of about 1,000 characters each, each its own chunk; `first` opens
    section 1."""
    sections = [
        f"# {w.title()}\n\n{long_text([w]).decode()}" for w in ("alpha", "brine", "cocoa", "delta")
    ]
    sections[0] = sections[0].replace("\n\n", f"\n\n{first}\n\n", 1) if first else sections[0]
    return "\n\n".join(sections).encode()


PROPOSING = lambda p: '{"topic": ["enzymes"], "method": ["assay"], "common": ["rate"]}'  # noqa: E731


def free_disabled_model(client, *roles: str) -> None:
    """A $0 model entered by hand and switched off, set for the roles: priced at nothing,
    it still cannot be called."""
    ai = client.app.state.ai
    ai.add_connection("gw", "openai", "Gateway", base_url="https://gw.example/v1")
    ai.add_model("gw", "llama", price_in=0, price_out=0)
    ai.update_model("gw", "llama", enabled=False)
    for role in roles:
        ai.set_role(role, "gw/llama")


MCQ = {"id": "a", "text": "which one?", "choices": ["x", "y"], "gold_answers": ["A"]}


def put_questions(client, name, questions, base=None):
    body = {"questions": questions, "base_version": base}
    return client.put(f"/eval-sets/{name}/questions", json=body)


def chat_body(question="enzyme rate", **extra):
    return {"collection": "c", "messages": [{"role": "user", "content": question}], **extra}


def new_chat(client, stores=("c",), **body) -> dict:
    r = client.post("/chats", json={"stores": list(stores), **body})
    assert r.status_code == 201, r.text
    return r.json()


def add_documents(client, name="c"):
    r = client.post("/collections", json={"name": name, "qa_type": "documents"})
    assert r.status_code == 201, r.text
    return r.json()


def terms_reply(prompt: str) -> str:
    return '{"terms": ["alpha", "brine", "cocoa"]}'


def documents_with_passages(client) -> None:
    """Store c: a Documents store of three passages, without keyword tags."""
    add_documents(client)
    store_import(client, "x.md", long_text(["alpha", "brine", "cocoa"]))
    assert client.app.state.store.count("c") == 3


def add_tags(client, confirm: bool):
    return client.post("/collections/c/keyword-tags", json={"confirm": confirm})


def passages(client, name="c") -> list:
    return [h for batch in client.app.state.store.scroll(name) for h in batch]


def new_set(client, name="e", qa_type="mcq"):
    assert client.post("/eval-sets", json={"name": name, "qa_type": qa_type}).status_code == 201


def count_ai(client) -> list[str]:
    """Every AI call from here on, by kind; the calls still go through."""
    ai, calls = client.app.state.ai, []
    complete, complete_many = ai.complete, ai.complete_many

    def one(*args, **kw):
        calls.append("complete")
        return complete(*args, **kw)

    def many(*args, **kw):
        calls.append("complete_many")
        return complete_many(*args, **kw)

    ai.complete, ai.complete_many = one, many
    return calls


def answer_store(client) -> list[str]:
    """Store c holding handbook("") as x.md and the linked Documents set cq; the ids of c."""
    add_documents(client)
    store_import(client, "x.md", handbook(""))
    r = client.post("/eval-sets", json={"name": "cq", "qa_type": "documents", "store": "c"})
    assert r.status_code == 201, r.text
    return sorted(h.id for h in passages(client))


def ask_answer(client, ids, confirm=False, name="cq", **body):
    body = {"question": "How many days?", "passages": ids, "confirm": confirm, **body}
    return client.post(f"/eval-sets/{name}/answer", json=body)


def unloadable(client, name: str = "c") -> None:
    """Records the store as built with gone/model, which this service can't load."""
    models = client.app.state.models
    real = models.dense_dim

    def dense_dim(model=None):
        if model == "gone/model":
            raise ValueError("gone/model is not in fastembed's supported models")
        return real(model)

    models.dense_dim = dense_dim
    conn = client.app.state.conn
    conn.execute("update collections set dense_model = 'gone/model' where name = ?", (name,))


def bare_client(tmp_path) -> TestClient:
    """A fresh install: no connection, no model and no job set up at all."""
    app = create_app(settings(tmp_path))
    fake_embeddings(app.state.models)
    app.state.worker.idle_seconds = 0.01
    return TestClient(app)


def add_text(client, filename, body, source_name="docs", name="c"):
    """A text file added the chat's way: upload with accept=text, then import under the
    preview's mapping."""
    up = client.post(
        f"/collections/{name}/uploads",
        params={"filename": filename, "kind": "items", "accept": "text"},
        content=body,
    )
    assert up.status_code == 200, up.text
    r = do_import(client, up.json(), source_name, name=name, confirm=True)
    assert r.status_code == 202, r.text
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "done", job
    return job


def source_ids(client, source, name="c"):
    return client.app.state.store.source_ids(name, source)


SCIBENCH = Path(__file__).parents[3] / "packages" / "core" / "tests" / "data" / "scibench_mini"


def building(prompt: str) -> str:
    if '"principle"' in prompt:
        return '{"principle": "P.", "method": "M.", "answer": "A"}'
    return '{"terms": ["kinematics"]}'


def scibench_source(client) -> list[Item]:
    """Collection p (paper 2024, science): the scibench mini items, and ITEMS under a second
    source "hand" that is switched off; a question asked of each source."""
    from ragqa_core.datasets import load_scibench

    app = client.app.state
    r = client.post("/collections", json={"name": "p", "qa_type": "science"})
    assert r.status_code == 201, r.text
    items, _ = load_scibench(app.settings.home, {p.name: p for p in SCIBENCH.glob("*.json")})
    spec = app.indexer.spec("p")
    app.indexer.add(spec, items, "scibench")
    app.indexer.add(spec, [Item(**i) for i in ITEMS], "hand")
    db.save_questions(
        app.conn,
        "p",
        [
            Question("q1", "a ball falls", None, ["3"], [items[0].id], {"source": "scibench"}),
            Question("q2", "enzyme rate", None, ["A"], ["enzyme-item"], {"source": "hand"}),
        ],
    )
    db.set_source_enabled(app.conn, "p", "hand", False)
    return items


TIDES = jsonl({"id": "n1", "text": "so the tides follow the moon each day we saw"})


HAND = {
    **MCQ,
    "id": "h1",
    "text": "the tides follow the moon each day",
    "meta": {"solution": "Rates rise."},
}


OTHER = {**MCQ, "id": "h2", "text": "an unrelated question", "meta": {"solution": "Rates rise."}}


def small_model(client, monkeypatch) -> None:
    """fake/small embeds at 64 dimensions and is offered alongside fastembed's own list."""
    fake_embeddings(client.app.state.models, dims={"fake/small": 64})
    offered = {"fake/small": {"dim": 64, "size_gb": 0.1}}
    monkeypatch.setattr(search_defaults, "supported_embedders", lambda: offered)


def reembed(client, name="c", model="fake/small"):
    return client.post(f"/collections/{name}/reembed", json={"model": model})


NO_KEYWORDS_MODEL = "connect a model for keywords first"
