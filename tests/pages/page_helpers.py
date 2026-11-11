"""What the page tests share: starting pages with fixed routes, and building stores and chats."""

import json
from contextlib import contextmanager

import httpx
from playwright.sync_api import expect

from ragqa_core import db, search_defaults
from ragqa_core.fakes import fake_replies


def interrupt(seeded, run_id):
    db.update_job(seeded.app.state.conn, run_id, status="interrupted")


@contextmanager
def worker_stopped(seeded, name):
    """Stop the shared worker and queue an eval on its own collection `name`. Always leaves the
    worker running again with that job no longer queued."""
    worker = seeded.app.state.worker
    seeded.build(name)  # its lists are keyed by the worker, so before stopping it
    worker.stop()
    try:
        yield seeded.queue_eval(name)
    finally:
        try:
            for job in httpx.get(f"{seeded.url}/jobs", params={"status": "queued"}).json():
                httpx.post(f"{seeded.url}/jobs/{job['id']}/cancel")
        finally:
            worker.start()


def route_load(page, name, reply):
    """Answer POST .../{name}/load with `reply(body)` -> (status, json); returns the bodies seen."""
    seen = []

    def handle(route):
        if route.request.method != "POST":
            return route.continue_()
        body = route.request.post_data_json
        seen.append(body)
        status, payload = reply(body)
        route.fulfill(status=status, json=payload)

    page.route(f"**/collections/{name}/load", handle)
    return seen


def jobs_done(seeded, name):
    for j in httpx.get(f"{seeded.url}/jobs", params={"collection": name, "kind": "import"}).json():
        assert seeded.wait(j["id"])["status"] == "done"


def upload_and_preview(page, kind, path):
    page.locator(f"#imp-{kind}-file").set_input_files(str(path))
    page.locator(f"#imp-{kind}-go").click()


def sources(seeded, name):
    return httpx.get(f"{seeded.url}/collections/{name}/sources").json()


@contextmanager
def search_defaults_restored(seeded):
    """Puts what a test changes back: store c's candidates, and every default but the fixture's."""
    try:
        yield
    finally:
        httpx.patch(seeded.url + "/collections/c", json={"chat_k": None})
        httpx.put(
            seeded.url + "/search-defaults", json={f: None for f in search_defaults.RECOMMENDED}
        )
        httpx.put(seeded.url + "/search-defaults", json={"rewrite": False})


def ai_state(seeded):
    return httpx.get(seeded.url + "/ai").json()


def mcq_set(seeded, name, n):
    """An mcq eval set of `n` questions (ids s0..), made through the API."""
    r = httpx.post(f"{seeded.url}/eval-sets", json={"name": name, "qa_type": "mcq"})
    assert r.status_code == 201, r.text
    qs = [
        {"id": f"s{i}", "text": f"question {i}", "choices": ["x", "y"], "gold_answers": ["A"]}
        for i in range(n)
    ]
    r = httpx.put(f"{seeded.url}/eval-sets/{name}/questions", json={"questions": qs})
    assert r.status_code == 200, r.text


def two_sections() -> str:
    """A .md of two sections, each too long to share a chunk: two passages."""
    return "# Alpha\n\n" + "alpha " * 160 + "\n\n# Brine\n\n" + "brine " * 160


def documents_store(seeded, name, passages=True):
    """A Documents store `name`; with `passages`, holding two_sections() as x.md."""
    r = httpx.post(f"{seeded.url}/collections", json={"name": name, "qa_type": "documents"})
    assert r.status_code == 201, r.text
    if not passages:
        return
    up = httpx.post(
        f"{seeded.url}/collections/{name}/uploads",
        params={"filename": "x.md", "kind": "items"},
        content=two_sections().encode(),
    ).json()
    body = {
        "upload_id": up["upload_id"],
        "kind": "items",
        "source_name": "x",
        "mapping": up["preview"]["mapping"],
        "confirm": True,
    }
    r = httpx.post(f"{seeded.url}/collections/{name}/imports", json=body)
    assert r.status_code == 202, r.text
    assert seeded.wait(r.json()["job_id"])["status"] == "done"


def tag_jobs(seeded, name):
    return httpx.get(f"{seeded.url}/jobs", params={"collection": name, "kind": "tag_build"}).json()


def question_map(seeded, name):
    return {q["id"]: q for q in httpx.get(f"{seeded.url}/eval-sets/{name}").json()["questions"]}


CHAT_ATTEMPT = {
    "n": 1,
    "attempt": 1,
    "search_text": "asked",
    "decisions": [],
    "evidence": [],
    "answer": "Rates rise [1] with <b>heat</b> [2].",
    "citations": [
        {
            "n": 1,
            "id": "enzyme-item",
            "text": "enzyme " + "kinetics " * 60,
            "source": "hand",
            "part": None,
            "parts": None,
            "store": "c",
        },
        {
            "n": 2,
            "id": "other",
            "text": "short <i>passage</i>",
            "source": None,
            "part": None,
            "parts": None,
            "store": "c",
        },
    ],
    "usd": 0.0123,
    "error": None,
    "model": "anthropic/fake",
    "pipeline": {
        "search_from": "message",
        "rerank_text": "asked",
        "merged_by": "rerank",
        "rerank": True,
        "rewrite": False,
        "follow_up_turns": 2,
        "evidence_size": 8,
        "rrf_k": 60,
        "pool": 2,
        "notes": [],
        "passages_only": False,
        "rerank_model": "jinaai/jina-reranker-v1-tiny-en",
        "rerank_pool": 20,
    },
}

# The passages the answer was given, each with its number in the prompt: what [n] links to.
CHAT_ATTEMPT["evidence"] = [
    {**c, "rank": c["n"], "score": 1.2, "relevance": 0.77, "channels": ["meaning", "words"]}
    for c in CHAT_ATTEMPT["citations"]
]


def make_chat(seeded, stores=("c",), title=None) -> str:
    body = {"stores": list(stores), **({"title": title} if title else {})}
    r = httpx.post(f"{seeded.url}/chats", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def open_chat(page, seeded, stores=("c",), title=None) -> str:
    """A new saved chat on `stores`, opened on the page."""
    chat_id = make_chat(seeded, stores, title)
    page.goto(f"{seeded.url}/chat#{chat_id}")
    expect(page.locator("#chips .chip")).to_have_count(len(stores))
    return chat_id


def chat_route(page, handler):
    """Answer each POST of a message to a chat."""
    page.route("**/chats/*/messages", lambda route: handler(route))


def chat_message(route, n=1, **attempt):
    """What the server sends back for the message POSTed at `route`: its attempt is
    CHAT_ATTEMPT with `attempt`'s changes."""
    content = json.loads(route.request.post_data)["content"]
    return {"n": n, "content": content, "attempts": [{**CHAT_ATTEMPT, "n": n, **attempt}]}


def chat_ask(page, text):
    page.locator("#box").fill(text)
    page.locator("#send").click()


REBUILD_MODEL = "BAAI/bge-small-en-v1.5"


def your_app(page):
    """The chat's Use from your app section, opened: its two <pre> lines."""
    page.locator("#chat-head summary", has_text="Use from your app").click()
    return page.locator("#chat-head details pre")


def labelling_set(seeded, name, store=True):
    """A Documents store `name` holding two passages of x.md, and an eval set `<name>q` of
    Documents linked to it (with `store`) or to none."""
    documents_store(seeded, name)
    body = {"name": f"{name}q", "qa_type": "documents"}
    if store:
        body["store"] = name
    r = httpx.post(f"{seeded.url}/eval-sets", json=body)
    assert r.status_code == 201, r.text


BUTTON = {"right": "Right", "wrong": "Wrong"}  # a mark, as its button reads


def mark(passage, which):
    passage.get_by_role("button", name=BUTTON[which], exact=True).click()


def pressed(passage, which):
    return passage.get_by_role("button", name=BUTTON[which], exact=True)


def labelled_questions(seeded, name, texts):
    """Adds questions of `texts` to the set `<name>q`; returns their ids in order."""
    ids = []
    for text in texts:
        r = httpx.post(
            f"{seeded.url}/eval-sets/{name}q/questions", json={"question": {"text": text}}
        )
        assert r.status_code == 201, r.text
        ids.append(r.json()["id"])
    return ids


@contextmanager
def replies(seeded, reply):
    """The fake AI answers every prompt with reply(prompt) until the block ends."""
    fake_replies(seeded.app.state.ai, reply)
    try:
        yield
    finally:
        fake_replies(seeded.app.state.ai, lambda p: "Answer: A")


def put_questions(seeded, name, questions, base=None):
    r = httpx.put(
        f"{seeded.url}/eval-sets/{name}/questions",
        json={"questions": questions, "base_version": base},
    )
    assert r.status_code == 200, r.text
    return r.json()["version"]


def documents_run(
    seeded, name, methods, judged_all=True, answer=False, judge=False, references=0, held=True
):
    """A Documents store `name` and its linked set `<name>q` of two questions, each right on
    the store's first passage and judged on both passages (with `judged_all`) or on that one
    alone; then a finished run of `methods` on the set. The first `references` questions carry a
    reference answer; with `held=False` the right passage is not in the store. Returns the run id
    and the ids of the two passages, the right one first."""
    labelling_set(seeded, name)
    pool = httpx.post(f"{seeded.url}/eval-sets/{name}q/pool", json={"question": "alpha"})
    assert pool.status_code == 200, pool.text
    ids = [p["id"] for p in pool.json()["passages"]]
    assert len(ids) == 2, ids
    judged = ids if judged_all else ids[:1]
    gold = ids[:1] if held else [f"{name}-gone"]
    for n, text in enumerate(("how long does alpha last?", "what does alpha hold?")):
        question = {"text": f"{text} ({name})", "gold_ids": gold, "judged_ids": judged}
        if n < references:
            question["gold_answers"] = ["Ten days"]
        r = httpx.post(f"{seeded.url}/eval-sets/{name}q/questions", json={"question": question})
        assert r.status_code == 201, r.text
    body = {
        "collection": name,
        "eval_set": f"{name}q",
        "split": "full",
        "methods": methods,
        "answer": answer,
        "judge": judge,
        "confirm": True,
    }
    r = httpx.post(f"{seeded.url}/eval", json=body)
    assert r.status_code == 202, r.text
    run_id = r.json()["job_id"]
    assert seeded.wait(run_id)["status"] == "done"
    return run_id, ids
