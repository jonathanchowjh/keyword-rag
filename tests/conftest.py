"""Fixtures for the page tests: the real app on fakes, served over HTTP to a real browser."""

import socket
import threading
import time
from dataclasses import dataclass, replace

import httpx
import pytest
import uvicorn

from ragqa_api.app import create_app
from ragqa_core import db
from ragqa_core.fakes import fake_embeddings, fake_replies
from ragqa_core.settings import load_settings
from ragqa_core.types import Question

ITEMS = [
    {
        "id": "enzyme-item",
        "text": "enzyme kinetics rate experiment",
        "answer": "A",
        "explanation": "Rates rise.",
        "meta": {"skill": "enzymes"},
    },
    {
        "id": "market-item",
        "text": "stock market price demand",
        "answer": "B",
        "explanation": "Prices fall.",
        "meta": {"skill": "markets"},
    },
    {
        "id": "volcano-item",
        "text": "volcano magma lava pressure",
        "answer": "C",
        "explanation": "Magma rises.",
        "meta": {"skill": "volcanoes"},
    },
]
LISTS = {
    "topic": ["enzyme", "market", "volcano", "optics"],
    "method": ["experiment", "survey", "sampling"],
    "common": ["rate", "price", "lava", "light"],
}
TOPICS = ["enzyme", "market", "volcano"]


@dataclass
class Seeded:
    url: str
    app: object
    run_id: str
    vector_only_run_id: str = ""

    def build(
        self, name: str, run: bool = False, summarize: bool = False, qa_type: str = "mcq"
    ) -> str:
        """A collection (mcq unless `qa_type` says otherwise) with items, lists and 40 questions;
        with `run`, one answered run of kw_individual, hybrid and random_example, done. Returns
        the run id ("" without)."""
        url = self.url
        r = httpx.post(
            f"{url}/collections", json={"name": name, "qa_type": qa_type, "summarize": summarize}
        )
        assert r.status_code == 201, r.text
        r = httpx.post(
            f"{url}/collections/{name}/documents", json={"items": ITEMS, "confirm": True}
        )
        assert r.status_code == 200, r.text
        r = httpx.put(f"{url}/collections/{name}/keyword-lists", json=LISTS)
        assert r.status_code == 202, r.text
        assert wait_job(url, r.json()["job_id"])["status"] == "done"
        db.save_questions(self.app.state.conn, name, questions())  # type: ignore[attr-defined]
        if not run:
            return ""
        return self.run(name, ["kw_individual", "hybrid", "random_example"])

    def wait(self, job_id: str) -> dict:
        return wait_job(self.url, job_id)

    def queue_eval(self, name: str) -> str:
        """Queue one answered eval on `name`; it stays queued while the worker is stopped."""
        body = {"collection": name, "methods": ["kw_individual"], "answer": True, "confirm": True}
        r = httpx.post(f"{self.url}/eval", json=body)
        assert r.status_code == 202, r.text
        return r.json()["job_id"]

    def run(self, name: str, methods: list[str]) -> str:
        """One answered run of `methods` on `name`, done. Returns its id."""
        body = {"collection": name, "methods": methods, "answer": True, "confirm": True}
        r = httpx.post(f"{self.url}/eval", json=body)
        assert r.status_code == 202, r.text
        run_id = r.json()["job_id"]
        assert wait_job(self.url, run_id)["status"] == "done"
        return run_id


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_job(url: str, job_id: str, seconds: float = 30) -> dict:
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = httpx.get(f"{url}/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def questions(n: int = 40) -> list[Question]:
    return [
        Question(
            f"q{i}",
            f"{TOPICS[i % 3]} question {i}",
            ["x", "y", "z"],
            ["A"],
            [],
            {"skill": TOPICS[i % 3], "solution": "Rates rise."},
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    """Collection `c` (mcq, 3 items, lists, 40 questions) with one answered run, done."""
    home = tmp_path_factory.mktemp("home")
    base = load_settings({"ANTHROPIC_API_KEY": "test"})
    fake = {"dense_model": "fake/dense", "judge_model": "fake/judge", "rerank_model": "fake/rerank"}
    app = create_app(replace(base, home=home, **fake))
    fake_embeddings(app.state.models)
    fake_replies(app.state.ai, lambda p: "Answer: A")
    app.state.worker.idle_seconds = 0.01
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(f"{url}/health").status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.05)
        else:
            raise AssertionError("server did not start")
        seeded = Seeded(url, app, "")
        seeded.run_id = seeded.build("c", run=True)
        # On its own collection: a vector-only run on `c` would become its newest answer.
        seeded.build("vo")
        seeded.vector_only_run_id = seeded.run("vo", ["vector"])
        yield seeded
    finally:
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def page(context):
    """A page that fails its test on any console error or failed request."""
    p = context.new_page()
    p.set_default_timeout(5000)
    problems: list[str] = []
    p.on("console", lambda m: problems.append(f"console: {m.text}") if m.type == "error" else None)
    p.on("pageerror", lambda e: problems.append(f"page error: {e}"))
    p.on("requestfailed", lambda r: problems.append(f"request failed: {r.url}"))
    p.on("response", lambda r: problems.append(f"{r.status} {r.url}") if r.status >= 400 else None)
    p.expected = []  # a test that provokes a 4xx on purpose lists its status here, e.g. "409"
    yield p
    # A route handler still fetching when the test ends must not fail the next test's setup.
    p.unroute_all(behavior="ignoreErrors")
    problems = [x for x in problems if not any(e in x for e in p.expected)]
    assert not problems, problems
