import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from api_helpers import (
    PAIR,
    TURBO,
    add_collection,
    make_client,
    set_running,
    settings,
    wait,
)
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import TypeAdapter
from starlette.routing import Match, Mount, Route

from ragqa_api.app import create_app
from ragqa_core import db, search_defaults
from ragqa_core.retriever import METHOD_NAMES
from ragqa_core.types import CollectionSpec, Item


def test_health(client):
    assert client.get("/health").json() == {"ok": True, "ai": True}


def test_health_reports_missing_key(tmp_path):
    assert make_client(tmp_path, api_key=None).get("/health").json()["ai"] is False


def test_basic_auth(tmp_path):
    c = TestClient(create_app(settings(tmp_path, username="u", password="p")))
    assert c.get("/health").status_code == 401
    assert c.get("/health").headers["www-authenticate"] == "Basic"
    assert c.get("/health", auth=("u", "wrong")).status_code == 401
    assert c.get("/health", auth=("u", "p")).status_code == 200


def test_basic_auth_handles_non_ascii(tmp_path):
    import base64

    c = TestClient(create_app(settings(tmp_path, username="ü", password="p")))

    def header(user, password):
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}

    assert c.get("/health", headers=header("é", "x")).status_code == 401
    assert c.get("/health", headers=header("ü", "p")).status_code == 200


def test_half_set_auth_is_503(tmp_path):
    r = TestClient(create_app(settings(tmp_path, username="u"))).get("/health")
    assert r.status_code == 503 and "RAGQA_PASSWORD" in r.json()["detail"]


SNAPSHOT = Path(__file__).with_name("api_snapshot.json")


PAGE_PATHS = {
    "/",
    "/c/{name}",
    "/e/{name}",
    "/c/{name}/runs/{id}",
    "/worker",
    "/lab",
    "/chat",
    "/settings",
    "/board",
    "/static",
}


def api_shape(app) -> str:
    """The paths and schemas of the app's OpenAPI document, without `info` (its version
    changes on release)."""
    spec = app.openapi()
    shape = {"paths": spec["paths"], "schemas": spec["components"]["schemas"]}
    return json.dumps(shape, indent=1, sort_keys=True)


def sample_path(path: str) -> str:
    """A URL the route answers: x1 for each path parameter, and a/b for a :path one."""
    return re.sub(r"\{\w+(:path)?\}", lambda m: "a/b" if m.group(1) else "x1", path)


def in_schema(route) -> bool:
    return isinstance(route, APIRoute) and route.include_in_schema


def test_api_matches_the_snapshot(tmp_path):
    """RAGQA_WRITE_SNAPSHOT=1 rewrites the file first; never set it in CI."""
    shape = api_shape(create_app(settings(tmp_path / "a")))
    assert api_shape(create_app(settings(tmp_path / "b"))) == shape  # the home is not in it
    if os.environ.get("RAGQA_WRITE_SNAPSHOT") == "1":
        SNAPSHOT.write_text(shape + "\n")
    assert json.loads(shape) == json.loads(SNAPSHOT.read_text())


def test_every_sample_path_reaches_one_route(tmp_path):
    app = create_app(settings(tmp_path))
    for route in filter(in_schema, app.routes):
        assert isinstance(route, APIRoute)
        path = sample_path(route.path)
        for method in route.methods:
            scope = {"type": "http", "path": path, "method": method}
            hits = [r for r in app.routes if r.matches(scope)[0] == Match.FULL]
            assert hits == [route], (method, path, hits)


DOCS = [Path(__file__).parents[3] / name for name in ("README.md", "REFERENCE.md")]
REFERENCE = DOCS[1]


def doc_curl_commands() -> list[str]:
    """Each curl command in the sh blocks of DOCS, its continuation lines joined."""
    text = "".join(doc.read_text() for doc in DOCS)
    blocks = "".join(re.findall(r"```sh\n(.*?)```", text, re.S))
    lines = blocks.replace("\\\n", " ").splitlines()
    return [line.strip() for line in lines if line.strip().startswith("curl ")]


def test_doc_curl_commands_fit_the_routes_and_request_models(tmp_path):
    app = create_app(settings(tmp_path))
    commands = doc_curl_commands()
    assert len(commands) == 18
    checked = 0
    for command in commands:
        method = re.search(r"-X (\w+)", command)
        sends = " -d " in command or "--data-binary" in command
        verb = method.group(1) if method else "POST" if sends else "GET"
        url = re.search(r"""["']?(?:\$api|https?://[^/\s"']+)(/[^\s"'?]*)""", command)
        assert url, command
        path = re.sub(r"<[^>]*>|\{[^}]*\}", "x1", url.group(1))
        scope = {"type": "http", "path": path, "method": verb}
        routes = [r for r in app.routes if r.matches(scope)[0] == Match.FULL]
        assert any(in_schema(r) for r in routes), command  # the method and path are a route
        body = re.search(r" -d '([^']*)'", command)
        if body is None:  # an upload's raw file, not a JSON body
            continue
        (route,) = routes
        assert isinstance(route, APIRoute) and route.body_field, command
        sent = json.loads(re.sub(r"<[^>]*>", "x1", body.group(1)))
        TypeAdapter(route.body_field.field_info.annotation).validate_python(sent)
        checked += 1
    assert checked == 16


def route_key(method: str, path: str) -> tuple[str, str]:
    """A method and path with every path parameter as {} and no query string."""
    return method, re.sub(r"\{[^}]*\}", "{}", path.split("?")[0])


def test_reference_lists_every_route(tmp_path):
    app = create_app(settings(tmp_path))
    routes = {
        route_key(method, r.path)
        for r in app.routes
        if isinstance(r, APIRoute) and in_schema(r)
        for method in r.methods
    }
    table = REFERENCE.read_text().split("\n## Endpoints\n")[1].split("\n## ")[0]
    rows = [line for line in table.splitlines() if line.startswith("| `")]
    listed: set[tuple[str, str]] = set()
    for row in rows:
        first = row.split("|")[1]
        spans = re.findall(r"`(GET|POST|PUT|PATCH|DELETE) (/[^`]*)`", first)
        named = {route_key(method, path) for method, path in spans}
        assert named and named <= routes, row  # each row names real routes, and only those
        listed |= named
    assert sorted(routes - listed) == []


def test_every_request_model_forbids_extra_fields(tmp_path):
    spec = create_app(settings(tmp_path)).openapi()
    schemas = spec["components"]["schemas"]

    def refs(schema) -> set[str]:
        """The components a schema names, through $ref, anyOf and any nesting."""
        if isinstance(schema, list):
            return set().union(*map(refs, schema))
        if not isinstance(schema, dict):
            return set()
        named = {schema["$ref"].rsplit("/", 1)[1]} if "$ref" in schema else set()
        return named.union(*map(refs, schema.values()))

    todo: set[str] = set()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if (method, path) == ("put", "/collections/{name}/search"):
                continue  # a free dict, checked by search_settings (decision D leaves it)
            if "requestBody" in op:
                named = refs(op["requestBody"]["content"]["application/json"]["schema"])
                assert named, (method, path)  # a body with no model would be unchecked
                todo |= named
    models: set[str] = set()
    while todo:
        name = todo.pop()
        models.add(name)
        todo |= refs(schemas[name]) - models
    assert {"ChatTurn", "ItemIn", "QuestionIn"} <= models  # nested models count
    open_ = sorted(m for m in models if schemas[m].get("additionalProperties") is not False)
    assert open_ == []


def test_pages_and_static_come_after_the_api(tmp_path):
    app = create_app(settings(tmp_path))
    docs = {app.openapi_url, app.docs_url, f"{app.docs_url}/oauth2-redirect", app.redoc_url}
    routes = [r for r in app.routes if not (isinstance(r, Route) and r.path in docs)]
    last_api = max(i for i, r in enumerate(routes) if in_schema(r))
    assert all(in_schema(r) for r in routes[:last_api])
    after = routes[last_api + 1 :]
    paths = [r.path for r in after if isinstance(r, (APIRoute, Mount))]
    assert len(paths) == len(after) and set(paths) == PAGE_PATHS


def test_every_route_needs_the_login(tmp_path):
    app = create_app(settings(tmp_path, username="u", password="p"))
    c = TestClient(app)
    calls = []
    for route in app.routes:
        if isinstance(route, Mount):
            calls.append(("GET", f"{route.path}/ui.js"))
        else:
            assert isinstance(route, Route)
            calls += [(method, sample_path(route.path)) for method in route.methods or ()]
    assert ("GET", "/health") in calls and ("GET", "/") in calls
    for method, path in calls:
        assert c.request(method, path).status_code == 401, (method, path)


STATIC = Path(__file__).resolve().parents[1] / "src/ragqa_api/static"


PUBLIC = {
    "/collections",
    "/eval-sets",
    "/query",
    "/eval",
    "/jobs",
    "/budget",
    "/health",
    "/catalog",
    "/status",
    "/ai",
    "/chat",
    "/chats",
    "/search-defaults",
}


PAGES = ("/", "/c/x", "/c/x/runs/abc", "/e/x", "/worker", "/lab", "/chat")


def test_pages_are_served(client):
    for path in PAGES:
        r = client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    board = client.get("/board", follow_redirects=False)
    assert board.status_code == 307 and board.headers["location"] == "/"
    for path in PAGES:  # /favicon.ico is not served, so every page opts out of asking for it
        assert 'rel="icon"' in client.get(path).text, path
    assert not (STATIC / "board.html").exists()


def test_eval_sets_path_is_a_page_for_a_browser_and_the_list_for_the_api(client):
    page = client.get("/eval-sets", headers={"accept": "text/html,application/xhtml+xml"})
    assert page.headers["content-type"].startswith("text/html")
    assert page.headers["vary"] == "Accept"
    api = client.get("/eval-sets")
    assert api.json() == [] and api.headers["vary"] == "Accept"


def test_html_patterns_survive_the_v_flag():
    """Chrome compiles `pattern` with the v flag, where an unescaped - in a class is a syntax error
    unless it is a range (a-z): the constraint is dropped with a console error. Python has no v
    flag, so check that rule. A js template literal halves each double backslash, so undo that."""
    found = 0
    for path in [*STATIC.glob("*.html"), *STATIC.glob("*.js")]:
        for value in re.findall(r'pattern="([^"]*)"', path.read_text()):
            found += 1
            if path.suffix == ".js":
                value = value.replace("\\\\", "\\")
            for cls in re.findall(r"\[((?:\\.|[^\]\\])*)\]", value):
                body = re.sub(r"(?:\\.|[^\\-])-(?:\\.|[^\\-])", "", cls)  # ranges are fine
                assert "-" not in re.sub(r"\\.", "", body), (path.name, value)
    assert found >= 2


def test_static_assets_are_served(client):
    assert client.get("/static/ui.css").status_code == 200
    assert "export" in client.get("/static/ui.js").text


def test_static_is_behind_basic_auth(tmp_path):
    c = TestClient(create_app(settings(tmp_path, username="u", password="p")))
    for path in ("/static/ui.js", "/static/ui.css", "/", "/lab"):
        assert c.get(path).status_code == 401
        assert c.get(path, auth=("u", "p")).status_code == 200


def test_moved_page_css_is_scoped():
    css = (STATIC / "ui.css").read_text()
    sections = re.split(r"/\* (page-[a-z]+) \*/", css)[1:]  # [name, rules, name, rules, ...]
    assert sections[0::2] == ["page-chat", "page-settings", "page-lab"]
    for name, rules in zip(sections[0::2], sections[1::2], strict=True):
        rules = re.sub(r"/\*.*?\*/", "", rules, flags=re.S)
        selectors = [s for head in re.findall(r"([^{}]+)\{", rules) if head.strip()[0] != "@"
                     for s in head.split(",")]  # fmt: skip
        assert selectors, name
        for selector in selectors:
            assert selector.strip().startswith(f".{name} "), (name, selector)
    for path in [*STATIC.glob("*.html"), *STATIC.glob("*.js")]:
        text = path.read_text()
        assert "<style" not in text and 'style="' not in text, path.name


def test_pages_only_call_public_endpoints():
    import re

    called: set[str] = set()
    for path in STATIC.glob("*.[hj][ts]*"):
        text = path.read_text()
        assert not re.search(r"(?:fetch|api)\(\s*`\$\{", text), path.name  # unreadable target
        for _, target in re.findall(r"""(?:fetch|api)\(\s*([`'"])(/[^`'"]*)""", text):
            called.add("/" + re.match(r"/?([A-Za-z0-9_-]*)", target).group(1))
        assert "http://" not in text and "https://" not in text  # no external requests
    assert called <= PUBLIC, called - PUBLIC
    assert {"/collections", "/query", "/catalog", "/status", "/budget"} <= called
    assert {p.name for p in STATIC.glob("*.html")} >= {"collections.html", "lab.html"}


NODE = shutil.which("node")


CHECK = """
import * as u from './ui.mjs';
const out = {
  a: u.pts(6.2, 1.8, 10.4), b: u.pts(-7, -12, -2), c: u.pts(1, -2, 4), d: u.pts(0.4, -3, 3),
  e: u.pct(0.776), f: u.esc('<a href="x">&</a>'),
  g: u.clock('2026-10-23T14:02:00', new Date('2026-10-23T09:00:00')),
  h: u.clock('2026-10-21T14:02:00', new Date('2026-10-23T09:00:00')),
  n: [u.pct(null), u.pct(undefined), u.pct(NaN), u.pct(Infinity), u.pct(0)],
  o: [u.pts(undefined, 1, 2), u.pts(null, 1, 2), u.pts(NaN, 1, 2), u.pts(-2.5, -4, -1)],
  p: [u.clock('garbage'), u.clock(null), u.clock(undefined)],
  q: [u.pts(2.5, 1.5, 3.5), u.pts(5, null, undefined)],
};
const labels = {methods: {kw_individual: {label: 'Keywords per list'}, hybrid: {label: 'Hybrid'}},
                metrics: {accuracy: {label: 'Answered correctly'}, em: {label: 'Exact match'}},
                metric_labels_by_type: {table: {em: 'Exact answer'}}};
const ans = (verdict, diff, lo, hi, metric = 'accuracy', qaType = 'mcq') => u.verdictRest(
  {candidate: 'kw_individual', reference: 'hybrid', metric, verdict, diff, lo, hi},
  qaType, labels);
out.j = ans('yes', 6.2, 1.8, 10.4);
out.k = ans('no', -7, -12, -2);
out.l = ans('cant_tell', 1, -2, 4);
out.t = ans('yes', 6, 2, 10, 'em', 'table');
out.u = ans('yes', 6, 2, 10, 'em', 'mcq');
out.v = ['rekeying', 'retagging', 'copying', 'stopped_budget', 'stopped_no_key', 'cancelled']
  .map(u.pill);
out.w = [u.count(11656), u.count(0)];
out.x = [0.004, 0, 0.4141, 12].map(u.usd);
const box = () => ({innerHTML: '', querySelector: () => ({addEventListener() {}})});
out.z = [{est_usd: null, calls: 3}, {est_usd: 0.4141, calls: 1}].map((estimate) => {
  const el = box();
  u.confirmSpend(el, estimate, () => {});
  return el.innerHTML;
});
const sim = {metric: 'solution_sim@1', proxy_trusted: true};
out.y = [{unit: 'raw'}, {unit: 'match'}].map((scale) => u.verdictNotes({...sim, scale}));
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_ui_formatting(tmp_path):
    (tmp_path / "ui.mjs").write_text((STATIC / "ui.js").read_text())
    (tmp_path / "check.mjs").write_text(CHECK)
    r = subprocess.run([str(NODE), "check.mjs"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout)
    assert got["a"] == "+6 pts (2–10)"
    assert got["b"] == "−7 pts (−12 to −2)"
    assert got["c"] == "+1 pt (−2 to 4)"
    assert got["d"] == "+0 pts (−3 to 3)"
    assert got["e"] == "78%"
    assert got["f"] == "&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;"
    assert got["g"] == "14:02" and got["h"] == "21 Oct 14:02"
    assert got["n"] == ["–", "–", "–", "–", "0%"]
    assert got["o"] == ["–", "–", "–", "−3 pts (−4 to −1)"]
    assert got["p"] == ["–", "–", "–"]
    assert got["q"] == ["+3 pts (2–4)", "+5 pts"]
    assert got["j"] == "Keywords per list +6 pts (2–10) answered correctly vs Hybrid"
    assert got["k"].startswith("Keywords per list −7 pts")
    assert got["l"].startswith("Keywords per list +1 pt")
    assert got["t"] == "Keywords per list +6 pts (2–10) exact answer vs Hybrid"
    assert got["u"].endswith("exact match vs Hybrid")  # no label for this type: the plain one
    assert [re.sub("<[^>]*>", "", v) for v in got["v"]] == [
        "Re-keying",
        "Re-tagging",
        "Copying",
        "Stopped budget",
        "Stopped no key",
        "Cancelled",
    ]
    assert "pill run" in got["v"][2]
    assert got["w"] == ["11,656", "0"]
    assert got["x"] == ["< $0.01", "$0.00", "$0.41", "$12.00"]
    unpriced, priced = got["z"]  # no model to price with: unknown, not $0.00
    assert "cost unknown" in unpriced and "$0.00" not in unpriced and "≈ $0.41" in priced
    raw, match = got["y"]  # a verdict's own scale marks a raw reasoning match
    assert len(raw) == 1 and "no chance level" in raw[0] and match == []


def test_restart_keeps_queued_jobs_and_interrupts_running(tmp_path):  # Review focus 5
    app = create_app(settings(tmp_path))
    conn = app.state.conn
    q = db.create_job(conn, "rekey", "c", {"version": "v"})
    r = db.create_job(conn, "load", "c", {})
    conn.execute("update jobs set status='running' where id=?", (r,))
    db.interrupt_running(conn)
    assert (
        db.get_job(conn, r)["status"] == "interrupted" and db.get_job(conn, q)["status"] == "queued"
    )
    # The lifespan does the same before it starts the worker (kept off here, so q stays queued).
    r2 = db.create_job(conn, "load", "c", {})
    set_running(conn, r2)
    app.state.worker.start = lambda: None
    app.state.models.warm_rerank_in_background = lambda model=None: None  # builds no model
    with TestClient(app):
        assert db.get_job(conn, r2)["status"] == "interrupted"
        assert db.get_job(conn, q)["status"] == "queued"


def test_catalog_lists_every_method(client):
    body = client.get("/catalog").json()
    assert set(body["methods"]) == set(METHOD_NAMES)
    assert body["methods"]["kw_individual"]["label"] == "Keywords per list"
    assert body["answer_pair"] == PAIR and body["qa_types"]["mcq"]["label"]


def test_catalog_serves_the_controls(client):
    assert client.get("/catalog").json()["controls"] == ["random_example", "fixed_example"]


def write_legacy(app, name="c"):
    models, store = app.state.models, app.state.store
    app.state.indexer.create(name, "mcq", False)
    items = [Item("sqa-1", "enzyme rate", "A", None, {}), Item("note-1", "market price", "B")]
    texts = [i.text for i in items]
    store.upsert_items(name, items, models.embed(texts), models.sparse(texts), [None, None])
    store.upsert_chunks(name, [i.id for i in items], texts, models.embed(texts), [None, None])


def backfills(conn):
    return [j for j in db.list_jobs(conn) if j["kind"] == "backfill"]


def test_startup_queues_one_backfill_for_unlabelled_collections(tmp_path):
    client = make_client(tmp_path)
    write_legacy(client.app)
    with client:
        jobs = backfills(client.app.state.conn)
        assert len(jobs) == 1
        assert wait(client, jobs[0]["id"])["status"] == "done"
        assert client.app.state.store.unsourced("c") == 0
    with client:
        assert len(backfills(client.app.state.conn)) == 1


def test_labelled_collections_queue_no_backfill(tmp_path):
    client = make_client(tmp_path)
    with client:
        add_collection(client, lists=False)
    with client:
        assert backfills(client.app.state.conn) == []


def test_startup_skips_collections_qdrant_lacks(tmp_path):
    client = make_client(tmp_path)
    conn = client.app.state.conn
    db.save_collection(conn, CollectionSpec("ghost", "mcq", False, "fake/dense", 256, None))
    with client:
        assert backfills(conn) == []


def test_startup_ensures_the_source_indexes_of_every_existing_collection(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    store = client.app.state.store
    store.create("old", 256)
    db.save_collection(client.app.state.conn, CollectionSpec("old", "mcq", False, "f", 256, None))
    db.save_collection(client.app.state.conn, CollectionSpec("ghost", "mcq", False, "f", 256, None))
    seen = []
    monkeypatch.setattr(store, "ensure_indexes", seen.append)
    with client:
        assert seen == ["old"]


def test_the_service_warms_the_settings_reranker_at_start(tmp_path):
    client = make_client(tmp_path)
    search_defaults.save(client.app.state.conn, {"rerank_model": TURBO})
    called = []
    client.app.state.models.warm_rerank_in_background = called.append  # type: ignore[attr-defined]
    with client:
        pass
    assert called == [TURBO]


def test_unknown_path_is_json_for_a_client_and_html_for_a_browser(client):
    assert client.get("/nothing").json() == {"detail": "Not Found"}
    r = client.get("/nothing", headers={"accept": "text/html"})
    assert r.status_code == 404 and "Page not found" in r.text
    # a route's own 404 stays JSON even for a browser
    assert client.get("/collections/nope", headers={"accept": "text/html"}).json()["detail"]
