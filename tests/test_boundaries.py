"""Source guards: the package boundary and the hooks that stand in for CI."""

import ast
import os
import re
import subprocess
from pathlib import Path

from ragqa_core import db
from ragqa_core.retriever import METHOD_NAMES

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "packages/core/src"
API_SRC = ROOT / "apps/api/src"
WORKER_SRC = ROOT / "apps/worker/src"
AICONNECT_SRC = ROOT / "packages/aiconnect/src"
WEB_LAYER = {"fastapi", "starlette", "uvicorn", "ragqa_api"}


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def forbidden_imports(src: Path) -> list[str]:
    found = []
    for path in sorted(src.rglob("*.py")):
        for name in imported_modules(path):
            if name.split(".")[0] in WEB_LAYER:
                found.append(f"{path.relative_to(ROOT)}: {name}")
    return found


def modules_importing(target: str, srcs: list[Path]) -> list[str]:
    found = []
    for src in srcs:
        for path in sorted(src.rglob("*.py")):
            if path.name == "fakes.py":
                continue
            names = imported_modules(path)
            if target in names or any(n.startswith(target + ".") for n in names):
                found.append(str(path.relative_to(ROOT)))
    return found


def test_core_never_imports_the_web_layer():
    assert forbidden_imports(CORE_SRC) == []


def test_worker_never_imports_the_web_layer():
    assert forbidden_imports(WORKER_SRC) == []


def test_aiconnect_imports_nothing_from_ragqa_or_web():
    found = []
    for path in sorted(AICONNECT_SRC.rglob("*.py")):
        for name in imported_modules(path):
            top = name.split(".")[0]
            if top.startswith("ragqa_") or top in WEB_LAYER:
                found.append(f"{path.relative_to(ROOT)}: {name}")
    assert found == []


def test_core_never_imports_the_worker():
    assert modules_importing("ragqa_worker", [CORE_SRC]) == []


def test_settings_live_in_core():
    assert (CORE_SRC / "ragqa_core/settings.py").exists()
    assert not (API_SRC / "ragqa_api/config.py").exists()


def test_only_tests_import_fakes():
    assert modules_importing("ragqa_core.fakes", [CORE_SRC, API_SRC, WORKER_SRC]) == []


def test_the_service_never_runs_the_owners_tools():
    # Authored outputs are put in place, and builds published to the Hub, by the owner inside
    # the container; never by a request. What both tools need lives in core, so neither
    # imports the other either.
    for tool in ["ragqa_core.apply_outputs", "ragqa_core.publish", "ragqa_core.replies"]:
        assert modules_importing(tool, [CORE_SRC, API_SRC, WORKER_SRC]) == [], tool


def test_pre_push_stays_fast():
    body = (ROOT / ".githooks/pre-push").read_text()
    code = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("#"))
    assert "pytest" not in code and "docker" not in code
    assert "ragqa-gate-receipt" in code


def test_compose_binds_localhost_only():
    c = (ROOT / "compose.yaml").read_text()  # plain text checks; no PyYAML dependency
    assert '"127.0.0.1:8080:8080"' in c and "image: qdrant/qdrant:v" in c
    qdrant_block = c.split("\n  qdrant:\n")[1].split("\nvolumes:")[0]
    assert "ports:" not in qdrant_block  # the Qdrant service publishes nothing


def test_gate_has_the_slow_steps():
    g = (ROOT / "scripts/gate.sh").read_text()
    assert "pytest -m integration" in g and "image-smoke.sh" in g and "ragqa-gate-receipt" in g
    assert "pytest -m pages" in g


def test_env_example_lists_every_setting():
    names = re.findall(r"^#?\s*([A-Z_]+)=", (ROOT / ".env.example").read_text(), re.M)
    assert set(names) >= {
        "ANTHROPIC_API_KEY",
        "RAGQA_MONTHLY_CAP_USD",
        "RAGQA_MODEL",
        "RAGQA_USERNAME",
        "RAGQA_PASSWORD",
        "RAGQA_DENSE_MODEL",
        "RAGQA_JUDGE_MODEL",
        "RAGQA_RERANK_MODEL",
    }


def test_docker_context_excludes_private_files():
    ignored = (ROOT / ".dockerignore").read_text().split()
    assert {"docs", ".env", ".superpowers", ".remember"} <= set(ignored)


def test_pages_name_no_provider_and_no_credit():
    for path in sorted((API_SRC / "ragqa_api/static").iterdir()):
        assert not re.search(r"claude|anthropic|credit", path.read_text(), re.I), path.name


def test_python_strings_name_no_provider():
    # Provider names live only in the preset and wire files. Lowercase ids
    # ("anthropic", "claude-haiku-5-5") and ANTHROPIC_API_KEY stay legal: they
    # are identifiers, not prose.
    srcs = [CORE_SRC, AICONNECT_SRC, *sorted((ROOT / "apps").glob("*/src"))]
    skip = {AICONNECT_SRC / "aiconnect/presets.py", AICONNECT_SRC / "aiconnect/wire.py"}
    found = []
    for src in srcs:
        for path in sorted(src.rglob("*.py")):
            if path in skip:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and re.search(r"Claude|Anthropic", node.value)
                ):
                    found.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert found == []


GONE = [
    r"store_bars",
    r"chat_bar",
    r"route_profile|route-profile|routeProfile",
    r"\brouting\b|\bRouting\b",
    r"\bRouter\b|ragqa_core\.router",
    r"[Ss]earch anyway",
    r"kept from the last message|_sticky|ON_MERIT",
    r"_drop_bars",
    r"\{name\}/bar|/routing/eval",
    r"check-routing|routing-check|routing-split|Check routing",
    r"bar_source|route_bar|\bownBar\b|\bsetBar\b|barFrom|route-text|routeText|routeDraft",
]
GONE_PY_ONLY = r"\bBar\b"
MIGRATION_TERMS = ("store_bars", "chat_bar", "route_profile", "routing")


def migration_lines(path: Path) -> set[int]:
    """Lines of the one migration function, `_drop_routing`, in db.py."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "_drop_routing":
            return set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return set()


def leftovers() -> list[str]:
    hits = []
    for src in (CORE_SRC, API_SRC, WORKER_SRC):
        for path in sorted(
            p for p in src.rglob("*") if p.suffix in {".py", ".js", ".html", ".css"}
        ):
            allowed = migration_lines(path) if path.name == "db.py" else set()
            for number, line in enumerate(path.read_text().splitlines(), 1):
                found = any(re.search(rx, line) for rx in GONE)
                found = found or (path.suffix == ".py" and re.search(GONE_PY_ONLY, line))
                if found and not (number in allowed and any(t in line for t in MIGRATION_TERMS)):
                    hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    return hits


def files_importing_the_router() -> list[str]:
    hits = []
    dirs = [ROOT / "tests", *ROOT.glob("packages/*/tests"), *ROOT.glob("apps/*/tests")]
    for folder in dirs:
        for path in sorted(folder.rglob("*.py")):
            if path == Path(__file__).resolve():
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"ragqa_core\.router", line):
                    hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    return hits


def test_routing_bars_and_profiles_are_gone():
    problems = leftovers() + files_importing_the_router()
    if (CORE_SRC / "ragqa_core/router.py").exists():
        problems.append("packages/core/src/ragqa_core/router.py still exists")
    for name in ("get_chat_bar", "last_item_job"):
        if hasattr(db, name):
            problems.append(f"db.{name} still exists")
    assert not problems, "\n".join(problems)


def test_compose_pins_project_name():
    c = (ROOT / "compose.yaml").read_text()  # no PyYAML dependency: check the top-level key by text
    assert "\nname: rag-qa\n" in c


def _services_qdrant_image() -> str:
    services = (ROOT / "compose.yaml").read_text().split("\nservices:\n")[1].split("\nvolumes:")[0]
    block = services.split("\n  qdrant:\n")[1]
    return re.search(r"^\s+image:\s*(\S+)", block, re.M).group(1)  # type: ignore[union-attr]


def test_dev_qdrant_reads_its_image_from_compose():
    script = ROOT / "scripts/dev-qdrant.sh"
    assert "qdrant/qdrant:" not in script.read_text()
    env = {k: v for k, v in os.environ.items() if k != "RAGQA_DEV_QDRANT_IMAGE"}
    run = lambda e: subprocess.run(  # noqa: E731
        [str(script), "image"], capture_output=True, text=True, check=True, env=e
    ).stdout.strip()
    assert run(env) == _services_qdrant_image()
    assert run({**env, "RAGQA_DEV_QDRANT_IMAGE": "x"}) == "x"


def _gate_lines() -> list[str]:
    return (ROOT / "scripts/gate.sh").read_text().splitlines()


def test_gate_runs_the_commit_checks_first():
    lines = _gate_lines()
    first_check = next(
        i
        for i, ln in enumerate(lines)
        if ".githooks/pre-commit" in ln and not ln.lstrip().startswith("#")
    )
    first_qdrant = next(
        i for i, ln in enumerate(lines) if "dev-qdrant.sh" in ln and not ln.lstrip().startswith("#")
    )
    assert first_check < first_qdrant


def test_gate_stops_qdrant_only_if_it_started_it():
    code = [ln for ln in _gate_lines() if not ln.lstrip().startswith("#")]
    status = next(i for i, ln in enumerate(code) if "dev-qdrant.sh status" in ln)
    up = next(i for i, ln in enumerate(code) if "dev-qdrant.sh up" in ln)
    assert status < up
    traps = [ln for ln in code if ln.lstrip().startswith("trap ") and " EXIT" in ln]
    assert len(traps) == 1 and "dev-qdrant.sh down" in traps[0]
    assert [ln for ln in code if "down" in ln] == traps


def _repo_map() -> str:
    text = (ROOT / "CONTRIBUTING.md").read_text()
    return text.split("## The shape of the repo")[1].split("\n## ")[0]


def test_contributing_map_names_real_paths():
    section = _repo_map()
    paths = re.findall(r"`([^`\s]+/[^`\s]*)`", section)
    assert paths
    missing = [p for p in paths if not (ROOT / p).exists()]
    assert not missing, f"CONTRIBUTING names paths that do not exist: {missing}"
    unnamed = [
        f"packages/core/src/ragqa_core/{p.name}"
        for p in sorted((CORE_SRC / "ragqa_core").glob("*.py"))
        if p.name != "__init__.py" and f"packages/core/src/ragqa_core/{p.name}" not in paths
    ]
    assert not unnamed, f"CONTRIBUTING's map leaves out: {unnamed}"
    count = re.search(r"(\d+) methods", section)
    assert count and int(count.group(1)) == len(METHOD_NAMES)


def test_no_stale_names():
    """The old name of the sister project and the -p flag compose.yaml makes needless.
    The needles are built in two halves so this file does not contain them."""
    needles = ["pg" + "-ops", "-p " + "rag-qa"]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\0")
    found = []
    for name in filter(None, tracked):
        if name.startswith("docs/") or name in ("CHANGELOG.md", "uv.lock"):
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        found += [f"{name}: {n}" for n in needles if n in text]
    assert not found, f"stale names: {found}"


LINKED_DOCS = [
    "README.md",
    "REFERENCE.md",
    "RESEARCH.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "packages/aiconnect/README.md",
]
# The labels the README bolds that are buttons or links on the pages, beside every
# bolded `+ ` button.
UI_LABELS = [
    "+ New store",
    "Find passages",
    "Right",
    "Wrong",
    "Estimate",
    "Chat with this store",
    "Copy",
    "Create",
    "Import",
    "Create one",
    "Label",
    "Get the AI's answer",
    "Save question",
    "Save and write the next",
    "Start",
    "Re-run",
    "Answer from this attempt",
    "Set as default",
    "Use from your app",
    "Have the AI answer the questions",
    "Judge answers",
    "Add keyword tags",
    "Theme",
]


def _prose(text: str) -> str:
    """Markdown without its fenced code blocks, whose `#` lines are not headings."""
    return re.sub(r"^```.*?^```", "", text, flags=re.S | re.M)


def _heading_slugs(path: Path) -> set[str]:
    """GitHub's anchors for a file's headings: lowercase, spaces to -, other punctuation
    dropped, and a repeated heading numbered -1, -2."""
    slugs: set[str] = set()
    seen: dict[str, int] = {}
    for line in _prose(path.read_text()).splitlines():
        heading = re.match(r"#{1,6} (.+)", line)
        if heading:
            slug = re.sub(r"[^\w\- ]", "", heading.group(1).strip().lower()).replace(" ", "-")
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            slugs.add(slug if n == 0 else f"{slug}-{n}")
    return slugs


def test_doc_links_resolve():
    links = 0
    broken = []
    for name in LINKED_DOCS:
        doc = ROOT / name
        text = re.sub(r"`[^`\n]*`", "", _prose(doc.read_text()))  # code is not a link
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+:", target):
                continue  # https:, mailto:
            file, _, anchor = target.partition("#")
            path = (doc.parent / file).resolve() if file else doc
            if path.is_relative_to(ROOT / "docs"):
                continue  # untracked
            links += 1
            if not path.exists():
                broken.append(f"{name}: {target}")
            elif anchor and anchor not in _heading_slugs(path):
                broken.append(f"{name}: {target}")
    assert links > 20
    assert not broken, f"links that resolve to nothing: {broken}"


def test_readme_stays_short():
    assert len((ROOT / "README.md").read_text().splitlines()) <= 320


def test_readme_names_real_ui_labels_and_env_vars():
    readme = (ROOT / "README.md").read_text()
    bold = {" ".join(b.split()) for b in re.findall(r"\*\*(.+?)\*\*", readme, re.S)}
    assert set(UI_LABELS) <= bold  # the list names labels the README shows
    pages = "\n".join(p.read_text() for p in sorted((API_SRC / "ragqa_api/static").iterdir()))
    labels = sorted(b for b in bold if b.startswith("+ ") or b in UI_LABELS)
    assert [b for b in labels if b not in pages] == []
    srcs = [*sorted(ROOT.glob("packages/*/src")), *sorted(ROOT.glob("apps/*/src"))]
    code = "\n".join(
        p.read_text() for src in srcs for p in sorted(src.rglob("*")) if p.suffix in {".py", ".js"}
    )
    docs = readme + (ROOT / "REFERENCE.md").read_text()
    named = sorted(set(re.findall(r"RAGQA_[A-Z_]+", docs)))
    assert named
    assert [n for n in named if n not in code] == []


def test_changelog_keeps_every_breaking_line():
    text = (ROOT / "CHANGELOG.md").read_text()
    section = re.split(r"^## ", text, flags=re.M)[1]
    assert section.startswith("0.2.0")
    breaking = [line for line in section.splitlines() if "(breaking)" in line]
    # 15 `(breaking)` lines in `## Unreleased` at 11eeeac, plus the one then written
    # `Breaking:` (keyword-lists/propose needing confirm), now marked like the rest.
    assert len(breaking) == 16
    assert "### Upgrading from 0.1.0" in section


def test_versions_agree():
    import tomllib

    pyprojects = [
        ROOT / "pyproject.toml",
        *sorted(ROOT.glob("packages/*/pyproject.toml")),
        *sorted(ROOT.glob("apps/*/pyproject.toml")),
    ]
    assert len(pyprojects) == 5
    versions = {
        str(p.relative_to(ROOT)): tomllib.loads(p.read_text())["project"]["version"]
        for p in pyprojects
    }
    # Read the literal from source: building an app would open the real state dir.
    app_source = (ROOT / "apps/api/src/ragqa_api/app.py").read_text()
    versions["app.version"] = re.search(r"FastAPI\([^)]*version=\"([^\"]+)\"", app_source).group(1)
    readme = (ROOT / "README.md").read_text()
    versions["README.md"] = re.search(r"\*\*Status:\*\* version (\d+\.\d+\.\d+)", readme).group(1)
    changelog = (ROOT / "CHANGELOG.md").read_text()
    versions["CHANGELOG.md"] = re.search(r"^## (\d+\.\d+\.\d+)", changelog, re.M).group(1)
    assert len(set(versions.values())) == 1, versions
