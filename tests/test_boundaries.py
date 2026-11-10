"""Source guards: the package boundary and the hooks that stand in for CI."""

import ast
import re
from pathlib import Path

from ragqa_core import db

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
    assert '"127.0.0.1:8080:8080"' in c and "qdrant/qdrant:v1.20.0" in c
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
