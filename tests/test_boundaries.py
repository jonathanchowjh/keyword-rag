"""Source guards: the package boundary and the hooks that stand in for CI."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "packages/core/src"
API_SRC = ROOT / "apps/api/src"
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


def test_only_tests_import_fakes():
    assert modules_importing("ragqa_core.fakes", [CORE_SRC, API_SRC]) == []


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
