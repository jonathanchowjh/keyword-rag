"""Source guards: the package boundary and the hooks that stand in for CI."""

import ast
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
