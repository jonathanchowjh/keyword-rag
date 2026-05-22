"""Read an uploaded file into rows, map its columns, build items and questions."""

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

from ragqa_core.datasets import mask_template
from ragqa_core.prompts import LETTERS
from ragqa_core.types import Item, QaType, Question

UPLOAD_LIMIT = 200 * 1024 * 1024
ZIP_LIMIT = 500 * 1024 * 1024  # total uncompressed size
ZIP_MEMBERS = 20_000
LEAK_MIN_WORDS = 6
UPLOAD_ID = re.compile(r"^[0-9a-f]{32}$")
SOURCE_NAME = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")

FORMATS = {
    ".jsonl": "jsonl",
    ".json": "json",
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "text",
    ".md": "text",
    ".zip": "zip",
}
FIELDS = {
    "items": ("id", "text", "answer", "explanation", "skill", "topic", "meta"),
    "questions": ("id", "text", "choices", "answer", "gold_ids", "solution", "skill", "scale"),
}
SYNONYMS = {
    "id": ("id", "uid", "qid", "problemid", "file", "key"),
    "text": ("text", "question", "problem", "problem_text", "content", "passage", "body"),
    "answer": ("answer", "answers", "gold", "gold_answer", "answer_number", "label"),
    "explanation": ("explanation", "solution", "rationale", "worked_solution"),
    "solution": ("solution", "explanation", "rationale", "worked_solution"),
    "choices": ("choices", "options"),
    "gold_ids": ("gold_ids", "supporting_ids", "evidence_ids", "evidence"),
    "skill": ("skill", "chapter", "category"),
    "topic": ("topic", "subject"),
    "scale": ("scale",),
    "meta": ("meta",),
}
TEXT_SUFFIXES = (".txt", ".md")


def read_rows(path: Path, filename: str) -> tuple[str, list[dict], list[str]]:
    fmt = FORMATS.get(Path(filename).suffix.lower())
    if fmt is None:
        raise ValueError(
            f"unsupported file type {Path(filename).suffix or filename!r}; "
            "use JSONL, JSON, CSV, TSV, .txt, .md, or a .zip of .txt/.md"
        )
    if fmt == "zip":
        return (fmt, *read_zip(path))
    if fmt == "text":
        return fmt, [{"file": filename, "text": decode(path.read_bytes())}], []
    if fmt == "json":
        data = json.loads(decode(path.read_bytes()))
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError("a JSON file must be an array of objects")
        return fmt, data, []
    if fmt == "jsonl":
        return (fmt, *read_jsonl(path))
    csv.field_size_limit(1 << 30)
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t" if fmt == "tsv" else ",")
        rows = [{k: v for k, v in r.items() if k is not None} for r in reader]
    return fmt, rows, []


def decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for n, line in enumerate(decode(path.read_bytes()).splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if isinstance(row, dict):
            rows.append(row)
        else:
            problems.append(f"line {n} is not a JSON object")
    return rows, problems


def read_zip(path: Path) -> tuple[list[dict], list[str]]:
    rows, skipped = [], 0
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) > ZIP_MEMBERS:
            raise ValueError(f"the zip has {len(infos)} files; the limit is {ZIP_MEMBERS}")
        if sum(i.file_size for i in infos) > ZIP_LIMIT:
            raise ValueError(f"the zip is over {ZIP_LIMIT} bytes uncompressed")
        for info in infos:
            name = info.filename
            base = name.rsplit("/", 1)[-1]
            if info.is_dir() or name.startswith("__MACOSX/") or base.startswith("."):
                continue
            if not name.lower().endswith(TEXT_SUFFIXES):
                skipped += 1
                continue
            rows.append({"file": name, "text": decode(z.read(info))})
    problems = [f"{skipped} files skipped; only .txt and .md are read"] if skipped else []
    return rows, problems


def columns(rows: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for row in rows:
        for key in row:
            seen.setdefault(key, None)
    return list(seen)


def required(kind: str, qa_type: QaType) -> tuple[str, ...]:
    if kind == "items":
        return ("text",)
    return ("text", "choices", "answer") if qa_type == "mcq" else ("text", "answer")


def guess_mapping(cols: list[str], kind: str) -> dict[str, str | None]:
    by_lower = {c.lower(): c for c in cols}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for field in FIELDS[kind]:
        mapping[field] = None
        for name in SYNONYMS[field]:
            col = by_lower.get(name)
            if col is not None and col not in used:
                mapping[field] = col
                used.add(col)
                break
    return mapping


def split_list(value) -> list[str]:
    if isinstance(value, str):
        pieces = None
        if value.lstrip().startswith("["):
            try:
                pieces = json.loads(value)
            except ValueError:
                pass
        if not isinstance(pieces, list):
            pieces = value.split("|")
    elif isinstance(value, list):
        pieces = value
    else:
        pieces = [] if value is None else [value]
    return [s for s in (str(p).strip() for p in pieces) if s]


def to_letter(answer: str, choices: list[str]) -> str | None:
    a = str(answer).strip()
    if len(a) == 1 and a.upper() in LETTERS[: len(choices)]:
        return a.upper()
    for i, c in enumerate(choices):
        if c.strip().lower() == a.lower():
            return LETTERS[i]
    if a.isdigit() and int(a) < len(choices):
        return LETTERS[int(a)]
    return None


def make_id(prefix: str, text: str) -> str:
    return prefix + hashlib.sha1(text.encode()).hexdigest()[:16]


def cell(row: dict, mapping: dict, field: str):
    col = mapping.get(field)
    value = row.get(col) if col else None
    return value.strip() if isinstance(value, str) else value


def cell_text(row: dict, mapping: dict, field: str) -> str:
    value = cell(row, mapping, field)
    return "" if value is None else str(value).strip()


def meta_object(value) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def build_items(rows: list[dict], mapping: dict) -> tuple[list[Item], dict]:
    report = {"rows": len(rows), "no_text": 0, "duplicate_ids": 0}
    items: dict[str, Item] = {}
    for row in rows:
        text = cell_text(row, mapping, "text")
        if not text:
            report["no_text"] += 1
            continue
        id_ = cell_text(row, mapping, "id") or make_id("h-", text)
        if id_ in items:
            report["duplicate_ids"] += 1
            continue
        meta = {k: v for k in ("skill", "topic") if (v := cell_text(row, mapping, k))}
        meta.update(meta_object(cell(row, mapping, "meta")))
        items[id_] = Item(
            id_,
            text,
            cell_text(row, mapping, "answer") or None,
            cell_text(row, mapping, "explanation") or None,
            meta,
        )
    return list(items.values()), report


def build_questions(
    rows: list[dict], mapping: dict, qa_type: QaType
) -> tuple[list[Question], dict]:
    report = {
        "rows": len(rows),
        "no_text": 0,
        "duplicate_ids": 0,
        "no_answer": 0,
        "no_choices": 0,
        "unreadable_answers": 0,
    }
    questions: dict[str, Question] = {}
    for row in rows:
        text = cell_text(row, mapping, "text")
        answer = cell_text(row, mapping, "answer")
        choices = split_list(cell(row, mapping, "choices")) or None
        if not text:
            report["no_text"] += 1
            continue
        if not answer:
            report["no_answer"] += 1
            continue
        if qa_type == "mcq" and not choices:
            report["no_choices"] += 1
            continue
        if choices:
            letter = to_letter(answer, choices)
            if letter is None:
                report["unreadable_answers"] += 1
                continue
            gold = [letter]
        else:
            gold = split_list(answer)
        id_ = cell_text(row, mapping, "id") or make_id("q-", text)
        if id_ in questions:
            report["duplicate_ids"] += 1
            continue
        meta = {k: v for k in ("solution", "skill", "scale") if (v := cell_text(row, mapping, k))}
        gold_ids = split_list(cell(row, mapping, "gold_ids"))
        questions[id_] = Question(id_, text, choices, gold, gold_ids, meta)
    return list(questions.values()), report


def preview(
    path: Path,
    filename: str,
    kind: str,
    qa_type: QaType,
    mapping: dict | None = None,
    known_ids: set[str] | None = None,
) -> dict:
    fmt, rows, problems = read_rows(path, filename)
    cols = columns(rows)
    mapping = mapping or guess_mapping(cols, kind)
    missing = [f for f in required(kind, qa_type) if not mapping.get(f)]
    usable = 0
    if not missing:
        if kind == "items":
            built, report = build_items(rows, mapping)
        else:
            built, report = build_questions(rows, mapping, qa_type)
            if known_ids is not None:
                unknown = sum(1 for q in built for g in q.gold_ids if g not in known_ids)
                if unknown:
                    problems.append(f"{unknown} gold ids name no item in this collection")
            if report["no_answer"]:
                problems.append(f"{report['no_answer']} rows have no answer and will be skipped")
            if report["no_choices"]:
                problems.append(f"{report['no_choices']} rows have no choices and will be skipped")
            if report["unreadable_answers"]:
                problems.append(f"{report['unreadable_answers']} answers match no choice")
        usable = len(built)
        if report["no_text"]:
            problems.append(f"{report['no_text']} rows have no text and will be skipped")
        if report["duplicate_ids"]:
            problems.append(f"{report['duplicate_ids']} duplicate ids; the first of each is kept")
    sample = [{k: str(v)[:200] for k, v in row.items()} for row in rows[:5]]
    return {
        "format": fmt,
        "rows": len(rows),
        "columns": cols,
        "mapping": mapping,
        "missing": missing,
        "sample": sample,
        "problems": problems,
        "usable": usable,
        "ready": not missing and usable > 0,
    }


def leaked_ids(questions: list[Question], texts: list[str]) -> set[str]:
    joined = "\n".join(" ".join(mask_template(t).split()) for t in texts)
    leaked = set()
    for q in questions:
        key = " ".join(mask_template(q.text).split())
        if len(key.split()) >= LEAK_MIN_WORDS and key in joined:
            leaked.add(q.id)
    return leaked
