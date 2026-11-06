"""Read an uploaded file into rows, map its columns, build items and questions."""

import csv
import hashlib
import json
import re
import shutil
import uuid
import zipfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path

from ragqa_core.datasets import mask_template
from ragqa_core.db import now
from ragqa_core.prompts import CHAT_PASSAGE_CHARS, LETTERS
from ragqa_core.types import Item, QaType, Question

UPLOAD_LIMIT = 200 * 1024 * 1024
ZIP_LIMIT = 500 * 1024 * 1024  # total uncompressed size
ZIP_MEMBERS = 20_000
LEAK_MIN_WORDS = 6
UPLOAD_ID = re.compile(r"[0-9a-f]{32}")
SOURCE_NAME = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")


def source_name(filename: str) -> str:
    """A source name from a file name: the stem with each run of characters outside
    SOURCE_NAME's set made a '-', trimmed, cut to 64 and trimmed again; 'upload' when
    nothing is left."""
    stem = re.sub(r"[^A-Za-z0-9 _.-]+", "-", Path(filename).stem)
    return stem.strip(" -")[:64].strip(" -") or "upload"


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
CHUNK_CHARS = CHAT_PASSAGE_CHARS  # a chunk, breadcrumb included, is never cut in the prompt
MIN_CHUNK_CHARS = 200
CRUMB_CHARS = 200
CRUMB = " › "


class TooLarge(Exception):
    pass


def save_upload(home: Path, collection: str, filename: str, chunks: Iterable[bytes]) -> str:
    """Write the chunks to home/uploads/<id>/file with a meta.json beside it; past
    UPLOAD_LIMIT nothing is kept. Only the generated id forms a path: filenames are data."""
    upload_id = uuid.uuid4().hex
    folder = home / "uploads" / upload_id
    folder.mkdir(parents=True)
    try:
        size = 0
        with (folder / "file").open("wb") as f:
            for chunk in chunks:
                size += len(chunk)
                if size > UPLOAD_LIMIT:
                    raise TooLarge(f"the file is over {UPLOAD_LIMIT} bytes")
                f.write(chunk)
        meta = {"collection": collection, "filename": filename, "size": size, "created_at": now()}
        (folder / "meta.json").write_text(json.dumps(meta))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return upload_id


def open_upload(home: Path, collection: str, upload_id: str) -> tuple[Path, dict]:
    if not UPLOAD_ID.fullmatch(upload_id):
        raise KeyError(upload_id)
    folder = home / "uploads" / upload_id
    try:
        meta = json.loads((folder / "meta.json").read_text())
    except (OSError, ValueError) as e:
        raise KeyError(upload_id) from e
    if meta.get("collection") != collection:
        raise KeyError(upload_id)
    return folder / "file", meta


def delete_uploads(home: Path, collection: str) -> None:
    """Remove every upload folder whose meta.json names the collection."""
    for folder in (home / "uploads").glob("*"):
        try:
            named = json.loads((folder / "meta.json").read_text()).get("collection")
        except (OSError, ValueError, AttributeError):
            continue
        if named == collection:
            shutil.rmtree(folder, ignore_errors=True)


def read_rows(path: Path, filename: str) -> tuple[str, list[dict], list[str]]:
    fmt = FORMATS.get(Path(filename).suffix.lower())
    if fmt is None:
        raise ValueError(
            f"unsupported file type {Path(filename).suffix or filename!r}; "
            "use JSONL, JSON, CSV, TSV, .txt, .md, or a .zip of .txt/.md"
        )
    # Untrusted bytes: whatever the parsers raise, the caller sees a ValueError.
    try:
        return read_format(path, filename, fmt)
    except (
        zipfile.BadZipFile,
        json.JSONDecodeError,
        RuntimeError,
        NotImplementedError,
        csv.Error,
        OSError,
        EOFError,
        zlib.error,
    ) as e:
        raise ValueError(f"could not read {filename}: {e}") from e


def read_format(path: Path, filename: str, fmt: str) -> tuple[str, list[dict], list[str]]:
    if fmt == "zip":
        return (fmt, *read_zip(path))
    if fmt == "text":
        return fmt, text_rows(filename, decode(path.read_bytes())), []
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


def text_rows(filename: str, text: str) -> list[dict]:
    """A text file's rows: one {file, text} as it is when it fits in one chunk, else one row
    per chunk, with the id `<file>#<i>` and its place in the file in meta."""
    chunks = chunk_text(text, filename)
    if len(chunks) <= 1:
        return [{"file": filename, "text": text}]
    n = len(chunks)
    return [
        {"file": f"{filename}#{i}", "text": c, "meta": {"file": filename, "part": i, "parts": n}}
        for i, c in enumerate(chunks, 1)
    ]


def text_files(rows: list[dict]) -> set[str]:
    """The file names text rows were read from, as text_rows names them."""
    return {r["meta"]["file"] if "meta" in r else r["file"] for r in rows}


# -- chunking a text file --------------------------------------------------------------------

HEADING = re.compile(r"^(#{1,6}) (.*)$")
FENCE = re.compile(r"^\s*```")
LIST_ITEM = re.compile(r"^\s*(?:[-*] |\d+\. )")
TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def chunk_text(text: str, filename: str, size: int = CHUNK_CHARS) -> list[str]:
    """The text in chunks of at most `size` characters. A file that fits with its breadcrumb
    is one chunk, as it is. Else a .md is cut at headings first (a chunk never spans two
    sections); within a section, paragraphs, code blocks, tables and lists are packed whole,
    and one too big alone is split at line, row, item or sentence ends. Each chunk starts
    with a breadcrumb line (`notes.md › Setup › Install`) and a blank line, both counted in
    `size`; a body under MIN_CHUNK_CHARS merges into a neighbour of its section that has
    room. No AI call."""
    if not text.strip():
        return []
    if len(text.strip()) + len(breadcrumb(filename, [])) + 2 <= size:
        return [text]
    chunks: list[tuple[str, str]] = []
    for path, lines in sections(text, filename.lower().endswith(".md")):
        crumb = breadcrumb(filename, path)
        room = max(size - len(crumb) - 2, 1)
        chunks += [(crumb, body) for body in merge_small(pack(units(lines), room), room)]
    if len(chunks) == 1:
        return [chunks[0][1]]
    return [f"{crumb}\n\n{body}" for crumb, body in chunks]


def breadcrumb(filename: str, path: list[str]) -> str:
    """The file name and heading path joined by ' › ', the deepest headings cut first to
    keep it within CRUMB_CHARS."""
    path = list(path)
    while path and len(CRUMB.join([filename, *path])) > CRUMB_CHARS:
        path.pop()
    return CRUMB.join([filename, *path])[:CRUMB_CHARS]


def sections(text: str, markdown: bool) -> list[tuple[list[str], list[str]]]:
    """(heading path, lines) for each section; a heading of level L ends every open heading
    of level L or deeper. A heading inside a code fence is text. A .txt is one section."""
    out: list[tuple[list[str], list[str]]] = [([], [])]
    stack: list[tuple[int, str]] = []
    fenced = False
    for line in text.splitlines():
        if FENCE.match(line):
            fenced = not fenced
        heading = HEADING.match(line) if markdown and not fenced else None
        if heading:
            level = len(heading.group(1))
            stack = [h for h in stack if h[0] < level]
            stack.append((level, heading.group(2).strip().rstrip("#").strip()))
            out.append(([t for _, t in stack], []))
        out[-1][1].append(line)
    return [(path, lines) for path, lines in out if any(line.strip() for line in lines)]


def units(lines: list[str]) -> list[tuple[str, str]]:
    """(kind, text) blocks: a fenced code block, a table, a list (items, indented lines and
    the blank lines between them) or a paragraph."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        start = i
        if not line.strip():
            i += 1
            continue
        if FENCE.match(line):
            i += 1
            while i < n and not FENCE.match(lines[i]):
                i += 1
            i = min(i + 1, n)
            kind = "code"
        elif line.lstrip().startswith("|"):
            while i < n and lines[i].lstrip().startswith("|"):
                i += 1
            kind = "table"
        elif LIST_ITEM.match(line):
            i += 1
            while i < n:
                if LIST_ITEM.match(lines[i]) or (lines[i][:1] in (" ", "\t") and lines[i].strip()):
                    i += 1
                    continue
                j = i
                while j < n and not lines[j].strip():
                    j += 1
                if j > i and j < n and (LIST_ITEM.match(lines[j]) or lines[j][:1] in (" ", "\t")):
                    i = j
                    continue
                break
            kind = "list"
        else:
            i += 1
            while i < n and lines[i].strip() and not starts_block(lines[i]):
                i += 1
            kind = "paragraph"
        out.append((kind, "\n".join(lines[start:i]).strip("\n").rstrip()))
    return out


def starts_block(line: str) -> bool:
    return bool(FENCE.match(line) or LIST_ITEM.match(line) or line.lstrip().startswith("|"))


def pack(blocks: list[tuple[str, str]], room: int) -> list[str]:
    """The blocks joined by blank lines into bodies of at most `room` characters. A block
    that does not fit what is left is split when it is too big alone: its first piece fills
    the body (when at least MIN_CHUNK_CHARS is left), and its last is packed on with what
    follows."""
    bodies: list[str] = []
    current = ""
    for kind, block in blocks:
        left = room - len(current) - 2 if current else room
        if len(block) <= left:
            current = f"{current}\n\n{block}" if current else block
            continue
        if len(block) <= room:
            bodies.append(current)
            current = block
            continue
        if left < MIN_CHUNK_CHARS:
            bodies.append(current)
            current, left = "", room
        pieces = split_block(kind, block, room, left)
        if current:
            pieces[0] = f"{current}\n\n{pieces[0]}"
        bodies += pieces[:-1]
        current = pieces[-1]
    bodies.append(current)
    return [b for b in (b.strip() for b in bodies) if b]


def merge_small(bodies: list[str], room: int) -> list[str]:
    """A body under MIN_CHUNK_CHARS joins the previous body when both fit in `room`, else
    the next; one that fits nowhere stays."""
    bodies, out = list(bodies), []
    for i, body in enumerate(bodies):
        if len(body) < MIN_CHUNK_CHARS and out and len(out[-1]) + 2 + len(body) <= room:
            out[-1] += "\n\n" + body
        elif (
            len(body) < MIN_CHUNK_CHARS
            and i + 1 < len(bodies)
            and len(body) + 2 + len(bodies[i + 1]) <= room
        ):
            bodies[i + 1] = body + "\n\n" + bodies[i + 1]
        else:
            out.append(body)
    return out


def split_block(kind: str, block: str, room: int, first: int) -> list[str]:
    """A block too big for one chunk in pieces of at most `room`, the first at most `first`:
    code at line ends, a table between rows (each piece repeats its header and rule), a list
    between items, a paragraph at sentence ends or spaces."""
    lines = block.split("\n")
    if kind == "code":
        return pack_lines(lines, room, first)
    if kind == "table":
        head = lines[:2] if len(lines) > 2 and TABLE_RULE.match(lines[1]) else []
        if head and len("\n".join(head)) + 1 < first // 2:
            return pack_lines(lines[len(head) :], room, first, head)
        return pack_lines(lines, room, first)
    if kind == "list":
        items: list[str] = []
        for line in lines:
            if LIST_ITEM.match(line) or not items:
                items.append(line)
            else:
                items[-1] += "\n" + line
        return pack_lines(items, room, first)
    return split_text(block, room, first)


def pack_lines(lines: list[str], room: int, first: int, head: list[str] | None = None) -> list[str]:
    """Lines packed into pieces of at most `room` (the first at most `first`), each starting
    with `head`; a line too long for a piece is split as text."""
    top = "\n".join(head or [])
    extra = len(top) + 1 if top else 0
    pieces: list[str] = []
    current = ""
    for line in lines:
        space = (first if not pieces else room) - extra
        if current and len(current) + 1 + len(line) <= space:
            current += "\n" + line
            continue
        if current:
            pieces.append(current)
            current, space = "", room - extra
        parts = [line] if len(line) <= space else split_text(line, room - extra, space)
        pieces += parts[:-1]
        current = parts[-1]
    pieces.append(current)
    return [f"{top}\n{p}" if top else p for p in pieces if p]


def split_text(text: str, room: int, first: int | None = None) -> list[str]:
    """Text in pieces of at most `room` (the first at most `first`): each cut after the last
    sentence end ('. ', '? ', '! ') that fits, else at the last whitespace, else hard."""
    pieces: list[str] = []
    rest = text.strip()
    limit = room if first is None else first
    while len(rest) > limit:
        window = rest[: limit + 1]
        end = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
        if end >= 0:
            cut = end + 1
        else:
            space = max((m.start() for m in re.finditer(r"\s", window)), default=-1)
            cut = space if space > 0 else limit
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
        limit = room
    if rest:
        pieces.append(rest)
    return [p for p in pieces if p]


def read_jsonl(path: Path) -> tuple[list[dict], list[str]]:
    rows, problems = [], []
    for n, line in enumerate(decode(path.read_bytes()).splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, RecursionError):
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
            rows.extend(text_rows(name, decode(z.read(info))))
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
    if a.isascii() and a.isdigit() and len(a) < 10 and int(a) < len(choices):
        return LETTERS[int(a)]
    return None


def make_id(prefix: str, text: str) -> str:
    return prefix + hashlib.sha1(text.encode(errors="replace")).hexdigest()[:16]


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


def question_problem(q: Question, qa_type: QaType) -> str | None:
    """Why the question is not usable, or None. No gold answer is fine for every type: it
    leaves the answers unmeasured, as with BRIGHT."""
    if not q.id.strip():
        return "no id"
    if not q.text.strip():
        return "no text"
    if qa_type == "mcq":
        choices = q.choices or []
        if len(choices) < 2:
            return "an mcq question needs at least 2 choices"
        letters = LETTERS[: len(choices)]
        if any(a not in letters for a in q.gold_answers):
            return f"the answer must be one of {', '.join(letters)}"
    return None


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
        answer = cell_text(row, mapping, "answer")  # for the empty check and the letter
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
            gold = split_list(cell(row, mapping, "answer"))
        id_ = cell_text(row, mapping, "id") or make_id("q-", text)
        if id_ in questions:
            report["duplicate_ids"] += 1
            continue
        meta = {k: v for k in ("solution", "skill", "scale") if (v := cell_text(row, mapping, k))}
        gold_ids = split_list(cell(row, mapping, "gold_ids"))
        question = Question(id_, text, choices, gold, gold_ids, meta)
        if question_problem(question, qa_type):
            report["unreadable_answers"] += 1
            continue
        questions[id_] = question
    return list(questions.values()), report


def preview(
    path: Path,
    filename: str,
    kind: str,
    qa_type: QaType,
    mapping: dict | None = None,
    known_ids: set[str] | None = None,
    source: str | None = None,
    owner_of: Callable[[list[str]], dict[str, str | None]] | None = None,
) -> dict:
    """owner_of names the source each id already has, to warn of ids that would move."""
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
        if owner_of and source:
            moving = Counter(
                o for o in owner_of([b.id for b in built]).values() if o and o != source
            )
            noun = "question" if kind == "questions" else "item"
            for other, n in moving.items():
                what = f"{n} {noun}s already belong" if n != 1 else f"{n} {noun} already belongs"
                problems.append(f"{what} to {other} and would move to this source")
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
        "required": list(required(kind, qa_type)),
        "sample": sample,
        "problems": problems,
        "usable": usable,
        "ready": not missing and usable > 0,
    }


def leaked_ids(questions: list[Question], texts: list[str]) -> set[str]:
    joined = "\n".join(" " + " ".join(mask_template(t).split()) + " " for t in texts)
    leaked = set()
    for q in questions:
        key = " ".join(mask_template(q.text).split())
        if len(key.split()) >= LEAK_MIN_WORDS and f" {key} " in joined:
            leaked.add(q.id)
    return leaked
