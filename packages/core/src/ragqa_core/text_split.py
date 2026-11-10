"""Split a text file into chunks, and name each chunk by its text."""

import hashlib
import re

from ragqa_core.prompts import CHAT_PASSAGE_CHARS

CHUNK_CHARS = CHAT_PASSAGE_CHARS  # a chunk, breadcrumb included, is never cut in the prompt
MIN_CHUNK_CHARS = 200
CRUMB_CHARS = 200
CRUMB = " › "


def chunk_id(filename: str, text: str, seen: dict[str, int]) -> str:
    """`<file>#<first 12 hex of the text's sha256>`; a text met again in the same file (as
    `seen` counts) takes `-2`, `-3`, and so on, in file order."""
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    seen[digest] = seen.get(digest, 0) + 1
    return f"{filename}#{digest}" if seen[digest] == 1 else f"{filename}#{digest}-{seen[digest]}"


CHUNK_MARK = re.compile(r"[0-9]+|[0-9a-f]{12}(?:-[0-9]+)?")


def chunk_of(item_id: str) -> str | None:
    """The file a chunk id names: `<file>#<digits>` (as loads before content ids named a
    chunk by its place) or `<file>#<12 hex>`, maybe with `-<digits>`; None for any other id."""
    base, mark, part = item_id.rpartition("#")
    return base if mark and CHUNK_MARK.fullmatch(part) else None


# -- chunking a text file --------------------------------------------------------------------

HEADING = re.compile(r"^(#{1,6}) (.*)$")
FENCE = re.compile(r"^\s*```")
LIST_ITEM = re.compile(r"^\s*(?:[-*] |\d+\. )")
TABLE_RULE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


def chunk_text(text: str, filename: str, size: int = CHUNK_CHARS) -> list[str]:
    """The text in chunks of at most `size` characters. A file that fits is one chunk with
    no breadcrumb, as it is (stripped, when only its padding is over). Else a .md is cut at
    headings first (a chunk never spans two sections); within a section, paragraphs, code
    blocks, tables and lists are packed whole, and one too big alone is split at line, row,
    item or sentence ends. Each chunk starts with a breadcrumb line (`notes.md › Setup ›
    Install`) and a blank line, both counted in `size`; a body under MIN_CHUNK_CHARS merges
    into a neighbour of its section that has room. No AI call."""
    if not text.strip():
        return []
    if len(text.strip()) <= size:
        return [text if len(text) <= size else text.strip()]
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
    of level L or deeper. A heading inside a code fence is text. A .txt is one section.

    A section that is only its heading line makes no section of its own. Its line is
    dropped when the section that follows with a body is under it (that breadcrumb names
    it), else carried into that section as leading text; with no such section after it,
    it goes on the end of the last one, and a file of headings only is one plain section."""
    out: list[tuple[list[str], list[str]]] = []
    carried: list[tuple[list[str], str]] = []  # (path, heading line) waiting for a body
    for path, lines, headed in heading_sections(text, markdown):
        if sum(1 for line in lines if line.strip()) == (1 if headed else 0):
            if headed:
                carried.append((path, next(line for line in lines if line.strip())))
            continue
        under = [p for p, _ in carried if len(p) < len(path) and path[: len(p)] == p]
        # Each carried line its own paragraph, so packing never cuts one in two.
        lead = [part for p, line in carried if p not in under for part in (line, "")]
        out.append((path, [*lead, *lines]))
        carried = []
    rest = [part for _, line in carried for part in ("", line)]
    if rest and out:
        out[-1] = (out[-1][0], [*out[-1][1], *rest])
    elif rest:
        out.append(([], rest))
    return out


def heading_sections(text: str, markdown: bool) -> list[tuple[list[str], list[str], bool]]:
    """(heading path, lines, headed) for each section, its heading line first, empty ones
    included; the first is what comes before any heading."""
    out: list[tuple[list[str], list[str], bool]] = [([], [], False)]
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
            out.append(([t for _, t in stack], [], True))
        out[-1][1].append(line)
    return out


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
