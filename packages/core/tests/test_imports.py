import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from ragqa_core import imports
from ragqa_core.imports import question_problem
from ragqa_core.types import Question


def write(path: Path, content: str | bytes) -> Path:
    if isinstance(content, str):
        content = content.encode()
    path.write_bytes(content)
    return path


def test_jsonl_skips_blank_lines_and_reports_bad_ones(tmp_path):
    p = write(tmp_path / "a.jsonl", '{"text": "one"}\n\n[1, 2]\n{"text": "two"}\n')
    fmt, rows, problems = imports.read_rows(p, "a.jsonl")
    assert fmt == "jsonl"
    assert [r["text"] for r in rows] == ["one", "two"]
    assert problems == ["line 3 is not a JSON object"]


def test_json_must_be_an_array_of_objects(tmp_path):
    ok = write(tmp_path / "a.json", json.dumps([{"text": "x"}]))
    assert imports.read_rows(ok, "a.json")[1] == [{"text": "x"}]
    bad = write(tmp_path / "b.json", json.dumps({"text": "x"}))
    with pytest.raises(ValueError):
        imports.read_rows(bad, "b.json")


def test_csv_with_bom_and_quoted_newline(tmp_path):
    p = write(tmp_path / "a.csv", '﻿id,text\n1,"line one\nline two"\n')
    fmt, rows, problems = imports.read_rows(p, "A.CSV")
    assert fmt == "csv"
    assert imports.columns(rows) == ["id", "text"]
    assert rows[0]["text"] == "line one\nline two"
    assert problems == []


def test_tsv(tmp_path):
    p = write(tmp_path / "a.tsv", "id\ttext\n1\thello, world\n")
    fmt, rows, _ = imports.read_rows(p, "a.tsv")
    assert fmt == "tsv"
    assert rows == [{"id": "1", "text": "hello, world"}]


def test_text_file_is_one_row(tmp_path):
    p = write(tmp_path / "n.md", "# Title\nbody")
    fmt, rows, _ = imports.read_rows(p, "n.md")
    assert fmt == "text"
    assert rows == [{"file": "n.md", "text": "# Title\nbody"}]


def test_zip_reads_text_members_only(tmp_path):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("top.txt", "alpha")
        z.writestr("docs/inner.md", "beta")
        z.writestr("__MACOSX/docs/._inner.md", "junk")
        z.writestr(".hidden.txt", "junk")
        z.writestr("pic.png", b"\x89PNG")
    fmt, rows, problems = imports.read_rows(p, "a.zip")
    assert fmt == "zip"
    assert {r["file"]: r["text"] for r in rows} == {"top.txt": "alpha", "docs/inner.md": "beta"}
    assert len(problems) == 1 and "1" in problems[0]


def test_zip_over_limit_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.txt", "x" * 100)
    monkeypatch.setattr(imports, "ZIP_LIMIT", 10)
    with pytest.raises(ValueError):
        imports.read_rows(p, "a.zip")


def test_zip_member_count_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.txt", "x")
        z.writestr("b.txt", "y")
    monkeypatch.setattr(imports, "ZIP_MEMBERS", 1)
    with pytest.raises(ValueError):
        imports.read_rows(p, "a.zip")


def test_unknown_type_is_refused(tmp_path):
    p = write(tmp_path / "a.pdf", "x")
    with pytest.raises(ValueError, match="unsupported file type"):
        imports.read_rows(p, "a.pdf")


def test_guess_mapping_scibench_columns():
    cols = ["problem_text", "answer_number", "unit", "solution", "problemid", "source"]
    q = imports.guess_mapping(cols, "questions")
    assert q["text"] == "problem_text"
    assert q["answer"] == "answer_number"
    assert q["solution"] == "solution"
    assert q["id"] == "problemid"
    assert q["choices"] is None
    i = imports.guess_mapping(cols, "items")
    assert i["explanation"] == "solution"


def test_guess_mapping_uses_each_column_once_and_ignores_case():
    m = imports.guess_mapping(["Text", "Solution"], "questions")
    assert m["text"] == "Text"
    assert m["solution"] == "Solution"


def test_science_question_without_choices_keeps_its_number():
    rows = [{"problem_text": "What is the mass?", "answer_number": "9.13e-35"}]
    mapping = imports.guess_mapping(imports.columns(rows), "questions")
    qs, report = imports.build_questions(rows, mapping, "science")
    assert qs[0].gold_answers == ["9.13e-35"]
    assert qs[0].choices is None
    assert report["rows"] == 1


def test_a_list_answer_keeps_each_string_as_gold():
    rows = [
        {"text": "Who wrote Hamlet?", "answer": ["Shakespeare", "William Shakespeare"]},
        {"text": "Who wrote Emma?", "answer": "Austen|Jane Austen"},
    ]
    qs, _ = imports.build_questions(rows, {"text": "text", "answer": "answer"}, "science")
    assert qs[0].gold_answers == ["Shakespeare", "William Shakespeare"]
    assert qs[1].gold_answers == ["Austen", "Jane Austen"]


def test_mcq_answers_to_letters():
    choices = ["Chloride", "Sulfate", "Nitrate"]
    assert imports.to_letter("b", choices) == "B"
    assert imports.to_letter("Sulfate", choices) == "B"
    assert imports.to_letter("2", ["1", "2", "3"]) == "B"
    assert imports.to_letter("0", choices) == "A"
    assert imports.to_letter("Z", choices) is None
    assert imports.to_letter("7", choices) is None

    rows = [
        {"text": "q1", "choices": "Chloride|Sulfate|Nitrate", "answer": "Sulfate"},
        {"text": "q2", "choices": "Chloride|Sulfate|Nitrate", "answer": 0},
        {"text": "q3", "choices": "Chloride|Sulfate|Nitrate", "answer": "Z"},
        {"text": "q4", "answer": "A"},
        {"text": "q5", "choices": "a|b"},
        {"text": "", "choices": "a|b", "answer": "A"},
    ]
    mapping = imports.guess_mapping(imports.columns(rows), "questions")
    qs, report = imports.build_questions(rows, mapping, "mcq")
    assert [q.gold_answers for q in qs] == [["B"], ["A"]]
    assert qs[0].choices == ["Chloride", "Sulfate", "Nitrate"]
    assert report == {
        "rows": 6,
        "no_text": 1,
        "duplicate_ids": 0,
        "no_answer": 1,
        "no_choices": 1,
        "unreadable_answers": 1,
    }


def test_split_list():
    assert imports.split_list(["a", " ", 3]) == ["a", "3"]
    assert imports.split_list('["a", "b"]') == ["a", "b"]
    assert imports.split_list("a | b||c") == ["a", "b", "c"]
    assert imports.split_list("[not json|x]") == ["[not json", "x]"]


def test_hash_ids_are_stable():
    assert imports.make_id("h-", "hello") == imports.make_id("h-", "hello")
    assert imports.make_id("h-", "hello") != imports.make_id("h-", "hello!")
    assert imports.make_id("q-", "hello").startswith("q-")
    assert len(imports.make_id("q-", "hello")) == 18


def test_build_items_report_and_meta():
    rows = [
        {"id": "a", "text": "one", "skill": "algebra", "meta": '{"k": 1}'},
        {"id": "a", "text": "dup"},
        {"id": "b", "text": ""},
        {"text": "no id"},
    ]
    mapping = imports.guess_mapping(imports.columns(rows), "items")
    items, report = imports.build_items(rows, mapping)
    assert [i.id for i in items] == ["a", imports.make_id("h-", "no id")]
    assert items[0].text == "one"
    assert items[0].meta == {"skill": "algebra", "k": 1}
    assert report == {"rows": 4, "no_text": 1, "duplicate_ids": 1}


def test_preview_not_ready_without_text(tmp_path):
    p = write(tmp_path / "a.csv", "title,answer\nx,y\n")
    out = imports.preview(p, "a.csv", "questions", "science")
    assert out["missing"] == ["text"]
    assert out["ready"] is False
    assert out["usable"] == 0


def test_preview_ready_with_sample_and_problems(tmp_path):
    p = write(tmp_path / "a.csv", "text,answer\n" + "x" * 300 + ",1\n,2\nq,3\n")
    out = imports.preview(p, "a.csv", "questions", "science")
    assert out["format"] == "csv"
    assert out["rows"] == 3
    assert out["usable"] == 2
    assert out["ready"] is True
    assert len(out["sample"][0]["text"]) == 200
    assert "1 rows have no text and will be skipped" in out["problems"]


def test_preview_counts_unknown_gold_ids(tmp_path):
    p = write(tmp_path / "a.csv", "text,answer,gold_ids\nq1,1,i1|i2\nq2,2,i3\n")
    out = imports.preview(p, "a.csv", "questions", "multihop", known_ids={"i1"})
    assert "2 gold ids name no item in this collection" in out["problems"]


def test_preview_warns_about_ids_that_belong_to_another_source(tmp_path):
    p = write(tmp_path / "a.csv", "id,text\na,one\nb,two\nc,three\n")
    owners = {"a": "upload:old", "b": "upload:old", "c": "upload:mine", "z": "hand"}
    out = imports.preview(
        p,
        "a.csv",
        "items",
        "mcq",
        source="upload:mine",
        owner_of=lambda ids: {i: owners[i] for i in ids if i in owners},
    )
    assert "2 items already belong to upload:old and would move to this source" in out["problems"]
    assert len(out["problems"]) == 1
    p = write(tmp_path / "b.csv", "id,text,answer\na,one,1\n")
    out = imports.preview(
        p, "b.csv", "questions", "science", source="x", owner_of=lambda ids: {"a": "hand"}
    )
    assert "1 question already belongs to hand and would move to this source" in out["problems"]


def test_leaked_ids():
    items = [
        "Yesterday Alice paid 40 dollars for 3 apples at the Market on Monday morning.",
        "A long passage. The quick brown fox jumps over the lazy dog near the river bank today.",
    ]
    qs = [
        Question(
            "masked",
            "Yesterday Bob paid 55 dollars for 9 apples at the Store on Friday morning.",
            None,
            [],
            [],
        ),
        Question("inside", "the quick brown fox jumps over the lazy dog", None, [], []),
        Question("short", "brown fox jumps over", None, [], []),
        Question("other", "What is the boiling point of water at sea level?", None, [], []),
    ]
    assert imports.leaked_ids(qs, items) == {"masked", "inside"}


def test_exotic_digit_answers_are_unreadable_not_fatal():
    choices = ["a", "b"]
    assert imports.to_letter("²", choices) is None
    assert imports.to_letter("9" * 5000, choices) is None
    rows = [
        {"text": "q1", "choices": "a|b", "answer": "²"},
        {"text": "q2", "choices": "a|b", "answer": "9" * 5000},
    ]
    mapping = imports.guess_mapping(imports.columns(rows), "questions")
    qs, report = imports.build_questions(rows, mapping, "mcq")
    assert qs == [] and report["unreadable_answers"] == 2


def test_corrupt_zip_is_a_value_error(tmp_path):
    p = write(tmp_path / "x.zip", "not a zip")
    with pytest.raises(ValueError, match="could not read x.zip"):
        imports.read_rows(p, "x.zip")


def test_zip_bad_crc_is_a_value_error(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.txt", "hello world")
    data = bytearray(p.read_bytes())
    data[data.index(b"hello world")] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="could not read x.zip"):
        imports.read_rows(p, "x.zip")


def test_deeply_nested_json_is_a_value_error(tmp_path):
    p = write(tmp_path / "x.json", "[" * 100000 + "]" * 100000)
    with pytest.raises(ValueError, match="could not read x.json"):
        imports.read_rows(p, "x.json")


def test_malformed_json_is_a_value_error(tmp_path):
    p = write(tmp_path / "x.json", "[{")
    with pytest.raises(ValueError, match="could not read x.json"):
        imports.read_rows(p, "x.json")


def test_deeply_nested_jsonl_line_is_a_reported_bad_line(tmp_path):
    p = write(tmp_path / "x.jsonl", '{"text": "ok"}\n' + "[" * 100000 + "\n")
    _, rows, problems = imports.read_rows(p, "x.jsonl")
    assert len(rows) == 1
    assert problems == ["line 2 is not a JSON object"]


def test_lone_surrogate_text_still_gets_an_id():
    assert imports.make_id("h-", "a\ud800b").startswith("h-")
    rows = [{"text": "a\ud800b"}]
    items, _ = imports.build_items(rows, imports.guess_mapping(["text"], "items"))
    assert len(items) == 1


def test_leak_does_not_match_inside_words():
    items = ["the quick brown fox jumps over the lazy doge near the river"]
    q = Question("q", "the quick brown fox jumps over the lazy dog", None, [], [])
    assert imports.leaked_ids([q], items) == set()
    q2 = Question("q2", "the quick brown fox jumps over the lazy doge", None, [], [])
    assert imports.leaked_ids([q2], items) == {"q2"}


def deflated_zip(path: Path) -> bytes:
    text = " ".join(f"word{i * 7919 % 1013}" for i in range(2000))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.txt", text)
    return path.read_bytes()


def test_corrupt_deflate_stream_is_a_value_error(tmp_path):
    p = tmp_path / "x.zip"
    data = bytearray(deflated_zip(p))
    start = 30 + len("a.txt") + 100
    data[start : start + 20] = b"\xff" * 20
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="could not read x.zip"):
        imports.read_rows(p, "x.zip")


def test_truncated_deflate_stream_is_a_value_error(tmp_path):
    p = tmp_path / "x.zip"
    data = bytearray(deflated_zip(p))
    # shorten the compressed size in the central directory so the stream ends early
    cd = data.index(b"PK\x01\x02")
    size = int.from_bytes(data[cd + 20 : cd + 24], "little")
    data[cd + 20 : cd + 24] = (size // 2).to_bytes(4, "little")
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="could not read x.zip"):
        imports.read_rows(p, "x.zip")


def test_upload_ids_cannot_escape(tmp_path):
    other = imports.save_upload(tmp_path, "other", "a.txt", [b"hello"])
    mine = imports.save_upload(tmp_path, "c", "a.txt", [b"hel", b"lo"])
    path, meta = imports.open_upload(tmp_path, "c", mine)
    assert path.read_bytes() == b"hello"
    assert meta["collection"] == "c" and meta["filename"] == "a.txt" and meta["size"] == 5
    for bad in ("../x", other, "0" * 32, ""):
        with pytest.raises(KeyError):
            imports.open_upload(tmp_path, "c", bad)


def test_save_upload_over_limit_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(imports, "UPLOAD_LIMIT", 10)
    with pytest.raises(imports.TooLarge):
        imports.save_upload(tmp_path, "c", "a.txt", [b"123456", b"7890123"])
    assert list((tmp_path / "uploads").iterdir()) == []


def test_preview_names_the_required_fields(tmp_path):
    p = write(tmp_path / "a.csv", "text,answer\nq,A\n")
    assert imports.preview(p, "a.csv", "items", "mcq")["required"] == ["text"]
    shown = imports.preview(p, "a.csv", "questions", "mcq")
    assert shown["required"] == ["text", "choices", "answer"]
    assert imports.preview(p, "a.csv", "questions", "science")["required"] == ["text", "answer"]


def test_question_problem():
    ok = Question("a", "which one?", ["x", "y"], ["B"], [], {})
    assert question_problem(ok, "mcq") is None
    assert question_problem(replace(ok, text="  "), "mcq") == "no text"
    assert question_problem(replace(ok, id=""), "mcq") == "no id"
    assert question_problem(replace(ok, id=" "), "mcq") == "no id"
    assert "choices" in question_problem(replace(ok, choices=["x"]), "mcq")
    assert "choices" in question_problem(replace(ok, choices=None), "mcq")
    assert "answer" in question_problem(replace(ok, gold_answers=["Z"]), "mcq")
    # Unmeasured answers are fine for every type.
    assert question_problem(replace(ok, gold_answers=[]), "mcq") is None
    plain = Question("b", "how warm?", None, [], [], {})
    assert question_problem(plain, "science") is None
    assert question_problem(replace(plain, gold_answers=["28 C"]), "multihop") is None


def test_source_name_cleans_file_names():
    assert imports.source_name("Q3 (final).md") == "Q3 -final"
    assert imports.source_name("(((.md") == "upload"
    assert imports.source_name("notes.txt") == "notes"
    name = imports.source_name("a" * 100 + ".txt")
    assert name == "a" * 64 and imports.SOURCE_NAME.match(name)
    assert imports.source_name("a" * 63 + " b.txt") == "a" * 63  # no space left at the cut


# -- chunking text uploads -------------------------------------------------------------------


def sentences(n: int, word: str = "word") -> str:
    """n sentences of 50 characters each, joined by spaces."""
    return " ".join([(word * 49)[:49] + "."] * n)


def body(chunk: str) -> str:
    return chunk.split("\n\n", 1)[1]


def test_chunk_text_packs_paragraphs():
    paras = [sentences(10, w) for w in ("ant", "bee", "cat", "dog")]  # 509 characters each
    chunks = imports.chunk_text("\n\n".join(paras), "n.txt")
    assert [body(c) for c in chunks] == ["\n\n".join(paras[:2]), "\n\n".join(paras[2:])]
    assert all(c.startswith("n.txt\n\n") for c in chunks)


def test_chunk_text_splits_a_long_paragraph_at_a_sentence_end():
    text = sentences(40)  # 2,039 characters, one paragraph
    chunks = imports.chunk_text(text, "n.txt")
    assert len(chunks) == 2 and all(body(c).endswith(".") for c in chunks)
    assert " ".join(body(c) for c in chunks) == text


def test_chunk_text_hard_cuts_a_run_with_no_spaces():
    chunks = imports.chunk_text("x" * 4000, "n.txt")
    assert len(chunks) == 3 and all(len(c) <= 1500 for c in chunks)
    assert "".join(body(c) for c in chunks) == "x" * 4000


def test_chunk_text_short_text_is_one_chunk_as_it_is_and_empty_is_none():
    assert imports.chunk_text("# Title\nbody\n", "n.md") == ["# Title\nbody\n"]
    assert imports.chunk_text("", "n.md") == [] and imports.chunk_text(" \n\t\n", "n.md") == []


def test_chunk_text_never_packs_two_sections_and_names_the_heading_path():
    text = (
        f"# A\n\n{sentences(20, 'aaa')}\n\n## B\n\n{sentences(20, 'bbb')}\n\n"
        f"# C\n\n{sentences(5, 'ccc')}"
    )
    chunks = imports.chunk_text(text, "x.md")
    assert not any("aaa" in c and "ccc" in c for c in chunks)
    under_b = [c for c in chunks if "bbb" in c]
    assert under_b and all(c.startswith("x.md › A › B\n\n") for c in under_b)
    assert [c for c in chunks if "ccc" in c][0].startswith("x.md › C\n\n# C")


def test_chunk_text_breadcrumb_is_capped_cutting_the_deepest_headings():
    text = f"# {'a' * 120}\n\n## {'b' * 120}\n\n{sentences(40)}"
    crumbs = {c.split("\n", 1)[0] for c in imports.chunk_text(text, "x.md")}
    assert crumbs == {f"x.md › {'a' * 120}"}


def test_chunk_text_keeps_a_code_block_whole_and_splits_a_big_one_at_line_ends():
    code = "```\nline one\n\nline two\n```"
    chunks = imports.chunk_text(f"{sentences(20)}\n\n{code}\n\n{sentences(20)}", "x.md")
    assert sum(code in c for c in chunks) == 1
    lines = [f"print({i})  # {'y' * 30}" for i in range(80)]
    chunks = imports.chunk_text("```\n" + "\n".join(lines) + "\n```", "x.md")
    assert len(chunks) > 1
    assert all(
        line in lines or line.startswith(("```", "x.md")) or not line
        for c in chunks
        for line in c.split("\n")
    )


def test_chunk_text_splits_a_long_table_between_rows_repeating_its_header():
    rows = [f"| r{i} | {'v' * 40} |" for i in range(80)]
    chunks = imports.chunk_text("| key | value |\n|---|---|\n" + "\n".join(rows), "t.md")
    assert len(chunks) > 1
    for c in chunks:
        assert body(c).startswith("| key | value |\n|---|---|\n| r")
    held = [line for c in chunks for line in body(c).split("\n")[2:]]
    assert held == rows


def test_chunk_text_merges_a_small_trailing_paragraph_into_the_previous_chunk():
    tail = "A short closing line of exactly fifty characters.."
    chunks = imports.chunk_text(f"{sentences(40)}\n\n{tail}", "n.txt")
    assert len(chunks) == 2 and not any(body(c) == tail for c in chunks)
    assert chunks[-1].endswith(tail) and len(body(chunks[-1])) >= imports.MIN_CHUNK_CHARS


def test_chunk_text_merges_a_small_chunk_into_the_next_one_with_room():
    # 1,478 + 101 does not fit 1,500 less the breadcrumb, so the 101 starts a chunk, and
    # the long paragraph after it fills the rest of that chunk rather than leave it small.
    first, small, long = sentences(29, "a"), sentences(2, "b"), sentences(40, "c")
    chunks = imports.chunk_text(f"{first}\n\n{small}\n\n{long}", "n.txt")
    assert body(chunks[0]) == first and body(chunks[1]).startswith(small + "\n\n" + "c")
    assert all(len(body(c)) >= imports.MIN_CHUNK_CHARS for c in chunks)
    # The floor itself: into the previous body when it fits, else the next, else alone.
    assert len(imports.merge_small(["x" * 900, "s" * 50, "y" * 900], 1000)) == 2
    assert imports.merge_small(["s" * 50, "y" * 900], 1000) == ["s" * 50 + "\n\n" + "y" * 900]
    assert imports.merge_small(["x" * 990, "s" * 50, "y" * 990], 1000)[1] == "s" * 50


def test_chunk_text_keeps_a_list_whole():
    items = "\n".join(f"- item {i}\n  more about it" for i in range(5)) + "\n\n- after a blank"
    chunks = imports.chunk_text(f"{sentences(25)}\n\n{items}\n\n{sentences(5)}", "n.md")
    assert sum(items in c for c in chunks) == 1


def test_chunk_text_every_chunk_fits_the_prompt():
    text = "\n\n".join(
        [
            f"# H{i}\n\n{sentences(i * 7)}\n\n| a | b |\n|-|-|\n" + "| x | y |\n" * i * 30
            for i in range(1, 6)
        ]
        + ["```\n" + "z" * 3000 + "\n```", "- " + "w " * 2000]
    )
    chunks = imports.chunk_text(text, "big.md")
    assert len(chunks) > 5 and all(len(c) <= imports.CHUNK_CHARS for c in chunks)
    assert imports.CHUNK_CHARS == 1500


def test_a_long_text_file_reads_as_one_row_per_chunk(tmp_path):
    p = write(tmp_path / "x.md", f"# A\n\n{sentences(40)}\n\n# B\n\n{sentences(40)}")
    fmt, rows, _ = imports.read_rows(p, "x.md")
    assert fmt == "text" and len(rows) > 2
    n = len(rows)
    assert [r["file"] for r in rows] == [f"x.md#{i}" for i in range(1, n + 1)]
    assert [r["meta"] for r in rows] == [
        {"file": "x.md", "part": i, "parts": n} for i in range(1, n + 1)
    ]
    mapping = imports.guess_mapping(imports.columns(rows), "items")
    assert mapping["id"] == "file" and mapping["meta"] == "meta"
    items, _ = imports.build_items(rows, mapping)
    assert [i.id for i in items] == [f"x.md#{i}" for i in range(1, n + 1)]
    assert items[1].meta == {"file": "x.md", "part": 2, "parts": n}
    assert imports.text_files(rows) == {"x.md"}


def test_a_long_text_file_inside_a_zip_is_chunked_too(tmp_path):
    p = tmp_path / "x.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("docs/long.txt", "x" * 4000)
        z.writestr("short.md", "beta")
    _, rows, _ = imports.read_rows(p, "x.zip")
    assert [r["file"] for r in rows] == [
        "docs/long.txt#1",
        "docs/long.txt#2",
        "docs/long.txt#3",
        "short.md",
    ]
    assert imports.text_files(rows) == {"docs/long.txt", "short.md"}


def test_chunk_text_drops_a_section_that_is_only_its_heading():
    text = f"# Guide\n\n## Install\n\n{sentences(25, 'i')}\n\n## Use\n\n{sentences(25, 'u')}"
    chunks = imports.chunk_text(text, "x.md")
    assert all(body(c).strip("#").strip() not in ("Guide", "") for c in chunks)
    assert all(
        any(line and not line.startswith("#") for line in body(c).split("\n")) for c in chunks
    )
    under = [c for c in chunks if "iiii" in c]
    assert under and all(c.startswith("x.md › Guide › Install\n\n") for c in under)
    # a heading with text of its own keeps its section
    kept = imports.chunk_text(f"# Guide\n\nwelcome\n\n## Use\n\n{sentences(40, 'u')}", "x.md")
    assert body(kept[0]) == "# Guide\n\nwelcome"


def test_chunk_text_one_chunk_needs_no_room_for_a_breadcrumb():
    text = sentences(29)  # 1,478 characters
    assert len(text) == 1478
    assert imports.chunk_text(text, "n" * 60 + ".md") == [text]


def test_a_one_chunk_text_file_is_never_longer_than_a_chunk(tmp_path):
    text = ("para. " * 40 + "\n\n\n\n") * 6 + "\n" * 200  # over 1,500 only by its padding
    assert len(text) > 1500 and len(text.strip()) <= 1500
    rows = imports.text_rows("w.txt", text)
    assert rows == [{"file": "w.txt", "text": text.strip()}]
    assert imports.text_rows("w.txt", "short\n") == [{"file": "w.txt", "text": "short\n"}]
    assert imports.text_rows("w.txt", "  ") == [{"file": "w.txt", "text": "  "}]
