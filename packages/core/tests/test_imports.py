import json
import zipfile
from pathlib import Path

import pytest

from ragqa_core import imports
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
