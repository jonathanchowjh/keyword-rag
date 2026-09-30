import time

from ragqa_core.program import (
    OUTPUT_CAP,
    PROGRAM_TIMEOUT,
    extract_program,
    program_answer,
    run_program,
)


def test_extract_program_takes_the_last_python_block():
    text = "```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```\nAnswer: 2"
    assert extract_program(text) == "print(2)\n"
    assert extract_program("```\nprint(1)\n```") is None
    assert extract_program("no code at all") is None


def test_program_runs_and_answers():
    out, rec = program_answer("plan\n```python\nprint(6*7)\n```\nAnswer: 41")
    assert out == "42" and rec["ran"] is True
    assert rec == {"ran": True, "output": "42\n", "error": None}


def test_program_answer_is_the_last_non_empty_line():
    out, _ = program_answer("```python\nprint('working')\nprint('3.5')\nprint()\n```")
    assert out == "3.5"


def test_program_without_code_or_output_has_no_answer():
    out, rec = program_answer("Answer: 41")
    assert out is None and rec["ran"] is False and rec["error"] == "no program"
    out, rec = program_answer("```python\nx = 1\n```")
    assert out is None and rec["ran"] is True


def test_program_timeout_falls_back():
    start = time.monotonic()
    out, rec = program_answer("```python\nwhile True:\n    pass\n```\nAnswer: 7")
    assert time.monotonic() - start < PROGRAM_TIMEOUT + 2
    assert out is None and rec["ran"] is False and "timed out" in rec["error"]


def test_program_output_is_capped():
    rec = run_program('print("x" * 10**6)')
    assert rec["output"] is not None and len(rec["output"]) <= OUTPUT_CAP


def test_program_error_is_recorded():
    rec = run_program('raise ValueError("boom")')
    assert rec["ran"] is False and "boom" in (rec["error"] or "")


def test_program_runs_in_a_temp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = run_program('open("f", "w").write("hi")\nimport os\nprint(os.listdir("."))')
    assert rec["ran"] is True and "'f'" in (rec["output"] or "")
    assert list(tmp_path.iterdir()) == []


def test_program_sees_no_environment(monkeypatch):
    monkeypatch.setenv("RAGQA_SECRET", "s3cret")
    rec = run_program('import os\nprint(os.environ.get("RAGQA_SECRET"))')
    assert rec["output"] == "None\n"
