import os
import time

import pytest

from ragqa_core import program
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
    assert rec["ran"] is True and rec["error"] == f"output cut at {OUTPUT_CAP} characters"


def test_cut_output_gives_no_answer():
    # The answer is the last line, and a cut output has lost it: a fragment is no answer.
    out, rec = program_answer('```python\nprint("x" * 5000)\nprint(42)\n```\nAnswer: 42')
    assert out is None and rec["ran"] is True and "cut" in rec["error"]


def test_a_program_that_cannot_start_did_not_run():
    rec = run_program("print(1)\x00")
    assert rec["ran"] is False and rec["output"] is None and "null" in rec["error"]
    out, rec = program_answer("```python\nprint(1)\x00\n```\nAnswer: 1")
    assert out is None and rec["ran"] is False


@pytest.mark.skipif(os.name != "posix", reason="process limits are POSIX")
def test_a_program_cannot_start_a_process_to_outlive_the_timeout():
    code = (
        "import subprocess\n"
        'subprocess.Popen(["sleep", "30"], start_new_session=True)\n'
        'print("escaped")\n'
    )
    rec = run_program(code)
    assert rec["ran"] is False and "escaped" not in (rec["output"] or "")


def test_a_program_can_use_numpy():
    rec = run_program("import numpy as np\nprint(np.ones((2, 2)).sum())")
    assert rec["ran"] is True and rec["output"] == "4.0\n", rec


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_timeout_kills_the_programs_children(monkeypatch):
    # The group kill is the backstop where the process limit does not bind (root).
    monkeypatch.setattr(program, "limit_resources", lambda: None)
    monkeypatch.setattr(program, "PROGRAM_TIMEOUT", 1)
    code = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(30)\n"
        "    os._exit(0)\n"
        "print(pid, flush=True)\n"
        "while True:\n"
        "    pass\n"
    )
    rec = run_program(code)
    assert rec["ran"] is False and "timed out" in rec["error"]
    child = int((rec["output"] or "").strip())
    deadline = time.monotonic() + 3  # the orphan is reaped by init, not by us
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(child, 9)
    raise AssertionError("the program's child outlived the timeout")


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
