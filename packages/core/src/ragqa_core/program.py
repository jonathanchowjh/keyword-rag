"""Program-and-run answers (spec D14): the answering model writes a short Python program,
it runs here, and its last printed line is the answer.

The sandbox bounds time, memory, file size and output and gives the program an empty
environment and a fresh temp dir. It is NOT network-isolated, so it only ever runs code
written by the user's own configured model."""

import os
import re
import subprocess
import sys
import tempfile

if os.name == "posix":
    import resource

PROGRAM_TIMEOUT = 10  # seconds of wall time
OUTPUT_CAP = 2000  # characters of stdout kept
CPU_SECONDS = 10
MEMORY_BYTES = 512 * 1024 * 1024
FILE_BYTES = 1024 * 1024
ERROR_CAP = 500  # the tail of stderr kept as the error
PYTHON_BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.S)


def extract_program(text: str) -> str | None:
    """The last ```python fenced block, or None."""
    blocks = PYTHON_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def limit_resources() -> None:
    # IN THE CHILD, BEFORE EXEC. A limit the platform refuses (macOS and RLIMIT_AS) is
    # skipped: the wall timeout still bounds the run.
    for limit, value in (
        (resource.RLIMIT_CPU, CPU_SECONDS),
        (resource.RLIMIT_AS, MEMORY_BYTES),
        (resource.RLIMIT_FSIZE, FILE_BYTES),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def run_program(code: str) -> dict:
    """Run `code` as `python -I -c` in a fresh temp dir with an empty environment.
    `ran` is True only for a clean exit within the timeout."""
    # STDOUT GOES TO A FILE, NOT A PIPE: RLIMIT_FSIZE then bounds what a runaway print
    # can write, and nothing has to drain a pipe while the program runs.
    with (
        tempfile.TemporaryDirectory() as cwd,
        tempfile.TemporaryFile() as out,
        tempfile.TemporaryFile() as err,
    ):
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", code],
            cwd=cwd,
            env={},
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            preexec_fn=limit_resources if os.name == "posix" else None,
        )
        try:
            exit_code = proc.wait(timeout=PROGRAM_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            exit_code = None
        out.seek(0)
        output = out.read(OUTPUT_CAP * 4).decode("utf-8", "replace")[:OUTPUT_CAP]
        err.seek(0, os.SEEK_END)
        err.seek(max(0, err.tell() - ERROR_CAP * 4))
        stderr = err.read().decode("utf-8", "replace")[-ERROR_CAP:].strip()
    if exit_code is None:
        return {"ran": False, "output": output, "error": f"timed out after {PROGRAM_TIMEOUT} s"}
    if exit_code != 0:
        return {"ran": False, "output": output, "error": stderr or f"exited with code {exit_code}"}
    return {"ran": True, "output": output, "error": None}


def program_answer(text: str) -> tuple[str | None, dict]:
    """The last non-empty line the reply's program printed, or None, and the run record."""
    code = extract_program(text)
    if code is None:
        return None, {"ran": False, "output": None, "error": "no program"}
    record = run_program(code)
    lines = [line.strip() for line in (record["output"] or "").splitlines() if line.strip()]
    return (lines[-1] if record["ran"] and lines else None), record
