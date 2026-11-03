"""Program-and-run answers (spec D14): the answering model writes a short Python program,
it runs here, and its last printed line is the answer.

The sandbox bounds time, memory, file size and output, stops the program starting any
process or thread, and gives it a fresh temp dir and no secrets in its environment.

It is NOT network-isolated and it can read what the service can read, the database of
stored API keys included. The code comes from the user's configured model, but that model
reads the question and the passages, so text in a corpus can steer what it writes."""

import os
import re
import signal
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
# ONE BLAS THREAD: with no threads allowed, numpy's default pool dies on import on Linux.
PROGRAM_ENV = {"OPENBLAS_NUM_THREADS": "1"}
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
        # No new process, so nothing can leave the group the timeout kills. Root ignores it.
        (resource.RLIMIT_NPROC, 0),
    ):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def run_program(code: str) -> dict:
    """Run `code` as `python -I -c` in a fresh temp dir with only PROGRAM_ENV set.
    `ran` is True only for a clean exit within the timeout. A program that cannot be
    started (a NUL byte in it, no process to spare) did not run: it never raises."""
    try:
        return launch(code)
    except (OSError, ValueError) as e:
        return {"ran": False, "output": None, "error": str(e) or type(e).__name__}


def launch(code: str) -> dict:
    # STDOUT GOES TO A FILE, NOT A PIPE: RLIMIT_FSIZE then bounds what a runaway print
    # can write, and nothing has to drain a pipe while the program runs.
    posix = os.name == "posix"
    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cwd,
        tempfile.TemporaryFile() as out,
        tempfile.TemporaryFile() as err,
    ):
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", code],
            cwd=cwd,
            env=PROGRAM_ENV,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            preexec_fn=limit_resources if posix else None,
            start_new_session=posix,  # its own process group, so a timeout reaches its children
        )
        try:
            exit_code = proc.wait(timeout=PROGRAM_TIMEOUT)
        except subprocess.TimeoutExpired:
            stop(proc, posix)
            exit_code = None
        size = out.seek(0, os.SEEK_END)
        out.seek(0)
        raw = out.read(OUTPUT_CAP * 4)
        text = raw.decode("utf-8", "replace")
        output, cut = text[:OUTPUT_CAP], size > len(raw) or len(text) > OUTPUT_CAP
        err.seek(max(0, err.seek(0, os.SEEK_END) - ERROR_CAP * 4))
        stderr = err.read().decode("utf-8", "replace")[-ERROR_CAP:].strip()
    if exit_code is None:
        return {"ran": False, "output": output, "error": f"timed out after {PROGRAM_TIMEOUT} s"}
    if exit_code != 0:
        return {"ran": False, "output": output, "error": stderr or f"exited with code {exit_code}"}
    # A CUT OUTPUT HAS LOST ITS LAST LINE, the answer: it ran, but answers nothing.
    error = f"output cut at {OUTPUT_CAP} characters" if cut else None
    return {"ran": True, "output": output, "error": error}


def stop(proc: subprocess.Popen, posix: bool) -> None:
    """Kill the program and, on POSIX, every process it started (its process group)."""
    try:
        if posix:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    proc.wait()


def program_answer(text: str) -> tuple[str | None, dict]:
    """The last non-empty line the reply's program printed, or None, and the run record.
    None too when the output was cut, so the reply's `Answer:` line is used instead."""
    code = extract_program(text)
    if code is None:
        return None, {"ran": False, "output": None, "error": "no program"}
    record = run_program(code)
    lines = [line.strip() for line in (record["output"] or "").splitlines() if line.strip()]
    answered = record["ran"] and record["error"] is None and lines
    return (lines[-1] if answered else None), record
