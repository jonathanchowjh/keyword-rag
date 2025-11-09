import sqlite3
import threading

from ragqa_core import db
from ragqa_core.evaluator import Evaluator
from ragqa_core.indexer import Indexer
from ragqa_core.llm import BudgetExceeded, Cancelled, Claude, NoApiKey
from ragqa_core.loader import Loader


class Worker:
    """Runs queued jobs one at a time on its own thread, so every cost lands on its job."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        claude: Claude,
        indexer: Indexer,
        loader: Loader,
        evaluator: Evaluator,
        idle_seconds: float = 0.5,
    ):
        self.conn = conn
        self.claude = claude
        self.indexer = indexer
        self.loader = loader
        self.evaluator = evaluator
        self.idle_seconds = idle_seconds
        self.running_job_id: str | None = None
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_one(self) -> str | None:
        """Claim the oldest queued job and run it to a final status; None if the queue is empty."""
        job = db.claim_next_job(self.conn)
        if job is None:
            return None
        job_id = job["id"]
        self.running_job_id = job_id
        try:
            self._run(job)
        finally:
            self.running_job_id = None
        return job_id

    def _run(self, job: dict) -> None:
        conn, job_id = self.conn, job["id"]
        step = {"last": 0}

        def check() -> None:
            if db.is_cancel_requested(conn, job_id):
                raise Cancelled()

        def log(message: str) -> None:
            db.log_job(conn, job_id, message)

        def progress(p: float) -> None:
            check()
            db.update_job(conn, job_id, progress=p)
            pct = int(p * 10) * 10
            if pct > step["last"]:
                step["last"] = pct
                log(f"progress {pct}%")

        with self.claude.for_job(job_id, check, log):
            log("started")
            try:
                self._dispatch(job, check, progress)
            except Cancelled:
                db.update_job(conn, job_id, status="cancelled")
            except BudgetExceeded as e:
                db.update_job(conn, job_id, status="stopped_budget", error=str(e))
            except NoApiKey:
                db.update_job(conn, job_id, status="stopped_no_key")
            except Exception as e:
                db.update_job(conn, job_id, status="failed", error=repr(e))
            final = db.get_job(conn, job_id)
            log(f"finished: {final['status'] if final else 'unknown'}")

    def _dispatch(self, job: dict, check, progress) -> None:
        job_id, kind = job["id"], job["kind"]
        if kind == "load":
            result = self.loader.run(job, check)
            db.update_job(self.conn, job_id, status="done", progress=1.0, result=result)
        elif kind == "rekey":
            version = job["setup"]["version"]
            self.indexer.rekey(job["collection"], version, progress)
            db.update_job(
                self.conn, job_id, status="done", progress=1.0, result={"version": version}
            )
        elif kind == "eval":
            self.evaluator.run(job_id, check)
        else:
            db.update_job(self.conn, job_id, status="failed", error=f"unknown job kind {kind!r}")

    def loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.run_one() is None:
                    try:
                        self.claude.collect_pending()
                    except Exception as e:
                        self.error = repr(e)
                    self._stop.wait(self.idle_seconds)
            except Exception as e:
                self.error = repr(e)
                self._stop.wait(self.idle_seconds)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.loop, name="ragqa-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
