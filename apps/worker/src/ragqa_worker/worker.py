import threading

from ragqa_core import db
from ragqa_core.connection import Connection
from ragqa_core.evaluator import Evaluator
from ragqa_core.indexer import Indexer
from ragqa_core.llm import BudgetExceeded, Cancelled, Claude, NoApiKey
from ragqa_core.loader import Loader


class Worker:
    """Runs queued jobs one at a time on its own thread, so every cost lands on its job."""

    def __init__(
        self,
        conn: Connection,
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
        self.error: str | None = None  # the last cycle's error, until a clean cycle
        self._cycle_failed = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_one(self) -> str | None:
        """Claim the oldest queued job and run it to a final status; None if the queue is empty."""
        job_id = db.claim_next_job_id(self.conn)
        if job_id is None:
            return None
        self.running_job_id = job_id
        try:
            self._run(job_id)
        finally:
            self.running_job_id = None
        return job_id

    def _run(self, job_id: str) -> None:
        conn = self.conn
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

        with self.claude.for_job(job_id, check, log), self.indexer.models.for_job(log):
            reason = "job ended without a final status"
            try:
                log("started")
                job = db.get_job(conn, job_id)
                if job is None:
                    raise KeyError(job_id)
                self._dispatch(job, check, progress)
            except Exception as e:
                try:
                    self._record_failure(job_id, e)
                except Exception as e2:
                    self._failed(e2)
                    reason = f"worker could not record a final status: {e2!r}"
            self._ensure_final(job_id, reason, log)

    def _record_failure(self, job_id: str, e: Exception) -> None:
        conn = self.conn
        if isinstance(e, Cancelled):
            db.update_job(conn, job_id, status="cancelled")
        elif isinstance(e, BudgetExceeded):
            db.update_job(conn, job_id, status="stopped_budget", error=str(e))
        elif isinstance(e, NoApiKey):
            db.update_job(conn, job_id, status="stopped_no_key")
        else:
            db.update_job(conn, job_id, status="failed", error=repr(e))

    def _ensure_final(self, job_id: str, reason: str, log) -> None:
        """Last resort: a claimed job never stays 'running'; failures here are kept, not raised."""
        try:
            job = db.get_job(self.conn, job_id)
            if job is not None and job["status"] == "running":
                db.update_job(self.conn, job_id, status="failed", error=reason)
                job = db.get_job(self.conn, job_id)
            log(f"finished: {job['status'] if job else 'unknown'}")
        except Exception as e:
            self._failed(e)

    def _dispatch(self, job: dict, check, progress) -> None:
        job_id, kind = job["id"], job["kind"]
        if kind == "load":
            result = self.loader.run(job, progress)
            db.update_job(self.conn, job_id, status="done", progress=1.0, result=result)
        elif kind == "rekey":
            version = job["setup"]["version"]
            self.indexer.rekey(job["collection"], version, progress)
            db.update_job(
                self.conn, job_id, status="done", progress=1.0, result={"version": version}
            )
        elif kind == "backfill":
            result = self.indexer.backfill(job["collection"], progress)
            db.update_job(self.conn, job_id, status="done", progress=1.0, result=result)
        elif kind == "eval":
            self.evaluator.run(job_id, check)
        else:
            db.update_job(self.conn, job_id, status="failed", error=f"unknown job kind {kind!r}")

    def _failed(self, e: Exception) -> None:
        self.error = repr(e)
        self._cycle_failed = True

    def cycle(self) -> None:
        """Run one job, or collect pending batches and idle; a clean cycle clears the error."""
        self._cycle_failed = False
        try:
            if self.run_one() is None:
                try:
                    self.claude.collect_pending()
                except Exception as e:
                    self._failed(e)
                self._stop.wait(self.idle_seconds)
        except Exception as e:
            self._failed(e)
            self._stop.wait(self.idle_seconds)
        if not self._cycle_failed:
            self.error = None

    def loop(self) -> None:
        while not self._stop.is_set():
            self.cycle()

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
