import sqlite3
import threading
from pathlib import Path


class Connection:
    """The process's one SQLite connection, safe to share across threads.

    SQLite's serialized mode guards the database, but not Python's cursors and
    statement cache: two threads stepping statements on one sqlite3 connection
    raise InterfaceError. Every call here runs under one lock and returns its
    rows already fetched, so no live cursor ever leaves the lock.
    """

    def __init__(self, path: Path):
        self.raw = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.raw.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.execute("pragma journal_mode=wal")

    def execute(self, sql: str, params: tuple | list | dict = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.raw.execute(sql, params).fetchall()

    def executemany(self, sql: str, rows: list) -> None:
        with self.lock:
            self.raw.executemany(sql, rows)

    def executescript(self, sql: str) -> None:
        with self.lock:
            self.raw.executescript(sql)

    def close(self) -> None:
        with self.lock:
            self.raw.close()

    def changes(self) -> int:
        """Rows changed by the last statement; call it inside `with conn.lock` with that
        statement, or another thread's statement may come between."""
        with self.lock:
            return self.raw.execute("select changes()").fetchone()[0]
