"""SQLite helper: a connection whose `with` block commits / rolls back AND closes (the stdlib one leaves it open)."""
import sqlite3


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)          # commit on success, rollback on error
        finally:
            self.close()


def connect(path) -> sqlite3.Connection:
    c = sqlite3.connect(path, factory=ClosingConnection)
    c.row_factory = sqlite3.Row
    return c
