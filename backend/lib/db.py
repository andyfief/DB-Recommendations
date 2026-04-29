"""
DB connection manager for the DrinkAdvisor rule engine.

open_connection() returns a fresh sqlite3 connection each time, used as a
context manager so it is always closed after the call. SQLite reads are fast
enough that a per-call connection is correct and avoids state pollution across
requests.

The DB file is initialized once on first use (schema → tokens → rules).
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

_LIB_DIR = Path(__file__).resolve().parent
_DB_PATH = _LIB_DIR / "schema" / "rules.db"


def ensure_initialized() -> None:
    """Create and seed the rules DB if it does not exist yet."""
    if _DB_PATH.exists():
        return

    from backend.lib.seed_data import seed_tokens
    from backend.lib.seed_rules import seed_rules

    schema_path = _LIB_DIR / "schema" / "schema.sql"
    conn = sqlite3.connect(_DB_PATH)
    try:
        with open(schema_path, "r") as f:
            conn.executescript(f.read())
        conn.commit()
        seed_tokens(conn)
        seed_rules(conn)
    finally:
        conn.close()


@contextmanager
def open_connection():
    """Yield a fresh, isolated sqlite3 connection. Always closed on exit."""
    ensure_initialized()
    conn = sqlite3.connect(_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


# Keep a module-level helper for the startup event so main.py doesn't need to change.
def get_connection() -> sqlite3.Connection:
    """Return a one-off connection (caller is responsible for closing it).
    Prefer open_connection() context manager in pipeline code."""
    ensure_initialized()
    return sqlite3.connect(_DB_PATH)
