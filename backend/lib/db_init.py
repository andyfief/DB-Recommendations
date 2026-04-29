# Extracted from DrinkBuilder/src/db_init.py.
# Kept: database initialization from schema.sql.
# Paths updated for backend/lib/ location.

import sqlite3
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
DB_PATH = LIB_DIR / "schema" / "rules.db"
SCHEMA_PATH = LIB_DIR / "schema" / "schema.sql"


def init_db() -> None:
    with open(SCHEMA_PATH, 'r') as f:
        schema = f.read()

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(schema)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized")
