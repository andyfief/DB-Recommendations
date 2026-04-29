# Extracted from DrinkBuilder/src/seed_data.py.
# Kept: token seeding from tokens.csv into the rules DB.
# Paths updated for backend/lib/ location; uses trimmed tokens.csv.

import sqlite3
import pandas as pd
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
DB_PATH = LIB_DIR / "schema" / "rules.db"
TOKENS_CSV = LIB_DIR / "data" / "tokens.csv"
INGREDIENTS_CSV = LIB_DIR / "data" / "ingredients.csv"


def seed_tokens(conn: sqlite3.Connection) -> int:
    df = pd.read_csv(TOKENS_CSV)
    df["name"] = df["name"].str.strip().str.lower()

    cur = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
            INSERT OR IGNORE INTO tokens (name, category)
            VALUES (?, ?)
        """, (row["name"], row.get("category", "standard")))

    conn.commit()
    return len(df)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    count = seed_tokens(conn)
    conn.close()
    print(f"Loaded {count} tokens")
