# Extracted from DrinkBuilder/src/seed_rules.py.
# Kept: rule loading from JSON files into the rules DB.
# Paths updated for backend/lib/ location.

import sqlite3
import json
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
DB_PATH = LIB_DIR / "schema" / "rules.db"
RULES_DIR = LIB_DIR / "rules"


def load_token_map(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM tokens WHERE active = 1")
    return {name: tid for tid, name in cur.fetchall()}


def resolve_token(token_map: dict, name: str) -> int:
    if name not in token_map:
        raise ValueError(f"Unknown token: '{name}'")
    return token_map[name]


def insert_rule(conn: sqlite3.Connection, rule: dict, token_map: dict) -> None:
    cur = conn.cursor()
    cur.execute("SAVEPOINT insert_rule")
    try:
        payload_json = json.dumps(rule.get('payload')) if 'payload' in rule else None

        cur.execute("""
            INSERT INTO rules (rule_type, description, priority, payload_json, active)
            VALUES (?, ?, ?, ?, 1)
        """, (rule['type'], rule['description'], rule['priority'], payload_json))

        rule_id = cur.lastrowid

        for token_name in rule['triggers']:
            token_id = resolve_token(token_map, token_name)
            cur.execute("""
                INSERT INTO rule_tokens (rule_id, token_id, role)
                VALUES (?, ?, 'trigger')
            """, (rule_id, token_id))

        if 'results' in rule:
            for token_name in rule['results']:
                token_id = resolve_token(token_map, token_name)
                cur.execute("""
                    INSERT INTO rule_tokens (rule_id, token_id, role)
                    VALUES (?, ?, 'result')
                """, (rule_id, token_id))

        cur.execute("RELEASE SAVEPOINT insert_rule")
    except ValueError:
        cur.execute("ROLLBACK TO SAVEPOINT insert_rule")
        raise  # re-raise so seed_rules can count skipped


def seed_rules(conn: sqlite3.Connection) -> int:
    token_map = load_token_map(conn)

    rule_files = [
        'classify.json',
        'profiles.json',
        'quantities.json',
        'assignments.json',
        'modifiers.json',
    ]

    total = 0
    for filename in rule_files:
        filepath = RULES_DIR / filename
        with open(filepath, 'r') as f:
            rules = json.load(f)

        skipped = 0
        for rule in rules:
            try:
                insert_rule(conn, rule, token_map)
                total += 1
            except ValueError:
                # Token in rule not in trimmed tokens.csv — skip silently
                skipped += 1

        conn.commit()

    return total


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    count = seed_rules(conn)
    conn.close()
    print(f"Loaded {count} rules")
