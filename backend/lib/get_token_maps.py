# Extracted from DrinkBuilder/tools/tokens/get_token_maps.py.
# Kept: extract_tokens_from_name() — greedy token matching from a drink name string.
# Removed: CSV batch processing (process_all_drinks, save_to_jsonl) — not needed here.
# Paths updated for backend/lib/ location; uses trimmed tokens.csv.

import re
from pathlib import Path

import pandas as pd

_LIB_DIR = Path(__file__).resolve().parent
_TOKENS_CSV = _LIB_DIR / "data" / "tokens.csv"

_tokens_df = pd.read_csv(_TOKENS_CSV)
KNOWN_TOKENS: list[str] = _tokens_df['name'].tolist()
KNOWN_TOKENS_SORTED: list[str] = sorted(KNOWN_TOKENS, key=len, reverse=True)

_FRACTIONS = {
    r"\b1\s*/\s*4\b": "quarter",
    r"\b1\s*/\s*2\b": "half",
    r"\b3\s*/\s*4\b": "three quarter",
}


def extract_tokens_from_name(name: str) -> dict[str, float]:
    """
    Greedily match known tokens in a drink name string.

    Returns {token: count}. Longest tokens are matched first to avoid
    substring collisions (e.g. 'cold_brew' before 'cold').
    """
    if not name or (isinstance(name, float) and pd.isna(name)):
        return {}

    name_clean = name.lower().strip()

    for pattern, replacement in _FRACTIONS.items():
        name_clean = re.sub(pattern, replacement, name_clean, flags=re.IGNORECASE)

    name_clean = name_clean.replace("%", " percent")
    name_clean = re.sub(r"[^\w\s]", "", name_clean)
    name_clean = re.sub(r"\s+", " ", name_clean)
    name_clean = re.sub(r"\badd\b", "", name_clean, flags=re.IGNORECASE).strip()

    matched: dict[str, float] = {}
    remaining = name_clean

    for token in KNOWN_TOKENS_SORTED:
        token_phrase = token.replace('_', ' ')
        while token_phrase in remaining:
            matched[token] = matched.get(token, 0) + 1
            remaining = remaining.replace(token_phrase, '', 1)
            remaining = re.sub(r"\s+", " ", remaining).strip()

    return matched
