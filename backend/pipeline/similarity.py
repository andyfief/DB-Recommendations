"""
Step 2a: Similarity path.

Queries the vector DB for the N nearest neighbor drinks, then produces a
DrinkObject for each neighbor via the rule engine. Each neighbor is resolved
by passing its name as a single-token dict {name: 1} — the same way DrinkBuilder
test inputs work — so the classify cascade produces the correct ingredient set.
"""

from __future__ import annotations

import sys
from pathlib import Path

from backend.config import DRINK_DB_PATH
from backend.models import DrinkObject, NeighborResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_neighbor_drink(name: str) -> tuple[DrinkObject, dict]:
    """Return (DrinkObject, raw_engine_result) for a neighbor drink name."""
    from backend.lib.db import open_connection
    from backend.lib.engine import process_order

    token_quantities = {name: 1}
    with open_connection() as conn:
        raw = process_order(token_quantities, conn)
    return DrinkObject.from_engine_result(raw), raw


async def find_similar(drink_name: str, n: int = 5) -> tuple[list[NeighborResult], list[dict]]:
    """
    Return (neighbors, engine_results) where engine_results[i] is the raw
    process_order output for neighbors[i], used for fired-rule audit logging.

    Raises KeyError if drink_name is not in the vector DB.
    """
    from recommend import recommend_from_db

    neighbors_raw = recommend_from_db(
        drink_name,
        k=n,
        method="semantic",
        db_path=DRINK_DB_PATH,
    )

    results: list[NeighborResult] = []
    engine_results: list[dict] = []
    for entry in neighbors_raw:
        neighbor_name = entry["name"]
        try:
            drink, engine_raw = _build_neighbor_drink(neighbor_name)
        except Exception:
            drink = DrinkObject(flavors={}, toppings={}, milk=[], base=[], coffee=[])
            engine_raw = {}
        results.append(NeighborResult(name=neighbor_name, drink=drink))
        engine_results.append(engine_raw)

    return results, engine_results
