"""
Step 3: Assignment check.

After modification, checks whether the modified drink matches any named drink
in the vector DB by comparing flavor sets.
"""

from __future__ import annotations

import json

from backend.config import DRINK_DB_PATH
from backend.models import AssignmentResult, DrinkObject


def _load_db() -> dict:
    with open(DRINK_DB_PATH, encoding="utf-8") as f:
        return json.load(f)


async def check_assignment(modified_drink: DrinkObject, what_changed: str) -> AssignmentResult:
    """
    Check if modified_drink's flavor set matches any known named drink.

    Returns AssignmentResult with matched_name set if a match is found,
    or None if no named drink matches.
    """
    db = _load_db()
    builds = db.get("builds", {})

    modified_flavors = set(modified_drink.flavors.keys())

    matched_name: str | None = None
    for build_name, entry in builds.items():
        build_flavors = set(entry.get("flavors", []))
        if build_flavors and build_flavors == modified_flavors:
            matched_name = build_name
            break

    return AssignmentResult(matched_name=matched_name, what_changed=what_changed)
