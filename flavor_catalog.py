import csv
import json
from pathlib import Path


def load_filter_list(filter_csv_path: str | Path) -> set[str]:
    """
    Read a CSV whose first column contains token names (one per row, header ignored).
    Returns the set of stripped, non-empty names.
    """
    names: set[str] = set()
    with open(filter_csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if row and row[0].strip():
                names.add(row[0].strip())
    return names


def load_flavor_catalog(
    assignments_path: str | Path,
    filter_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """
    Parse assignments.json and return {drink_name: [flavor_ingredients]}
    for all rules with payload.role == 'flavor'.

    Args:
        assignments_path: Path to assignments.json.
        filter_path: Optional path to a CSV whitelist (e.g. flavors.csv).
                     When provided, only builds whose trigger name appears in
                     the whitelist are included.
    """
    allowed: set[str] | None = None
    if filter_path is not None:
        allowed = load_filter_list(filter_path)

    with open(assignments_path, encoding="utf-8") as f:
        rules = json.load(f)

    catalog: dict[str, list[str]] = {}
    for rule in rules:
        payload = rule.get("payload", {})
        if payload.get("role") != "flavor":
            continue
        triggers = rule.get("triggers", [])
        if not triggers:
            continue
        name = triggers[0]
        if allowed is not None and name not in allowed:
            continue
        items = []
        for item in payload.get("items", []):
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, dict):
                items.append(item["name"])
        if items:
            catalog[name] = items

    return catalog
