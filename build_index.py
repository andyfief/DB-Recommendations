"""
Pipeline step 1: Vectorize drink builds and persist to vector_db.json.

Reads flavors.csv as a whitelist — only builds whose trigger name appears
in that file are indexed. Tokens in flavors.csv that have no matching flavor
assignment in assignments.json are reported but not treated as errors.

Run this once (or whenever assignments.json, flavor_vectors.csv, or flavors.csv changes):
    python -m drink_recommendations.build_index

Produces drink_recommendations/vector_db.json, which is the source of truth for
all downstream recommendation queries.
"""

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from flavor_catalog import load_flavor_catalog, load_filter_list
from flavor_vectorize import (
    DIMS,
    build_raw_vector,
    build_semantic_vector,
    load_semantic_vectors,
)

_HERE = Path(__file__).parent
DEFAULT_ASSIGNMENTS = _HERE.parent / "rules" / "assignments.json"
DEFAULT_VECTORS_CSV = _HERE / "flavor_vectors.csv"
DEFAULT_FILTER_CSV = _HERE / "flavors.csv"
DEFAULT_DB_PATH = _HERE / "vector_db.json"


def build_vector_db(
    assignments_path: Path = DEFAULT_ASSIGNMENTS,
    vectors_csv_path: Path = DEFAULT_VECTORS_CSV,
    filter_path: Path | None = DEFAULT_FILTER_CSV,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """
    Vectorize every build in the (optionally filtered) flavor catalog and write
    to db_path as JSON.

    Args:
        assignments_path: Path to assignments.json.
        vectors_csv_path: Path to flavor_vectors.csv.
        filter_path: Path to a whitelist CSV (e.g. flavors.csv). Pass None to
                     index all builds without filtering.
        db_path: Output path for vector_db.json.

    Returns the in-memory DB dict.
    """
    # Load whitelist and report coverage
    allowed: set[str] | None = None
    if filter_path is not None and Path(filter_path).exists():
        allowed = load_filter_list(filter_path)
        print(f"Filter list loaded from {filter_path}: {len(allowed)} tokens")

    catalog = load_flavor_catalog(assignments_path, filter_path=filter_path)

    # Report which whitelist tokens had no flavor assignment in assignments.json
    if allowed is not None:
        unmatched = sorted(allowed - set(catalog.keys()))
        if unmatched:
            print(f"  Tokens in filter list with no flavor assignment ({len(unmatched)}):")
            for t in unmatched:
                print(f"    - {t}")

    semantic_vectors = load_semantic_vectors(vectors_csv_path)

    # Sorted list of all unique flavor ingredients across all indexed builds.
    # Defines the raw vector index; stored in DB for query-time reconstruction.
    all_flavors: list[str] = sorted({fl for flavors in catalog.values() for fl in flavors})

    builds: dict = {}
    missing_sem: set[str] = set()
    for name, flavors in catalog.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sem_vec = build_semantic_vector(flavors, semantic_vectors, warn_missing=True)
            for w in caught:
                if "No semantic vector" in str(w.message):
                    # Extract the flavor name from the warning message
                    msg = str(w.message)
                    fl = msg.split("'")[1] if "'" in msg else msg
                    missing_sem.add(fl)
        raw_vec = build_raw_vector(flavors, all_flavors)
        builds[name] = {
            "flavors": flavors,
            "semantic_vec": sem_vec.tolist(),
            "raw_vec": raw_vec.tolist(),
        }

    db = {
        "_meta": {
            "generated_at": str(date.today()),
            "n_builds": len(builds),
            "n_semantic_dims": len(DIMS),
            "semantic_dim_labels": DIMS,
            "n_raw_dims": len(all_flavors),
            "raw_flavor_index": all_flavors,
            "assignments_source": str(assignments_path),
            "vectors_source": str(vectors_csv_path),
            "filter_source": str(filter_path) if filter_path else None,
        },
        "builds": builds,
    }

    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    print(f"\nVector DB written to {db_path}")
    print(f"  Builds indexed : {len(builds)}")
    print(f"  Semantic dims  : {len(DIMS)}")
    print(f"  Raw flavor dims: {len(all_flavors)}")
    if missing_sem:
        print(f"  Flavor ingredients without semantic vector ({len(missing_sem)}) — contributing zero:")
        for fl in sorted(missing_sem):
            print(f"    - {fl}")

    return db


if __name__ == "__main__":
    build_vector_db()
