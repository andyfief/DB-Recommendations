"""
Flavor-based drink recommendation system.

Two modes:
  DB mode  (preferred) — recommend_from_db(build_name, k, method)
      Reads precomputed vectors from vector_db.json (built by build_index.py).
      Fast; no re-vectorization at query time.

  Live mode — recommend(query_flavors, k, method)
      Computes vectors on-the-fly from assignments.json + flavor_vectors.csv.
      Useful for custom flavor lists not in the catalog.

Similarity methods:
  semantic — cosine similarity on 28-dim aggregated semantic vectors
  raw      — Jaccard similarity on flavor ingredient sets
  hybrid   — HYBRID_SEMANTIC_WEIGHT * semantic + HYBRID_RAW_WEIGHT * raw
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flavor_catalog import load_flavor_catalog
from flavor_vectorize import (
    build_raw_vector,
    build_semantic_vector,
    cosine_similarity,
    jaccard_similarity,
    load_semantic_vectors,
)

_HERE = Path(__file__).parent
_DEFAULT_DB = _HERE / "vector_db.json"
_DEFAULT_CATALOG = _HERE.parent / "rules" / "assignments.json"
_DEFAULT_VECTORS = _HERE / "flavor_vectors.csv"

HYBRID_SEMANTIC_WEIGHT = 0.6
HYBRID_RAW_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_db(db_path: Path) -> dict:
    with open(db_path, encoding="utf-8") as f:
        return json.load(f)


def _compute_similarity(
    query_flavors: list[str],
    query_sem: np.ndarray,
    catalog_flavors: list[str],
    catalog_sem: np.ndarray,
    method: str,
) -> float:
    if method == "semantic":
        return cosine_similarity(query_sem, catalog_sem)
    if method == "raw":
        return jaccard_similarity(query_flavors, catalog_flavors)
    if method == "hybrid":
        return (
            HYBRID_SEMANTIC_WEIGHT * cosine_similarity(query_sem, catalog_sem)
            + HYBRID_RAW_WEIGHT * jaccard_similarity(query_flavors, catalog_flavors)
        )
    raise ValueError(f"Unknown method '{method}'. Choose 'semantic', 'raw', or 'hybrid'.")


def _rank(results: list[dict], k: int) -> list[dict]:
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results[:k]


# ---------------------------------------------------------------------------
# DB-backed recommendation (primary interface)
# ---------------------------------------------------------------------------

def recommend_from_db(
    build_name: str,
    k: int = 5,
    method: str = "semantic",
    db_path: str | Path = _DEFAULT_DB,
    user_profile=None,
) -> list[dict]:
    """
    Return the top-K most similar builds to build_name using precomputed vectors.

    Args:
        build_name: Exact name of a build present in vector_db.json
                    (e.g. 'caramelizer', 'aftershock').
        k: Number of recommendations to return.
        method: 'semantic' | 'raw' | 'hybrid'
        db_path: Path to vector_db.json (produced by build_index.py).
        user_profile: Optional UserProfile. When provided, modifier suggestions
                      are appended to each result.

    Returns:
        List of dicts sorted by similarity (descending):
        [{'name', 'flavors', 'similarity', 'method', ['modifier_suggestions']}]

    Raises:
        KeyError: If build_name is not found in the DB.
        FileNotFoundError: If vector_db.json does not exist (run build_index.py first).
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {db_path}. Run build_index.py first:\n"
            f"  python -m drink_recommendations.build_index"
        )

    db = _load_db(db_path)
    builds = db["builds"]

    if build_name not in builds:
        available = sorted(builds.keys())
        raise KeyError(
            f"Build '{build_name}' not in vector DB.\n"
            f"Available builds ({len(available)}): {available}"
        )

    query = builds[build_name]
    query_flavors = query["flavors"]
    query_sem = np.array(query["semantic_vec"])

    results = []
    for name, entry in builds.items():
        if name == build_name:
            continue
        cat_sem = np.array(entry["semantic_vec"])
        sim = _compute_similarity(query_flavors, query_sem, entry["flavors"], cat_sem, method)
        results.append({
            "name": name,
            "flavors": entry["flavors"],
            "similarity": round(sim, 4),
            "method": method,
        })

    ranked = _rank(results, k)

    if user_profile is not None:
        suggestions = user_profile.get_modifier_suggestions()
        for r in ranked:
            r["modifier_suggestions"] = suggestions

    return ranked


# ---------------------------------------------------------------------------
# Live recommendation (on-the-fly, for arbitrary flavor lists)
# ---------------------------------------------------------------------------

def recommend(
    query_flavors: list[str],
    k: int = 5,
    method: str = "semantic",
    catalog_path: str | Path = _DEFAULT_CATALOG,
    vectors_path: str | Path = _DEFAULT_VECTORS,
    user_profile=None,
) -> list[dict]:
    """
    Recommend builds for an arbitrary flavor list (not required to be in catalog).

    Vectorizes on-the-fly — use recommend_from_db() when querying catalog builds.
    """
    catalog = load_flavor_catalog(catalog_path)
    semantic_vectors = load_semantic_vectors(vectors_path)

    query_sem = build_semantic_vector(query_flavors, semantic_vectors)
    query_set = set(query_flavors)

    results = []
    for name, flavors in catalog.items():
        if set(flavors) == query_set:
            continue
        cat_sem = build_semantic_vector(flavors, semantic_vectors, warn_missing=False)
        sim = _compute_similarity(query_flavors, query_sem, flavors, cat_sem, method)
        results.append({"name": name, "flavors": flavors, "similarity": round(sim, 4), "method": method})

    ranked = _rank(results, k)

    if user_profile is not None:
        suggestions = user_profile.get_modifier_suggestions()
        for r in ranked:
            r["modifier_suggestions"] = suggestions

    return ranked


def recommend_from_engine(
    token_quantities: dict,
    conn,
    k: int = 5,
    method: str = "semantic",
    db_path: str | Path = _DEFAULT_DB,
    user_profile=None,
) -> list[dict]:
    """
    Run engine pipeline on token_quantities, extract flavor set, then query DB.

    Falls back to live recommend() if the resolved flavor set doesn't match
    any named build in the DB.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from engine import process_order

    result = process_order(token_quantities, conn)
    query_flavors = list(result.get("flavor", {}).keys())

    db_path = Path(db_path)
    if db_path.exists():
        db = _load_db(db_path)
        # Try to find a DB build whose flavor set matches exactly
        for name, entry in db["builds"].items():
            if set(entry["flavors"]) == set(query_flavors):
                return recommend_from_db(name, k=k, method=method, db_path=db_path, user_profile=user_profile)

    # Fall back to live vectorization for custom orders
    return recommend(query_flavors, k=k, method=method, user_profile=user_profile)
