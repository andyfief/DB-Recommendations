import csv
import warnings
from pathlib import Path

import numpy as np

DIMS = [
    "sweet", "fruity", "citrus", "berry", "tropical", "stone_fruit",
    "tart", "sour", "bitter", "salty", "nutty", "creamy", "chocolatey",
    "floral", "herbal", "minty", "spicy", "caramel", "candy_like",
    "dessert_like", "freshness", "richness",
    "red", "blue", "green", "yellow_orange", "purple", "white_brown",
]

# Maps non-standard or aliased ingredient names to their CSV flavor name.
# SF variants strip "sugar_free_" prefix; smoothie mixes map to base flavor.
FLAVOR_ALIAS: dict[str, str] = {
    "sugar_free_caramel": "caramel",
    "sugar_free_chocolate": "chocolate",
    "sugar_free_chocolate_macadamia_nut": "chocolate_macadamia_nut",
    "sugar_free_coconut": "coconut",
    "sugar_free_hazelnut": "hazelnut",
    "sugar_free_irish_cream": "irish_cream",
    "sugar_free_peach": "peach",
    "sugar_free_peppermint": "peppermint",
    "sugar_free_pumpkin": "pumpkin",
    "sugar_free_raspberry": "red_raspberry",
    "sugar_free_salted_caramel": "salted_caramel",
    "sugar_free_strawberry": "strawberry",
    "sugar_free_vanilla": "vanilla",
    "sugar_free_white_chocolate": "white_chocolate",
    "sf_vanilla": "vanilla",
    "blue_raz": "blue_raspberry",
    "mango_smoothie_mix": "mango",
    "strawberry_smoothie_mix": "strawberry",
    "apple_smoothie_mix": "green_apple",
    "lemon_concentrate": "lemon",
    "almond": "almond_orgeat",
}

_ZERO = np.zeros(len(DIMS), dtype=float)


def load_semantic_vectors(csv_path: str | Path) -> dict[str, np.ndarray]:
    """
    Parse flavor_vectors.csv and return {flavor_name: 29-dim numpy array}.

    Handles the malformed banana row where name and first value are space-separated
    instead of comma-separated.
    """
    vectors: dict[str, np.ndarray] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # header: ['name', ' sweet', ' fruity', ...] — strip whitespace
        col_names = [h.strip() for h in header]
        value_cols = col_names[1:]  # should match DIMS order

        n_dims = len(value_cols)

        for row in reader:
            if not row:
                continue
            # Some rows may have name and first value joined by a space (e.g. "banana 0.8")
            raw_name = row[0].strip()
            if " " in raw_name:
                parts = raw_name.split(" ", 1)
                name = parts[0]
                first_val = parts[1]
                values = [first_val] + [v.strip() for v in row[1:]]
            else:
                name = raw_name
                values = [v.strip() for v in row[1:]]

            if len(values) < n_dims:
                warnings.warn(f"flavor_vectors.csv: row '{name}' has only {len(values)} values, expected {n_dims} — skipping")
                continue
            # Truncate any trailing extra columns silently
            values = values[:n_dims]

            vectors[name] = np.array([float(v) for v in values], dtype=float)

    return vectors


def _resolve(flavor: str, semantic_vectors: dict[str, np.ndarray]) -> np.ndarray | None:
    """Return the semantic vector for a flavor, resolving aliases. None if unknown."""
    if flavor in semantic_vectors:
        return semantic_vectors[flavor]
    alias = FLAVOR_ALIAS.get(flavor)
    if alias and alias in semantic_vectors:
        return semantic_vectors[alias]
    return None


def build_semantic_vector(
    flavors: list[str],
    semantic_vectors: dict[str, np.ndarray],
    warn_missing: bool = True,
) -> np.ndarray:
    """
    Average semantic vectors for a flavor list.
    Flavors absent from the CSV (after alias resolution) contribute a zero vector
    and emit a warning.
    """
    if not flavors:
        return _ZERO.copy()

    vecs = []
    for fl in flavors:
        v = _resolve(fl, semantic_vectors)
        if v is None:
            if warn_missing:
                warnings.warn(f"No semantic vector for flavor '{fl}' — contributing zero")
            v = _ZERO
        vecs.append(v)

    return np.mean(vecs, axis=0)


def build_raw_vector(flavors: list[str], all_flavors: list[str]) -> np.ndarray:
    """
    Binary presence vector over the full flavor vocabulary (sorted list all_flavors).
    1.0 where flavor is present, 0.0 elsewhere.
    """
    flavor_set = set(flavors)
    return np.array([1.0 if f in flavor_set else 0.0 for f in all_flavors], dtype=float)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def jaccard_similarity(flavors_a: list[str], flavors_b: list[str]) -> float:
    """Jaccard similarity on flavor sets."""
    set_a, set_b = set(flavors_a), set(flavors_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
