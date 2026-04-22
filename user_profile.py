"""
User profile: maintains a customer's presence in semantic flavor space.

Designed now as an extensibility hook; integrate with real order history
when available. The semantic centroid enables centroid-biased recommendations
and artificial customer profile generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flavor_vectorize import (
    DIMS,
    build_semantic_vector,
    load_semantic_vectors,
)


class UserProfile:
    """Tracks a customer's flavor history and position in semantic space."""

    def __init__(self):
        self.drink_history: list[dict] = []
        # Rolling average of drink semantic vectors (29 dims).
        # None until first order is added.
        self.semantic_centroid: np.ndarray | None = None
        self.modifier_counts: dict[str, int] = {}
        self._n_orders: int = 0

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def add_order(
        self,
        drink_name: str,
        flavors: list[str],
        semantic_vectors: dict[str, np.ndarray],
        modifiers: dict | None = None,
    ) -> None:
        """
        Record a drink order and update the centroid.

        Args:
            drink_name: Canonical build name (e.g. "caramelizer").
            flavors: Flavor ingredient list for this drink.
            semantic_vectors: Loaded from flavor_vectorize.load_semantic_vectors().
            modifiers: Optional dict of modifiers, e.g. {"milk": "oat_milk", "toppings": ["whip_cream"]}.
        """
        entry: dict = {"name": drink_name, "flavors": flavors, "modifiers": modifiers or {}}
        self.drink_history.append(entry)

        vec = build_semantic_vector(flavors, semantic_vectors, warn_missing=False)
        n = self._n_orders
        if self.semantic_centroid is None:
            self.semantic_centroid = vec.copy()
        else:
            # Incremental mean: new_centroid = (n * old + new_vec) / (n + 1)
            self.semantic_centroid = (n * self.semantic_centroid + vec) / (n + 1)
        self._n_orders += 1

        if modifiers:
            milk = modifiers.get("milk")
            if milk:
                self.modifier_counts[milk] = self.modifier_counts.get(milk, 0) + 1
            for topping in modifiers.get("toppings", []):
                self.modifier_counts[topping] = self.modifier_counts.get(topping, 0) + 1

    # ------------------------------------------------------------------
    # Read-only methods
    # ------------------------------------------------------------------

    def get_modifier_suggestions(self, threshold: float = 0.5) -> dict[str, float]:
        """
        Return modifiers whose frequency (count / total_orders) exceeds threshold.
        Returns {modifier_name: frequency}.
        """
        if self._n_orders == 0:
            return {}
        return {
            mod: count / self._n_orders
            for mod, count in self.modifier_counts.items()
            if count / self._n_orders >= threshold
        }

    def dim_labels(self) -> list[str]:
        """Return the semantic dimension names in centroid order."""
        return DIMS.copy()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "drink_history": self.drink_history,
            "semantic_centroid": self.semantic_centroid.tolist() if self.semantic_centroid is not None else None,
            "modifier_counts": self.modifier_counts,
            "n_orders": self._n_orders,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        profile = cls()
        profile.drink_history = data.get("drink_history", [])
        centroid = data.get("semantic_centroid")
        profile.semantic_centroid = np.array(centroid, dtype=float) if centroid is not None else None
        profile.modifier_counts = data.get("modifier_counts", {})
        profile._n_orders = data.get("n_orders", 0)
        return profile

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "UserProfile":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
