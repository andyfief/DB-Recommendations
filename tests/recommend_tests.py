"""
Sanity-check tests for the flavor recommendation system.
Validates that similarity clusters make intuitive sense.
"""

import sys
import pytest
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
CATALOG = ROOT / "rules" / "assignments.json"
VECTORS = ROOT / "drink_recommendations" / "flavor_vectors.csv"
DB = ROOT / "drink_recommendations" / "vector_db.json"

sys.path.insert(0, str(ROOT / "drink_recommendations"))

from recommend import recommend, recommend_from_db
from build_index import build_vector_db
from flavor_catalog import load_flavor_catalog
from flavor_vectorize import (
    load_semantic_vectors,
    build_semantic_vector,
    cosine_similarity,
    jaccard_similarity,
    FLAVOR_ALIAS,
)
from user_profile import UserProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def top_names(results):
    return [r["name"] for r in results]


def _rec(flavors, method="semantic", top_n=10):
    return recommend(flavors, k=top_n, method=method,
                     catalog_path=CATALOG, vectors_path=VECTORS)


# ---------------------------------------------------------------------------
# flavor_vectorize unit tests
# ---------------------------------------------------------------------------

class TestSemanticVectors:
    def setup_method(self):
        self.vecs = load_semantic_vectors(VECTORS)

    def test_loads_expected_flavors(self):
        assert "strawberry" in self.vecs
        assert "dark_chocolate" in self.vecs
        assert "caramel" in self.vecs
        assert "vanilla" in self.vecs

    def test_vector_length(self):
        assert len(self.vecs) > 0, "No vectors loaded"
        n_dims = len(next(iter(self.vecs.values())))
        for name, vec in self.vecs.items():
            assert len(vec) == n_dims, f"{name} has wrong dim {len(vec)}"

    def test_values_in_range(self):
        for name, vec in self.vecs.items():
            assert vec.min() >= 0.0 and vec.max() <= 1.0, f"{name} out of range"

    def test_banana_parsed_correctly(self):
        # Banana row has a space instead of comma between name and first value
        vec = self.vecs["banana"]
        assert vec[0] == pytest.approx(0.8)  # sweet

    def test_dark_chocolate_more_chocolatey_than_strawberry(self):
        from drink_recommendations.flavor_vectorize import DIMS
        choc_idx = DIMS.index("chocolatey")
        assert self.vecs["dark_chocolate"][choc_idx] > self.vecs["strawberry"][choc_idx]

    def test_sf_alias_map_covers_all_sf_flavors(self):
        """Every SF flavor in the catalog should have an alias entry."""
        import json
        with open(CATALOG) as f:
            rules = [r for r in json.load(f) if r.get("payload", {}).get("role") == "flavor"]
        sf_flavors = set()
        for r in rules:
            for item in r["payload"]["items"]:
                name = item if isinstance(item, str) else item["name"]
                if "sugar_free" in name or name == "sf_vanilla":
                    sf_flavors.add(name)
        missing = sf_flavors - set(FLAVOR_ALIAS.keys())
        assert not missing, f"SF flavors missing from FLAVOR_ALIAS: {missing}"


class TestBuildSemanticVector:
    def setup_method(self):
        self.vecs = load_semantic_vectors(VECTORS)

    def test_single_flavor_equals_lookup(self):
        v = build_semantic_vector(["strawberry"], self.vecs)
        import numpy as np
        assert (v == self.vecs["strawberry"]).all()

    def test_empty_flavors_returns_zero(self):
        import numpy as np
        v = build_semantic_vector([], self.vecs)
        assert (v == 0).all()

    def test_sf_flavor_maps_to_base(self):
        sf_v = build_semantic_vector(["sugar_free_caramel"], self.vecs, warn_missing=False)
        base_v = build_semantic_vector(["caramel"], self.vecs)
        assert cosine_similarity(sf_v, base_v) == pytest.approx(1.0)

    def test_average_of_two(self):
        import numpy as np
        v = build_semantic_vector(["caramel", "vanilla"], self.vecs)
        expected = (self.vecs["caramel"] + self.vecs["vanilla"]) / 2
        assert v == pytest.approx(expected)


class TestSimilarityFunctions:
    def test_cosine_identical(self):
        import numpy as np
        a = np.array([1.0, 0.5, 0.2])
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_zero_vector(self):
        import numpy as np
        a = np.zeros(5)
        b = np.array([1.0, 0.5, 0.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_jaccard_identical(self):
        assert jaccard_similarity(["a", "b"], ["a", "b"]) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        assert jaccard_similarity(["a"], ["b"]) == pytest.approx(0.0)

    def test_jaccard_partial(self):
        assert jaccard_similarity(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# flavor_catalog tests
# ---------------------------------------------------------------------------

class TestFlavorCatalog:
    def setup_method(self):
        self.catalog = load_flavor_catalog(CATALOG)

    def test_catalog_not_empty(self):
        assert len(self.catalog) > 50

    def test_known_builds_present(self):
        assert "caramelizer" in self.catalog
        assert "aftershock" in self.catalog
        assert "trifecta" in self.catalog

    def test_caramelizer_has_caramel(self):
        assert "caramel" in self.catalog["caramelizer"]

    def test_aftershock_multi_flavor(self):
        flavors = self.catalog["aftershock"]
        assert len(flavors) >= 3

    def test_trifecta_has_chocolate_and_caramel(self):
        flavors = self.catalog["trifecta"]
        assert "dark_chocolate" in flavors or "chocolate" in flavors
        assert "caramel" in flavors


# ---------------------------------------------------------------------------
# Recommendation cluster tests
# ---------------------------------------------------------------------------

class TestRecommendClusters:
    """Validates that semantically similar drinks cluster together."""

    # -- Chocolate cluster --

    def test_caramelizer_returns_results(self):
        results = _rec(["caramel"])
        assert len(results) > 0

    def test_chocolate_drinks_cluster_semantic(self):
        # Query: dark_chocolate → should surface other chocolate-forward drinks
        results = _rec(["dark_chocolate"], method="semantic", top_n=10)
        names = top_names(results)
        chocolate_builds = {"trifecta", "double_torture", "black_forest", "caramelizer"}
        overlap = chocolate_builds & set(names)
        assert len(overlap) >= 1, f"Expected chocolate builds in top 10, got: {names}"

    def test_multi_choc_query_cluster(self):
        results = _rec(["dark_chocolate", "caramel"], method="semantic", top_n=10)
        names = top_names(results)
        # trifecta = dark_chocolate + white_chocolate + caramel — should be near top
        assert "trifecta" in names, f"trifecta not in top 10: {names}"

    # -- Fruit cluster --

    def test_berry_drinks_cluster(self):
        results = _rec(["strawberry", "red_raspberry"], method="semantic", top_n=10)
        names = top_names(results)
        # aftershock contains red_raspberry + strawberry + lime + blackberry
        assert "aftershock" in names, f"aftershock not in top 10: {names}"

    def test_tropical_drinks_cluster(self):
        results = _rec(["mango", "pineapple"], method="semantic", top_n=10)
        names = top_names(results)
        # Should surface other tropical builds
        assert len(names) > 0

    # -- Raw Jaccard cluster --

    def test_exact_flavor_match_raw(self):
        # caramelizer = [caramel]; query [caramel, dark_chocolate]
        # raw should favor drinks sharing caramel
        results = _rec(["caramel"], method="raw", top_n=10)
        names = top_names(results)
        # Any drink with caramel should appear
        catalog = load_flavor_catalog(CATALOG)
        caramel_builds = {n for n, fl in catalog.items() if "caramel" in fl and n != "caramelizer"}
        overlap = caramel_builds & set(names)
        assert len(overlap) >= 1, f"No caramel builds in top-10 raw results: {names}"

    def test_sf_raw_isolation(self):
        # SF strawberry should NOT match regular strawberry in raw Jaccard
        results = _rec(["sugar_free_strawberry"], method="raw", top_n=5)
        names = top_names(results)
        catalog = load_flavor_catalog(CATALOG)
        # Builds using regular strawberry (not SF) should not be top match
        for name in names[:2]:
            flavors = catalog.get(name, [])
            assert "sugar_free_strawberry" in flavors or "strawberry" not in flavors, \
                f"Raw Jaccard matched SF query to non-SF build '{name}' with flavors {flavors}"

    def test_sf_semantic_proximity(self):
        # SF caramel should be semantically close to regular caramel drinks
        sf_results = _rec(["sugar_free_caramel"], method="semantic", top_n=5)
        reg_results = _rec(["caramel"], method="semantic", top_n=5)
        sf_names = set(top_names(sf_results))
        reg_names = set(top_names(reg_results))
        overlap = sf_names & reg_names
        assert len(overlap) >= 2, f"SF and regular caramel top-5 barely overlap: {sf_names} vs {reg_names}"

    # -- Hybrid --

    def test_hybrid_returns_results(self):
        results = _rec(["vanilla", "caramel"], method="hybrid", top_n=5)
        assert len(results) == 5

    def test_hybrid_scores_between_0_and_1(self):
        results = _rec(["strawberry"], method="hybrid", top_n=10)
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0, f"Score out of range: {r}"

    # -- Custom order (non-named build) --

    def test_custom_order_matches_named_build(self):
        # vanilla + caramel → should match golden_eagle if it uses those flavors
        results = _rec(["vanilla", "caramel"], method="raw", top_n=5)
        names = top_names(results)
        catalog = load_flavor_catalog(CATALOG)
        # At least one result should contain both vanilla and caramel
        for name in names:
            flavors = set(catalog.get(name, []))
            if "vanilla" in flavors and "caramel" in flavors:
                return  # pass
        # Or at least contain one of them
        for name in names:
            flavors = set(catalog.get(name, []))
            if "vanilla" in flavors or "caramel" in flavors:
                return
        pytest.fail(f"No vanilla/caramel builds in results: {names}")

    # -- Exact match exclusion --

    def test_exact_match_excluded(self):
        # Query exactly matching a catalog entry should not appear in results
        catalog = load_flavor_catalog(CATALOG)
        caramelizer_flavors = catalog["caramelizer"]
        results = _rec(caramelizer_flavors, method="raw", top_n=20)
        names = top_names(results)
        assert "caramelizer" not in names

    # -- Result structure --

    def test_result_structure(self):
        results = _rec(["vanilla"])
        assert len(results) > 0
        r = results[0]
        assert "name" in r
        assert "flavors" in r
        assert "similarity" in r
        assert "method" in r

    def test_results_sorted_descending(self):
        results = _rec(["dark_chocolate", "caramel"], top_n=10)
        scores = [r["similarity"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# UserProfile tests
# ---------------------------------------------------------------------------

class TestUserProfile:
    def setup_method(self):
        self.vecs = load_semantic_vectors(VECTORS)

    def test_empty_profile(self):
        p = UserProfile()
        assert p.semantic_centroid is None
        assert p._n_orders == 0
        assert p.get_modifier_suggestions() == {}

    def test_add_single_order_sets_centroid(self):
        import numpy as np
        p = UserProfile()
        p.add_order("caramelizer", ["caramel"], self.vecs)
        expected = build_semantic_vector(["caramel"], self.vecs)
        assert p.semantic_centroid == pytest.approx(expected)
        assert p._n_orders == 1

    def test_add_two_orders_centroid_is_mean(self):
        import numpy as np
        p = UserProfile()
        p.add_order("caramelizer", ["caramel"], self.vecs)
        p.add_order("vanilla_latte", ["vanilla"], self.vecs)
        v_caramel = build_semantic_vector(["caramel"], self.vecs)
        v_vanilla = build_semantic_vector(["vanilla"], self.vecs)
        expected = (v_caramel + v_vanilla) / 2
        assert p.semantic_centroid == pytest.approx(expected)

    def test_modifier_counts_tracked(self):
        p = UserProfile()
        p.add_order("x", ["vanilla"], self.vecs, modifiers={"milk": "oat_milk", "toppings": ["whip_cream"]})
        p.add_order("y", ["caramel"], self.vecs, modifiers={"milk": "oat_milk"})
        assert p.modifier_counts["oat_milk"] == 2
        assert p.modifier_counts["whip_cream"] == 1

    def test_modifier_suggestions_threshold(self):
        p = UserProfile()
        p.add_order("a", ["vanilla"], self.vecs, modifiers={"milk": "oat_milk"})
        p.add_order("b", ["caramel"], self.vecs, modifiers={"milk": "oat_milk"})
        p.add_order("c", ["strawberry"], self.vecs, modifiers={"toppings": ["whip_cream"]})
        suggestions = p.get_modifier_suggestions(threshold=0.5)
        assert "oat_milk" in suggestions  # 2/3 ≈ 0.67 ≥ 0.5
        assert "whip_cream" not in suggestions  # 1/3 ≈ 0.33 < 0.5

    def test_profile_shows_in_recommendations(self):
        p = UserProfile()
        p.add_order("a", ["vanilla"], self.vecs, modifiers={"milk": "oat_milk"})
        p.add_order("b", ["vanilla"], self.vecs, modifiers={"milk": "oat_milk"})
        results = recommend(
            ["vanilla"], k=3, method="semantic",
            user_profile=p, catalog_path=CATALOG, vectors_path=VECTORS
        )
        for r in results:
            assert "modifier_suggestions" in r
            assert "oat_milk" in r["modifier_suggestions"]

    def test_serialization_roundtrip(self, tmp_path):
        p = UserProfile()
        p.add_order("caramelizer", ["caramel"], self.vecs, modifiers={"milk": "oat_milk"})
        path = tmp_path / "profile.json"
        p.save(path)
        p2 = UserProfile.load(path)
        assert p2._n_orders == 1
        assert p2.semantic_centroid == pytest.approx(p.semantic_centroid)
        assert p2.modifier_counts == p.modifier_counts


# ---------------------------------------------------------------------------
# Vector DB pipeline tests
# ---------------------------------------------------------------------------

class TestVectorDB:
    """Tests the full build_index → recommend_from_db pipeline."""

    @pytest.fixture(scope="class")
    def fresh_db(self, tmp_path_factory):
        """Build a fresh vector DB in a temp dir for isolation."""
        tmp = tmp_path_factory.mktemp("db")
        db_path = tmp / "vector_db.json"
        build_vector_db(
            assignments_path=CATALOG,
            vectors_csv_path=VECTORS,
            db_path=db_path,
        )
        return db_path

    def test_db_file_created(self, fresh_db):
        assert fresh_db.exists()

    def test_db_structure(self, fresh_db):
        import json
        with open(fresh_db) as f:
            db = json.load(f)
        assert "_meta" in db
        assert "builds" in db
        meta = db["_meta"]
        assert meta["n_builds"] > 50
        assert meta["n_semantic_dims"] == 28
        assert len(meta["semantic_dim_labels"]) == 28
        assert meta["n_raw_dims"] > 0
        assert len(meta["raw_flavor_index"]) == meta["n_raw_dims"]

    def test_known_builds_indexed(self, fresh_db):
        import json
        with open(fresh_db) as f:
            builds = json.load(f)["builds"]
        assert "caramelizer" in builds
        assert "aftershock" in builds
        assert "trifecta" in builds

    def test_build_entry_structure(self, fresh_db):
        import json
        with open(fresh_db) as f:
            db = json.load(f)
        entry = db["builds"]["caramelizer"]
        assert "flavors" in entry
        assert "semantic_vec" in entry
        assert "raw_vec" in entry
        assert len(entry["semantic_vec"]) == db["_meta"]["n_semantic_dims"]
        assert len(entry["raw_vec"]) == db["_meta"]["n_raw_dims"]

    def test_recommend_from_db_returns_k(self, fresh_db):
        results = recommend_from_db("caramelizer", k=5, db_path=fresh_db)
        assert len(results) == 5

    def test_recommend_from_db_excludes_self(self, fresh_db):
        results = recommend_from_db("caramelizer", k=20, db_path=fresh_db)
        names = [r["name"] for r in results]
        assert "caramelizer" not in names

    def test_recommend_from_db_sorted(self, fresh_db):
        results = recommend_from_db("trifecta", k=10, db_path=fresh_db)
        scores = [r["similarity"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_from_db_result_structure(self, fresh_db):
        results = recommend_from_db("aftershock", k=3, db_path=fresh_db)
        for r in results:
            assert "name" in r
            assert "flavors" in r
            assert "similarity" in r
            assert "method" in r
            assert 0.0 <= r["similarity"] <= 1.0

    def test_chocolate_cluster_db(self, fresh_db):
        results = recommend_from_db("caramelizer", k=10, method="semantic", db_path=fresh_db)
        names = [r["name"] for r in results]
        # caramelizer = [caramel] → should find other caramel-heavy builds
        catalog = load_flavor_catalog(CATALOG)
        caramel_builds = {n for n, fl in catalog.items() if "caramel" in fl and n != "caramelizer"}
        assert len(caramel_builds & set(names)) >= 2, f"Expected caramel builds in top 10: {names}"

    def test_berry_cluster_db(self, fresh_db):
        results = recommend_from_db("aftershock", k=10, method="semantic", db_path=fresh_db)
        names = [r["name"] for r in results]
        # aftershock = [red_raspberry, strawberry, lime, blackberry]
        # berry/fruit builds should dominate top 10
        catalog = load_flavor_catalog(CATALOG)
        berry_builds = {
            n for n, fl in catalog.items()
            if any(f in fl for f in ["strawberry", "red_raspberry", "blackberry", "raspberry"])
            and n != "aftershock"
        }
        overlap = berry_builds & set(names)
        assert len(overlap) >= 3, f"Expected berry builds in top 10: {names}"

    def test_all_methods_work_db(self, fresh_db):
        for method in ("semantic", "raw", "hybrid"):
            results = recommend_from_db("trifecta", k=5, method=method, db_path=fresh_db)
            assert len(results) == 5, f"method={method} returned {len(results)} results"

    def test_missing_build_raises_key_error(self, fresh_db):
        with pytest.raises(KeyError, match="not in vector DB"):
            recommend_from_db("nonexistent_drink_xyz", db_path=fresh_db)

    def test_missing_db_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_index"):
            recommend_from_db("caramelizer", db_path=tmp_path / "no_db.json")

    def test_hybrid_scores_bounded(self, fresh_db):
        results = recommend_from_db("caramelizer", k=10, method="hybrid", db_path=fresh_db)
        for r in results:
            assert 0.0 <= r["similarity"] <= 1.0
