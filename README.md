# Drink Recommendations

A flavor-based recommendation system for Dutch Bros drink builds. Given a drink name, it returns the K most similar builds based on their flavor compositions.

Similarity is computed purely on **flavor ingredients** — milk, base, toppings, and size are ignored. This gives the clearest semantic signal for taste-based matching.

---

## How It Works

Every named drink build maps to a list of flavor ingredients (e.g. `caramelizer → [caramel]`, `trifecta → [dark_chocolate, white_chocolate, caramel]`). These flavor lists are sourced directly from `rules/assignments.json`.

Each build gets two vector representations:

- **Semantic vector (28 dims)** — aggregated from `flavor_vectors.csv`, where each ingredient has a hand-scored profile across taste/sensory dimensions (sweet, bitter, fruity, chocolatey, richness, etc.). Similarity = cosine distance. Captures *how a drink tastes*, even across different ingredient names.
- **Raw vector (36 dims)** — binary presence over the full flavor vocabulary. Similarity = Jaccard overlap. Captures *exactly which ingredients* two drinks share.

Both are precomputed and stored in `vector_db.json` at build time.

---

## Pipeline

### Step 1 — Build the index

Run once, or whenever `assignments.json`, `flavor_vectors.csv`, or `flavors.csv` changes:

```
python -m drink_recommendations.build_index
```

This reads `flavors.csv` as a whitelist, matches each token to its flavor assignment in `assignments.json`, vectorizes every matched build, and writes `vector_db.json`.

Output:
```
Filter list loaded: 159 tokens
  Tokens with no flavor assignment (17):
    - birthday_cake
    - mudslide
    ...

Vector DB written to drink_recommendations/vector_db.json
  Builds indexed : 142
  Semantic dims  : 28
  Raw flavor dims: 36
  Flavor ingredients without semantic vector (1): cinnamon
```

Tokens listed under "no flavor assignment" are in `flavors.csv` but have no matching rule in `assignments.json` — they are skipped.

### Step 2 — Query

```
python -m drink_recommendations.query <build_name> [--k K] [--method METHOD]
```

`build_name` must be a build present in `vector_db.json` (i.e. it appeared in `flavors.csv` and had a flavor assignment).

---

## CLI Usage

### Basic query

```
python -m drink_recommendations.query caramelizer
```
```
Top 5 recommendations for 'caramelizer' [semantic]
----------------------------------------------------
   1. caramel                        1.0000  [caramel]
   2. golden_eagle                   0.9512  [caramel, vanilla]
   3. french_vanilla_bean            0.9512  [vanilla, caramel]
   4. horchata                       0.9462  [caramel, white_chocolate, cinnamon]
   5. salted_caramel                 0.9377  [salted_caramel]
```

### More results

```
python -m drink_recommendations.query aftershock --k 10
```

### Change similarity method

```
python -m drink_recommendations.query trifecta --method raw
python -m drink_recommendations.query caramelizer --method hybrid
```

### JSON output (for programmatic use)

```
python -m drink_recommendations.query caramelizer --k 3 --json
```
```json
[
  {"name": "caramel", "flavors": ["caramel"], "similarity": 1.0, "method": "semantic"},
  {"name": "golden_eagle", "flavors": ["caramel", "vanilla"], "similarity": 0.9512, "method": "semantic"},
  ...
]
```

### All options

```
python -m drink_recommendations.query --help
```

| Flag | Default | Description |
|---|---|---|
| `build` | (required) | Build name to query |
| `--k` | 5 | Number of recommendations |
| `--method` | semantic | `semantic`, `raw`, or `hybrid` |
| `--db` | vector_db.json | Path to a different DB file |
| `--json` | off | Output raw JSON |

---

## Similarity Methods

| Method | How it works | Best for |
|---|---|---|
| `semantic` | Cosine similarity on 28-dim aggregated taste vectors | "Tastes similar" — works across different ingredient names |
| `raw` | Jaccard similarity on exact flavor ingredient sets | "Same ingredients" — strict overlap matching |
| `hybrid` | 60% semantic + 40% raw | Balance of taste feel and ingredient overlap |

**When to use which:**
- `semantic` — default, best for customer-facing recommendations. A `sugar_free_caramel` drink will surface near `caramel` drinks because they map to the same sensory profile.
- `raw` — useful for finding builds that literally share ingredients. SF and non-SF variants will *not* match each other.
- `hybrid` — useful when you want ingredient overlap to anchor the result while still allowing taste-based reach.

---

## Python API

### Recommend from the precomputed DB (preferred)

```python
from drink_recommendations.recommend import recommend_from_db

results = recommend_from_db("caramelizer", k=5, method="semantic")
for r in results:
    print(r["name"], r["similarity"], r["flavors"])
```

### Recommend from an arbitrary flavor list (live, no DB required)

```python
from drink_recommendations.recommend import recommend

results = recommend(["caramel", "dark_chocolate"], k=5, method="semantic")
```

Useful for custom orders that don't map to a named build. Vectorizes on-the-fly against the full catalog.

### Recommend from raw engine tokens

```python
from drink_recommendations.recommend import recommend_from_engine
import sqlite3

conn = sqlite3.connect("schema/rules.db")
results = recommend_from_engine(
    {"medium": 1, "iced": 1, "caramelizer": 1},
    conn,
    k=5,
    method="semantic"
)
```

Runs the full engine pipeline on the token dict, extracts the resolved flavor set, then queries the DB (or falls back to live vectorization for custom orders).

### With a user profile (modifier suggestions)

```python
from drink_recommendations.recommend import recommend_from_db
from drink_recommendations.user_profile import UserProfile
from drink_recommendations.flavor_vectorize import load_semantic_vectors

vecs = load_semantic_vectors("drink_recommendations/flavor_vectors.csv")

profile = UserProfile()
profile.add_order("caramelizer", ["caramel"], vecs, modifiers={"milk": "oat_milk"})
profile.add_order("golden_eagle", ["caramel", "vanilla"], vecs, modifiers={"milk": "oat_milk", "toppings": ["whip_cream"]})

results = recommend_from_db("trifecta", k=5, user_profile=profile)
for r in results:
    print(r["name"], r.get("modifier_suggestions"))
# → {"oat_milk": 1.0, "whip_cream": 0.5}
```

---

## Files

```
drink_recommendations/
│
├── flavors.csv             Whitelist of build names to index. First column is
│                           the token name (header row skipped). Tokens must match
│                           triggers in assignments.json to be included in the DB.
│
├── flavor_vectors.csv      Semantic profiles for individual flavor ingredients.
│                           35 flavors × 28 sensory dimensions (0–1 normalized).
│                           Dimensions: sweet, fruity, citrus, berry, tropical,
│                           stone_fruit, tart, sour, bitter, salty, nutty, creamy,
│                           chocolatey, floral, herbal, minty, spicy, caramel,
│                           candy_like, dessert_like, freshness, richness,
│                           red, blue, green, yellow_orange, purple, white_brown.
│
├── vector_db.json          Generated — do not edit by hand. Precomputed semantic
│                           and raw vectors for all indexed builds, plus metadata.
│                           Rebuilt by running build_index.py.
│
├── build_index.py          Pipeline step 1. Reads flavors.csv + assignments.json
│                           + flavor_vectors.csv, vectorizes all matched builds,
│                           writes vector_db.json. Run this to refresh the DB.
│
├── query.py                CLI interface. Reads vector_db.json and returns ranked
│                           recommendations for a given build name.
│
├── recommend.py            Core recommendation logic. Three entry points:
│                             recommend_from_db()   — query precomputed DB
│                             recommend()           — live, arbitrary flavor list
│                             recommend_from_engine() — full engine pipeline input
│
├── flavor_catalog.py       Parses assignments.json into {build: [flavors]} dict.
│                           Supports optional whitelist filtering via flavors.csv.
│
├── flavor_vectorize.py     Vector math utilities:
│                             load_semantic_vectors() — parse flavor_vectors.csv
│                             build_semantic_vector() — aggregate flavors → 28-dim
│                             build_raw_vector()      — binary flavor presence vec
│                             cosine_similarity()     — for semantic method
│                             jaccard_similarity()    — for raw method
│                             FLAVOR_ALIAS            — maps SF variants and
│                                                       smoothie mixes to their
│                                                       base flavor for semantic lookup
│
├── user_profile.py         UserProfile class. Tracks order history, maintains a
│                           rolling semantic centroid (average of all ordered drink
│                           vectors), and counts modifier frequency. Supports
│                           JSON serialization for persistence. Designed as an
│                           extensibility hook for future customer profile work.
│
└── tests/
    └── recommend_tests.py  56 tests covering: CSV parsing, vector math, catalog
                            loading, cluster sanity checks (chocolate/berry/tropical
                            families), SF flavor handling, DB pipeline end-to-end,
                            UserProfile centroid and serialization.
```

---

## Data Sources

| File | Sourced from | Purpose |
|---|---|---|
| `flavor_vectors.csv` | Hand-scored (ChatGPT-assisted, manually reviewed) | Semantic taste dimensions per ingredient |
| `flavors.csv` | Real Dutch Bros order data / menu | Whitelist of builds to index |
| `rules/assignments.json` | Rule engine (parent project) | Maps build names → flavor ingredient lists |

---

## Extending with Customer Profiles

`UserProfile` is designed as a foundation for future work. Each profile maintains:

- **Semantic centroid** — a 28-dim vector representing the customer's average taste position. Updated incrementally per order: `new_centroid = (n × old + new_vec) / (n + 1)`.
- **Modifier counts** — frequency of each modifier (milk type, toppings) across all orders.
- **`get_modifier_suggestions(threshold)`** — returns modifiers used in ≥ `threshold` fraction of orders.

Profiles serialize to/from JSON:

```python
profile.save("customer_123.json")
profile = UserProfile.load("customer_123.json")
```

The planned next step is building artificial customer profiles from multiple orders and using the centroid to bias recommendations toward a customer's established taste region.
