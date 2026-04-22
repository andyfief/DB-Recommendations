"""
Drink recommendation query interface.

Run from the drink_recommendations/ directory:
    python query.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from recommend import recommend_from_db

_HERE = Path(__file__).parent
_DEFAULT_DB = _HERE / "vector_db.json"

METHODS = ("semantic", "raw", "hybrid")


def _prompt(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value if value else default


def _load_build_names(db_path: Path) -> list[str]:
    with open(db_path, encoding="utf-8") as f:
        return sorted(json.load(f)["builds"].keys())


def main() -> None:
    if not _DEFAULT_DB.exists():
        print("vector_db.json not found. Run build_index.py first.")
        sys.exit(1)

    build_names = _load_build_names(_DEFAULT_DB)
    print(f"\n{len(build_names)} builds available. Examples: {', '.join(build_names[:6])}, ...")

    build = ""
    while build not in build_names:
        build = input("\nBuild name: ").strip()
        if build not in build_names:
            print(f"  '{build}' not found. Check spelling or run build_index.py to refresh.")

    k_raw = _prompt("Number of recommendations", "5")
    try:
        k = int(k_raw)
    except ValueError:
        k = 5

    method_raw = _prompt("Method (semantic / raw / hybrid)", "semantic")
    method = method_raw if method_raw in METHODS else "semantic"

    results = recommend_from_db(build, k=k, method=method)

    print(f"\nTop {len(results)} recommendations for '{build}' [{method}]")
    print("-" * 52)
    for i, r in enumerate(results, 1):
        flavors = ", ".join(r["flavors"])
        print(f"  {i:2}. {r['name']:<30} {r['similarity']:.4f}  [{flavors}]")
    print()


if __name__ == "__main__":
    main()
