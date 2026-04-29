import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Path to vector_db.json — defaults to the project root (sibling of backend/)
_HERE = Path(__file__).resolve().parent
DRINK_DB_PATH: Path = Path(
    os.environ.get("DRINK_DB_PATH", str(_HERE.parent / "vector_db.json"))
)
