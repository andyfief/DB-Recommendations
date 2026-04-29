"""
Step 1: Intent extraction.

Calls OpenAI to identify the drink name, matched modifier tokens, and whether
the user wants similar drinks or a modification. The LLM returns a tokens dict
(e.g. {"caramelizer": 1, "oat_milk": 1}) that is passed directly to process_order,
exactly as DrinkBuilder test inputs are structured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import OpenAI

from backend.config import DRINK_DB_PATH, OPENAI_API_KEY, OPENAI_MODEL
from backend.models import DrinkObject, IntentResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intent_extraction.txt"

_client = OpenAI(api_key=OPENAI_API_KEY)


class DrinkNotFoundError(Exception):
    pass


class IntentParseError(Exception):
    pass


def _load_drink_list() -> list[str]:
    with open(DRINK_DB_PATH, encoding="utf-8") as f:
        db = json.load(f)
    return sorted(db["builds"].keys())


def _load_token_list() -> list[str]:
    from backend.lib.get_token_maps import KNOWN_TOKENS
    return sorted(KNOWN_TOKENS)


def _call_llm(user_input: str, drink_list: list[str], token_list: list[str]) -> str:
    prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")
    system_prompt = prompt_template.replace(
        "{drink_list}", "\n".join(f"- {n}" for n in drink_list)
    ).replace(
        "{token_list}", "\n".join(f"- {t}" for t in token_list)
    )

    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return response.choices[0].message.content


def _parse_intent(raw: str) -> IntentResult:
    try:
        data = json.loads(raw)
        return IntentResult(
            drink_name=data.get("drink_name"),
            tokens={k: int(v) for k, v in data.get("tokens", {}).items()},
            method=data.get("method"),
            confidence=float(data.get("confidence", 0.0)),
            error=data.get("error"),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise IntentParseError(f"Could not parse intent JSON: {exc}") from exc


def _run_engine(tokens: dict[str, int]) -> tuple[DrinkObject, dict]:
    """Run process_order and return (DrinkObject, raw_engine_result)."""
    from backend.lib.db import open_connection
    from backend.lib.engine import process_order

    with open_connection() as conn:
        raw = process_order(tokens, conn)
    return DrinkObject.from_engine_result(raw), raw


async def extract_intent(user_input: str) -> tuple[IntentResult, DrinkObject, dict]:
    """
    Returns (IntentResult, DrinkObject, raw_engine_result).
    raw_engine_result contains fired_rules for audit logging.
    Raises DrinkNotFoundError if the drink cannot be matched.
    Raises IntentParseError if the LLM response cannot be parsed after retry.
    """
    drink_list = _load_drink_list()
    token_list = _load_token_list()
    raw = _call_llm(user_input, drink_list, token_list)

    try:
        intent = _parse_intent(raw)
    except IntentParseError:
        raw = _call_llm(user_input, drink_list, token_list)
        intent = _parse_intent(raw)

    if intent.error == "DRINK_NOT_FOUND" or intent.drink_name is None:
        raise DrinkNotFoundError(f"Could not match drink in: {user_input!r}")

    drink_obj, engine_result = _run_engine(intent.tokens)
    return intent, drink_obj, engine_result
