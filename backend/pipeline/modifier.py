"""
Step 2b: Modification path.

Calls OpenAI with the original drink object and the user's change request.
Returns a modified DrinkObject and a plain-English description of what changed.
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL
from backend.models import DrinkObject, ModifierResult

_client = OpenAI(api_key=OPENAI_API_KEY)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "modifier_system.txt"
_KNOWLEDGE_PATH = _PROMPTS_DIR / "drink_knowledge.txt"


class ModifierParseError(Exception):
    pass


def _build_system_prompt() -> str:
    system_template = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    knowledge = _KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return system_template.replace("{drink_knowledge}", knowledge)


def _call_llm(system_prompt: str, original_drink: DrinkObject, user_input: str) -> str:
    user_message = (
        f"Original drink:\n{original_drink.model_dump_json(indent=2)}\n\n"
        f"Customer request: {user_input}"
    )

    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return response.choices[0].message.content


def _parse_result(raw: str) -> ModifierResult:
    try:
        data = json.loads(raw)
        mod_obj = data["modified_drink_object"]
        drink = DrinkObject(
            flavors=mod_obj.get("flavors", {}),
            toppings=mod_obj.get("toppings", {}),
            milk=mod_obj.get("milk", []),
            base=mod_obj.get("base", []),
            coffee=mod_obj.get("coffee", []),
            shots=mod_obj.get("shots"),
            scoops=mod_obj.get("scoops"),
        )
        return ModifierResult(
            modified_drink=drink,
            change_description=data["change_description"],
            llm_raw=raw,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        raise ModifierParseError(f"Could not parse modifier JSON: {exc}") from exc


async def modify_drink(original_drink: DrinkObject, user_input: str) -> ModifierResult:
    """
    Apply user-requested modifications to original_drink.
    Raises ModifierParseError if LLM response cannot be parsed after retry.
    """
    system_prompt = _build_system_prompt()

    raw = _call_llm(system_prompt, original_drink, user_input)
    try:
        return _parse_result(raw)
    except ModifierParseError:
        raw = _call_llm(system_prompt, original_drink, user_input)
        return _parse_result(raw)
