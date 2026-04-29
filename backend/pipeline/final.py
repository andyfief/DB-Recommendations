"""
Step 4: Final LLM response.

Generates the user-facing natural language recommendation from the combined
pipeline context (similarity or modification path).
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL
from backend.models import FinalLLMInput

_client = OpenAI(api_key=OPENAI_API_KEY)
_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "final_system.txt"


def _build_user_message(final_input: FinalLLMInput) -> str:
    lines = [
        f"Original drink: {final_input.original_drink_name}",
        f"Method: {final_input.method}",
        f"Customer said: {final_input.user_prompt}",
        "",
    ]

    if final_input.method == "similar" and final_input.neighbors:
        lines.append("Similar drinks found:")
        for n in final_input.neighbors:
            flavor_list = ", ".join(n.drink.flavors.keys()) or "(no flavors resolved)"
            lines.append(f"  - {n.name}: flavors [{flavor_list}]")

    elif final_input.method == "modify" and final_input.assignment:
        a = final_input.assignment
        lines.append(f"Change made: {a.what_changed}")
        if a.matched_name:
            lines.append(f"The modified drink matches a known menu drink: {a.matched_name}")

    return "\n".join(lines)


async def generate_response(final_input: FinalLLMInput) -> str:
    system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
    user_message = _build_user_message(final_input)

    response = _client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()
