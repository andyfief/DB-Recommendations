from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UserRequest(BaseModel):
    user_input: str


class IntentResult(BaseModel):
    drink_name: str | None
    tokens: dict[str, int]  # matched token dict passed directly to process_order
    method: Literal["similar", "modify"] | None
    confidence: float
    error: str | None = None


class DrinkObject(BaseModel):
    flavors: dict[str, float | str]
    toppings: dict[str, float]
    milk: list[str]
    base: list[str]
    coffee: list[str]
    shots: int | None = None
    scoops: int | None = None

    @classmethod
    def from_engine_result(cls, result: dict) -> "DrinkObject":
        """Build a DrinkObject from a raw process_order() result dict."""
        return cls(
            flavors=result.get("flavor", {}),
            toppings=result.get("toppings", {}),
            milk=result.get("milk", []),
            base=result.get("base", []),
            coffee=result.get("coffee", []),
            shots=result.get("shots"),
            scoops=result.get("scoops"),
        )


class ModifierResult(BaseModel):
    modified_drink: DrinkObject
    change_description: str
    llm_raw: str = ""  # full JSON string returned by the modifier LLM, for auditing


class AssignmentResult(BaseModel):
    matched_name: str | None
    what_changed: str


class NeighborResult(BaseModel):
    name: str
    drink: DrinkObject


class FinalLLMInput(BaseModel):
    original_drink_name: str
    method: Literal["similar", "modify"]
    user_prompt: str
    # Similarity path
    neighbors: list[NeighborResult] | None = None
    # Modification path
    assignment: AssignmentResult | None = None


class RecommendResponse(BaseModel):
    response: str
    error: str | None = None
