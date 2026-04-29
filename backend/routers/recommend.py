from __future__ import annotations

from fastapi import APIRouter

from backend.models import FinalLLMInput, RecommendResponse, UserRequest
from backend.pipeline.assignment import check_assignment
from backend.pipeline.audit_log import AuditLog
from backend.pipeline.final import generate_response
from backend.pipeline.intent import DrinkNotFoundError, IntentParseError, extract_intent
from backend.pipeline.modifier import ModifierParseError, modify_drink
from backend.pipeline.similarity import find_similar

router = APIRouter()


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: UserRequest) -> RecommendResponse:
    log = AuditLog(req.user_input)

    # ── Step 1: intent extraction ─────────────────────────────────────────────
    try:
        intent, drink_obj, engine_result = await extract_intent(req.user_input)
    except DrinkNotFoundError:
        log.step1_not_found(req.user_input)
        log.flush()
        return RecommendResponse(
            response="Sorry, we couldn't find that drink in our system. Try naming a Dutch Bros drink like 'Golden Eagle', 'Annihilator', or 'Caramelizer'.",
            error="DRINK_NOT_FOUND",
        )
    except (IntentParseError, Exception) as exc:
        log.step_error("step1", str(exc))
        log.flush()
        return RecommendResponse(
            response="Something went wrong while understanding your request. Please try again.",
            error=str(exc),
        )

    log.step1(intent, drink_obj)
    log.step1_fired_rules(engine_result)

    final_input = FinalLLMInput(
        original_drink_name=intent.drink_name,
        method=intent.method,
        user_prompt=req.user_input,
    )

    try:
        if intent.method == "similar":
            # ── Step 2a: similarity ───────────────────────────────────────────
            neighbors, neighbor_engine_results = await find_similar(intent.drink_name)
            log.step2_similar(neighbors, neighbor_engine_results)
            final_input.neighbors = neighbors

        elif intent.method == "modify":
            # ── Step 2b: modification ─────────────────────────────────────────
            modifier_result = await modify_drink(drink_obj, req.user_input)
            log.step2_modify(modifier_result.llm_raw, modifier_result)

            # ── Step 3: assignment check ──────────────────────────────────────
            assignment = await check_assignment(
                modifier_result.modified_drink,
                modifier_result.change_description,
            )
            log.step3_assignment(assignment)
            final_input.assignment = assignment

        # ── Step 4: final response ────────────────────────────────────────────
        response_text = await generate_response(final_input)
        log.step4(response_text)
        log.flush()
        return RecommendResponse(response=response_text)

    except ModifierParseError as exc:
        log.step_error("step2b", str(exc))
        log.flush()
        return RecommendResponse(
            response="Something went wrong while modifying the drink. Please try rephrasing your request.",
            error=str(exc),
        )
    except Exception as exc:
        log.step_error("pipeline", str(exc))
        log.flush()
        return RecommendResponse(
            response="An unexpected error occurred. Please try again.",
            error=str(exc),
        )
