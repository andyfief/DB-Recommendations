"""
Audit logger for the DrinkAdvisor pipeline.

Writes one JSON record per request to logs/pipeline_audit.jsonl (newline-delimited JSON).
Each record captures all inputs and outputs at every pipeline step so the full
decision chain can be reconstructed for debugging.

Usage (from the router):
    log = AuditLog(user_input)
    log.step1(intent, drink_obj)
    log.step2_similar(neighbors)          # similarity path
    log.step2_modify(raw_llm_json, result) # modification path
    log.step3_assignment(result)
    log.step4(response_text)
    log.flush()                           # writes the record
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.models import AssignmentResult, DrinkObject, IntentResult, ModifierResult, NeighborResult

_LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "pipeline_audit.jsonl"

# Also mirror pipeline events to stderr via standard logging so they show in
# the uvicorn terminal without needing to open the file.
_logger = logging.getLogger("drinkadvisor.pipeline")
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    _logger.addHandler(_handler)


def _drink_dict(d: DrinkObject) -> dict:
    return d.model_dump()


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    return obj


def _log_fired_rules(logger: logging.Logger, label: str, fired: dict) -> None:
    """
    Mirror of DrinkBuilder's print_detailed_output() fired-rule formatting,
    written to the audit logger instead of stdout.
    """
    logger.debug("%s fired rules:", label)

    classify = fired.get("classify", [])
    logger.debug("  [Classify]")
    if classify:
        for r in classify:
            logger.debug(
                "    pass %s | pri %s | %s | triggers=%s",
                r.get("pass_num", "?"), r.get("priority"), r.get("description"), sorted(r.get("triggers", [])),
            )
    else:
        logger.debug("    (none)")

    profile = fired.get("profile", [])
    logger.debug("  [Profile]")
    if profile:
        for r in profile:
            logger.debug(
                "    pass %s | pri %s | %s | triggers=%s | profile=%s",
                r.get("pass_num", "?"), r.get("priority"), r.get("description"),
                sorted(r.get("triggers", [])), r.get("payload", {}).get("profile"),
            )
    else:
        logger.debug("    (none)")

    assign = fired.get("assign", [])
    logger.debug("  [Assign]")
    if assign:
        for r in assign:
            p = r.get("payload", {})
            logger.debug(
                "    pass %s | pri %s | %s | triggers=%s | role=%s | items=%s",
                r.get("pass_num", "?"), r.get("priority"), r.get("description"),
                sorted(r.get("triggers", [])), p.get("role"), p.get("items"),
            )
    else:
        logger.debug("    (none)")

    quantity = fired.get("quantity", [])
    logger.debug("  [Quantity]")
    if quantity:
        for r in quantity:
            p = r.get("payload", {})
            logger.debug(
                "    pass %s | pri %s | %s | triggers=%s | target=%s | value=%s | mode=%s",
                r.get("pass_num", "?"), r.get("priority"), r.get("description"),
                sorted(r.get("triggers", [])), p.get("target"), p.get("value"), p.get("mode"),
            )
    else:
        logger.debug("    (none)")

    modifier = fired.get("modifier", [])
    logger.debug("  [Modifier]")
    if modifier:
        for r in modifier:
            p = r.get("payload", {})
            logger.debug(
                "    pri %s | %s | triggers=%s | target=%s | op=%s | value=%s | qty=%s",
                r.get("priority"), r.get("description"),
                sorted(r.get("triggers", [])), p.get("target"), p.get("operation"),
                p.get("value"), p.get("quantity", "N/A"),
            )
    else:
        logger.debug("    (none)")


class AuditLog:
    """Accumulates pipeline step data for one request, then flushes to disk."""

    def __init__(self, user_input: str) -> None:
        self._record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_input": user_input,
        }
        _logger.info("─" * 60)
        _logger.info("USER INPUT: %s", user_input)

    # ── Step 1 ────────────────────────────────────────────────────────────────

    def step1(self, intent: IntentResult, drink_obj: DrinkObject) -> None:
        self._record["step1_intent"] = {
            "drink_name": intent.drink_name,
            "tokens": intent.tokens,
            "method": intent.method,
            "confidence": intent.confidence,
            "error": intent.error,
        }
        self._record["step1_drink_object"] = _drink_dict(drink_obj)

        _logger.info(
            "STEP 1 — drink_name=%r  tokens=%s  method=%r  confidence=%.2f",
            intent.drink_name,
            intent.tokens,
            intent.method,
            intent.confidence,
        )
        _logger.info("STEP 1 — drink object: %s", json.dumps(_drink_dict(drink_obj), indent=2))

    def step1_fired_rules(self, engine_result: dict) -> None:
        """Log the fired rules from the engine for the primary drink resolution."""
        fired = engine_result.get("fired_rules", {})
        self._record["step1_fired_rules"] = _make_serializable(fired)
        _log_fired_rules(_logger, "STEP 1 ENGINE", fired)

    def step1_not_found(self, user_input: str) -> None:
        self._record["step1_intent"] = {"error": "DRINK_NOT_FOUND"}
        _logger.warning("STEP 1 — DRINK_NOT_FOUND for input: %r", user_input)

    # ── Step 2a: similarity ───────────────────────────────────────────────────

    def step2_similar(self, neighbors: list[NeighborResult], engine_results: list[dict]) -> None:
        neighbor_data = [
            {
                "name": n.name,
                "drink_object": _drink_dict(n.drink),
                "fired_rules": _make_serializable(er.get("fired_rules", {})),
            }
            for n, er in zip(neighbors, engine_results)
        ]
        self._record["step2_similar_neighbors"] = neighbor_data

        _logger.info("STEP 2a — %d nearest neighbors:", len(neighbors))
        for n, er in zip(neighbors, engine_results):
            flavors = list(n.drink.flavors.keys())
            _logger.info("  • %s  flavors=%s", n.name, flavors)
            _log_fired_rules(_logger, f"  STEP 2a ENGINE [{n.name}]", er.get("fired_rules", {}))

    # ── Step 2b: modification ─────────────────────────────────────────────────

    def step2_modify(self, raw_llm_response: str, result: ModifierResult) -> None:
        self._record["step2_modify_llm_raw"] = raw_llm_response
        self._record["step2_modify_result"] = {
            "modified_drink_object": _drink_dict(result.modified_drink),
            "change_description": result.change_description,
        }

        _logger.info("STEP 2b — modifier LLM raw response:\n%s", raw_llm_response)
        _logger.info("STEP 2b — change description: %s", result.change_description)
        _logger.info("STEP 2b — modified drink object: %s", json.dumps(_drink_dict(result.modified_drink), indent=2))

    # ── Step 3: assignment check ──────────────────────────────────────────────

    def step3_assignment(self, result: AssignmentResult) -> None:
        self._record["step3_assignment"] = {
            "matched_name": result.matched_name,
            "what_changed": result.what_changed,
        }

        if result.matched_name:
            _logger.info("STEP 3 — matched named drink: %r", result.matched_name)
        else:
            _logger.info("STEP 3 — no match for modified drink object in assignments")

    # ── Step 4: final response ────────────────────────────────────────────────

    def step4(self, response_text: str) -> None:
        self._record["step4_final_response"] = response_text
        _logger.info("STEP 4 — final response:\n%s", response_text)

    def step_error(self, step: str, error: str) -> None:
        self._record.setdefault("errors", []).append({"step": step, "error": error})
        _logger.error("ERROR at %s: %s", step, error)

    # ── Flush ─────────────────────────────────────────────────────────────────

    def flush(self) -> None:
        try:
            _LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._record) + "\n")
        except Exception as exc:
            _logger.error("Failed to write audit log: %s", exc)
