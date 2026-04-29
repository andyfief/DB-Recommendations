# Extracted from DrinkBuilder/src/engine.py.
# Kept: multi-phase rule engine (classify, profile, quantity, assign, modifier)
# and process_order() — the core function that maps token_quantities to a drink object.
# Paths updated for backend/lib/ location.

import sqlite3
import json
from typing import Set, Dict, List, Optional, Tuple


def load_rules_by_type(conn: sqlite3.Connection, rule_type: str) -> List[Dict]:
    """Load all active rules of a specific type with their triggers."""
    cur = conn.cursor()

    cur.execute("""
        SELECT r.id, r.description, r.priority, r.payload_json
        FROM rules r
        WHERE r.rule_type = ? AND r.active = 1
        ORDER BY r.priority ASC
    """, (rule_type,))

    rules = []
    for rule_id, description, priority, payload_json in cur.fetchall():
        cur.execute("""
            SELECT t.name
            FROM rule_tokens rt
            JOIN tokens t ON rt.token_id = t.id
            WHERE rt.rule_id = ? AND rt.role = 'trigger'
        """, (rule_id,))
        triggers = {row[0] for row in cur.fetchall()}

        results = set()
        if rule_type == 'classify':
            cur.execute("""
                SELECT t.name
                FROM rule_tokens rt
                JOIN tokens t ON rt.token_id = t.id
                WHERE rt.rule_id = ? AND rt.role = 'result'
            """, (rule_id,))
            results = {row[0] for row in cur.fetchall()}

        rule = {
            'id': rule_id,
            'description': description,
            'priority': priority,
            'triggers': triggers,
            'results': results,
            'payload': json.loads(payload_json) if payload_json else {}
        }

        rules.append(rule)

    return rules


def classify_phase(detected: Set[str], rules: List[Dict], fired_descriptions: Set[str], pass_num: int, max_iterations: int = 10) -> Tuple[Set[str], List[Dict]]:
    detected = detected.copy()
    newly_fired = []

    for iteration in range(max_iterations):
        changed = False

        for rule in rules:
            if rule['description'] in fired_descriptions:
                continue

            if rule['triggers'].issubset(detected):
                new_tokens = rule['results'] - detected
                if new_tokens:
                    detected |= new_tokens
                    rule_copy = rule.copy()
                    rule_copy['pass_num'] = pass_num
                    newly_fired.append(rule_copy)
                    fired_descriptions.add(rule['description'])
                    changed = True

        if not changed:
            break

    return detected, newly_fired


def profile_phase(detected: Set[str], rules: List[Dict]) -> Tuple[Optional[str], List[Dict]]:
    PROFILE_ORDER = {'1111': 1, '1112': 2, '1123': 3, '1234': 4, '2222': 5}

    matching_rules = [r for r in rules if r['triggers'].issubset(detected)]

    if not matching_rules:
        return None, []

    set_rules = [r for r in matching_rules if r['payload'].get('mode', 'set') == 'set']
    min_rules = [r for r in matching_rules if r['payload'].get('mode') == 'min']
    max_rules = [r for r in matching_rules if r['payload'].get('mode') == 'max']

    current_profile = None

    if set_rules:
        set_rules.sort(key=lambda r: r['priority'], reverse=True)
        current_profile = set_rules[0]['payload']['profile']

    for rule in max_rules:
        rule_profile = rule['payload']['profile']
        if current_profile is None:
            current_profile = rule_profile
        elif PROFILE_ORDER[rule_profile] > PROFILE_ORDER[current_profile]:
            current_profile = rule_profile

    for rule in min_rules:
        rule_profile = rule['payload']['profile']
        if current_profile is None:
            current_profile = rule_profile
        elif PROFILE_ORDER[rule_profile] < PROFILE_ORDER[current_profile]:
            current_profile = rule_profile

    return current_profile, matching_rules


def quantity_phase(
    detected: Set[str],
    token_quantities: Dict[str, float],
    profile: Optional[str],
    rules: List[Dict],
    fired_descriptions: Set[str],
    current_quantities: Dict[str, float],
    pass_num: int
) -> Tuple[Dict[str, float], List[Dict]]:
    newly_fired = []

    for rule in rules:
        if rule['description'] in fired_descriptions:
            continue

        payload = rule['payload']

        ignore_on = payload.get('ignore_on', [])
        if any(token in token_quantities for token in ignore_on):
            continue

        if not rule['triggers'].issubset(detected):
            continue

        required_profile = payload.get('requires_profile')
        if required_profile and required_profile != profile:
            continue

        target = payload['target']
        value = payload['value']
        mode = payload.get('mode', 'set')

        if mode == 'set':
            current_quantities[target] = value
        elif mode == 'max':
            if target in current_quantities:
                current_quantities[target] = min(current_quantities[target], value)
            else:
                current_quantities[target] = value
        elif mode == 'min':
            if target in current_quantities:
                current_quantities[target] = max(current_quantities[target], value)
            else:
                current_quantities[target] = value

        rule_copy = rule.copy()
        rule_copy['pass_num'] = pass_num
        newly_fired.append(rule_copy)
        fired_descriptions.add(rule['description'])

    return current_quantities, newly_fired


def assign_phase(
    detected: Set[str],
    token_quantities: Dict[str, float],
    rules: List[Dict],
    fired_descriptions: Set[str],
    current_assignments: Dict[str, any],
    pass_num: int
):
    newly_fired = []

    QUANTITY_ROLES = {"topping", "milk_adds", "flavor"}

    for rule in rules:
        if rule['description'] in fired_descriptions:
            continue

        if not rule['triggers'].issubset(detected):
            continue

        payload = rule['payload']
        role = payload['role']
        items = payload['items']

        if role not in current_assignments:
            current_assignments[role] = {} if role in QUANTITY_ROLES else []

        for item in items:
            if isinstance(item, dict):
                name = item["name"]
                qty = item.get("quantity", 1)

                if role in QUANTITY_ROLES:
                    current_assignments[role][name] = (
                        current_assignments[role].get(name, 0) + qty
                    )
                else:
                    current_assignments[role].extend([name] * qty)

            else:
                if role == "flavor":
                    if item in token_quantities:
                        current_assignments[role][item] = token_quantities[item]
                    elif item not in current_assignments[role]:
                        current_assignments[role][item] = "default"

                elif role in QUANTITY_ROLES:
                    current_assignments[role][item] = (
                        current_assignments[role].get(item, 0) + 1
                    )
                else:
                    if item not in current_assignments[role]:
                        current_assignments[role].append(item)

        rule_copy = rule.copy()
        rule_copy['pass_num'] = pass_num
        newly_fired.append(rule_copy)
        fired_descriptions.add(rule['description'])

    return current_assignments, newly_fired


def modifier_phase(
    detected: Set[str],
    token_quantities: Dict[str, float],
    rules: List[Dict],
    quantities: Dict[str, float],
    assignments: Dict[str, any],
    modifier_allowed_tokens: Set[str]
):
    fired_rules = []

    for role in ('topping', 'milk_adds', 'flavor'):
        if role not in assignments:
            assignments[role] = {}
        elif isinstance(assignments[role], list):
            assignments[role] = {item: 1 for item in assignments[role]}

    for rule in rules:
        if len(rule['triggers']) > 1:
            if not rule['triggers'].issubset(detected):
                continue
        else:
            if not rule['triggers'].issubset(modifier_allowed_tokens):
                continue

        trigger_quantity = min(
            (token_quantities.get(t, 1) for t in rule['triggers']),
            default=1,
        )

        payload = rule['payload']

        fired_rules.append({
            "id": rule["id"],
            "description": rule["description"],
            "priority": rule["priority"],
            "triggers": rule["triggers"],
            "payload": payload,
            "pass_num": 0
        })

        target = payload['target']
        operation = payload['operation']
        value = payload.get('value')
        quantity = payload.get('quantity', 1)

        if target in ('topping', 'milk_adds', 'flavor'):
            bucket = assignments[target]

            topping_name = (
                value[0] if isinstance(value, list) and value else value
            )

            if not topping_name:
                continue

            if operation == 'add':
                bucket[topping_name] = bucket.get(topping_name, 0) + quantity * trigger_quantity

            elif operation == 'mul':
                if topping_name in bucket:
                    bucket[topping_name] *= quantity

            elif operation == 'remove':
                bucket.pop(topping_name, None)

            elif operation == 'override':
                bucket[topping_name] = quantity * trigger_quantity

        elif target in ("shots", "scoops"):
            if operation == 'add':
                quantities[target] = quantities.get(target, 0) + value * trigger_quantity

            elif operation == 'mul':
                quantities[target] = quantities.get(target, 0) * value

            elif operation == 'set':
                quantities[target] = value * trigger_quantity

            elif operation == 'override':
                quantities[target] = value * trigger_quantity

        else:
            if target not in assignments or not isinstance(assignments[target], list):
                assignments[target] = []

            if operation == 'remove':
                if value in assignments[target]:
                    assignments[target].remove(value)

            elif operation == 'override':
                assignments[target] = [value]

            elif operation == 'add':
                assignments[target].append(value)

    return quantities, assignments, fired_rules


def process_order(token_quantities: Dict[str, float], conn: sqlite3.Connection) -> Dict:
    """
    Main entry point. Maps token_quantities dict to a structured drink object.

    Args:
        token_quantities: {token_name: quantity} e.g. {"vanilla": 1, "latte": 1}
        conn: SQLite connection to the seeded rules DB

    Returns dict with keys: flavor, toppings, milk, base, coffee, shots, scoops,
    size, style, mix_line, milk_adds, profile, detected_tokens, fired_rules
    """
    if isinstance(token_quantities, list):
        token_quantities = {t: 1 for t in token_quantities}
    elif isinstance(token_quantities, dict):
        token_quantities = token_quantities.copy()
    else:
        raise TypeError(f"Invalid tokens type: {type(token_quantities)}")

    detected = set(token_quantities.keys())
    modifier_allowed_tokens = detected.copy()

    classify_rules = load_rules_by_type(conn, 'classify')
    profile_rules = load_rules_by_type(conn, 'profile')
    quantity_rules = load_rules_by_type(conn, 'quantity')
    assign_rules = load_rules_by_type(conn, 'assign')
    modifier_rules = load_rules_by_type(conn, 'modifier')

    fired_descriptions = set()

    all_fired = {
        'classify': [],
        'profile': [],
        'quantity': [],
        'assign': [],
        'modifier': []
    }

    profile = None
    quantities = {}
    assignments = {}

    max_passes = 5

    for pass_num in range(max_passes):
        changed = False

        detected_before_classify = detected.copy()
        detected, classify_fired = classify_phase(detected, classify_rules, fired_descriptions, pass_num)
        if classify_fired:
            all_fired['classify'].extend(classify_fired)
            classification_added = detected - detected_before_classify
            modifier_allowed_tokens |= classification_added
            changed = True

        profile, profile_rules_evaluated = profile_phase(detected, profile_rules)
        if profile_rules_evaluated:
            all_fired['profile'] = [
                {**r, 'pass_num': 0} for r in profile_rules_evaluated
            ]

        assignments, assign_fired = assign_phase(detected, token_quantities, assign_rules, fired_descriptions, assignments, pass_num)
        if assign_fired:
            all_fired['assign'].extend(assign_fired)

            new_ingredients = set()
            for rule in assign_fired:
                for item in rule['payload']['items']:
                    if isinstance(item, dict):
                        new_ingredients.add(item["name"])
                    else:
                        new_ingredients.add(item)

            if not new_ingredients.issubset(detected):
                detected |= new_ingredients
                changed = True

        quantities, quantity_fired = quantity_phase(detected, token_quantities, profile, quantity_rules, fired_descriptions, quantities, pass_num)
        if quantity_fired:
            all_fired['quantity'].extend(quantity_fired)
            changed = True

        if not changed:
            break

    quantities, assignments, modifier_fired = modifier_phase(detected, token_quantities, modifier_rules, quantities, assignments, modifier_allowed_tokens)
    if modifier_fired:
        all_fired["modifier"].extend(modifier_fired)

    payload = {
        'original_tokens': token_quantities,
        'size': assignments.get('size', []),
        'style': assignments.get('style', []),
        'mix_line': assignments.get('mix_line', 'default'),
        'base': assignments.get('base', []),
        'milk': assignments.get('milk', []),
        'flavor': assignments.get('flavor', {}),
        'toppings': assignments.get('topping', {}),
        'milk_adds': assignments.get('milk_adds', {}),
        'coffee': assignments.get('coffee', []),
        'shots': quantities.get('shots'),
        'scoops': quantities.get('scoops'),
        'profile': profile,
        'detected_tokens': sorted(detected),
        'fired_rules': all_fired
    }

    return payload
