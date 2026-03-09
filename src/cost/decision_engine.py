from src.utils.logger import get_logger
from src.cost.price_lookup import PriceLookup
from config import (
    SEVERITY_MULTIPLIERS,
    REPLACE_THRESHOLD,
    CURRENCY
)

logger = get_logger(__name__)

# Load pricing table once
price_lookup = PriceLookup()


def decide_action(
    base_action:    str,
    severity_label: str,
    area_ratio:     float
) -> str:
    """
    Determines whether to repair or replace a part.

    Logic:
        - If CSV says replace → always replace
        - If damage covers more than REPLACE_THRESHOLD of image → replace
        - If severity is severe → replace
        - Otherwise → repair
    """
    if base_action == "replace":
        return "replace"

    if area_ratio > REPLACE_THRESHOLD:
        logger.info(
            f"Upgrading to replace — area ratio "
            f"{area_ratio:.2%} exceeds threshold {REPLACE_THRESHOLD:.2%}"
        )
        return "replace"

    if severity_label == "severe":
        return "replace"

    return "repair"


def calculate_final_cost(
    base_price:          dict,
    severity_multiplier: float,
    action:              str
) -> dict:
    """
    Applies severity multiplier to base price.
    Replace action adds 30% premium over repair.

    Returns:
        {
            "min_cost": int,
            "max_cost": int,
        }
    """
    min_cost = int(base_price["min_price"] * severity_multiplier)
    max_cost = int(base_price["max_price"] * severity_multiplier)

    # Replace costs more than repair
    if action == "replace":
        min_cost = int(min_cost * 1.3)
        max_cost = int(max_cost * 1.3)

    return {
        "min_cost": min_cost,
        "max_cost": max_cost
    }


def process_damage_costs(matched_results: list[dict]) -> list[dict]:
    """
    Main function — processes all matched damage-part pairs
    and adds cost estimates to each one.

    Input:  matched results from severity_calc (with severity fields)
    Output: same list with added cost fields:
        - base_price    : raw price from CSV
        - action        : "repair" or "replace"
        - min_cost      : minimum estimated cost in PKR
        - max_cost      : maximum estimated cost in PKR
    """
    processed = []

    for item in matched_results:
        damage_type      = item["damage_type"]
        part_name        = item["part_name"]
        severity_label   = item["severity_label"]
        severity_mult    = item["severity_multiplier"]
        area_ratio       = item["area_ratio"]

        # STEP 1: Get base price from CSV
        base_price = price_lookup.get_price(part_name, damage_type)

        # STEP 2: Decide repair or replace
        action = decide_action(
            base_action    = base_price["action"],
            severity_label = severity_label,
            area_ratio     = area_ratio
        )

        # STEP 3: Calculate final cost
        final_cost = calculate_final_cost(
            base_price          = base_price,
            severity_multiplier = severity_mult,
            action              = action
        )

        # STEP 4: Add to result
        item["action"]   = action
        item["min_cost"] = final_cost["min_cost"]
        item["max_cost"] = final_cost["max_cost"]

        logger.info(
            f"{damage_type} on {part_name} → "
            f"{action.upper()} | "
            f"{CURRENCY} {final_cost['min_cost']:,} – "
            f"{final_cost['max_cost']:,}"
        )

        processed.append(item)

    return processed


def calculate_total_cost(processed_results: list[dict]) -> dict:
    """
    Sums up all individual costs into a total estimate.

    Returns:
        {
            "total_min": int,
            "total_max": int,
            "currency":  "PKR"
        }
    """
    total_min = sum(item["min_cost"] for item in processed_results)
    total_max = sum(item["max_cost"] for item in processed_results)

    logger.info(
        f"Total estimate: {CURRENCY} "
        f"{total_min:,} – {total_max:,}"
    )

    return {
        "total_min": total_min,
        "total_max": total_max,
        "currency":  CURRENCY
    }
