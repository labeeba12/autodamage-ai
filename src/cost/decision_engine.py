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

# Car model price multipliers based on Pakistani market parts pricing
CAR_PRICE_MULTIPLIERS = {
    "suzuki_mehran":      0.7,
    "suzuki_alto":        0.8,
    "suzuki_cultus":      0.85,
    "suzuki_swift":       0.9,
    "suzuki_wagonr":      0.85,
    "suzuki_highroof":    0.8,
    "suzuki_kyber":       0.85,
    "honda_city":         1.0,
    "honda_civic":        1.2,
    "toyota_corolla":     1.1,
    "toyota_vigo":        1.4,
    "toyota_aqua":        1.3,
    "toyota_passo":       1.0,
    "toyota_vitz":        1.0,
    "toyota_hiace":       1.3,
    "daihatsu_mira":      0.85,
    "daihatsu_core":      0.85,
}

def get_car_multiplier(car_model: str) -> float:
    """Returns price multiplier based on car model."""
    if not car_model:
        return 1.0
    key = car_model.lower().replace(" ", "_")
    for k, v in CAR_PRICE_MULTIPLIERS.items():
        if k in key:
            return v
    return 1.0


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
    action:              str,
    car_model:           str = None
) -> dict:
    """
    Applies severity multiplier and car model multiplier to base price.
    Replace action adds 30% premium over repair.
    """
    car_mult = get_car_multiplier(car_model)

    min_cost = int(base_price["min_price"] * severity_multiplier * car_mult)
    max_cost = int(base_price["max_price"] * severity_multiplier * car_mult)

    if action == "replace":
        min_cost = int(min_cost * 1.3)
        max_cost = int(max_cost * 1.3)

    logger.info(f"Car multiplier applied: {car_model} → {car_mult}x")

    return {
        "min_cost": min_cost,
        "max_cost": max_cost
    }


def process_damage_costs(matched_results: list[dict], car_model: str = None) -> list[dict]:
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

        # Get base price from CSV
        base_price = price_lookup.get_price(part_name, damage_type)

        # Decide repair or replace
        action = decide_action(
            base_action    = base_price["action"],
            severity_label = severity_label,
            area_ratio     = area_ratio
        )

        # Calculate final cost
        final_cost = calculate_final_cost(
            base_price          = base_price,
            severity_multiplier = severity_mult,
            action              = action,
            car_model           = car_model
        )

        # Add to result
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
