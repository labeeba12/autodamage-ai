import numpy as np
from src.utils.logger import get_logger
from config import SEVERITY_THRESHOLDS, SEVERITY_MULTIPLIERS

logger = get_logger(__name__)


def get_severity_label(area_ratio: float) -> str:
    """
    Classifies damage severity based on area ratio.
    area_ratio = damage pixels / total image pixels

    Returns: "minor", "moderate", or "severe"
    """
    for label, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= area_ratio < high:
            return label
    return "severe"


def get_severity_multiplier(severity_label: str) -> float:
    """
    Returns the cost multiplier for a given severity label.
    minor    → 1.0  (base price)
    moderate → 1.5  (50% more)
    severe   → 2.2  (120% more)
    """
    return SEVERITY_MULTIPLIERS.get(severity_label, 1.0)


def calculate_severity(matched_results: list[dict]) -> list[dict]:
    """
    Adds severity label and multiplier to each matched damage-part pair.

    Input:  list of matched results from overlap_engine
    Output: same list with added severity fields:
        - severity_label    : "minor" / "moderate" / "severe"
        - severity_multiplier: float used for cost calculation
    """
    for item in matched_results:
        area_ratio   = item["area_ratio"]
        label        = get_severity_label(area_ratio)
        multiplier   = get_severity_multiplier(label)

        item["severity_label"]        = label
        item["severity_multiplier"]   = multiplier

        logger.info(
            f"{item['damage_type']} on {item['part_name']} "
            f"→ Severity: {label.upper()} "
            f"(area: {area_ratio:.2%}, multiplier: {multiplier}x)"
        )

    return matched_results


def get_overall_severity(matched_results: list[dict]) -> str:
    """
    Returns the overall severity of the entire car damage.
    Based on the worst single damage found.

    Logic:
        if any damage is severe   → overall is severe
        if any damage is moderate → overall is moderate
        otherwise                 → overall is minor
    """
    if not matched_results:
        return "none"

    labels = [item.get("severity_label", "minor") for item in matched_results]

    if "severe" in labels:
        return "severe"
    elif "moderate" in labels:
        return "moderate"
    else:
        return "minor"