import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Calculates Intersection over Union between two binary masks.
    IoU = (area of overlap) / (area of union)
    Returns a value between 0.0 (no overlap) and 1.0 (perfect overlap)
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union        = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def calculate_overlap_ratio(damage_mask: np.ndarray, part_mask: np.ndarray) -> float:
    """
    Calculates what percentage of the damage falls within the part.
    This is more useful than IoU for our case because:
        - A small scratch on a large hood should still be linked to the hood
        - IoU would be very low in that case (small scratch vs large hood)
        - Overlap ratio correctly shows 100% of scratch is ON the hood
    """
    intersection = np.logical_and(damage_mask, part_mask).sum()
    damage_area  = damage_mask.sum()

    if damage_area == 0:
        return 0.0

    return float(intersection / damage_area)


def find_damage_on_parts(
    parts_detections:  list[dict],
    damage_detections: list[dict],
    overlap_threshold: float = 0.3
) -> list[dict]:
    """
    Main function — matches each damage to the car part it sits on.

    Logic:
        For each damage mask:
            → Check overlap with every part mask
            → Find the part with highest overlap ratio
            → If overlap > threshold, link damage to that part
            → If no overlap found, label as 'Unknown Part'

    Returns a list of matched results:
        [
            {
                "damage_type":    "scratch",
                "part_name":      "Front-bumper",
                "confidence":     0.87,
                "overlap_ratio":  0.91,
                "pixel_area":     4521,
                "area_ratio":     0.032,
                "damage_mask":    np.ndarray,
                "part_mask":      np.ndarray,
            },
            ...
        ]
    """
    matched_results = []

    if not parts_detections:
        logger.warning("No car parts detected — cannot map damage to parts")

    if not damage_detections:
        logger.warning("No damage detected in image")
        return matched_results

    for damage in damage_detections:
        best_match      = None
        best_overlap    = 0.0

        # Compare this damage against every detected part
        for part in parts_detections:
            overlap = calculate_overlap_ratio(
                damage["mask"],
                part["mask"]
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_match   = part

        # MATCH FOUND
        if best_match and best_overlap >= overlap_threshold:
            matched_results.append({
                "damage_type":   damage["class_name"],
                "part_name":     best_match["class_name"],
                "confidence":    damage["confidence"],
                "overlap_ratio": round(best_overlap, 4),
                "pixel_area":    damage["pixel_area"],
                "area_ratio":    damage["area_ratio"],
                "damage_mask":   damage["mask"],
                "part_mask":     best_match["mask"],
            })
            logger.info(
                f"Matched: {damage['class_name']} "
                f"ON {best_match['class_name']} "
                f"(overlap: {best_overlap:.2%})"
            )

        # NO MATCH FOUND
        else:
            matched_results.append({
                "damage_type":   damage["class_name"],
                "part_name":     "Unknown Part",
                "confidence":    damage["confidence"],
                "overlap_ratio": round(best_overlap, 4),
                "pixel_area":    damage["pixel_area"],
                "area_ratio":    damage["area_ratio"],
                "damage_mask":   damage["mask"],
                "part_mask":     None,
            })
            logger.warning(
                f"No part match for {damage['class_name']} "
                f"(best overlap: {best_overlap:.2%})"
            )

    logger.info(f"Overlap engine completed — {len(matched_results)} damage-part pairs found")
    return matched_results

