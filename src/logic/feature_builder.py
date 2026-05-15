import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Encoding maps (Convert string labels to numbers for XGBoost input)
DAMAGE_TYPE_MAP = {
    "scratch":       0,
    "dent":          1,
    "crack":         2,
    "glass shatter": 3,
    "lamp broken":   4,
    "tire flat":     5
}

SEVERITY_MAP = {
    "minor":    0,
    "moderate": 1,
    "severe":   2
}

ACTION_MAP = {
    "repair":  0,
    "replace": 1
}

PART_MAP = {
    "front-bumper":   0,  "back-bumper":    1,
    "front-door":     2,  "back-door":      3,
    "front-wheel":    4,  "back-wheel":     5,
    "front-window":   6,  "back-window":    7,
    "back-windshield":8,  "fender":         9,
    "grille":         10, "headlight":      11,
    "hood":           12, "license-plate":  13,
    "mirror":         14, "quarter-panel":  15,
    "rocker-panel":   16, "roof":           17,
    "tail-light":     18, "trunk":          19,
    "windshield":     20, "unknown part":   21
}


def build_feature_vector(damage_item: dict) -> np.ndarray:
    """
    Converts a single damage item into a numerical feature vector.

    Input:  single damage dict from matched_results
    Output: numpy array of shape (7,)
        [damage_type, part_name, severity, action,
         area_ratio, overlap_ratio, confidence]
    """
    damage_type  = DAMAGE_TYPE_MAP.get(
        damage_item["damage_type"].lower(), -1
    )
    
    clean_part_name = damage_item["part_name"].lower().strip().replace(" ", "-")
    
    part_name = PART_MAP.get(clean_part_name, 21)
    severity     = SEVERITY_MAP.get(
        damage_item.get("severity_label", "minor").lower(), 0
    )
    action       = ACTION_MAP.get(
        damage_item.get("action", "repair").lower(), 0
    )
    area_ratio   = float(damage_item.get("area_ratio",   0.0))
    overlap_ratio= float(damage_item.get("overlap_ratio",0.0))
    confidence   = float(damage_item.get("confidence",   0.0))

    feature_vector = np.array([
        damage_type,
        part_name,
        severity,
        action,
        area_ratio,
        overlap_ratio,
        confidence
    ], dtype=np.float32)

    logger.info(
        f"Feature vector built for "
        f"{damage_item['damage_type']} on "
        f"{damage_item['part_name']}: {feature_vector}"
    )
    return feature_vector


def build_feature_matrix(matched_results: list[dict]) -> np.ndarray:
    """
    Builds feature matrix for all damage items.
    Each row = one damage item.

    Output: numpy array of shape (n_damages, 7)
    """
    if not matched_results:
        return np.empty((0, 7), dtype=np.float32)

    matrix = np.array([
        build_feature_vector(item)
        for item in matched_results
    ])

    logger.info(
        f"Feature matrix built: "
        f"{matrix.shape[0]} damages × "
        f"{matrix.shape[1]} features"
    )
    return matrix