import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import BASELINE_PRICES_PATH

# ENCODING MAPS (must match feature_builder.py)
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
    "front-bumper":    0,  "back-bumper":     1,
    "front-door":      2,  "back-door":       3,
    "front-wheel":     4,  "back-wheel":      5,
    "front-window":    6,  "back-window":     7,
    "back-windshield": 8,  "fender":          9,
    "grille":          10, "headlight":       11,
    "hood":            12, "license-plate":   13,
    "mirror":          14, "quarter-panel":   15,
    "rocker-panel":    16, "roof":            17,
    "tail-light":      18, "trunk":           19,
    "windshield":      20, "unknown part":    21
}

# SEVERITY AREA RANGES
# Realistic area_ratio ranges per severity
SEVERITY_AREA_RANGES = {
    "minor":    (0.001, 0.019),
    "moderate": (0.020, 0.079),
    "severe":   (0.080, 0.400),
}

SEVERITY_MULTIPLIERS = {
    "minor":    1.0,
    "moderate": 1.5,
    "severe":   2.2,
}

SAMPLES_PER_ROW = 100  # generates 100 variations per CSV row


def generate_training_data(output_path: str = "data/xgboost_training_data.csv"):
    """
    Reads baseline_prices.csv and generates synthetic training data
    by simulating realistic variations in:
        - area_ratio      (how large the damage is)
        - overlap_ratio   (how well damage overlaps with part)
        - confidence      (YOLO detection confidence)
        - severity        (minor/moderate/severe)
        - actual_cost     (realistic cost with variation)
    """

    print(f"Loading baseline prices from {BASELINE_PRICES_PATH}...")
    df = pd.read_csv(BASELINE_PRICES_PATH)
    print(f"Loaded {len(df)} base price entries")

    records = []
    np.random.seed(42)  # reproducible results

    for _, row in df.iterrows():
        part        = row["part"].strip().lower()
        damage_type = row["damage_type"].strip().lower()
        min_price   = row["min_price"]
        max_price   = row["max_price"]
        base_action = row["repair_or_replace"].strip().lower()

        part_id        = PART_MAP.get(part, 21)
        damage_type_id = DAMAGE_TYPE_MAP.get(damage_type, -1)

        if damage_type_id == -1:
            continue

        for _ in range(SAMPLES_PER_ROW):

            # RANDOMLY ASSIGN SEVERITY
            severity = np.random.choice(
                ["minor", "moderate", "severe"],
                p=[0.4, 0.4, 0.2]  # minor most common, severe least
            )

            severity_id   = SEVERITY_MAP[severity]
            multiplier    = SEVERITY_MULTIPLIERS[severity]

            # GENERATE REALISTIC AREA RATIO
            area_low, area_high = SEVERITY_AREA_RANGES[severity]
            area_ratio = np.random.uniform(area_low, area_high)

            # GENERATE OVERLAP AND CONFIDENCE
            overlap_ratio = np.random.uniform(0.50, 1.00)
            confidence    = np.random.uniform(0.40, 0.99)

            # DETERMINE ACTION
            if base_action == "replace" or severity == "severe" or area_ratio > 0.35:
                action    = "replace"
                action_id = 1
            else:
                action    = "repair"
                action_id = 0

            # CALCULATE REALISTIC COST
            # Base price × multiplier × random variation
            # Replace adds 30% premium
            base_cost = np.random.uniform(min_price, max_price)
            cost      = base_cost * multiplier

            if action == "replace":
                cost *= 1.3

            # Add ±15% realistic workshop variation
            variation = np.random.uniform(0.85, 1.15)
            cost      = int(cost * variation)
            cost      = max(cost, 500)  # minimum PKR 500

            records.append({
                "damage_type_id": damage_type_id,
                "part_id":        part_id,
                "severity_id":    severity_id,
                "action_id":      action_id,
                "area_ratio":     round(area_ratio,    6),
                "overlap_ratio":  round(overlap_ratio, 4),
                "confidence":     round(confidence,    4),
                "actual_cost":    cost,
                # human readable — not used for training
                "damage_type":    damage_type,
                "part_name":      part,
                "severity":       severity,
                "action":         action,
            })

    # Save to CSV
    output_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(output_df)} training samples")
    print(f"Saved to: {output_path}")
    print(f"\nSample data:")
    print(output_df[["damage_type", "part_name", "severity",
                      "action", "area_ratio", "actual_cost"]].head(10))
    print(f"\nCost statistics:")
    print(output_df["actual_cost"].describe())

    return output_path


if __name__ == "__main__":
    generate_training_data()