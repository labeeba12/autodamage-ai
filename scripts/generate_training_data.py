import pandas as pd
import numpy as np
import random

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# REALISTIC 2026 PKR BASE PRICES
# Source: PakWheels forums, Karachi/Lahore workshop quotes,
#         market research (March-June 2024, adjusted for 2026)
# Format: (part, damage_type) → (min_repair, max_repair, min_replace, max_replace)
# ─────────────────────────────────────────────────────────────
BASE_PRICES = {
    # FRONT BUMPER
    ("front-bumper", "dent"):          (8000,  25000,  18000,  55000),
    ("front-bumper", "scratch"):       (5000,  15000,  12000,  32000),
    ("front-bumper", "crack"):         (6000,  20000,  15000,  45000),

    # BACK BUMPER
    ("back-bumper",  "dent"):          (7000,  22000,  16000,  50000),
    ("back-bumper",  "scratch"):       (4000,  13000,  10000,  28000),
    ("back-bumper",  "crack"):         (5000,  18000,  13000,  40000),

    # HOOD
    ("hood",         "dent"):          (10000, 30000,  22000,  65000),
    ("hood",         "scratch"):       (6000,  18000,  14000,  38000),
    ("hood",         "crack"):         (8000,  25000,  18000,  55000),

    # FRONT DOOR
    ("front-door",   "dent"):          (9000,  28000,  20000,  60000),
    ("front-door",   "scratch"):       (6000,  18000,  12000,  35000),
    ("front-door",   "crack"):         (7000,  22000,  15000,  45000),

    # BACK DOOR
    ("back-door",    "dent"):          (8000,  25000,  18000,  55000),
    ("back-door",    "scratch"):       (5000,  16000,  11000,  32000),
    ("back-door",    "crack"):         (6000,  20000,  14000,  42000),

    # FENDER
    ("fender",       "dent"):          (7000,  22000,  15000,  45000),
    ("fender",       "scratch"):       (4000,  13000,  9000,   28000),
    ("fender",       "crack"):         (5000,  18000,  12000,  38000),

    # QUARTER PANEL
    ("quarter-panel","dent"):          (8000,  25000,  18000,  55000),
    ("quarter-panel","scratch"):       (5000,  16000,  11000,  32000),
    ("quarter-panel","crack"):         (6000,  20000,  14000,  42000),

    # TRUNK
    ("trunk",        "dent"):          (9000,  28000,  20000,  60000),
    ("trunk",        "scratch"):       (6000,  18000,  13000,  35000),
    ("trunk",        "crack"):         (7000,  22000,  15000,  45000),

    # ROOF
    ("roof",         "dent"):          (12000, 38000,  25000,  75000),
    ("roof",         "scratch"):       (8000,  25000,  18000,  55000),
    ("roof",         "crack"):         (10000, 32000,  22000,  65000),

    # WINDSHIELD
    ("windshield",       "glass shatter"): (20000, 60000,  45000, 120000),
    ("windshield",       "crack"):         (8000,  25000,  20000,  60000),
    ("back-windshield",  "glass shatter"): (15000, 45000,  35000,  90000),
    ("back-windshield",  "crack"):         (6000,  20000,  15000,  45000),

    # HEADLIGHT / TAIL LIGHT
    ("headlight",    "lamp broken"):   (12000, 35000,  28000,  80000),
    ("headlight",    "crack"):         (5000,  18000,  12000,  35000),
    ("tail-light",   "lamp broken"):   (8000,  25000,  18000,  55000),
    ("tail-light",   "crack"):         (4000,  14000,  9000,   28000),

    # GRILLE
    ("grille",       "crack"):         (4000,  15000,  10000,  30000),
    ("grille",       "dent"):          (5000,  18000,  12000,  35000),

    # ROCKER PANEL
    ("rocker-panel", "dent"):          (6000,  20000,  13000,  40000),
    ("rocker-panel", "scratch"):       (3000,  12000,  8000,   25000),

    # MIRROR
    ("mirror",       "crack"):         (3000,  10000,  7000,   20000),
    ("mirror",       "dent"):          (2000,  8000,   5000,   15000),

    # WHEEL / TIRE
    ("front-wheel",  "tire flat"):     (3000,  12000,  8000,   25000),
    ("back-wheel",   "tire flat"):     (3000,  12000,  8000,   25000),
    ("front-wheel",  "dent"):          (4000,  15000,  10000,  30000),
    ("back-wheel",   "dent"):          (4000,  15000,  10000,  30000),
}

# ─────────────────────────────────────────────────────────────
# CAR MODEL MULTIPLIERS
# Based on parts availability and workshop rates in Pakistan 2026
# ─────────────────────────────────────────────────────────────
CAR_MODELS = {
    # Budget (cheapest parts, most available)
    "Suzuki_Mehran":        0.65,
    "Daihatsu_Mira":         0.70,
    "Daihatsu_Core":         0.72,
    "Suzuki_alto_2007":     0.75,

    # Economy
    "Suzuki_alto_2019":     0.80,
    "Suzuki_highroof":      0.78,
    "Suzuki_kyber":         0.82,
    "Toyota_Passo":         0.85,
    "Toyota_Vitz":          0.85,
    "Toyota_Aqua":          0.88,

    # Mid-range
    "Suzuki_cultus_2018":   0.88,
    "Suzuki_cultus_2019":   0.90,
    "Suzuki_wagonR_2015":   0.88,
    "Suzuki_swift":         0.90,
    "Honda_city_2000":      0.90,
    "Honda_City_aspire":    0.95,
    "Toyota_corolla_2007":  0.95,

    # Upper mid-range
    "Toyota_corolla_2011":  1.05,
    "Toyota_corolla_2016":  1.15,
    "Honda_civic_2007":     1.10,
    "Honda_civic_2018":     1.25,

    # Premium / High cost
    "Toyota_Hiace_2012":    1.30,
    "Toyota_Vigo":          1.45,
}

# ─────────────────────────────────────────────────────────────
# SEVERITY MULTIPLIERS
# ─────────────────────────────────────────────────────────────
SEVERITY_MULTIPLIERS = {
    "minor":    1.0,
    "moderate": 1.5,
    "severe":   2.2,
}

# ─────────────────────────────────────────────────────────────
# ENCODE CATEGORIES
# ─────────────────────────────────────────────────────────────
DAMAGE_TYPES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
PARTS = sorted(set(p for p, _ in BASE_PRICES.keys()))
SEVERITIES = ["minor", "moderate", "severe"]
ACTIONS = ["repair", "replace"]

damage_type_id  = {d: i for i, d in enumerate(DAMAGE_TYPES)}
part_id_map     = {p: i for i, p in enumerate(PARTS)}
severity_id_map = {s: i for i, s in enumerate(SEVERITIES)}
action_id_map   = {a: i for i, a in enumerate(ACTIONS)}
car_model_id    = {c: i for i, c in enumerate(CAR_MODELS.keys())}

# ─────────────────────────────────────────────────────────────
# GENERATE DATASET
# ─────────────────────────────────────────────────────────────
records = []

for car_model, car_mult in CAR_MODELS.items():
    for (part, damage), (min_r, max_r, min_rep, max_rep) in BASE_PRICES.items():
        for severity in SEVERITIES:
            sev_mult = SEVERITY_MULTIPLIERS[severity]

            # Determine action
            if severity == "severe":
                action = "replace"
            elif damage in ["glass shatter", "lamp broken", "tire flat"]:
                action = "replace"
            elif severity == "minor" and damage == "scratch":
                action = "repair"
            else:
                action = random.choice(["repair", "replace"])

            # Calculate realistic cost
            if action == "repair":
                base_min = min_r
                base_max = max_r
            else:
                base_min = min_rep
                base_max = max_rep

            # Apply multipliers
            cost_min = base_min * sev_mult * car_mult
            cost_max = base_max * sev_mult * car_mult

            # Add realistic noise (±10%)
            noise_factor = random.uniform(0.90, 1.10)
            actual_cost = ((cost_min + cost_max) / 2) * noise_factor
            actual_cost = max(1500, int(actual_cost))

            # Area ratio based on severity
            if severity == "minor":
                area_ratio = random.uniform(0.005, 0.02)
            elif severity == "moderate":
                area_ratio = random.uniform(0.02, 0.10)
            else:
                area_ratio = random.uniform(0.10, 0.45)

            # Overlap ratio
            overlap_ratio = random.uniform(0.3, 1.0)

            # Confidence
            confidence = random.uniform(0.55, 0.98)

            records.append({
                # Features
                "car_model":      car_model,
                "car_model_id":   car_model_id[car_model],
                "damage_type":    damage,
                "damage_type_id": damage_type_id.get(damage, 0),
                "part_name":      part,
                "part_id":        part_id_map[part],
                "severity":       severity,
                "severity_id":    severity_id_map[severity],
                "action":         action,
                "action_id":      action_id_map[action],
                "area_ratio":     round(area_ratio, 6),
                "overlap_ratio":  round(overlap_ratio, 4),
                "confidence":     round(confidence, 4),
                # Target
                "actual_cost":    actual_cost,
            })

# Generate more samples with random combinations
for _ in range(3000):
    car_model = random.choice(list(CAR_MODELS.keys()))
    car_mult  = CAR_MODELS[car_model]
    part_damage = random.choice(list(BASE_PRICES.keys()))
    part, damage = part_damage
    severity = random.choice(SEVERITIES)
    sev_mult = SEVERITY_MULTIPLIERS[severity]
    min_r, max_r, min_rep, max_rep = BASE_PRICES[part_damage]

    if severity == "severe" or damage in ["glass shatter", "lamp broken", "tire flat"]:
        action = "replace"
        base_min, base_max = min_rep, max_rep
    else:
        action = random.choice(["repair", "replace"])
        base_min = min_r if action == "repair" else min_rep
        base_max = max_r if action == "repair" else max_rep

    cost_min = base_min * sev_mult * car_mult
    cost_max = base_max * sev_mult * car_mult
    noise    = random.uniform(0.88, 1.12)
    actual_cost = max(1500, int(((cost_min + cost_max) / 2) * noise))

    if severity == "minor":
        area_ratio = random.uniform(0.005, 0.02)
    elif severity == "moderate":
        area_ratio = random.uniform(0.02, 0.10)
    else:
        area_ratio = random.uniform(0.10, 0.45)

    records.append({
        "car_model":      car_model,
        "car_model_id":   car_model_id[car_model],
        "damage_type":    damage,
        "damage_type_id": damage_type_id.get(damage, 0),
        "part_name":      part,
        "part_id":        part_id_map[part],
        "severity":       severity,
        "severity_id":    severity_id_map[severity],
        "action":         action,
        "action_id":      action_id_map[action],
        "area_ratio":     round(area_ratio, 6),
        "overlap_ratio":  round(random.uniform(0.3, 1.0), 4),
        "confidence":     round(random.uniform(0.55, 0.98), 4),
        "actual_cost":    actual_cost,
    })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total samples: {len(df)}")
print(f"Car models: {df['car_model'].nunique()}")
print(f"Damage types: {df['damage_type'].nunique()}")
print(f"Parts: {df['part_name'].nunique()}")
print(f"\nCost statistics:")
print(f"  Min: PKR {df['actual_cost'].min():,}")
print(f"  Max: PKR {df['actual_cost'].max():,}")
print(f"  Mean: PKR {df['actual_cost'].mean():,.0f}")
print(f"  Median: PKR {df['actual_cost'].median():,.0f}")
print(f"\nSample rows:")
print(df[["car_model", "damage_type", "part_name", "severity", "action", "actual_cost"]].head(10).to_string())

df.to_csv("data/xgboost_training_data.csv", index=False)
print(f"\nSaved to data/xgboost_training_data.csv")