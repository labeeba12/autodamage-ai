import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Paths
TRAINING_DATA_PATH = "data/xgboost_training_data.csv"
MODEL_OUTPUT_PATH  = "models/xgboost_cost_model.json"

# Features
FEATURE_COLS = [
    "car_model_id",
    "car_multiplier",
    "damage_type_id",
    "part_id",
    "severity_id",
    "action_id",
    "area_ratio",
    "overlap_ratio",
    "confidence"
]

TARGET_COL = "actual_cost"


def train_model():

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------
    print(f"Loading dataset from: {TRAINING_DATA_PATH}")

    df = pd.read_csv(TRAINING_DATA_PATH)

    print(f"Loaded {len(df)} samples")

    # Remove invalid rows
    df = df[df[TARGET_COL] > 0]

    # Features / Target
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # ---------------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    print(f"Train Samples : {len(X_train)}")
    print(f"Test Samples  : {len(X_test)}")

    # ---------------------------------------------------------
    # LOG TRANSFORM TARGET
    # ---------------------------------------------------------
    y_train_log = np.log1p(y_train)
    y_test_log  = np.log1p(y_test)

    # ---------------------------------------------------------
    # XGBOOST MODEL
    # ---------------------------------------------------------
    print("\nTraining XGBoost Regressor...\n")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",

        n_estimators=1200,
        learning_rate=0.02,

        max_depth=6,
        min_child_weight=2,

        subsample=0.85,
        colsample_bytree=0.85,

        reg_alpha=0.3,
        reg_lambda=1.5,

        gamma=0.1,

        random_state=42,
        n_jobs=-1
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------
    model.fit(
        X_train,
        y_train_log,

        eval_set=[(X_test, y_test_log)],

        verbose=50
    )

    # ---------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------
    print("\nEvaluating Model...\n")

    y_pred_log = model.predict(X_test)

    # Convert back from log scale
    y_pred = np.expm1(y_pred_log)

    # Avoid negative predictions
    y_pred = np.maximum(y_pred, 0)

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------
    mae  = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    mape = np.mean(
        np.abs((y_test - y_pred) / y_test)
    ) * 100

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------
    print("=" * 50)

    print(f"MAE   : PKR {mae:,.0f}")
    print(f"RMSE  : PKR {rmse:,.0f}")
    print(f"MAPE  : {mape:.2f}%")
    print(f"R²    : {r2:.4f}")

    print("=" * 50)

    # ---------------------------------------------------------
    # SAMPLE PREDICTIONS
    # ---------------------------------------------------------
    print("\nSample Predictions vs Actual:\n")

    print(f"{'Actual':>12} {'Predicted':>12} {'Diff':>12}")

    print("-" * 42)

    for actual, pred in zip(y_test.values[:10], y_pred[:10]):

        diff = abs(actual - pred)

        print(
            f"PKR {actual:>8,.0f} "
            f"PKR {pred:>8,.0f} "
            f"PKR {diff:>8,.0f}"
        )

    # ---------------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------------
    print("\nFeature Importance:\n")

    importances = model.feature_importances_

    importance_pairs = sorted(
        zip(FEATURE_COLS, importances),
        key=lambda x: x[1],
        reverse=True
    )

    for feat, imp in importance_pairs:

        bar = "█" * int(imp * 50)

        print(f"{feat:<20} {bar} {imp:.4f}")

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------
    os.makedirs(
        os.path.dirname(MODEL_OUTPUT_PATH),
        exist_ok=True
    )

    model.save_model(MODEL_OUTPUT_PATH)

    print(f"\nModel saved to:")
    print(MODEL_OUTPUT_PATH)

    print("\nTraining complete.")


if __name__ == "__main__":
    train_model()