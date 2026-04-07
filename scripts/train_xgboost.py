import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from scripts.generate_training_data import generate_training_data

# Paths
TRAINING_DATA_PATH = "data/xgboost_training_data.csv"
MODEL_OUTPUT_PATH  = "models/xgboost_cost_model.json"

# Features used for training (must match feature_builder.py)
FEATURE_COLS = [
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
    """
    Full training pipeline:
    1. Generate synthetic data if not exists
    2. Load and prepare data
    3. Train XGBoost model
    4. Evaluate on test set
    5. Save model
    """

    # Generate data if needed
    if not os.path.exists(TRAINING_DATA_PATH):
        print("Training data not found — generating now...")
        generate_training_data(TRAINING_DATA_PATH)
    else:
        print(f"Training data found at {TRAINING_DATA_PATH}")

    # Load data
    print("\nLoading training data...")
    df = pd.read_csv(TRAINING_DATA_PATH)
    print(f"Loaded {len(df)} samples")

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42
    )
    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Train XGBoost
    print("\nTraining XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 3,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0
    )

    model.fit(
        X_train, y_train,
        eval_set              = [(X_test, y_test)],
        verbose               = 50
    )

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"  Mean Absolute Error : PKR {mae:,.0f}")
    print(f"  R² Score            : {r2:.4f}")
    print(f"{'='*40}")

    # Sample predictions vs actual
    print("\nSample Predictions vs Actual:")
    print(f"{'Actual':>12} {'Predicted':>12} {'Diff':>10}")
    print("-" * 36)
    for actual, pred in zip(y_test[:8], y_pred[:8]):
        diff = abs(actual - pred)
        print(f"PKR {actual:>8,.0f} PKR {pred:>8,.0f} PKR {diff:>6,.0f}")

    # Feature importance
    print("\nFeature Importance:")
    features    = FEATURE_COLS
    importances = model.feature_importances_
    for feat, imp in sorted(
        zip(features, importances),
        key=lambda x: x[1],
        reverse=True
    ):
        bar = "█" * int(imp * 50)
        print(f"  {feat:<20} {bar} {imp:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    model.save_model(MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    print(f"Ready to use in cost_regressor.py")


if __name__ == "__main__":
    train_model()