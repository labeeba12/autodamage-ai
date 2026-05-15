import numpy as np
import os
from src.utils.logger import get_logger
from config import XGBOOST_MODEL_PATH

logger = get_logger(__name__)


class CostRegressor:
    """
    XGBoost-based cost predictor.
    Uses trained model if available, falls back to
    rule-based pricing from decision_engine if not.
    """

    def __init__(self):
        self.model       = None
        self.is_available = False
        self._load_model()

    def _load_model(self):
        """
        Attempts to load XGBoost model.
        Fails gracefully if model file not found.
        """
        if not os.path.exists(XGBOOST_MODEL_PATH):
            logger.warning(
                f"XGBoost model not found at {XGBOOST_MODEL_PATH}. "
                f"Falling back to rule-based pricing."
            )
            return

        try:
            import xgboost as xgb
            self.model        = xgb.XGBRegressor()
            self.model.load_model(XGBOOST_MODEL_PATH)
            self.is_available = True
            logger.info("XGBoost cost model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")

    def predict(self, feature_vector: np.ndarray) -> dict | None:
        """
        Predicts repair cost from feature vector.

        Args:
            feature_vector: numpy array of shape (7,)
                from feature_builder.py

        Returns:
            {
                "predicted_cost": int,
                "source": "xgboost"
            }
            or None if model not available
        """
        if not self.is_available or self.model is None:
            return None

        try:
            # XGBoost expects 2D array
            X = feature_vector.reshape(1, -1)
            predicted = float(self.model.predict(X)[0])
            predicted = max(0, int(predicted))

            logger.info(f"XGBoost prediction: PKR {predicted:,}")

            return {
                "predicted_cost": predicted,
                "source":         "xgboost"
            }

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return None
        
        @staticmethod
    def calculate_accuracy_metrics(predicted_cost, actual_cost):
        """
        Ye function error ko percentage (%) mein badalta hai.
        """
        if actual_cost <= 0:
            return 0.0
            
        difference = abs(predicted_cost - actual_cost)
        mape = (difference / actual_cost) * 100
        return round(mape, 2)

    def predict_batch(
        self,
        feature_matrix: np.ndarray
    ) -> list[int] | None:
        """
        Predicts costs for multiple damages at once.

        Args:
            feature_matrix: numpy array of shape (n, 7)

        Returns:
            list of predicted costs or None
        """
        if not self.is_available or self.model is None:
            return None

        try:
            predictions = self.model.predict(feature_matrix)
            results     = [max(0, int(p)) for p in predictions]
            logger.info(f"Batch prediction: {len(results)} costs predicted")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return None