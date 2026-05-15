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

    def predict(self, feature_vector: np.ndarray, actual_cost: float = None) -> dict | None:
        if not self.is_available or self.model is None:
            return None

        try:
            # XGBoost expects 2D array
            X = feature_vector.reshape(1, -1)
            
            # Check Raw prediction 
            raw_prediction = float(self.model.predict(X)[0])
            logger.info(f"DEBUG: Raw XGBoost Output -> {raw_prediction}")

            # Calculate Final cost 
            predicted = max(0, int(raw_prediction))
            logger.info(f"XGBoost prediction: PKR {predicted:,}")

            # MAPE Logic (Accuracy check)
            mape_value = None
            if actual_cost is not None:
                # call function
                mape_value = self.calculate_accuracy_metrics(predicted, actual_cost)

            return {
                "predicted_cost": predicted,
                "source":         "xgboost",
                "mape_error":     f"{mape_value}%" if mape_value is not None else "N/A"
            }

        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return None
        
    @staticmethod
    def calculate_accuracy_metrics(predicted_cost, actual_cost):
        """
        This function converts error to percentage (%) 
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