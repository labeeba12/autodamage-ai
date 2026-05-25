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

    def predict(self, feature_matrix):
        """
        Predicts total repair cost using the tuned Log-Transformed XGBoost model.
        """
        try:
            # Validate feature_matrix
            if not isinstance(feature_matrix, np.ndarray):
                logger.error("Feature matrix must be a numpy array.")
                return 0

            # Ensure each row is a valid array
            for row in feature_matrix:
                if not isinstance(row, np.ndarray):
                    logger.error("Each row in the feature matrix must be a numpy array.")
                    return 0

            total_predicted = 0
            
            if feature_matrix is None or len(feature_matrix) == 0:
             return 0

            # Loop through each row in the matrix (each detected damage)
            for row in feature_matrix:
                X = row.reshape(1, -1)
                
                log_prediction = float(self.model.predict(X)[0])
                
                actual_pkr = np.expm1(log_prediction)
                
                total_predicted += max(0, actual_pkr)

            final_total = int(total_predicted)
            logger.info(f"XGBoost Total Prediction (Log-Decoded): PKR {final_total:,}")
            return final_total

        except Exception as e:
            logger.error(f"Error during cost estimation: {str(e)}")
            return 0
                        
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