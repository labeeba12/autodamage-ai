import pandas as pd
import os
from src.utils.logger import get_logger
from config import PAINT_CODES_PATH

logger = get_logger(__name__)


class PaintEstimator:
    """
    Loads paint_codes.csv and calculates additional
    paint costs based on paint type.
    """

    def __init__(self):
        if not os.path.exists(PAINT_CODES_PATH):
            raise FileNotFoundError(
                f"Paint codes file not found at: {PAINT_CODES_PATH}"
            )
        self.df = pd.read_csv(PAINT_CODES_PATH)
        logger.info(f"Paint codes loaded — {len(self.df)} paint types")

    def get_paint_cost(self, paint_code: str = "solid") -> dict:
        """
        Returns additional paint cost for a given paint type.

        Args:
            paint_code: "solid", "metallic", "pearl", "matte", "candy"

        Returns:
            {
                "paint_type": "Metallic",
                "min_price":  5000,
                "max_price":  12000
            }
        """
        paint_code = paint_code.strip().lower()

        row = self.df[self.df["paint_code"].str.lower() == paint_code]

        if row.empty:
            logger.warning(
                f"Paint code '{paint_code}' not found — "
                f"using solid as default"
            )
            row = self.df[self.df["paint_code"] == "solid"]

        result = {
            "paint_type": str(row["paint_type"].values[0]),
            "min_price":  int(row["min_price"].values[0]),
            "max_price":  int(row["max_price"].values[0])
        }

        logger.info(
            f"Paint cost: {result['paint_type']} → "
            f"PKR {result['min_price']:,} – {result['max_price']:,}"
        )
        return result

    def apply_paint_cost(
        self,
        min_cost:   int,
        max_cost:   int,
        paint_code: str = "solid"
    ) -> dict:
        """
        Adds paint cost to existing repair cost.

        Returns:
            {
                "min_cost": int,
                "max_cost": int,
                "paint_type": str
            }
        """
        paint = self.get_paint_cost(paint_code)

        return {
            "min_cost":   min_cost + paint["min_price"],
            "max_cost":   max_cost + paint["max_price"],
            "paint_type": paint["paint_type"]
        }