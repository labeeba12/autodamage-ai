import pandas as pd
import os
from src.utils.logger import get_logger
from config import BASELINE_PRICES_PATH

logger = get_logger(__name__)


class PriceLookup:
    """
    Loads the pricing CSV once and provides fast lookups.
    Usage:
        lookup = PriceLookup()
        price  = lookup.get_price("Front-bumper", "scratch")
    """

    def __init__(self):
        if not os.path.exists(BASELINE_PRICES_PATH):
            raise FileNotFoundError(
                f"Pricing table not found at: {BASELINE_PRICES_PATH}"
            )

        self.df = pd.read_csv(BASELINE_PRICES_PATH)
        logger.info(
            f"Pricing table loaded — "
            f"{len(self.df)} entries from {BASELINE_PRICES_PATH}"
        )

    def get_price(self, part_name: str, damage_type: str) -> dict | None:
        """
        Fetches base price for a specific part + damage combination.

        Returns:
            {
                "min_price": 5000,
                "max_price": 15000,
                "action":    "repair"   # or "replace"
            }
            or None if combination not found in CSV
        """
        # Normalize inputs to avoid case mismatch
        part_name   = part_name.strip().lower()
        damage_type = damage_type.strip().lower()

        # Filter dataframe
        row = self.df[
            (self.df["part"].str.lower()        == part_name) &
            (self.df["damage_type"].str.lower() == damage_type)
        ]

        if row.empty:
            logger.warning(
                f"No price found for: {part_name} + {damage_type} "
                f"— using default pricing"
            )
            return self._get_default_price(damage_type)

        result = {
            "min_price": int(row["min_price"].values[0]),
            "max_price": int(row["max_price"].values[0]),
            "action":    str(row["repair_or_replace"].values[0])
        }

        logger.info(
            f"Price found: {part_name} + {damage_type} → "
            f"PKR {result['min_price']:,} – {result['max_price']:,} "
            f"({result['action']})"
        )
        return result

    def _get_default_price(self, damage_type: str) -> dict:
        """
        Fallback prices when exact combination not in CSV.
        Prevents the system from crashing on unknown combinations.
        """
        defaults = {
            "scratch":       {"min_price": 3000,  "max_price": 10000, "action": "repair"},
            "dent":          {"min_price": 5000,  "max_price": 20000, "action": "repair"},
            "crack":         {"min_price": 8000,  "max_price": 25000, "action": "repair"},
            "glass shatter": {"min_price": 15000, "max_price": 50000, "action": "replace"},
            "lamp broken":   {"min_price": 10000, "max_price": 35000, "action": "replace"},
            "tire flat":     {"min_price": 5000,  "max_price": 15000, "action": "replace"},
        }
        return defaults.get(
            damage_type.lower(),
            {"min_price": 5000, "max_price": 20000, "action": "repair"}
        )

    def get_all_parts(self) -> list[str]:
        """Returns list of all parts in the pricing table."""
        return self.df["part"].unique().tolist()

    def get_all_damage_types(self) -> list[str]:
        """Returns list of all damage types in the pricing table."""
        return self.df["damage_type"].unique().tolist()