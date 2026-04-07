import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Color Definitions
# Each paint color mapped to: HSV range (hue, saturation, value), paint_code (matches paint_codes.csv), display name
COLOR_PROFILES = [
    {
        "name":       "White",
        "paint_code": "solid",
        "lower":      np.array([0,   0,   180]),
        "upper":      np.array([180, 40,  255]),
    },
    {
        "name":       "Black",
        "paint_code": "solid",
        "lower":      np.array([0,   0,   0  ]),
        "upper":      np.array([180, 60,  60 ]),
    },
    {
        "name":       "Silver",
        "paint_code": "metallic",
        "lower":      np.array([0,   0,   100]),
        "upper":      np.array([180, 40,  179]),
    },
    {
        "name":       "Red",
        "paint_code": "solid",
        "lower":      np.array([0,   100, 100]),
        "upper":      np.array([10,  255, 255]),
    },
    {
        "name":       "Red",
        "paint_code": "solid",
        "lower":      np.array([160, 100, 100]),
        "upper":      np.array([180, 255, 255]),
    },
    {
        "name":       "Blue",
        "paint_code": "metallic",
        "lower":      np.array([100, 100, 50 ]),
        "upper":      np.array([130, 255, 255]),
    },
    {
        "name":       "Dark Blue",
        "paint_code": "metallic",
        "lower":      np.array([100, 50,  20 ]),
        "upper":      np.array([130, 255, 100]),
    },
    {
        "name":       "Green",
        "paint_code": "solid",
        "lower":      np.array([40,  50,  50 ]),
        "upper":      np.array([80,  255, 255]),
    },
    {
        "name":       "Yellow",
        "paint_code": "solid",
        "lower":      np.array([20,  100, 100]),
        "upper":      np.array([35,  255, 255]),
    },
    {
        "name":       "Orange",
        "paint_code": "solid",
        "lower":      np.array([10,  100, 100]),
        "upper":      np.array([20,  255, 255]),
    },
    {
        "name":       "Brown/Beige",
        "paint_code": "solid",
        "lower":      np.array([10,  30,  80 ]),
        "upper":      np.array([30,  120, 200]),
    },
    {
        "name":       "Pearl White",
        "paint_code": "pearl",
        "lower":      np.array([0,   0,   200]),
        "upper":      np.array([180, 25,  255]),
    },
]


def detect_paint_color(image_path: str) -> dict:
    """
    Detects the dominant paint color of the car in the image.

    Strategy:
        1. Load image and convert to HSV
        2. Focus on center region (likely car body)
        3. Match against color profiles
        4. Return best matching color + paint code

    Returns:
        {
            "color_name":  "Silver",
            "paint_code":  "metallic",
            "confidence":  0.82,
            "hex_color":   "#C0C0C0"
        }
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Could not load image for color detection: {image_path}")
        return _default_color()

    h, w = img.shape[:2]

    # Focus on center region (Avoids background, sky, road)
    margin_h = int(h * 0.2)
    margin_w = int(w * 0.2)
    center   = img[margin_h:h-margin_h, margin_w:w-margin_w]

    # Convert to HSV for better color matching
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)

    total_pixels = center.shape[0] * center.shape[1]
    best_match   = None
    best_count   = 0

    # Match against color profiles
    color_counts = {}

    for profile in COLOR_PROFILES:
        mask  = cv2.inRange(hsv, profile["lower"], profile["upper"])
        count = int(np.sum(mask > 0))

        name = profile["name"]
        if name not in color_counts:
            color_counts[name] = {"count": 0, "paint_code": profile["paint_code"]}
        color_counts[name]["count"] += count

    # Find dominant color
    for name, data in color_counts.items():
        if data["count"] > best_count:
            best_count = data["count"]
            best_match = {"name": name, "paint_code": data["paint_code"]}

    if best_match is None or best_count < total_pixels * 0.05:
        logger.warning("Could not detect paint color confidently — using default")
        return _default_color()

    confidence = min(round(best_count / total_pixels, 4), 1.0)
    hex_color  = _get_hex_color(best_match["name"])

    result = {
        "color_name":  best_match["name"],
        "paint_code":  best_match["paint_code"],
        "confidence":  confidence,
        "hex_color":   hex_color
    }

    logger.info(
        f"Paint color detected: {result['color_name']} "
        f"({result['paint_code']}) — "
        f"confidence: {result['confidence']:.2%}"
    )

    return result


def _default_color() -> dict:
    """Returns default solid white when detection fails."""
    return {
        "color_name": "Unknown",
        "paint_code": "solid",
        "confidence": 0.0,
        "hex_color":  "#CCCCCC"
    }


def _get_hex_color(color_name: str) -> str:
    """Returns approximate hex color for display in frontend."""
    hex_map = {
        "White":       "#F5F5F5",
        "Pearl White": "#FAFAF0",
        "Black":       "#1A1A1A",
        "Silver":      "#C0C0C0",
        "Red":         "#CC0000",
        "Blue":        "#0044CC",
        "Dark Blue":   "#002288",
        "Green":       "#006600",
        "Yellow":      "#FFCC00",
        "Orange":      "#FF6600",
        "Brown/Beige": "#AA8855",
    }
    return hex_map.get(color_name, "#CCCCCC")