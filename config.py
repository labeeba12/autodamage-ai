import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Model paths
PARTS_MODEL_PATH  = os.getenv("PARTS_MODEL_PATH",  "models/car_parts_seg.pt")
DAMAGE_MODEL_PATH = os.getenv("DAMAGE_MODEL_PATH", "models/car_damage_seg.pt")
XGBOOST_MODEL_PATH = os.getenv("XGBOOST_MODEL_PATH", "models/xgboost_cost_model.json")
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "models/car_classifier.pt")

# Data paths
BASELINE_PRICES_PATH = "data/baseline_prices.csv"
PAINT_CODES_PATH     = "data/paint_codes.csv"

# Detection settings
CONFIDENCE_THRESH  = float(os.getenv("CONFIDENCE_THRESH", 0.40))
IMAGE_SIZE         = 1024       # inference image size
MIN_SEGMENT_AREA   = 0.001      # minimum mask area as % of image (filters noise)

# Image validation settings
MIN_IMAGE_WIDTH    = 224        # pixels
MIN_IMAGE_HEIGHT   = 224        # pixels
BLUR_THRESHOLD     = 50.0       # laplacian variance — below this = too blurry

# SEVERITY THRESHOLDS
# Based on damage area as % of total image
SEVERITY_THRESHOLDS = {
    "minor":    (0.000, 0.020),   # 0% – 2% of image
    "moderate": (0.020, 0.080),   # 2% – 8% of image
    "severe":   (0.080, 1.000),   # 8%+ of image
}

# Cost settings
SEVERITY_MULTIPLIERS = {
    "minor":    1.0,
    "moderate": 1.5,
    "severe":   2.2,
}

# If damage covers more than this % of a part → replace instead of repair
REPLACE_THRESHOLD = 0.35

# Currency label for output report
CURRENCY = "PKR"

# API settings
API_TITLE       = "AutoDamage AI"
API_VERSION     = "1.0.0"
API_DESCRIPTION = "Car Damage Detection and Cost Estimation API"
HOST            = "0.0.0.0"
PORT            = 8000

# Logging
LOG_DIR    = "logs"
LOG_LEVEL  = "INFO"