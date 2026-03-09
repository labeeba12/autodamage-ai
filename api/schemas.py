from pydantic import BaseModel
from typing import List, Optional


# INDIVIDUAL DAMAGE ITEM
# One damage detection with its cost
class DamageItem(BaseModel):
    damage_type:    str     # "scratch", "dent", "crack", etc.
    part_name:      str     # "Front-bumper", "Hood", etc.
    severity_label: str     # "minor", "moderate", "severe"
    action:         str     # "repair" or "replace"
    confidence:     float   # YOLO confidence score 0.0 – 1.0
    pixel_area:     int     # number of pixels in damage mask
    area_ratio:     float   # damage area / total image area
    overlap_ratio:  float   # how much damage overlaps with part
    min_cost:       int     # minimum repair cost in PKR
    max_cost:       int     # maximum repair cost in PKR


# FULL ESTIMATION RESPONSE
# Returned to user after analysis
class EstimationResponse(BaseModel):
    success:          bool
    total_min:        int           # total minimum cost in PKR
    total_max:        int           # total maximum cost in PKR
    currency:         str           # "PKR"
    overall_severity: str           # worst damage severity
    total_damages:    int           # number of damages found
    damages:          List[DamageItem]  # list of individual damages
    message:           Optional[str] = None
    result_image_path: Optional[str] = None
    paint_color:       Optional[str] = None
    paint_code:        Optional[str] = None
    paint_hex:         Optional[str] = None
    paint_cost_min:    Optional[int] = None
    paint_cost_max:    Optional[int] = None


# ERROR RESPONSE
# Returned when something goes wrong
class ErrorResponse(BaseModel):
    success: bool  = False
    error:   str              # human readable error message


#   HEALTH CHECK RESPONSE
#   Returned by /health endpoint
class HealthResponse(BaseModel):
    status:  str  # "ok"
    version: str  # API version
    message: str  # welcome message