from pydantic import BaseModel
from typing import List, Optional


# Individual damage item (One damage detection with its cost)
class DamageItem(BaseModel):
    damage_type:    str     
    part_name:      str     
    severity_label: str     
    action:         str     
    confidence:     float   
    pixel_area:     int     
    area_ratio:     float   
    overlap_ratio:  float   
    min_cost:       int     
    max_cost:       int     


# Full estimation response (Returned to user after analysis)
class EstimationResponse(BaseModel):
    success:          bool
    total_min:        int           
    total_max:        int           
    currency:         str           # "PKR"
    overall_severity: str           
    total_damages:    int           
    damages:          List[DamageItem]  
    message:           Optional[str] = None
    result_image_path: Optional[str] = None
    paint_color:       Optional[str] = None
    paint_code:        Optional[str] = None
    paint_hex:         Optional[str] = None
    paint_cost_min:    Optional[int] = None
    paint_cost_max:    Optional[int] = None
    heatmap_path:      Optional[str] = None
    car_model:             Optional[str]  = None
    car_model_confidence:  Optional[float] = None
    car_model_top3:        Optional[list] = None


# Error response
class ErrorResponse(BaseModel):
    success: bool  = False
    error:   str              


# Health check response
class HealthResponse(BaseModel):
    status:  str  # "ok"
    version: str  # API version
    message: str  # welcome message