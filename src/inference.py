from src.utils.logger import get_logger
from src.detection.validator import validate_image, load_image
from src.detection.segmenter import CarSegmenter
from src.logic.overlap_engine import find_damage_on_parts
from src.logic.severity_calc import calculate_severity, get_overall_severity
from src.cost.decision_engine import process_damage_costs, calculate_total_cost
from src.utils.visualizer import create_before_after
from src.utils.color_detector import detect_paint_color
from src.utils.heatmap_generator import generate_heatmap
from src.cost.paint_estimator import PaintEstimator
from config import CURRENCY
import os
import uuid

logger = get_logger(__name__)

# Load models once at startup (not on every request)
segmenter = CarSegmenter()


def run_pipeline(image_path: str) -> dict:
    """
    Main pipeline function.
    Takes an image path and returns a complete damage report.

    Returns:
    {
        "success": True,
        "total_min": 33000,
        "total_max": 52000,
        "currency": "PKR",
        "overall_severity": "moderate",
        "total_damages": 3,
        "damages": [
            {
                "damage_type":    "scratch",
                "part_name":      "Front-bumper",
                "severity_label": "moderate",
                "action":         "repair",
                "confidence":     0.87,
                "pixel_area":     4521,
                "area_ratio":     0.032,
                "overlap_ratio":  0.91,
                "min_cost":       7500,
                "max_cost":       22500,
            },
            ...
        ]
    }
    """

    logger.info(f"Pipeline started for: {image_path}")
    
    # DETECT PAINT COLOR
    paint_info     = detect_paint_color(image_path)
    paint_estimator = PaintEstimator()
    paint_cost     = paint_estimator.get_paint_cost(paint_info["paint_code"])
    logger.info(f"Paint detected: {paint_info['color_name']} → {paint_info['paint_code']}")

    # STEP 1: Validate Image
    is_valid, message = validate_image(image_path)
    if not is_valid:
        logger.error(f"Image validation failed: {message}")
        return {
            "success": False,
            "error":   message
        }
    logger.info("Step 1 complete — Image validated")

    # STEP 2: Run Both YOLO Models
    parts_detections, damage_detections = segmenter.detect_all(image_path)
    logger.info(
        f"Step 2 complete — "
        f"Parts: {len(parts_detections)} | "
        f"Damage: {len(damage_detections)}"
    )

    # No damage found — return early
    if not damage_detections:
        logger.info("No damage detected in image")
        return {
            "success":          True,
            "total_min":        0,
            "total_max":        0,
            "currency":         CURRENCY,
            "overall_severity": "none",
            "total_damages":    0,
            "damages":          [],
            "message":          "No damage detected in this image"
        }

    # STEP 3: Match Damage to Parts
    matched_results = find_damage_on_parts(
        parts_detections,
        damage_detections
    )
    logger.info(
        f"Step 3 complete — "
        f"{len(matched_results)} damage-part pairs matched"
    )

    # STEP 4: Calculate Severity
    matched_results  = calculate_severity(matched_results)
    overall_severity = get_overall_severity(matched_results)
    logger.info(
        f"Step 4 complete — "
        f"Overall severity: {overall_severity.upper()}"
    )

    # STEP 5: Calculate Costs
    processed_results = process_damage_costs(matched_results)
    total_cost        = calculate_total_cost(processed_results)
    logger.info(
        f"Step 5 complete — "
        f"Total: {CURRENCY} "
        f"{total_cost['total_min']:,} – "
        f"{total_cost['total_max']:,}"
    )

    # STEP 6: Generate Before/After Image
    os.makedirs("temp_outputs", exist_ok=True)
    output_filename = f"result_{uuid.uuid4().hex[:8]}.jpg"
    output_path     = os.path.join("temp_outputs", output_filename)

    try:
        create_before_after(
            original_path    = image_path,
            matched_results  = processed_results,
            total_min        = total_cost["total_min"],
            total_max        = total_cost["total_max"],
            overall_severity = overall_severity,
            output_path      = output_path
        )
        logger.info(f"Before/after image saved: {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate before/after image: {e}")
        output_path = None
        
    # STEP 7: Generate Heatmap
    heatmap_filename = f"heatmap_{uuid.uuid4().hex[:8]}.jpg"
    heatmap_path     = os.path.join("temp_outputs", heatmap_filename)

    try:
        generate_heatmap(
            matched_results = processed_results,
            output_path     = heatmap_path
        )
        logger.info(f"Heatmap saved: {heatmap_path}")
    except Exception as e:
        logger.warning(f"Could not generate heatmap: {e}")
        heatmap_path = None

    # STEP 8: Build Clean Output Report
    # Remove numpy masks from output (not JSON serializable)
    damages_output = []
    for item in processed_results:
        damages_output.append({
            "damage_type":    item["damage_type"],
            "part_name":      item["part_name"],
            "severity_label": item["severity_label"],
            "action":         item["action"],
            "confidence":     item["confidence"],
            "pixel_area":     item["pixel_area"],
            "area_ratio":     item["area_ratio"],
            "overlap_ratio":  item["overlap_ratio"],
            "min_cost":       item["min_cost"],
            "max_cost":       item["max_cost"],
        })

    logger.info("Pipeline completed successfully")

    return {
        "success":            True,
        "total_min":          total_cost["total_min"],
        "total_max":          total_cost["total_max"],
        "currency":           CURRENCY,
        "overall_severity":   overall_severity,
        "total_damages":      len(damages_output),
        "damages":            damages_output,
        "result_image_path":  output_path,
        "paint_color":        paint_info["color_name"],
        "paint_code":         paint_info["paint_code"],
        "paint_hex":          paint_info["hex_color"],
        "paint_cost_min":     paint_cost["min_price"],
        "paint_cost_max":     paint_cost["max_price"],
        "heatmap_path":       heatmap_path,
    }