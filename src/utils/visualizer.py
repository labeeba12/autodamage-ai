import cv2
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Color Map
# Each damage type gets a unique color, Format: BGR (Blue, Green, Red)
DAMAGE_COLORS = {
    "scratch":       (0,   255, 255),   # Yellow
    "dent":          (0,   165, 255),   # Orange
    "crack":         (0,   0,   255),   # Red
    "glass shatter": (255, 0,   0  ),   # Blue
    "lamp broken":   (255, 0,   255),   # Magenta
    "tire flat":     (0,   255, 0  ),   # Green
    "unknown":       (128, 128, 128),   # Gray
}

PART_COLOR  = (0, 255, 0)    # Green for part outlines
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.6
FONT_THICKNESS = 2
MASK_ALPHA  = 0.4            # transparency of mask overlay


def draw_damage_masks(
    image_path:      str,
    matched_results: list[dict],
    output_path:     str
) -> str:
    """
    Draws colored damage masks and labels on the image.
    Saves annotated image to output_path.

    Returns: output_path of saved image
    """
    # Load original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    overlay = img.copy()
    h, w    = img.shape[:2]

    for item in matched_results:
        damage_type = item["damage_type"]
        part_name   = item["part_name"]
        severity    = item.get("severity_label", "unknown")
        min_cost    = item.get("min_cost", 0)
        max_cost    = item.get("max_cost", 0)
        mask        = item.get("damage_mask")

        # Get color for this damage type
        color = DAMAGE_COLORS.get(
            damage_type.lower(),
            DAMAGE_COLORS["unknown"]
        )

        # Draw Filled Mask
        if mask is not None:
            # Resize mask to image size if needed
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Apply colored overlay
            overlay[mask] = color

            # Draw Label
            # Find centroid of mask for label placement
            ys, xs     = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))

                # Label text
                label = f"{damage_type} | {part_name}"
                cost  = f"PKR {min_cost:,}-{max_cost:,}"
                sev   = f"[{severity.upper()}]"

                # Draw background rectangle for label
                (lw, lh), _ = cv2.getTextSize(
                    label, FONT, FONT_SCALE, FONT_THICKNESS
                )
                cv2.rectangle(
                    overlay,
                    (cx - 5, cy - lh - 10),
                    (cx + lw + 5, cy + 5),
                    (0, 0, 0),
                    -1
                )

                # Draw label lines
                cv2.putText(
                    overlay, label,
                    (cx, cy - 5),
                    FONT, FONT_SCALE, color,
                    FONT_THICKNESS
                )
                cv2.putText(
                    overlay, cost,
                    (cx, cy + 20),
                    FONT, 0.5, (255, 255, 255),
                    1
                )
                cv2.putText(
                    overlay, sev,
                    (cx, cy + 38),
                    FONT, 0.5, color,
                    1
                )

    # Blend original image with overlay
    annotated = cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0)

    # Save Output Image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    logger.info(f"Annotated image saved: {output_path}")

    return output_path


def draw_summary_bar(
    image_path:       str,
    total_min:        int,
    total_max:        int,
    overall_severity: str,
    output_path:      str
) -> str:
    """
    Adds a summary bar at the bottom of the image showing:
    - Total cost estimate
    - Overall severity
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]

    # Add black bar at bottom
    bar_height = 60
    bar        = np.zeros((bar_height, w, 3), dtype=np.uint8)

    # Summary text
    summary = (
        f"TOTAL ESTIMATE: PKR {total_min:,} - {total_max:,}"
        f"  |  SEVERITY: {overall_severity.upper()}"
    )

    cv2.putText(
        bar, summary,
        (15, 40),
        FONT, 0.7,
        (255, 255, 255),
        2
    )

    # Stack bar below image
    final = np.vstack([img, bar])

    cv2.imwrite(output_path, final)
    logger.info(f"Summary bar added: {output_path}")

    return output_path